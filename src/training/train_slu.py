import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from src.pipeline.dataset import load_splits, get_stag_bifurcation
from src.pipeline.features import extract_spectrogram
from src.models.slu_dnn import CharacterTokenizer, SLUModel

class SpeechIMUKDDataset(Dataset):
    """
    Dataset to load Speech WAV files and corresponding upscaled Accelerometer spectrograms.
    """
    def __init__(self, metadata_rows, dataset_root, upscaler=None, tokenizer=None, max_len=150):
        self.rows = metadata_rows
        self.dataset_root = dataset_root
        self.upscaler = upscaler
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.max_len = max_len
        
        # Load and cache valid samples
        self.valid_samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        print(f"Caching and preprocessing {len(self.rows)} samples...")
        for row in self.rows:
            uuid = row[0]
            duration = float(row[1])
            wav_path_rel = row[2]
            
            # Transcription labels
            transcript = row[4]
            semantic_frame = row[3]
            
            # Target is the semantic frame JSON-like string
            target_text = semantic_frame
            target_tokens = self.tokenizer.encode(target_text)
            
            # Truncate or pad target tokens to max_len
            if len(target_tokens) > self.max_len:
                target_tokens = target_tokens[:self.max_len]
            else:
                target_tokens = target_tokens + [self.tokenizer.pad_id] * (self.max_len - len(target_tokens))
            
            base_dir = os.path.dirname(wav_path_rel)
            wav_path = os.path.join(self.dataset_root, wav_path_rel.replace('./', ''))
            acc_path = os.path.join(self.dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
            gyro_path = os.path.join(self.dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
            
            if not os.path.exists(wav_path) or not os.path.exists(acc_path) or not os.path.exists(gyro_path):
                continue
                
            self.valid_samples.append({
                'uuid': uuid,
                'duration': duration,
                'wav_path': wav_path,
                'acc_path': acc_path,
                'gyro_path': gyro_path,
                'target_tokens': np.array(target_tokens)
            })

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # 1. Load and process Audio (Speech signal)
        # Load audio at 16 kHz
        y, sr = librosa.load(sample['wav_path'], sr=16000)
        
        # Extract speech spectrogram (Mel Spectrogram)
        # To match input dimension we map to 30 bins
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30, n_fft=512, hop_length=320) # 320 hop = 20ms at 16kHz
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max).T # Shape: (frames, 30)
        
        # 2. Load and process IMU (Accelerometer signal)
        # Simulate STAG temporal misalignment and reconstruct Z-axis
        acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
            sample['acc_path'], sample['gyro_path'], sample['duration']
        )
        
        if self.upscaler is not None:
            # Reconstruct upscaled 400 Hz signal
            acc_z_400 = self.upscaler.reconstruct_signal(acc_odd, gyro_even, t_odd, t_even)
        else:
            # Fallback to simple interpolation if upscaler not trained yet
            from scipy.interpolate import interp1d
            acc_z_400 = interp1d(t_odd, acc_odd, kind='linear', fill_value="extrapolate")(t_even)
            
        # Extract features (spectrogram) from upscaled Z-axis signal
        imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
        
        # Ensure audio and imu sequence lengths match the model padding
        max_frames = 300 # Max temporal frames (6 seconds at 50 FPS)
        
        def pad_sequence(seq):
            if seq.shape[0] > max_frames:
                return seq[:max_frames, :]
            else:
                padding = np.zeros((max_frames - seq.shape[0], seq.shape[1]))
                return np.vstack([seq, padding])
                
        speech_feat = pad_sequence(log_mel_spec)
        imu_feat = pad_sequence(imu_spec)
        
        return {
            'speech_feat': torch.FloatTensor(speech_feat),
            'imu_feat': torch.FloatTensor(imu_feat),
            'target_tokens': torch.LongTensor(sample['target_tokens'])
        }

def train_kd_pipeline(metadata_file, dataset_root, upscaler_path, save_dir, epochs=5, batch_size=8, max_samples=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load splits
    train_rows, val_rows, _ = load_splits(metadata_file)
    
    if max_samples is not None:
        # Train on a small subset to run quickly and avoid memory limit
        train_rows = train_rows[:max_samples]
        val_rows = val_rows[:min(max_samples // 5, 50)]
    
    # Load upscaler
    with open(upscaler_path, 'rb') as f:
        upscaler = pickle.load(f)
        
    tokenizer = CharacterTokenizer()
    
    # Create datasets
    print("Loading Train Dataset...")
    train_dataset = SpeechIMUKDDataset(train_rows, dataset_root, upscaler=upscaler, tokenizer=tokenizer)
    print("Loading Val Dataset...")
    val_dataset = SpeechIMUKDDataset(val_rows, dataset_root, upscaler=upscaler, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    teacher_model = SLUModel(vocab_size=tokenizer.vocab_size).to(device)
    student_model = SLUModel(vocab_size=tokenizer.vocab_size).to(device)
    
    # 1. Train Teacher Model on Speech Spectrograms
    print("\n--- Phase 1: Training Teacher Model on Speech Spectrograms ---")
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    for epoch in range(epochs):
        teacher_model.train()
        total_loss = 0
        for batch in train_loader:
            speech_feat = batch['speech_feat'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            
            optimizer_teacher.zero_grad()
            # Forward: targets are shifted by 1 in cross entropy
            output = teacher_model(speech_feat, target_tokens, teacher_forcing_ratio=0.5)
            
            # Reshape output for loss calculation
            # output shape: (batch, trg_len-1, vocab_size) -> (batch * (trg_len-1), vocab_size)
            # targets shape: (batch, trg_len) -> target[:, 1:] shape: (batch * (trg_len-1))
            loss = criterion(output.view(-1, tokenizer.vocab_size), target_tokens[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer_teacher.step()
            
            total_loss += loss.item()
            
        print(f"Teacher Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
    # Save Teacher Model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(teacher_model.state_dict(), os.path.join(save_dir, "teacher_model.pt"))
    
    # 2. Train Student Model using Cross-Modal Knowledge Distillation
    print("\n--- Phase 2: Training Student Model via Cross-Modal KD ---")
    teacher_model.eval() # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    optimizer_student = optim.Adam(student_model.parameters(), lr=1e-3)
    
    # KL Divergence Loss for distillation
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    alpha = 0.5 # balance weight between NLL and KD
    temperature = 2.0 # KD temperature
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        for batch in train_loader:
            speech_feat = batch['speech_feat'].to(device)
            imu_feat = batch['imu_feat'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            
            optimizer_student.zero_grad()
            
            # Forward: get student predictions
            student_output = student_model(imu_feat, target_tokens, teacher_forcing_ratio=0.5)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(speech_feat, target_tokens, teacher_forcing_ratio=0.5)
                
            # Compute classification loss (Cross Entropy)
            loss_ce = criterion(student_output.view(-1, tokenizer.vocab_size), target_tokens[:, 1:].contiguous().view(-1))
            
            # Compute distillation loss (KL Divergence on soft targets)
            # apply log_softmax on student and softmax on teacher (divided by temperature)
            student_soft = F.log_softmax(student_output / temperature, dim=-1)
            teacher_soft = F.softmax(teacher_output / temperature, dim=-1)
            
            loss_kd = kl_loss_fn(student_soft.view(-1, tokenizer.vocab_size), teacher_soft.view(-1, tokenizer.vocab_size)) * (temperature ** 2)
            
            # Combined Loss
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
            
            loss.backward()
            optimizer_student.step()
            
            total_loss += loss.item()
            
        print(f"Student Epoch {epoch+1}/{epochs} - Combined Loss: {total_loss/len(train_loader):.4f}")
        
    # Save Student Model
    torch.save(student_model.state_dict(), os.path.join(save_dir, "student_model.pt"))
    print(f"Student model saved to {os.path.join(save_dir, 'student_model.pt')}")

if __name__ == "__main__":
    meta = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv"
    root = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset"
    upscaler = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models/upscaler.pkl"
    save = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models"
    train_kd_pipeline(meta, root, upscaler, save, epochs=1, batch_size=4)
