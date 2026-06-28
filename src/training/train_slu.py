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
from src.models.slu_dnn import CharacterTokenizer, PaperSLUModel as SLUModel

class CachedKDDataset(Dataset):
    """
    Dataset to load pre-computed Speech and IMU features.
    """
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.speech_feats = data['speech_feat']
        self.imu_feats = data['imu_feat']
        self.target_tokens = data['target_tokens']
        
    def __len__(self):
        return len(self.target_tokens)

    def __getitem__(self, idx):
        return {
            'speech_feat': self.speech_feats[idx],
            'imu_feat': self.imu_feats[idx],
            'target_tokens': self.target_tokens[idx]
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
    train_dataset = CachedKDDataset(os.path.join(save_dir, "train_data.pt"))
    print("Loading Val Dataset...")
    val_dataset = CachedKDDataset(os.path.join(save_dir, "val_data.pt"))
    
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
        for i, batch in enumerate(train_loader):
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
            
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}", flush=True)
            
        print(f"Teacher Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}", flush=True)
        
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
        for i, batch in enumerate(train_loader):
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
            
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} KD Loss: {loss.item():.4f}", flush=True)
            
        print(f"Student Epoch {epoch+1}/{epochs} - Combined Loss: {total_loss/len(train_loader):.4f}", flush=True)
        
    # Save Student Model
    torch.save(student_model.state_dict(), os.path.join(save_dir, "student_model.pt"))
    print(f"Student model saved to {os.path.join(save_dir, 'student_model.pt')}")

if __name__ == "__main__":
    meta = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv"
    root = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset"
    upscaler = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models/upscaler.pkl"
    save = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models"
    # We will use epoch=15 for teacher, and epoch=30 for student if we want, but let's pass it
    train_kd_pipeline(meta, root, upscaler, save, epochs=30, batch_size=256)
