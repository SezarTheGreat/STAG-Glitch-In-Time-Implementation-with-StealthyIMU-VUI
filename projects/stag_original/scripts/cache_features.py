import os
import sys
import pickle
import numpy as np
import torch
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.dataset import load_splits, get_stag_bifurcation
from src.pipeline.features import extract_spectrogram
from src.models.slu_dnn import CharacterTokenizer

def process_sample(row, dataset_root, upscaler, tokenizer, max_len=150, max_frames=300):
    uuid = row[0]
    duration = float(row[1])
    wav_path_rel = row[2]
    semantic_frame = row[3]
    
    target_text = semantic_frame
    target_tokens = tokenizer.encode(target_text)
    
    if len(target_tokens) > max_len:
        target_tokens = target_tokens[:max_len]
    else:
        target_tokens = target_tokens + [tokenizer.pad_id] * (max_len - len(target_tokens))
        
    base_dir = os.path.dirname(wav_path_rel)
    wav_path = os.path.join(dataset_root, wav_path_rel.replace('./', ''))
    acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
    gyro_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
    
    if not os.path.exists(wav_path) or not os.path.exists(acc_path) or not os.path.exists(gyro_path):
        return None
        
    try:
        # Audio Processing
        y, sr = librosa.load(wav_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30, n_fft=512, hop_length=320)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max).T
        
        # IMU Processing
        acc_odd, gyro_even, _, t_even, t_odd = get_stag_bifurcation(acc_path, gyro_path, duration)
        
        if upscaler is not None:
            acc_z_400 = upscaler.reconstruct_signal(acc_odd, gyro_even, t_odd, t_even)
        else:
            from scipy.interpolate import interp1d
            acc_z_400 = interp1d(t_odd, acc_odd, kind='linear', fill_value="extrapolate")(t_even)
            
        imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
        
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
            'target_tokens': torch.LongTensor(target_tokens)
        }
    except Exception as e:
        # Silently fail on bad files
        return None

def main():
    metadata_file = 'StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv'
    dataset_root = 'StealthyIMU_dataset'
    upscaler_path = 'models/upscaler.pkl'
    save_dir = 'models'
    
    print("Loading splits...")
    train_rows, val_rows, test_rows = load_splits(metadata_file)
    
    with open(upscaler_path, 'rb') as f:
        upscaler = pickle.load(f)
        
    tokenizer = CharacterTokenizer()
    
    splits = {
        'train': train_rows,
        'val': val_rows,
        'test': test_rows
    }
    
    os.makedirs(save_dir, exist_ok=True)
    num_cores = max(1, cpu_count() - 1)
    
    for split_name, rows in splits.items():
        print(f"\nProcessing {split_name} split ({len(rows)} samples) with {num_cores} cores...")
        
        process_func = partial(process_sample, dataset_root=dataset_root, upscaler=upscaler, tokenizer=tokenizer)
        
        valid_samples = []
        with Pool(num_cores) as pool:
            for result in tqdm(pool.imap(process_func, rows), total=len(rows)):
                if result is not None:
                    valid_samples.append(result)
                    
        # Stack into single tensors
        speech_feats = torch.stack([s['speech_feat'] for s in valid_samples])
        imu_feats = torch.stack([s['imu_feat'] for s in valid_samples])
        target_tokens = torch.stack([s['target_tokens'] for s in valid_samples])
        
        save_dict = {
            'speech_feat': speech_feats,
            'imu_feat': imu_feats,
            'target_tokens': target_tokens
        }
        
        save_path = os.path.join(save_dir, f"{split_name}_data.pt")
        torch.save(save_dict, save_path)
        print(f"Saved {split_name} data to {save_path}. Total samples: {len(valid_samples)}")

if __name__ == '__main__':
    main()
