import os
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.pipeline.dataset import load_splits, get_stag_bifurcation
from src.models.upscaler import StagUpscaler

def train_stag_upscaler(metadata_file, dataset_root, save_path, max_samples=None):
    print("Loading dataset splits...")
    train_rows, val_rows, _ = load_splits(metadata_file)
    
    if max_samples is not None:
        # Restrict samples to speed up training
        train_rows = train_rows[:max_samples]
        val_rows = val_rows[:min(max_samples // 5, 200)]
    
    print(f"Preparing training data for {len(train_rows)} samples...")
    acc_odd_list = []
    gyro_even_list = []
    acc_even_list = []
    t_odd_list = []
    t_even_list = []
    
    for idx, row in enumerate(train_rows):
        uuid = row[0]
        duration = float(row[1])
        wav_path = row[2] # e.g. ./data/cleanair/UUID/UUID.wav
        
        # Determine raw paths
        base_dir = os.path.dirname(wav_path)
        acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
        gyro_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
        
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            continue
            
        try:
            acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                acc_path, gyro_path, duration
            )
            acc_odd_list.append(acc_odd)
            gyro_even_list.append(gyro_even)
            acc_even_list.append(acc_even_target)
            t_odd_list.append(t_odd)
            t_even_list.append(t_even)
        except Exception as e:
            # Skip invalid files/corrupt samples
            continue

    print(f"Training LightGBM Upscaler on {len(acc_odd_list)} valid samples...")
    if not acc_odd_list:
        print("Error: No valid training samples found!")
        return None
        
    upscaler = StagUpscaler()
    upscaler.fit(acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(upscaler, f)
    print(f"Upscaler model saved to {save_path}")
    
    # Validate
    print("Evaluating upscaler on validation split...")
    val_targets = []
    val_preds = []
    val_interps = []
    
    for row in val_rows:
        uuid = row[0]
        duration = float(row[1])
        wav_path = row[2]
        
        base_dir = os.path.dirname(wav_path)
        acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
        gyro_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
        
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            continue
            
        try:
            acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                acc_path, gyro_path, duration
            )
            pred_even, acc_interp = upscaler.predict_even(acc_odd, gyro_even, t_odd, t_even)
            
            val_targets.append(acc_even_target)
            val_preds.append(pred_even)
            val_interps.append(acc_interp)
        except Exception:
            continue
            
    if val_targets:
        targets = np.concatenate(val_targets)
        preds = np.concatenate(val_preds)
        interps = np.concatenate(val_interps)
        
        r2_model = r2_score(targets, preds)
        mse_model = mean_squared_error(targets, preds)
        
        r2_interp = r2_score(targets, interps)
        mse_interp = mean_squared_error(targets, interps)
        
        print("\n--- Upscaling Validation Results ---")
        print(f"STAG Model R^2: {r2_model:.4f}, MSE: {mse_model:.4f}")
        print(f"Cubic Spline baseline R^2: {r2_interp:.4f}, MSE: {mse_interp:.4f}")
        print(f"Relative improvement in MSE: {(mse_interp - mse_model) / mse_interp * 100:.2f}%")
        
    return upscaler

if __name__ == "__main__":
    meta = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv"
    root = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/StealthyIMU_dataset"
    save = "c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models/upscaler.pkl"
    train_stag_upscaler(meta, root, save, max_samples=100)
