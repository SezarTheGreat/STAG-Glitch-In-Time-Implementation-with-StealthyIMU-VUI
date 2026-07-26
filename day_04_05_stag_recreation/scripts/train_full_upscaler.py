import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.upscaler import StagUpscaler
from src.pipeline.dataset import load_splits, get_stag_bifurcation

def train_upscaler():
    metadata_file = 'StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv'
    dataset_root = 'StealthyIMU_dataset'
    
    train_rows, val_rows, _ = load_splits(metadata_file)
    
    acc_odd_list = []
    gyro_even_list = []
    acc_even_list = []
    t_odd_list = []
    t_even_list = []
    
    print(f"Loading data for Upscaler... Total train rows: {len(train_rows)}")
    max_samples = 5000 
    
    # Shuffle and pick max_samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_rows), min(max_samples, len(train_rows)), replace=False)
    
    for idx in tqdm(sample_indices):
        row = train_rows[idx]
        uuid = row[0]
        duration = float(row[1])
        wav_path_rel = row[2]
        
        base_dir = os.path.dirname(wav_path_rel)
        acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
        gyro_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
        
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            continue
            
        try:
            acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                acc_path, gyro_path, duration
            )
        except Exception as e:
            continue
            
        acc_odd_list.append(acc_odd)
        gyro_even_list.append(gyro_even)
        acc_even_list.append(acc_even_target)
        t_odd_list.append(t_odd)
        t_even_list.append(t_even)
        
    print(f"Loaded {len(acc_odd_list)} samples for Upscaler training.")
    
    upscaler = StagUpscaler(n_estimators=300, max_depth=7, learning_rate=0.05, W=5)
    print("Training Upscaler...")
    upscaler.fit(acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list)
    
    print("Evaluating on the same subset...")
    y_true_all = []
    y_pred_all = []
    for acc_odd, gyro_even, acc_even, t_odd, t_even in tqdm(zip(acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list), total=len(acc_odd_list)):
        predicted_even, _ = upscaler.predict_even(acc_odd, gyro_even, t_odd, t_even)
        y_true_all.append(acc_even)
        y_pred_all.append(predicted_even)
        
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    
    r2 = r2_score(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    
    print(f"Upscaler R2: {r2:.4f}, MSE: {mse:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/upscaler.pkl', 'wb') as f:
        pickle.dump(upscaler, f)
        
    print("Saved Upscaler to models/upscaler.pkl")

if __name__ == '__main__':
    train_upscaler()
