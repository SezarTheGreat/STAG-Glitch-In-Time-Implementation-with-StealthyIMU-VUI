import os
import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score

# Add directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))
sys.path.append(os.path.dirname(__file__))

from projects.combined_reconstruction.stacking import StackingUpscaler
from projects.combined_reconstruction.models import load_student_regression_engine
from projects.stag_original.src.pipeline.dataset import load_splits, get_stag_bifurcation
from projects.stag_original.src.pipeline.features import extract_spectrogram

def train_student_baseline(student_engine, train_rows, dataset_root, epochs=3):
    print("Training Student baseline model regression head...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_engine.to(device)
    student_engine.regressor.train()
    opt = torch.optim.Adam(student_engine.regressor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        samples_count = 0
        for row in train_rows:
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
                acc_interp = interp1d(t_odd, acc_odd, kind='cubic', fill_value="extrapolate")(t_even)
                
                acc_z_400 = np.zeros(len(acc_odd) + len(acc_interp))
                acc_z_400[0::2] = acc_odd
                acc_z_400[1::2] = acc_interp
                imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
                
                spec_tensor = torch.FloatTensor(imu_spec).unsqueeze(0).to(device)
                target_tensor = torch.FloatTensor(acc_even_target).to(device)
                
                opt.zero_grad()
                pred = student_engine(spec_tensor, len(acc_even_target))
                loss = criterion(pred, target_tensor)
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                samples_count += 1
            except Exception:
                continue
        print(f"  Student Base Head Epoch {epoch+1}/{epochs} Loss: {total_loss/max(1, samples_count):.4f}")

def evaluate_reconstruction():
    metadata_file = 'common/data/StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv'
    dataset_root = 'common/data/StealthyIMU_dataset'
    teacher_ckpt = 'common/models/teacher_model.pt'
    student_ckpt = 'common/models/student_model.pt'
    
    print("Loading data splits...")
    train_rows, val_rows, test_rows = load_splits(metadata_file)
    
    # Train on a small fast subset to fit CPU time limits
    train_subset = train_rows[:400]
    test_subset = test_rows[:150]
    
    print(f"Subsampled train set: {len(train_subset)}, test set: {len(test_subset)}")
    
    # 1. Initialize and train student baseline
    student_engine = load_student_regression_engine(student_ckpt)
    train_student_baseline(student_engine, train_subset, dataset_root, epochs=3)
    
    # 2. Initialize and train stacking champion
    stacking_upscaler = StackingUpscaler(teacher_ckpt, W=5)
    stacking_upscaler.train_components(train_subset, dataset_root, epochs=3)
    
    import pickle
    os.makedirs("common/models", exist_ok=True)
    with open("common/models/stacking_upscaler.pkl", "wb") as f:
        pickle.dump(stacking_upscaler, f)
    print("Saved trained StackingUpscaler to common/models/stacking_upscaler.pkl")
    
    # 3. Side-by-side inference loop on test subset
    print("\nRunning side-by-side inference and benchmarking...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_engine.to(device)
    student_engine.eval()
    
    y_true_all = []
    y_pred_student_all = []
    y_pred_stacked_all = []
    
    with torch.no_grad():
        for row in test_subset:
            uuid = row[0]
            duration = float(row[1])
            wav_path = row[2]
            
            base_dir = os.path.dirname(wav_path)
            acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
            gyro_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.gyro")
            
            if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
                continue
                
            try:
                # Ground truth
                acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                    acc_path, gyro_path, duration
                )
                
                # Student Baseline prediction
                acc_interp = interp1d(t_odd, acc_odd, kind='cubic', fill_value="extrapolate")(t_even)
                acc_z_400 = np.zeros(len(acc_odd) + len(acc_interp))
                acc_z_400[0::2] = acc_odd
                acc_z_400[1::2] = acc_interp
                imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
                
                spec_tensor = torch.FloatTensor(imu_spec).unsqueeze(0).to(device)
                pred_student = student_engine(spec_tensor, len(acc_even_target)).cpu().numpy()
                
                # Stacking Champion prediction
                pred_stacked = stacking_upscaler.predict_even(acc_odd, gyro_even, t_odd, t_even)
                
                y_true_all.append(acc_even_target)
                y_pred_student_all.append(pred_student)
                y_pred_stacked_all.append(pred_stacked)
            except Exception:
                continue
                
    if not y_true_all:
        print("Error: No test samples were evaluated!")
        return
        
    y_true = np.concatenate(y_true_all)
    y_pred_student = np.concatenate(y_pred_student_all)
    y_pred_stacked = np.concatenate(y_pred_stacked_all)
    
    # Metrics
    mse_student = mean_squared_error(y_true, y_pred_student)
    r2_student = r2_score(y_true, y_pred_student)
    
    mse_stacked = mean_squared_error(y_true, y_pred_stacked)
    r2_stacked = r2_score(y_true, y_pred_stacked)
    
    improvement_mse = (mse_student - mse_stacked) / mse_student * 100
    
    print("\n" + "=" * 60)
    print("  STAG SIGNAL RECONSTRUCTION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  {'Model Architecture':<30} | {'MSE':<8} | {'R-squared':<8}")
    print("-" * 60)
    print(f"  {'Student KD Baseline':<30} | {mse_student:.5f} | {r2_student:.5f}")
    print(f"  {'Teacher-led Stacking Champion':<30} | {mse_stacked:.5f} | {r2_stacked:.5f}")
    print("=" * 60)
    print(f"  MSE Reduction: {improvement_mse:.2f}%")
    print(f"  R-squared Increase: {r2_stacked - r2_student:.5f}")
    print("=" * 60)
    
    # Save a summary txt file for reporting
    with open("projects/combined_reconstruction/evaluation_results.txt", "w") as f:
        f.write(f"Student KD Baseline - MSE: {mse_student:.6f}, R2: {r2_student:.6f}\n")
        f.write(f"Teacher-led Stacking Champion - MSE: {mse_stacked:.6f}, R2: {r2_stacked:.6f}\n")
        f.write(f"MSE Improvement: {improvement_mse:.2f}%\n")
        f.write(f"R-squared Gain: {r2_stacked - r2_student:.6f}\n")

if __name__ == '__main__':
    evaluate_reconstruction()
