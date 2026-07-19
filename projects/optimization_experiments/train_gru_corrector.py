import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))

from projects.stag_original.src.pipeline.dataset import load_splits, get_stag_bifurcation
from projects.interpolation_experiments.pipeline_variants import extract_features_from_interp

class ResidualGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1):
        super(ResidualGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (1, seq_len, input_dim)
        out, _ = self.gru(x)  # out shape: (1, seq_len, hidden_dim * 2)
        res = self.fc(out)    # res shape: (1, seq_len, output_dim)
        return res

def main():
    metadata_file = "projects/stag_original/results/slu_baseline_paper/1235/train-type=direct.csv"
    dataset_root = "common/data/StealthyIMU_dataset/"
    upscaler_path = "common/models/upscaler.pkl"
    save_path = "common/models/gru_corrector.pt"
    
    print("Loading datasets and model...")
    import csv
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        train_rows = list(reader)
    # Skip header if it exists
    if train_rows[0][0] == "ID":
        train_rows = train_rows[1:]
    with open(upscaler_path, 'rb') as f:
        upscaler = pickle.load(f)
    
    lgb_model = upscaler.model
    W = upscaler.W
    
    # Restrict samples to train quickly on CPU
    train_rows = train_rows[:250]
    
    print(f"Preparing training data for {len(train_rows)} sequences...")
    training_data = []
    
    for row in tqdm(train_rows):
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
            # Perform Cubic Spline interpolation of odd acc
            import scipy.interpolate as interpolate
            cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
            acc_interp = cs(t_even)
            
            # Predict using LightGBM
            feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
            pred_even = lgb_model.predict(feats)
            
            # Prepare inputs to GRU: (N, 4) -> pred_even + gyro_even (3 channels)
            # gyro_even shape: (3, N) -> transpose to (N, 3)
            inputs = np.column_stack([pred_even, gyro_even.T])
            targets = (acc_even_target - pred_even).reshape(-1, 1)  # Target is the residual
            
            training_data.append((
                torch.FloatTensor(inputs).unsqueeze(0), # (1, seq_len, 4)
                torch.FloatTensor(targets).unsqueeze(0) # (1, seq_len, 1)
            ))
        except Exception:
            continue

    print(f"Collected {len(training_data)} training sequences.")
    
    # Initialize GRU
    model = ResidualGRU()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train Loop
    epochs = 4
    print(f"Training Residual GRU Corrector on CPU for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in training_data:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} -> Average MSE Loss: {epoch_loss/len(training_data):.6f}")
        
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Trained GRU corrector model saved to {save_path}")

if __name__ == "__main__":
    main()
