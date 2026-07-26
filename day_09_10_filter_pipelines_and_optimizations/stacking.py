import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))

from projects.combined_reconstruction.features import extract_hybrid_features
from projects.combined_reconstruction.models import (
    CNNUpscaler,
    RNNUpscaler,
    load_teacher_regression_engine,
    load_student_regression_engine
)
from projects.stag_original.src.pipeline.dataset import get_stag_bifurcation
from projects.stag_original.src.pipeline.features import extract_spectrogram

class StackingUpscaler:
    """
    Stacking Ensemble that fuses:
    1. Pre-trained StealthyIMU Teacher model (Encoder + Regressor Head)
    2. 1D CNN Upscaler
    3. RNN (GRU) Upscaler
    4. LightGBM Regressor (with engineered features)
    Using Ridge regression as the meta-regressor.
    """
    def __init__(self, teacher_ckpt_path, W=5):
        self.W = W
        self.teacher_engine = load_teacher_regression_engine(teacher_ckpt_path)
        self.cnn = CNNUpscaler(window_size=2*W+1, num_channels=4)
        self.rnn = RNNUpscaler(num_channels=4)
        
        # Import lightgbm inside to prevent import errors if not present (defensive check)
        import lightgbm as lgb
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
        self.meta_regressor = Ridge(alpha=1.0)
        
    def _prepare_windows(self, acc_interp, gyro_even):
        """
        Slices local windows for CNN and RNN input.
        """
        N = len(acc_interp)
        W = self.W
        W_size = 2 * W + 1
        
        gyro_padded = np.pad(gyro_even, ((0, 0), (W, W)), mode='edge')
        acc_padded = np.pad(acc_interp, (W, W), mode='edge')
        
        cnn_inputs = []
        rnn_inputs = []
        
        for t in range(N):
            acc_win = acc_padded[t : t + W_size]
            gyro_win = gyro_padded[:, t : t + W_size]
            
            # 4 channels: acc_z, gyro_x, gyro_y, gyro_z
            win_data = np.vstack([acc_win.reshape(1, -1), gyro_win]) # shape: (4, W_size)
            cnn_inputs.append(win_data)
            rnn_inputs.append(win_data.T) # shape: (W_size, 4)
            
        return np.array(cnn_inputs), np.array(rnn_inputs)

    def train_components(self, train_rows, dataset_root, epochs=3, batch_size=64):
        """
        Trains CNN, RNN, LightGBM, and Teacher regression head on training data.
        """
        print("Preparing dataset for training components...")
        
        all_lgb_feats = []
        all_cnn_inputs = []
        all_rnn_inputs = []
        all_targets = []
        
        # We will also collect sequences to train the teacher head sequence-wise
        teacher_train_sequences = []
        
        for idx, row in enumerate(train_rows):
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
                
                # Extract engineered features for LightGBM
                feats = extract_hybrid_features(gyro_even, acc_interp, W=self.W)
                
                # Extract sliding window inputs for CNN/RNN
                cnn_win, rnn_win = self._prepare_windows(acc_interp, gyro_even)
                
                all_lgb_feats.append(feats)
                all_cnn_inputs.append(cnn_win)
                all_rnn_inputs.append(rnn_win)
                all_targets.append(acc_even_target)
                
                # For Teacher engine
                # Construct temporary reconstructed signal (using cubic spline) for spectrogram
                acc_z_400 = np.zeros(len(acc_odd) + len(acc_interp))
                acc_z_400[0::2] = acc_odd
                acc_z_400[1::2] = acc_interp
                imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
                
                teacher_train_sequences.append({
                    'spec': torch.FloatTensor(imu_spec).unsqueeze(0), # (1, frames, 30)
                    'target': torch.FloatTensor(acc_even_target),
                    'target_len': len(acc_even_target)
                })
            except Exception as e:
                continue
                
        if not all_targets:
            raise ValueError("No valid training samples loaded!")
            
        X_lgb = np.vstack(all_lgb_feats)
        X_cnn = np.vstack(all_cnn_inputs)
        X_rnn = np.vstack(all_rnn_inputs)
        Y = np.concatenate(all_targets)
        
        # 1. Train LightGBM
        print("Training LightGBM component...")
        self.lgb_model.fit(X_lgb, Y)
        
        # 2. Train PyTorch CNN and RNN
        print("Training PyTorch CNN and RNN components...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn.to(device)
        self.rnn.to(device)
        
        cnn_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_cnn), torch.FloatTensor(Y))
        rnn_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_rnn), torch.FloatTensor(Y))
        
        cnn_loader = torch.utils.data.DataLoader(cnn_dataset, batch_size=batch_size, shuffle=True)
        rnn_loader = torch.utils.data.DataLoader(rnn_dataset, batch_size=batch_size, shuffle=True)
        
        opt_cnn = torch.optim.Adam(self.cnn.parameters(), lr=1e-3)
        opt_rnn = torch.optim.Adam(self.rnn.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.cnn.train()
            total_loss = 0
            for bx, by in cnn_loader:
                bx, by = bx.to(device), by.to(device)
                opt_cnn.zero_grad()
                pred = self.cnn(bx)
                loss = criterion(pred, by)
                loss.backward()
                opt_cnn.step()
                total_loss += loss.item()
            # print(f"  CNN Epoch {epoch+1}/{epochs} Loss: {total_loss/len(cnn_loader):.4f}")
            
        for epoch in range(epochs):
            self.rnn.train()
            total_loss = 0
            for bx, by in rnn_loader:
                bx, by = bx.to(device), by.to(device)
                opt_rnn.zero_grad()
                pred = self.rnn(bx)
                loss = criterion(pred, by)
                loss.backward()
                opt_rnn.step()
                total_loss += loss.item()
            # print(f"  RNN Epoch {epoch+1}/{epochs} Loss: {total_loss/len(rnn_loader):.4f}")
            
        # 3. Train Teacher Regression Head
        print("Training Teacher model regression head...")
        self.teacher_engine.to(device)
        self.teacher_engine.regressor.train()
        opt_teacher = torch.optim.Adam(self.teacher_engine.regressor.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            # Shuffle a copy of the list to keep original aligned with other lists
            shuffled_seqs = list(teacher_train_sequences)
            np.random.shuffle(shuffled_seqs)
            for seq in shuffled_seqs:
                spec = seq['spec'].to(device)
                target = seq['target'].to(device)
                target_len = seq['target_len']
                
                opt_teacher.zero_grad()
                pred = self.teacher_engine(spec, target_len)
                loss = criterion(pred, target)
                loss.backward()
                opt_teacher.step()
                total_loss += loss.item()
            # print(f"  Teacher Head Epoch {epoch+1}/{epochs} Loss: {total_loss/len(teacher_train_sequences):.4f}")
            
        # 4. Train Stacking Meta-Regressor
        print("Training Stacking Meta-Regressor...")
        self.cnn.eval()
        self.rnn.eval()
        self.teacher_engine.eval()
        
        meta_preds = []
        meta_targets = []
        
        with torch.no_grad():
            for seq, feats, cnn_win, rnn_win, target in zip(
                teacher_train_sequences, all_lgb_feats, all_cnn_inputs, all_rnn_inputs, all_targets
            ):
                # LightGBM predictions
                pred_lgb = self.lgb_model.predict(feats)
                
                # CNN/RNN predictions
                pred_cnn = self.cnn(torch.FloatTensor(cnn_win).to(device)).cpu().numpy()
                pred_rnn = self.rnn(torch.FloatTensor(rnn_win).to(device)).cpu().numpy()
                
                # Teacher engine predictions
                spec = seq['spec'].to(device)
                pred_teacher = self.teacher_engine(spec, seq['target_len']).cpu().numpy()
                
                meta_preds.append(np.column_stack([pred_lgb, pred_cnn, pred_rnn, pred_teacher]))
                meta_targets.append(target)
                
        X_meta = np.vstack(meta_preds)
        Y_meta = np.concatenate(meta_targets)
        
        self.meta_regressor.fit(X_meta, Y_meta)
        print("Stacking meta-regressor trained successfully!")
        
    def predict_even(self, acc_odd, gyro_even, t_odd, t_even):
        """
        Predicts even accelerometer values for a single sample.
        """
        N = len(t_even)
        acc_interp = interp1d(t_odd, acc_odd, kind='cubic', fill_value="extrapolate")(t_even)
        
        # Prepare inputs
        feats = extract_hybrid_features(gyro_even, acc_interp, W=self.W)
        cnn_win, rnn_win = self._prepare_windows(acc_interp, gyro_even)
        
        acc_z_400 = np.zeros(len(acc_odd) + len(acc_interp))
        acc_z_400[0::2] = acc_odd
        acc_z_400[1::2] = acc_interp
        imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn.to(device)
        self.rnn.to(device)
        self.teacher_engine.to(device)
        
        self.cnn.eval()
        self.rnn.eval()
        self.teacher_engine.eval()
        
        with torch.no_grad():
            # LightGBM
            pred_lgb = self.lgb_model.predict(feats)
            
            # CNN
            pred_cnn = self.cnn(torch.FloatTensor(cnn_win).to(device)).cpu().numpy()
            
            # RNN
            pred_rnn = self.rnn(torch.FloatTensor(rnn_win).to(device)).cpu().numpy()
            
            # Teacher regression engine
            spec_tensor = torch.FloatTensor(imu_spec).unsqueeze(0).to(device)
            pred_teacher = self.teacher_engine(spec_tensor, N).cpu().numpy()
            
        # Stacking Fusion
        X_meta = np.column_stack([pred_lgb, pred_cnn, pred_rnn, pred_teacher])
        pred_stacked = self.meta_regressor.predict(X_meta)
        
        return pred_stacked

    def reconstruct_signal(self, acc_odd, gyro_even, t_odd, t_even):
        """
        Reconstructs the full 400 Hz accelerometer signal by interleaving 
        true odd samples and predicted even samples.
        """
        predicted_even = self.predict_even(acc_odd, gyro_even, t_odd, t_even)
        
        n_samples = len(acc_odd) + len(predicted_even)
        reconstructed = np.zeros(n_samples)
        
        reconstructed[0::2] = acc_odd
        reconstructed[1::2] = predicted_even
        
        return reconstructed
