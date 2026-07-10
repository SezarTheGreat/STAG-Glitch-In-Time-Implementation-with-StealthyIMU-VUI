import numpy as np
import os
from common.interpolation.interpolation import cubic_spline_interpolate
from projects.stag_original.src.pipeline.dataset import get_stag_bifurcation

def extract_windows(acc_interp, gyro_even, W=2):
    """
    Extracts sliding window features for each sample.
    For each time step t in [0, N-1], extracts a window of size 2*W + 1.
    
    acc_interp: shape (N,)
    gyro_even: shape (3, N)
    W: window radius
    
    Returns:
        X: shape (N, 2*W + 1, 4) - where the last dimension is:
           [acc_interp, gyro_x, gyro_y, gyro_z]
    """
    N = len(acc_interp)
    W_size = 2 * W + 1
    
    # Pad signals along boundaries to support shifts
    acc_padded = np.pad(acc_interp, (W, W), mode='edge')
    gyro_padded = np.pad(gyro_even, ((0, 0), (W, W)), mode='edge')
    
    X = np.zeros((N, W_size, 4), dtype=np.float32)
    
    for t in range(N):
        X[t, :, 0] = acc_padded[t : t + W_size]
        X[t, :, 1:4] = gyro_padded[:, t : t + W_size].T
        
    return X

def load_dataset_samples(rows, dataset_root, W=2):
    """
    Loads raw files, performs cubic spline interpolation, and extracts feature windows.
    Returns:
        X_all: concatenated features of shape (total_samples, 2*W + 1, 4)
        Y_all: concatenated targets of shape (total_samples,)
    """
    X_list = []
    Y_list = []
    
    for row in rows:
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
            # Perform cubic spline interpolation from common module
            acc_interp = cubic_spline_interpolate(t_odd, acc_odd, t_even)
            
            # Extract features
            X = extract_windows(acc_interp, gyro_even, W=W)
            
            X_list.append(X)
            Y_list.append(acc_even_target.astype(np.float32))
        except Exception:
            continue
            
    if not X_list:
        return np.zeros((0, 2*W + 1, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)
