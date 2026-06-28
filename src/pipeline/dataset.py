import os
import csv
import numpy as np
import scipy.interpolate as interpolate

def read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)

def load_splits(metadata_file, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Loads metadata and splits deterministically based on UUID hash.
    """
    rows = read_csv(metadata_file)
    np.random.seed(seed)
    
    # Let's shuffle rows deterministically
    indices = np.arange(len(rows))
    np.random.shuffle(indices)
    
    n_train = int(len(rows) * train_ratio)
    n_val = int(len(rows) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    
    return train_rows, val_rows, test_rows

def load_raw_sensor(filepath):
    """
    Reads a raw .acc or .gyro CSV file.
    Returns:
        timestamps: np.array of shape (N,)
        values: np.array of shape (3, N) containing X, Y, Z values
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sensor file not found: {filepath}")
    
    timestamps = []
    x, y, z = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None) # skip header
        for row in reader:
            if len(row) >= 4:
                try:
                    timestamps.append(float(row[0]))
                    x.append(float(row[1]))
                    y.append(float(row[2]))
                    z.append(float(row[3]))
                except ValueError:
                    pass
    return np.array(timestamps), np.vstack([x, y, z])

def resample_signal(timestamps, values, target_timestamps, kind='linear'):
    """
    Resamples a signal values (3, N) on target_timestamps (M,).
    """
    # Shift timestamps to start at 0
    t_shifted = timestamps - timestamps[0]
    
    # Standardize target timestamps
    t_target = target_timestamps - target_timestamps[0]
    
    # Boundary clip to avoid extrapolation errors
    t_target = np.clip(t_target, t_shifted[0], t_shifted[-1])
    
    resampled_values = []
    for axis in range(values.shape[0]):
        f = interpolate.interp1d(t_shifted, values[axis, :], kind=kind, fill_value="extrapolate")
        resampled_values.append(f(t_target))
        
    return np.vstack(resampled_values)

def get_stag_bifurcation(acc_path, gyro_path, duration_seconds, fs=400):
    """
    Simulates temporal misalignment of IMU sensors.
    Downsamples 400 Hz sensor readings into odd/even 200 Hz streams.
    Returns:
        acc_odd: Accelerometer Z-axis at odd timestamps (200 Hz)
        gyro_even: Gyroscope X, Y, Z axes at even timestamps (200 Hz)
        acc_even_target: True Accelerometer Z-axis at even timestamps (200 Hz)
        t_even: Time grid for even samples
        t_odd: Time grid for odd samples
    """
    t_acc, val_acc = load_raw_sensor(acc_path)
    t_gyro, load_gyro = load_raw_sensor(gyro_path)
    
    # Noise reduction (Median Filter) as per paper preprocessing
    import scipy.signal
    val_acc = scipy.signal.medfilt(val_acc, kernel_size=(1, 5))
    load_gyro = scipy.signal.medfilt(load_gyro, kernel_size=(1, 5))
    
    # Z-score normalize raw signals
    val_acc_norm = (val_acc - np.mean(val_acc, axis=1, keepdims=True)) / (np.std(val_acc, axis=1, keepdims=True) + 1e-8)
    val_gyro_norm = (load_gyro - np.mean(load_gyro, axis=1, keepdims=True)) / (np.std(load_gyro, axis=1, keepdims=True) + 1e-8)
    
    # Establish uniform grid at fs = 400 Hz (interval = 2.5 ms)
    # Total duration is duration_seconds
    n_samples = int(duration_seconds * fs)
    t_uniform = np.arange(n_samples) * (1000.0 / fs) # in milliseconds
    
    # Resample raw signal onto uniform grid
    acc_uniform = resample_signal(t_acc, val_acc_norm, t_uniform)
    gyro_uniform = resample_signal(t_gyro, val_gyro_norm, t_uniform)
    
    # Bifurcate:
    # Odd samples (index 0, 2, 4...) -> Acc_odd (200 Hz)
    # Even samples (index 1, 3, 5...) -> Gyro_even, Acc_even_target (200 Hz)
    odd_idx = np.arange(0, n_samples, 2)
    even_idx = np.arange(1, n_samples, 2)
    
    t_odd = t_uniform[odd_idx]
    t_even = t_uniform[even_idx]
    
    # Acc odd (Z-axis only)
    acc_odd = acc_uniform[2, odd_idx] # Z is index 2
    
    # Gyro even (X, Y, Z axes)
    gyro_even = gyro_uniform[:, even_idx]
    
    # Acc even target (Z-axis only)
    acc_even_target = acc_uniform[2, even_idx]
    
    return acc_odd, gyro_even, acc_even_target, t_even, t_odd
