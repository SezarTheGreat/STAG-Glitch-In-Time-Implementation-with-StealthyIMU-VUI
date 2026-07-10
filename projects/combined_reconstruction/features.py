import numpy as np

def safe_stats(window):
    """
    Computes statistical moments (mean, variance, skewness, kurtosis) safely.
    """
    n = len(window)
    if n == 0:
        return np.zeros(4)
    
    mean = np.mean(window)
    std = np.std(window)
    
    if std < 1e-8:
        return np.array([mean, 0.0, 0.0, 0.0])
    
    diff = window - mean
    var = std ** 2
    
    skew = np.mean(diff ** 3) / (std ** 3)
    kurt = np.mean(diff ** 4) / (std ** 4) - 3.0 # excess kurtosis
    
    return np.array([mean, var, skew, kurt])

def haar_dwt_1d(x, levels=2):
    """
    Computes manual 1D Discrete Haar Wavelet Transform.
    Pads signal to even lengths at each level if necessary.
    """
    coeffs = []
    curr_approx = np.array(x, dtype=float)
    
    for lvl in range(levels):
        n = len(curr_approx)
        if n <= 1:
            break
        if n % 2 != 0:
            curr_approx = np.append(curr_approx, curr_approx[-1])
            n += 1
            
        # Haar Decomposition
        approx = (curr_approx[0::2] + curr_approx[1::2]) / np.sqrt(2.0)
        detail = (curr_approx[0::2] - curr_approx[1::2]) / np.sqrt(2.0)
        
        coeffs.append(detail)
        curr_approx = approx
        
    coeffs.append(curr_approx) # append final approximation
    # Concatenate all coefficients into a single feature vector
    return np.concatenate(coeffs)

def extract_hybrid_features(gyro_even, acc_interp, W=5):
    """
    Extracts sliding window features for the signals.
    For each time step t, extracts context window, FFT, statistical moments, and DWT.
    
    gyro_even shape: (3, N)
    acc_interp shape: (N,)
    W: sliding window radius (window size = 2W + 1)
    
    Returns:
        features shape: (N, num_features)
    """
    N = len(acc_interp)
    W_size = 2 * W + 1
    
    # Pad signals along boundaries to support shifts
    gyro_padded = np.pad(gyro_even, ((0, 0), (W, W)), mode='edge') # shape: (3, N + 2W)
    acc_padded = np.pad(acc_interp, (W, W), mode='edge') # shape: (N + 2W,)
    
    feats_list = []
    
    for t in range(N):
        # Slice local windows
        acc_win = acc_padded[t : t + W_size] # length W_size
        gyro_win = gyro_padded[:, t : t + W_size] # shape (3, W_size)
        
        sample_feats = []
        
        # 1. Raw sliding window context
        sample_feats.append(acc_win)
        sample_feats.append(gyro_win.flatten())
        
        # 2. FFT magnitude features
        # FFT of accelerometer window
        acc_fft = np.abs(np.fft.rfft(acc_win))
        sample_feats.append(acc_fft)
        
        # FFT of gyroscope channels
        for axis in range(3):
            gyro_fft = np.abs(np.fft.rfft(gyro_win[axis]))
            sample_feats.append(gyro_fft)
            
        # 3. Statistical moments
        sample_feats.append(safe_stats(acc_win))
        for axis in range(3):
            sample_feats.append(safe_stats(gyro_win[axis]))
            
        # 4. Haar Wavelet transform features (DWT)
        sample_feats.append(haar_dwt_1d(acc_win, levels=2))
        for axis in range(3):
            sample_feats.append(haar_dwt_1d(gyro_win[axis], levels=2))
            
        # Combine all features for sample t
        sample_feats_concat = np.concatenate(sample_feats)
        feats_list.append(sample_feats_concat)
        
    return np.vstack(feats_list) # shape (N, num_features)
