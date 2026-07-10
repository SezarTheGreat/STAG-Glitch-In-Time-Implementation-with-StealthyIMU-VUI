import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.interpolate import make_interp_spline

# ----------------- Kalman Filter & RTS Smoother Helper -----------------
def rts_smoother_1d(y, q_noise=1e-3, r_noise=1e-1):
    """
    1D Rauch-Tung-Striebel (RTS) Smoother for signal y (N,).
    Uses a Constant Velocity (PV) state-space model:
    State: [position, velocity]^T
    """
    N = len(y)
    dt = 1.0  # Normalized time step
    
    # State transition matrix
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    # Measurement matrix
    H = np.array([[1.0, 0.0]])
    
    # Process noise covariance (Q) and Measurement noise covariance (R)
    Q = np.array([[dt**3/3.0, dt**2/2.0],
                  [dt**2/2.0, dt]]) * q_noise
    R = np.array([[r_noise]])
    
    # Pre-allocate arrays
    x_pred = np.zeros((N, 2))
    P_pred = np.zeros((N, 2, 2))
    x_filt = np.zeros((N, 2))
    P_filt = np.zeros((N, 2, 2))
    
    # Initial state
    x_filt[0] = [y[0], 0.0]
    P_filt[0] = np.eye(2) * 10.0
    
    # Forward Pass (Kalman Filter)
    for k in range(1, N):
        # Predict
        x_pred[k] = F @ x_filt[k-1]
        P_pred[k] = F @ P_filt[k-1] @ F.T + Q
        
        # Update
        y_meas = y[k]
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T / S[0, 0]
        
        x_filt[k] = x_pred[k] + K.squeeze() * (y_meas - H @ x_pred[k])
        P_filt[k] = (np.eye(2) - K @ H) @ P_pred[k]
        
    # Backward Pass (RTS Smoother)
    x_smooth = np.zeros((N, 2))
    P_smooth = np.zeros((N, 2, 2))
    
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for k in range(N-2, -1, -1):
        # Predict covariance step at next step
        J = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k+1])
        x_smooth[k] = x_filt[k] + J @ (x_smooth[k+1] - x_pred[k+1])
        P_smooth[k] = P_filt[k] + J @ (P_smooth[k+1] - P_pred[k+1]) @ J.T
        
    return x_smooth[:, 0]

def apply_kalman_rts_3d(val_3d, q_noise=1e-3, r_noise=1e-1):
    """
    Applies 1D RTS Smoother axis-by-axis to a (3, N) signal.
    """
    smoothed = []
    for axis in range(val_3d.shape[0]):
        smoothed.append(rts_smoother_1d(val_3d[axis, :], q_noise, r_noise))
    return np.vstack(smoothed)

# ----------------- Feature Extraction Helper (Locked Upstream Logic) -----------------
def extract_features_from_interp(gyro_even, acc_interp, W=2):
    """
    Extracts temporal window features exactly matching original StagUpscaler logic.
    gyro_even shape: (3, N)
    acc_interp shape: (N,)
    """
    N = len(acc_interp)
    
    # Pad signals along boundaries to support shifts
    gyro_padded = np.pad(gyro_even, ((0, 0), (W, W)), mode='edge')
    acc_padded = np.pad(acc_interp, (W, W), mode='edge')
    
    feats_list = []
    for shift in range(-W, W + 1):
        start_idx = shift + W
        end_idx = start_idx + N
        
        gyro_shift = gyro_padded[:, start_idx:end_idx].T
        acc_shift = acc_padded[start_idx:end_idx].reshape(-1, 1)
        feats_list.append(gyro_shift)
        feats_list.append(acc_shift)
        
    return np.hstack(feats_list)

# ----------------- Pipeline Reconstruction Variants -----------------
def reconstruct_baseline(acc_odd, gyro_even, t_odd, t_even, lgb_model, W=2):
    """
    Baseline pipeline (Cubic Spline interpolation + LightGBM prediction).
    """
    # Cubic spline interpolate odd acc to even grid
    cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
    acc_interp = cs(t_even)
    
    # Extract shift features and predict even acc values
    feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
    pred_even = lgb_model.predict(feats)
    
    # Interleave to produce 400 Hz output
    reconstructed = np.zeros(len(acc_odd) + len(pred_even))
    reconstructed[0::2] = acc_odd
    reconstructed[1::2] = pred_even
    return reconstructed

def reconstruct_variant1_bspline(acc_odd, gyro_even, t_odd, t_even, lgb_model, W=2, degree=5):
    """
    Variant 1: Higher-order B-Spline interpolation (degree 5) for pre-prediction alignment.
    """
    bspl = make_interp_spline(t_odd, acc_odd, k=degree)
    acc_interp = bspl(t_even)
    
    # Extract shift features and predict even acc values
    feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
    pred_even = lgb_model.predict(feats)
    
    # Interleave to produce 400 Hz output
    reconstructed = np.zeros(len(acc_odd) + len(pred_even))
    reconstructed[0::2] = acc_odd
    reconstructed[1::2] = pred_even
    return reconstructed

def reconstruct_variant2_kalman(acc_odd, gyro_even, t_odd, t_even, lgb_model, W=2):
    """
    Variant 2: Kalman (RTS) Smoothing applied to 200 Hz acc and gyro streams *before* spline interpolation.
    """
    acc_odd_filtered = rts_smoother_1d(acc_odd, q_noise=1e-3, r_noise=1e-1)
    gyro_even_filtered = apply_kalman_rts_3d(gyro_even, q_noise=1e-3, r_noise=1e-1)
    
    cs = interpolate.CubicSpline(t_odd, acc_odd_filtered, extrapolate=True)
    acc_interp = cs(t_even)
    
    feats = extract_features_from_interp(gyro_even_filtered, acc_interp, W=W)
    pred_even = lgb_model.predict(feats)
    
    reconstructed = np.zeros(len(acc_odd) + len(pred_even))
    reconstructed[0::2] = acc_odd_filtered
    reconstructed[1::2] = pred_even
    return reconstructed

def reconstruct_variant3_postfilter(acc_odd, gyro_even, t_odd, t_even, lgb_model, W=2):
    """
    Variant 3: Post-Correction Low-Pass Butterworth Filter applied to the 400 Hz reconstructed output.
    """
    cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
    acc_interp = cs(t_even)
    feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
    pred_even = lgb_model.predict(feats)
    
    reconstructed = np.zeros(len(acc_odd) + len(pred_even))
    reconstructed[0::2] = acc_odd
    reconstructed[1::2] = pred_even
    
    # 400 Hz sampling rate, cutoff at 80 Hz (Nyquist is 200 Hz)
    sos = signal.butter(4, 80.0, 'lowpass', fs=400.0, output='sos')
    filtered_reconstructed = signal.sosfiltfilt(sos, reconstructed)
    return filtered_reconstructed

def reconstruct_variant4_combined(acc_odd, gyro_even, t_odd, t_even, lgb_model, W=2):
    """
    Variant 4 (Combined): Pre-Interpolation Kalman (RTS) Smoothing + Post-Correction Butterworth Filter.
    """
    # 1. Pre-Interpolation Kalman filtering
    acc_odd_filtered = rts_smoother_1d(acc_odd, q_noise=1e-3, r_noise=1e-1)
    gyro_even_filtered = apply_kalman_rts_3d(gyro_even, q_noise=1e-3, r_noise=1e-1)
    
    # 2. Interpolate Odd Acc to Even grid
    cs = interpolate.CubicSpline(t_odd, acc_odd_filtered, extrapolate=True)
    acc_interp = cs(t_even)
    
    # 3. Extract shift features and predict even acc values
    feats = extract_features_from_interp(gyro_even_filtered, acc_interp, W=W)
    pred_even = lgb_model.predict(feats)
    
    # 4. Interleave
    reconstructed = np.zeros(len(acc_odd) + len(pred_even))
    reconstructed[0::2] = acc_odd_filtered
    reconstructed[1::2] = pred_even
    
    # 5. Post-Correction low-pass filtering
    sos = signal.butter(4, 80.0, 'lowpass', fs=400.0, output='sos')
    filtered_reconstructed = signal.sosfiltfilt(sos, reconstructed)
    return filtered_reconstructed

