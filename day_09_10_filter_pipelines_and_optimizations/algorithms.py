import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
from scipy.interpolate import make_interp_spline, Akima1DInterpolator

# ----------------- Time-Series Interpolators -----------------

def akima_interpolate(t_odd, val_odd, t_even):
    """
    Interpolates the 200Hz odd sensor stream onto the even timestamps using Akima Spline.
    Falls back to Cubic Spline for extrapolation boundary NaNs.
    """
    ak = Akima1DInterpolator(t_odd, val_odd)
    res = ak(t_even)
    if np.isnan(res).any():
        cs = interpolate.CubicSpline(t_odd, val_odd, extrapolate=True)
        cs_res = cs(t_even)
        res = np.where(np.isnan(res), cs_res, res)
    return res

def bspline_interpolate(t_odd, val_odd, t_even, degree=5):
    """
    Interpolates the 200Hz odd sensor stream onto the even timestamps using B-Spline.
    """
    bspl = make_interp_spline(t_odd, val_odd, k=degree)
    return bspl(t_even)

def lanczos_interpolate(t_odd, val_odd, t_even, a=3):
    """
    Interpolates the 200Hz odd sensor stream onto the even timestamps using Lanczos Resampling.
    """
    spacing = 5.0  # ms (200 Hz sample spacing)
    t_diff = (t_even[:, None] - t_odd[None, :]) / spacing
    
    sinc_x = np.sinc(t_diff)
    sinc_xa = np.sinc(t_diff / a)
    kernel = sinc_x * sinc_xa
    
    # Restrict to window support
    kernel = np.where((t_diff > -a) & (t_diff < a), kernel, 0.0)
    
    # Normalize to preserve signal gain
    kernel_sum = np.sum(kernel, axis=1, keepdims=True)
    kernel_sum = np.where(kernel_sum == 0, 1.0, kernel_sum)
    kernel = kernel / kernel_sum
    
    return kernel @ val_odd

def sinc_interpolate(t_odd, val_odd, t_even):
    """
    Interpolates using Whittaker-Shannon Sinc interpolation.
    """
    spacing = 5.0  # ms
    t_diff = (t_even[:, None] - t_odd[None, :]) / spacing
    kernel = np.sinc(t_diff)
    
    kernel_sum = np.sum(kernel, axis=1, keepdims=True)
    kernel_sum = np.where(kernel_sum == 0, 1.0, kernel_sum)
    kernel = kernel / kernel_sum
    
    return kernel @ val_odd

# ----------------- Pre-Processing Filters -----------------

def dwt_denoise(y, level=2, threshold=0.05):
    """
    Discrete Wavelet Transform (Haar Wavelet) Denoising.
    Selectively thresholds high-frequency details.
    """
    n = len(y)
    pad_len = 2**int(np.ceil(np.log2(n))) - n
    y_padded = np.pad(y, (0, pad_len), mode='edge')
    
    coeffs = []
    arr = y_padded.copy()
    for _ in range(level):
        approx = (arr[0::2] + arr[1::2]) / np.sqrt(2)
        detail = (arr[0::2] - arr[1::2]) / np.sqrt(2)
        # Soft thresholding
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        coeffs.append(detail)
        arr = approx
        
    for detail in reversed(coeffs):
        recond = np.zeros(len(arr) * 2)
        recond[0::2] = (arr + detail) / np.sqrt(2)
        recond[1::2] = (arr - detail) / np.sqrt(2)
        arr = recond
        
    return arr[:n]

def wiener_filter(y, mysize=5):
    """
    Applies an adaptive Wiener filter locally to the signal y.
    """
    return signal.wiener(y, mysize=mysize)

def kalman_denoise(y, q_noise=1.0, r_noise=1e-4):
    """
    Optimized RTS smoother with tuned process/measurement noise covariances.
    Avoids over-smoothing by keeping Q high relative to R.
    """
    N = len(y)
    dt = 1.0
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[dt**3/3.0, dt**2/2.0],
                  [dt**2/2.0, dt]]) * q_noise
    R = np.array([[r_noise]])
    
    x_pred = np.zeros((N, 2))
    P_pred = np.zeros((N, 2, 2))
    x_filt = np.zeros((N, 2))
    P_filt = np.zeros((N, 2, 2))
    
    x_filt[0] = [y[0], 0.0]
    P_filt[0] = np.eye(2) * 10.0
    
    for k in range(1, N):
        x_pred[k] = F @ x_filt[k-1]
        P_pred[k] = F @ P_filt[k-1] @ F.T + Q
        y_meas = y[k]
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T / S[0, 0]
        x_filt[k] = x_pred[k] + K.squeeze() * (y_meas - H @ x_pred[k])
        P_filt[k] = (np.eye(2) - K @ H) @ P_pred[k]
        
    x_smooth = np.zeros((N, 2))
    x_smooth[-1] = x_filt[-1]
    
    for k in range(N-2, -1, -1):
        J = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k+1])
        x_smooth[k] = x_filt[k] + J @ (x_smooth[k+1] - x_pred[k+1])
        
    return x_smooth[:, 0]

def apply_kalman_rts_3d_opt(val_3d, q_noise=1.0, r_noise=1e-4):
    smoothed = []
    for axis in range(val_3d.shape[0]):
        smoothed.append(kalman_denoise(val_3d[axis, :], q_noise, r_noise))
    return np.vstack(smoothed)
