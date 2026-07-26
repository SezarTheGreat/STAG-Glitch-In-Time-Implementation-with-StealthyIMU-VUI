import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate

def sinc_interpolate_1d(t_odd, x_odd, t_even):
    """
    Whittaker-Shannon (Sinc) Interpolation.
    Perfect reconstruction filter for band-limited signals.
    
    Parameters:
    -----------
    t_odd : np.ndarray
        Timestamps of the input signal (e.g., 200 Hz grid).
    x_odd : np.ndarray
        Signal values at t_odd.
    t_even : np.ndarray
        Target timestamps for interpolation (e.g., staggered 200 Hz grid).
    """
    if len(t_odd) == 0:
        return np.zeros_like(t_even)
        
    # Assume a regular sampling interval
    dt = np.mean(np.diff(t_odd))
    
    # Compute the sinc matrix
    # Sinc(x) = sin(pi * x) / (pi * x)
    # The term is: (t_even_j - t_odd_i) / dt
    t_diff = (t_even[:, np.newaxis] - t_odd[np.newaxis, :]) / dt
    sinc_matrix = np.sinc(t_diff)
    
    # Interpolated values
    x_even = np.dot(sinc_matrix, x_odd)
    return x_even


def lanczos_interpolate_1d(t_odd, x_odd, t_even, a=3):
    """
    Lanczos Interpolation.
    A windowed version of sinc interpolation that limits spatial support
    to prevent infinite-support ripples while retaining band-limiting properties.
    
    Parameters:
    -----------
    t_odd : np.ndarray
        Timestamps of the input signal.
    x_odd : np.ndarray
        Signal values at t_odd.
    t_even : np.ndarray
        Target timestamps for interpolation.
    a : int
        Lanczos kernel size parameter (typically 2 or 3).
    """
    if len(t_odd) == 0:
        return np.zeros_like(t_even)
        
    dt = np.mean(np.diff(t_odd))
    t_diff = (t_even[:, np.newaxis] - t_odd[np.newaxis, :]) / dt
    
    # Compute Lanczos kernel: sinc(x) * sinc(x/a) for -a < x < a, 0 otherwise
    # Avoid division by zero by using np.sinc
    val_sinc = np.sinc(t_diff)
    val_window = np.sinc(t_diff / a)
    
    kernel = val_sinc * val_window
    # Zero out elements where |t_diff| >= a
    kernel[np.abs(t_diff) >= a] = 0.0
    
    # Normalize rows to preserve signal amplitude (DC gain = 1)
    row_sums = np.sum(kernel, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    kernel = kernel / row_sums
    
    x_even = np.dot(kernel, x_odd)
    return x_even


def fourier_interpolate_1d(x_odd, num_target_samples):
    """
    Fourier (FFT-based) Interpolation.
    Zero-pads the DFT of the signal and takes the IDFT. 
    Equivalent to perfect sinc interpolation for periodic signals.
    
    Parameters:
    -----------
    x_odd : np.ndarray
        Input signal at uniformly spaced points.
    num_target_samples : int
        The target number of samples.
    """
    return signal.resample(x_odd, num_target_samples)


def apply_alternative_interpolation_3d(val_3d, t_odd, t_even, method="sinc", **kwargs):
    """
    Applies alternative interpolation methods axis-by-axis to (3, N) signals.
    """
    interpolated = []
    for axis in range(val_3d.shape[0]):
        if method == "sinc":
            interp_val = sinc_interpolate_1d(t_odd, val_3d[axis, :], t_even)
        elif method == "lanczos":
            a = kwargs.get("a", 3)
            interp_val = lanczos_interpolate_1d(t_odd, val_3d[axis, :], t_even, a=a)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        interpolated.append(interp_val)
    return np.vstack(interpolated)
