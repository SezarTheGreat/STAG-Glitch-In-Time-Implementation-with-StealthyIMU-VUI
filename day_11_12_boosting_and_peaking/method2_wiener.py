import scipy.signal as signal

def apply_wiener_filter(acc_odd):
    """
    Method 2: Adaptive Wiener Filtering
    Apply a Wiener filter to the raw 200 Hz stream to estimate local signal variance,
    preserving sharp speech transients while smoothing noisy segments.
    
    Parameters:
        acc_odd (np.ndarray): Z-axis accelerometer at odd timestamps (length N)
    Returns:
        np.ndarray: Filtered acc_odd signal
    """
    # Use a small window size of 5 for local variance estimation in 200 Hz stream
    return signal.wiener(acc_odd, mysize=5)
