import scipy.signal as signal

def apply_highpass_filter(acc_odd, cutoff=80.0, fs=200.0):
    """
    Method 4: Targeted High-Pass Filter
    Apply an 80 Hz High-Pass Filter to the raw 200 Hz stream to strip out low-frequency human motion interference
    while preserving the 85-255 Hz speech band.
    
    Parameters:
        acc_odd (np.ndarray): Z-axis accelerometer at odd timestamps (length N)
        cutoff (float): High-pass filter cutoff in Hz (80 Hz)
        fs (float): Sampling frequency of raw stream (200 Hz)
    Returns:
        np.ndarray: Filtered acc_odd signal
    """
    # 80 Hz cutoff at 200 Hz sampling rate requires a normalized frequency of 80 / 100 = 0.8
    # Since Nyquist is fs/2 = 100 Hz, 80 Hz is extremely close to Nyquist.
    # If the cutoff is too close to Nyquist, filter design can fail or blow up.
    # We will clip the normalized cutoff to 0.79 to ensure numerical stability.
    nyq = 0.5 * fs
    norm_cutoff = min(0.79, cutoff / nyq)
    
    sos = signal.butter(4, norm_cutoff, btype='highpass', output='sos')
    return signal.sosfiltfilt(sos, acc_odd)
