import numpy as np

def apply_coherence_multiplier(acc_odd, gyro_even, t_odd, t_even):
    """
    Method 1: Coherence Multiplier
    Multiplies the Z-axis of the 200 Hz accelerometer with the X-axis of the 200 Hz gyroscope
    to create a DC bias that isolates coherent speech.
    
    Parameters:
        acc_odd (np.ndarray): Z-axis accelerometer at odd timestamps (length N)
        gyro_even (np.ndarray): Gyroscope (3, M) at even timestamps
        t_odd (np.ndarray): Timestamps for odd accelerometer samples
        t_even (np.ndarray): Timestamps for even gyroscope samples
    Returns:
        np.ndarray: Boosted acc_odd signal
    """
    # Interpolate Gyroscope X-axis (index 0) to odd time grid to align with acc_odd
    gyro_x_aligned = np.interp(t_odd, t_even, gyro_even[0, :])
    
    # Create coherence multiplier (multiply Z-axis accelerometer with aligned X-axis gyroscope)
    # Adding a scaling/bias term to ensure we isolate coherent speech without destroying signal shape
    acc_odd_boosted = acc_odd * (1.0 + np.abs(gyro_x_aligned))
    
    return acc_odd_boosted
