import numpy as np
import scipy.signal
import logging
from config import PreprocessingConfig

# Configure module logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Day13_Module1")

def apply_wiener(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Applies adaptive Wiener filtering along the time axis."""
    logger.debug(f"Applying Wiener filter with window size {window_size}")
    if signal.ndim == 1:
        return scipy.signal.wiener(signal, mysize=window_size)
    else:
        denoised = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            denoised[i, :] = scipy.signal.wiener(signal[i, :], mysize=window_size)
        return denoised

def remove_stationary_noise(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies a median filter to remove stationary noise (spikes)."""
    logger.debug(f"Applying median filter with kernel size {kernel_size}")
    if signal.ndim == 1:
        return scipy.signal.medfilt(signal, kernel_size=kernel_size)
    else:
        return scipy.signal.medfilt(signal, kernel_size=(1, kernel_size))

def remove_dc_bias(signal: np.ndarray) -> np.ndarray:
    """Removes DC bias by subtracting the mean of each channel."""
    logger.debug("Removing DC bias")
    if signal.ndim == 1:
        return signal - np.mean(signal)
    else:
        return signal - np.mean(signal, axis=1, keepdims=True)

def process_raw_imu(accel: np.ndarray, gyro: np.ndarray, config: PreprocessingConfig = None) -> tuple:
    """
    Module 1 Main Pipeline:
    1. Stationary noise removal
    2. Wiener filtering
    3. DC bias removal
    
    Expected input shape: (Channels, Time) -> e.g., (3, T)
    Returns: (accel_clean, gyro_clean)
    """
    if config is None:
        config = PreprocessingConfig()
        
    logger.info(f"Module 1: Processing raw IMU data. Accel shape: {accel.shape}, Gyro shape: {gyro.shape}")
    
    # 1. Stationary Noise Removal
    accel_clean = remove_stationary_noise(accel, kernel_size=config.median_filter_kernel_size)
    gyro_clean = remove_stationary_noise(gyro, kernel_size=config.median_filter_kernel_size)
    
    # 2. Wiener Filtering
    accel_clean = apply_wiener(accel_clean, window_size=config.wiener_window_size)
    gyro_clean = apply_wiener(gyro_clean, window_size=config.wiener_window_size)
    
    # 3. DC Bias Removal
    accel_clean = remove_dc_bias(accel_clean)
    gyro_clean = remove_dc_bias(gyro_clean)
    
    logger.info("Module 1 processing complete.")
    return accel_clean, gyro_clean
