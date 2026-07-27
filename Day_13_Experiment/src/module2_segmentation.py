import numpy as np
import scipy.signal
import scipy.ndimage
import logging
import logging
from config import SegmentationConfig, PreprocessingConfig

# Configure module logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Day13_Module2")

def compute_energy_envelope(accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """
    Computes the interaction energy envelope between accelerometer and gyroscope.
    InertiEAR principle: Axis multiplication amplifies correlated speech vibrations
    while suppressing uncorrelated mechanical noise.
    
    Inputs:
        accel: shape (3, T)
        gyro: shape (3, T)
    Returns:
        energy: shape (T,)
    """
    logger.debug("Computing cross-axis energy envelope.")
    # Sum of absolute magnitudes across axes
    accel_mag = np.sum(np.abs(accel), axis=0)
    gyro_mag = np.sum(np.abs(gyro), axis=0)
    
    # InertiEAR axis multiplication
    energy = accel_mag * gyro_mag
    return energy

def apply_otsu_threshold(energy: np.ndarray, bins: int = 256) -> float:
    """
    Calculates Otsu's threshold dynamically based on the energy histogram.
    
    Inputs:
        energy: shape (T,)
        bins: number of histogram bins
    Returns:
        threshold: float
    """
    logger.debug(f"Calculating Otsu's threshold with {bins} bins.")
    if np.max(energy) == np.min(energy):
        return np.mean(energy)
        
    hist, bin_edges = np.histogram(energy, bins=bins, range=(np.min(energy), np.max(energy)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Otsu's method
    total_weight = np.sum(hist)
    best_thresh = bin_centers[0]
    max_variance = 0.0
    
    weight_bg = 0.0
    sum_bg = 0.0
    sum_all = np.sum(bin_centers * hist)
    
    for i in range(bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total_weight - weight_bg
        if weight_fg == 0:
            break
            
        sum_bg += bin_centers[i] * hist[i]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        
        # Between class variance
        var_between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        
        if var_between > max_variance:
            max_variance = var_between
            best_thresh = bin_centers[i]
            
    return best_thresh

def smooth_boundaries(mask: np.ndarray, fill_gaps_samples: int, min_duration_samples: int) -> np.ndarray:
    """
    Fills small gaps in the speech mask and removes isolated short spikes.
    """
    logger.debug("Smoothing speech boundaries.")
    # Fill gaps (Morphological closing 1D)
    if fill_gaps_samples > 0:
        struct = np.ones(fill_gaps_samples, dtype=bool)
        mask = scipy.ndimage.binary_closing(mask, structure=struct)
        
    # Remove short durations (Morphological opening 1D)
    if min_duration_samples > 0:
        struct = np.ones(min_duration_samples, dtype=bool)
        mask = scipy.ndimage.binary_opening(mask, structure=struct)
        
    return mask

def segment_speech(accel: np.ndarray, gyro: np.ndarray, 
                   prep_config: PreprocessingConfig = None, 
                   seg_config: SegmentationConfig = None) -> np.ndarray:
    """
    Module 2 Main Pipeline:
    1. Compute interaction energy envelope.
    2. Smooth envelope.
    3. Apply Otsu's thresholding.
    4. Smooth and format boolean mask.
    
    Returns:
        boolean mask of shape (T,) where True indicates speech.
    """
    if prep_config is None:
        prep_config = PreprocessingConfig()
    if seg_config is None:
        seg_config = SegmentationConfig()
        
    logger.info("Module 2: Starting speech segmentation.")
    
    # 1. Energy Envelope
    energy = compute_energy_envelope(accel, gyro)
    
    # 2. Smooth Energy Envelope (low-pass)
    kernel_size = seg_config.energy_smooth_kernel
    if kernel_size > 0:
        # Use a simple moving average or median
        energy = scipy.signal.medfilt(energy, kernel_size=kernel_size)
        
    # 3. Otsu Threshold
    threshold = apply_otsu_threshold(energy, bins=seg_config.otsu_bins)
    mask = energy > threshold
    
    # 4. Smooth Boundaries
    # Convert ms to samples
    fill_gaps_samples = int((seg_config.fill_gaps_ms / 1000.0) * prep_config.sampling_rate_in)
    min_duration_samples = int((seg_config.min_speech_duration_ms / 1000.0) * prep_config.sampling_rate_in)
    
    final_mask = smooth_boundaries(mask, fill_gaps_samples, min_duration_samples)
    
    logger.info(f"Module 2 processing complete. Detected {np.sum(final_mask)} speech samples out of {len(final_mask)}.")
    return final_mask
