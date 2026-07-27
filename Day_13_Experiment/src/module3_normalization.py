import numpy as np
import logging
from config import NormalizationConfig

logger = logging.getLogger("Day13_Module3")

def apply_device_independent_scaling(features: np.ndarray, speech_mask: np.ndarray, config: NormalizationConfig = None) -> np.ndarray:
    """
    Standardizes sensor features to generalize across hardware by calculating
    the mean and variance purely over the active speech regions, and applying
    it to the entire signal.
    
    Inputs:
        features: shape (Channels, Time)
        speech_mask: boolean mask shape (Time,)
    Returns:
        normalized: shape (Channels, Time)
    """
    if config is None:
        config = NormalizationConfig()
        
    logger.info(f"Module 3: Starting device independent normalization for {features.shape[0]} channels.")
    
    if not np.any(speech_mask):
        logger.warning("Speech mask is completely empty. Bypassing normalization to prevent zero division.")
        return features.copy()
        
    active_features = features[:, speech_mask]
    
    if config.use_robust_scaling:
        # Median and Interquartile Range
        median = np.median(active_features, axis=1, keepdims=True)
        q75, q25 = np.percentile(active_features, [75, 25], axis=1)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1.0, iqr)
        iqr = iqr.reshape(-1, 1)
        normalized = (features - median) / (iqr + config.epsilon)
    else:
        # Z-score using active regions
        mean = np.mean(active_features, axis=1, keepdims=True)
        std = np.std(active_features, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        normalized = (features - mean) / (std + config.epsilon)
        
    logger.info("Module 3 processing complete.")
    return normalized
