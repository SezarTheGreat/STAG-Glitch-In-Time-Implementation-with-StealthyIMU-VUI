import numpy as np
import scipy.interpolate as interpolate
import logging
from config import StagConfig

logger = logging.getLogger("Day13_Module4")

class StagEngine:
    """
    Module 4: STAG Reconstruction Engine.
    Upsamples 200 Hz normalized sensor features to a 400 Hz acoustic-equivalent waveform.
    Uses Cubic Spline Interpolation for the base upsampling and a two-stage 
    (mocked/wrapped) LightGBM fusion to refine the predictions.
    """
    def __init__(self, config: StagConfig = None):
        self.config = config if config else StagConfig()
        
        # Here we mock the presence of the two-stage LightGBM models.
        # In a full training cycle, these would be loaded from disk or trained.
        self.lgbm_stage1 = None
        self.lgbm_stage2 = None

    def _cubic_spline_upsample(self, target_signal: np.ndarray) -> np.ndarray:
        """
        Upsamples a 200Hz signal to 400Hz using Cubic Spline Interpolation.
        Input: (T,)
        Output: (2T,)
        """
        T = len(target_signal)
        t_200 = np.arange(T) * 5.0  # 5ms spacing for 200Hz
        t_400 = np.arange(T * 2) * 2.5 # 2.5ms spacing for 400Hz
        
        cs = interpolate.CubicSpline(t_200, target_signal, extrapolate=True)
        return cs(t_400)
        
    def _extract_features(self, features_200: np.ndarray, upsampled_target: np.ndarray) -> np.ndarray:
        """
        Constructs a feature matrix for the LightGBM models using sliding windows.
        Simplified version of the StagUpscaler feature extractor for inference.
        Returns features of shape (2T, num_features).
        """
        T_sub = features_200.shape[1]
        T_400 = T_sub * 2
        W = self.config.window_size
        
        # Num features = (6 channels) * (2W + 1)
        num_feats = features_200.shape[0] * (2 * W + 1)
        
        # Mock feature matrix for the API contract
        feat_matrix = np.zeros((T_400, num_feats), dtype=np.float32)
        return feat_matrix

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Expected Input: 
            features: shape (Channels, T)
        Expected Output: 
            reconstructed: shape (1, 2T)
        """
        logger.info(f"Module 4: Starting STAG Engine reconstruction on input shape {features.shape}.")
        
        target_axis = self.config.target_axis
        target_signal = features[target_axis, :]
        
        # 1. Base Interpolation
        base_400 = self._cubic_spline_upsample(target_signal)
        
        # 2. Feature Extraction for Fusion
        lgbm_features = self._extract_features(features, base_400)
        
        # 3. LightGBM Stage 1 & Stage 2 Fusion
        reconstructed = base_400.copy()
        
        # Return as (1, 2T)
        reconstructed = reconstructed.reshape(1, -1)
        
        logger.info(f"Module 4 processing complete. Output shape {reconstructed.shape}.")
        return reconstructed
