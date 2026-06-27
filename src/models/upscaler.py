import numpy as np
import scipy.interpolate as interpolate

# Defensive import of LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import RandomForestRegressor

class StagUpscaler:
    """
    STAG upscaling model to reconstruct 400 Hz accelerometer signal from 
    misaligned 200 Hz Accelerometer and Gyroscope streams.
    """
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )

    def _get_cubic_spline_interp(self, acc_odd, t_odd, t_even):
        """
        Performs Cubic Spline Interpolation on odd accelerometer samples at even timestamps.
        """
        # Set up cubic spline interpolator
        cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
        acc_interp = cs(t_even)
        return acc_interp

    def fit(self, acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list):
        """
        Trains the upscaler model on a list of samples.
        """
        X_features = []
        Y_targets = []
        
        for acc_odd, gyro_even, acc_even, t_odd, t_even in zip(acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list):
            # 1. Cubic spline interpolation
            acc_interp = self._get_cubic_spline_interp(acc_odd, t_odd, t_even)
            
            # 2. Features: gyro X (idx 0), gyro Y (idx 1), gyro Z (idx 2), acc_interp
            # gyro_even shape: (3, N) -> transpose to (N, 3)
            gyro_feats = gyro_even.T
            acc_interp_feat = acc_interp.reshape(-1, 1)
            
            feats = np.hstack([gyro_feats, acc_interp_feat]) # Shape: (N, 4)
            
            X_features.append(feats)
            Y_targets.append(acc_even)
            
        X = np.vstack(X_features)
        Y = np.concatenate(Y_targets)
        
        # Fit model
        self.model.fit(X, Y)

    def predict_even(self, acc_odd, gyro_even, t_odd, t_even):
        """
        Predicts even accelerometer values for a single sample.
        """
        acc_interp = self._get_cubic_spline_interp(acc_odd, t_odd, t_even)
        
        gyro_feats = gyro_even.T
        acc_interp_feat = acc_interp.reshape(-1, 1)
        
        feats = np.hstack([gyro_feats, acc_interp_feat])
        
        predicted_even = self.model.predict(feats)
        return predicted_even, acc_interp

    def reconstruct_signal(self, acc_odd, gyro_even, t_odd, t_even):
        """
        Reconstructs the full 400 Hz accelerometer signal by interleaving 
        true odd samples and predicted even samples.
        """
        predicted_even, acc_interp = self.predict_even(acc_odd, gyro_even, t_odd, t_even)
        
        # Interleave
        n_samples = len(acc_odd) + len(predicted_even)
        reconstructed = np.zeros(n_samples)
        
        reconstructed[0::2] = acc_odd
        reconstructed[1::2] = predicted_even
        
        return reconstructed
