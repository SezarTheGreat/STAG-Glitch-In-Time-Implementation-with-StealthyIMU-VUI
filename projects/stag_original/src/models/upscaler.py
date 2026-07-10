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
    Utilizes temporal sliding window context to improve modeling precision.
    """
    def __init__(self, n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42, W=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.W = W
        
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
        cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
        acc_interp = cs(t_even)
        return acc_interp

    def _extract_features(self, gyro_even, acc_interp):
        """
        Extracts temporal window features for a single sample.
        gyro_even shape: (3, N)
        acc_interp shape: (N,)
        """
        N = len(acc_interp)
        W = self.W
        
        # Pad signals along boundaries to support shifts
        gyro_padded = np.pad(gyro_even, ((0, 0), (W, W)), mode='edge') # shape: (3, N + 2W)
        acc_padded = np.pad(acc_interp, (W, W), mode='edge') # shape: (N + 2W,)
        
        feats_list = []
        for shift in range(-W, W + 1):
            start_idx = shift + W
            end_idx = start_idx + N
            
            gyro_shift = gyro_padded[:, start_idx:end_idx].T # shape: (N, 3)
            acc_shift = acc_padded[start_idx:end_idx].reshape(-1, 1) # shape: (N, 1)
            feats_list.append(gyro_shift)
            feats_list.append(acc_shift)
            
        return np.hstack(feats_list) # shape: (N, (3+1) * (2W+1))

    def fit(self, acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list, use_grid_search=False):
        """
        Trains the upscaler model on a list of samples.
        """
        X_features = []
        Y_targets = []
        
        for acc_odd, gyro_even, acc_even, t_odd, t_even in zip(acc_odd_list, gyro_even_list, acc_even_list, t_odd_list, t_even_list):
            acc_interp = self._get_cubic_spline_interp(acc_odd, t_odd, t_even)
            feats = self._extract_features(gyro_even, acc_interp)
            
            X_features.append(feats)
            Y_targets.append(acc_even)
            
        X = np.vstack(X_features)
        Y = np.concatenate(Y_targets)
        
        if use_grid_search and HAS_LIGHTGBM:
            print("Running 5-fold Grid Search for LightGBM...")
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [31, 63, 127]
            }
            grid_search = GridSearchCV(
                estimator=lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1),
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1
            )
            grid_search.fit(X, Y)
            self.model = grid_search.best_estimator_
            print(f"Best hyperparameters found: {grid_search.best_params_}")
            
            # Update instance parameters to reflect best params
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.learning_rate = grid_search.best_params_['learning_rate']
            self.max_depth = grid_search.best_params_['max_depth']
        else:
            # Fit model directly
            self.model.fit(X, Y)

    def predict_even(self, acc_odd, gyro_even, t_odd, t_even):
        """
        Predicts even accelerometer values for a single sample.
        """
        acc_interp = self._get_cubic_spline_interp(acc_odd, t_odd, t_even)
        feats = self._extract_features(gyro_even, acc_interp)
        
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
