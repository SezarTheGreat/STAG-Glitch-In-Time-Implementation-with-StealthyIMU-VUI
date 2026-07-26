import os
import pickle
import numpy as np
import scipy.interpolate as interpolate
from sklearn.linear_model import Ridge
from projects.stag_original.src.pipeline.dataset import get_stag_bifurcation
from projects.interpolation_experiments.pipeline_variants import extract_features_from_interp

class ResidualCorrectionLayer:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        
    def train(self, train_rows, data_root, max_samples=200, W=2):
        """
        Trains the residual regressor on the difference between the cubic spline interpolation
        and the ground-truth target values.
        """
        X_data = []
        y_data = []
        
        count = 0
        for row in train_rows:
            if count >= max_samples:
                break
                
            uuid = row[0]
            duration = float(row[1])
            wav_path_rel = row[2]
            
            base_dir = os.path.dirname(wav_path_rel)
            acc_path = os.path.join(data_root, base_dir.replace('./', ''), f"{uuid}.acc")
            gyro_path = os.path.join(data_root, base_dir.replace('./', ''), f"{uuid}.gyro")
            
            if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
                continue
                
            try:
                acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                    acc_path, gyro_path, duration
                )
                
                # Perform baseline Cubic Spline interpolation
                cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
                acc_interp = cs(t_even)
                
                # Extract context window features
                feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
                
                # Calculate residual
                residual = acc_even_target - acc_interp
                
                X_data.append(feats)
                y_data.append(residual)
                count += 1
            except Exception:
                continue
                
        if len(X_data) > 0:
            X = np.vstack(X_data)
            y = np.concatenate(y_data)
            print(f"[INFO] Training residual regressor on {X.shape[0]} feature vectors...")
            self.model.fit(X, y)
        else:
            raise RuntimeError("No training data collected for residual correction layer.")
            
    def predict_residual(self, feats):
        return self.model.predict(feats)

def load_or_train_residual_model(train_rows, data_root, model_path, max_samples=200, W=2):
    expected_feats = 44 if W == 5 else 20
    if os.path.exists(model_path):
        print(f"[INFO] Loading existing residual model from {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                regressor = pickle.load(f)
            # Test predict to see if dimensions match
            regressor.model.predict(np.zeros((1, expected_feats)))
            return regressor
        except Exception as e:
            print(f"[WARN] Error loading model or feature mismatch: {e}. Re-training...")
            
    print("[INFO] Residual model not found or invalid. Starting training...")
    regressor = ResidualCorrectionLayer()
    regressor.train(train_rows, data_root, max_samples=max_samples, W=W)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(regressor, f)
    print(f"[SUCCESS] Saved residual model to {model_path}")
    return regressor
