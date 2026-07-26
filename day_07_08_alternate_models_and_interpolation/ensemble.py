import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

class StackingEnsemble:
    def __init__(self):
        self.meta_learner = Ridge(alpha=1.0)
        
    def fit(self, base_preds, y_true):
        """
        Fits the Ridge meta-regressor on base predictions.
        base_preds: numpy array of shape (N, 4)
        y_true: numpy array of shape (N,)
        """
        self.meta_learner.fit(base_preds, y_true)
        
    def predict(self, base_preds):
        """
        Predicts final value from base predictions.
        """
        return self.meta_learner.predict(base_preds)

class WeightedAveragingEnsemble:
    def __init__(self):
        self.weights = None
        
    def fit(self, base_preds, y_true):
        """
        Finds optimal weights that minimize MSE, constrained to sum to 1.
        base_preds: numpy array of shape (N, 4)
        y_true: numpy array of shape (N,)
        """
        def loss_fn(w):
            weighted_pred = np.dot(base_preds, w)
            return np.mean((weighted_pred - y_true) ** 2)
            
        # Initial guess: equal weights
        init_weights = np.ones(4) / 4.0
        # Constraints: weights sum to 1
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # Bounds: weights between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(4)]
        
        res = minimize(loss_fn, init_weights, constraints=cons, bounds=bounds, method='SLSQP')
        self.weights = res.x
        
    def predict(self, base_preds):
        if self.weights is None:
            raise ValueError("WeightedAveragingEnsemble must be fitted first.")
        return np.dot(base_preds, self.weights)

class VotingEnsemble:
    def predict(self, base_preds):
        """
        Uniform average of base predictions.
        base_preds: numpy array of shape (N, 4)
        """
        return np.mean(base_preds, axis=1)
