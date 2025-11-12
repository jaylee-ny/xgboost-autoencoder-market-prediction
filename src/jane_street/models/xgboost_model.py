import numpy as np
import xgboost as xgb
from .base import BaseModel
import yaml


class XGBoostModel(BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__("XGBoost")
        
        if config:
            self.params = config['model']['xgboost'].copy()
            self.n_estimators = self.params.pop('n_estimators')
        else:
            # Fallback to kwargs for backward compatibility
            self.n_estimators = kwargs.get('n_estimators', 500)
            self.params = {
                'max_depth': kwargs.get('max_depth', 6),
                'learning_rate': kwargs.get('learning_rate', 0.1),
            }
        
        self.params.update({
            'objective': 'binary:logistic',
            'random_state': kwargs.get('random_state', 42)
        })
    
    def fit(self, X, y):
        """Train XGBoost."""
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        if X.isnull().any().any():
            raise ValueError("Features contain NaN values. Clean data first.")
        
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"Target must be binary (0/1), got values: {y.unique()}")
        
        if len(X) != len(y):
            raise ValueError(f"Feature/target length mismatch: {len(X)} vs {len(y)}")
        
        # Check for zero-variance features (XGBoost will fail silently)
        if (X.std() == 0).any():
            zero_var_cols = X.columns[X.std() == 0].tolist()
            raise ValueError(f"Zero-variance features detected: {zero_var_cols}")
    
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Return [prob_0, prob_1] for each sample."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        dtest = xgb.DMatrix(X)
        proba_pos = self.model.predict(dtest)
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
