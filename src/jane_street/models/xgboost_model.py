from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from .base import BaseModel
from ..constants import XGBOOST_DEFAULTS, DECISION_THRESHOLD


class XGBoostModel(BaseModel):
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        n_estimators: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__("XGBoost")
        
        self.params = XGBOOST_DEFAULTS.copy()
        
        if params:
            self.params.update(params)
            self.n_estimators = self.params.pop('n_estimators', XGBOOST_DEFAULTS['n_estimators'])
        else:
            self.n_estimators = n_estimators if n_estimators else XGBOOST_DEFAULTS['n_estimators']
        
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        self.params['objective'] = 'binary:logistic'
        self.params['random_state'] = random_state
        
        self.model: Optional[xgb.Booster] = None
        self.threshold = DECISION_THRESHOLD
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        self._validate_inputs(X, y)
        
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        dtest = xgb.DMatrix(X)
        proba_pos = self.model.predict(dtest)
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is None:
            threshold = self.threshold
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        if len(X) != len(y):
            raise ValueError(f"Shape mismatch: {len(X)} vs {len(y)}")
        
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"NaN values in: {nan_cols[:5]}")
        
        if y.isnull().any():
            raise ValueError("Target contains NaN")
        
        unique_values = set(y.unique())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Target must be binary, got: {unique_values}")
        
        # Zero-variance features will cause XGBoost to fail silently
        zero_var_cols = X.columns[X.std() == 0].tolist()
        if zero_var_cols:
            raise ValueError(f"Zero-variance features: {zero_var_cols[:5]}")
        
        if np.isinf(X.values).any():
            raise ValueError("Features contain infinite values")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        return pd.Series(importance_dict).sort_values(ascending=False)
