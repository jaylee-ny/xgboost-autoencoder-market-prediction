from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseModel
from .xgboost_model import XGBoostModel
from .autoencoder import AutoencoderMLP
from ..constants import XGB_WEIGHT, AE_WEIGHT


class Ensemble(BaseModel):
    
    def __init__(
        self,
        xgb_weight: float = XGB_WEIGHT,
        ae_weight: float = AE_WEIGHT,
        xgb_params: Optional[dict] = None,
        random_state: int = 42
    ):
        super().__init__("Ensemble")
        
        if not np.isclose(xgb_weight + ae_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {xgb_weight + ae_weight}"
            )
        
        self.xgb_weight = xgb_weight
        self.ae_weight = ae_weight
        
        self.xgb = XGBoostModel(params=xgb_params, random_state=random_state)
        self.ae = AutoencoderMLP(random_state=random_state)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Ensemble':
        print(f"  Training XGBoost (weight={self.xgb_weight:.1%})...")
        self.xgb.fit(X, y)
        
        print(f"  Training Autoencoder-MLP (weight={self.ae_weight:.1%})...")
        self.ae.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        xgb_proba = self.xgb.predict_proba(X)
        ae_proba = self.ae.predict_proba(X)
        
        ensemble_proba = self.xgb_weight * xgb_proba + self.ae_weight * ae_proba
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_individual_predictions(self, X: pd.DataFrame) -> dict:
        """Get predictions from each component for debugging."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        return {
            'xgb': self.xgb.predict_proba(X),
            'ae': self.ae.predict_proba(X),
            'ensemble': self.predict_proba(X)
        }
