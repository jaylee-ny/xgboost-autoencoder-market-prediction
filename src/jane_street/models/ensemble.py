import numpy as np
from .base import BaseModel
from .xgboost_model import XGBoostModel
from .autoencoder import AutoencoderMLP


class Ensemble(BaseModel):
    """Combine XGBoost and Autoencoder with fixed weights."""
    
    def __init__(self, xgb_weight=0.7, ae_weight=0.3, random_state=42):
        super().__init__("Ensemble")
        self.xgb_weight = xgb_weight
        self.ae_weight = ae_weight
        
        if not np.isclose(xgb_weight + ae_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {xgb_weight + ae_weight}")
        
        self.xgb = XGBoostModel(random_state=random_state)
        self.ae = AutoencoderMLP(random_state=random_state)
        
    def fit(self, X, y):
        """Train both models."""
        print(f"  Training XGBoost (weight={self.xgb_weight})...")
        self.xgb.fit(X, y)
        
        print(f"  Training Autoencoder-MLP (weight={self.ae_weight})...")
        self.ae.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Weighted average of both model predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        xgb_proba = self.xgb.predict_proba(X)
        ae_proba = self.ae.predict_proba(X)
        
        ensemble_proba = self.xgb_weight * xgb_proba + self.ae_weight * ae_proba
        
        return ensemble_proba
