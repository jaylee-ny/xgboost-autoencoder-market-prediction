from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_cols: Optional[list] = None
        
    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, any]]:
        """
        Load competition data.
        
        Returns:
            X: Features
            y: Binary target (resp > 0) for classification training
            weights: Sample importance weights
            returns: Continuous resp values for utility calculation
            metadata: Dataset statistics
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        if 'resp' not in df.columns:
            raise ValueError("Missing required column: resp")
        
        self.feature_cols = [c for c in df.columns if c.startswith('feature_')]
        
        if len(self.feature_cols) == 0:
            raise ValueError("No feature columns found")
        
        X = df[self.feature_cols].copy()
        y = (df['resp'] > 0).astype(int)
        weights = df['weight'].copy() if 'weight' in df.columns else pd.Series(1.0, index=df.index)
        returns = df['resp'].copy()
        
        metadata = {
            'n_samples': len(df),
            'n_features': len(self.feature_cols),
            'n_days': df['date'].nunique() if 'date' in df.columns else None,
            'feature_names': self.feature_cols,
            'target_balance': y.mean(),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'mean_weight': weights.mean(),
        }
        
        return X, y, weights, returns, metadata
    
    def validate_features(self, X: pd.DataFrame) -> None:
        """Validate data quality."""
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"NaN values in columns: {nan_cols[:5]}")
        
        if np.isinf(X.values).any():
            raise ValueError("Features contain infinite values")
        
        zero_var_cols = X.columns[X.std() == 0].tolist()
        if zero_var_cols:
            raise ValueError(f"Zero-variance features: {zero_var_cols[:5]}")
