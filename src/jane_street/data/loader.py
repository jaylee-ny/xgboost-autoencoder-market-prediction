import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.feature_cols = None
        
    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
        """
        Load Jane Street competition data and prepare for training.
        
        Key transformations:
        1. Binary target: resp > 0 (for classification loss)
        2. Actual returns: resp values (for utility metric)
        3. Weights: provided by competition (non-uniform sampling)
        
        Why both targets? XGBoost optimizes binary cross-entropy,
        but competition evaluates on weighted return utility.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data not found: {self.data_path}\n"
                "Download: kaggle competitions download -c jane-street-market-prediction"
            )
        
        df = pd.read_csv(self.data_path)
        
        self.feature_cols = [c for c in df.columns if c.startswith('feature_')]
        
        X = df[self.feature_cols]
        y = (df['resp'] > 0).astype(int)
        weights = df['weight'] if 'weight' in df.columns else None
        
        metadata = {
            'n_samples': len(df),
            'n_features': len(self.feature_cols),
            'n_days': df['date'].nunique() if 'date' in df.columns else None,
            'feature_names': self.feature_cols
        }
        
        return X, y, weights, metadata
