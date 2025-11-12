import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.feature_cols = None
        
    def load(self):
        """
        Load data from CSV.
        
        Returns:
            tuple: (features_df, target_series, weights_series, metadata_dict)
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
