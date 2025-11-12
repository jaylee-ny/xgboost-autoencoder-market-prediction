"""
data -> features -> model -> evaluation.
"""
from .data.loader import DataLoader
from .features.pca import PCAReducer
from .models.xgboost_model import XGBoostModel
from .models.ensemble import Ensemble
from .evaluation.cross_validation import cross_validate


class Pipeline:
    
    def __init__(self, data_path, apply_pca=True, use_ensemble=True, random_state=42):
        self.data_path = data_path
        self.apply_pca = apply_pca
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        self.loader = DataLoader(data_path)
        self.pca = PCAReducer(random_state=random_state) if apply_pca else None
        
        if use_ensemble:
            self.model = Ensemble(random_state=random_state)
        else:
            self.model = XGBoostModel(random_state=random_state)
        
        self.X = None
        self.y = None
        self.weights = None
        self.returns = None
        self.metadata = None
        
    def load_data(self):
        """Load and preprocess data."""
        X, y, weights, metadata = self.loader.load()
        
        if self.pca:
            print(f"Applying PCA: {X.shape[1]} features -> ", end="")
            X = self.pca.fit_transform(X)
            print(f"{X.shape[1]} features")
            metadata['n_components'] = X.shape[1]
            metadata['variance_explained'] = self.pca.pca.explained_variance_ratio_.sum()
        
        self.X = X
        self.y = y
        self.weights = weights
        self.returns = y.copy()  # TODO: use actual returns if available
        self.metadata = metadata
        
        return X, y, weights, metadata
    
    def train(self):
        """Train model on loaded data."""
        if self.X is None:
            raise ValueError("Must load data first using load_data()")
        
        print(f"Training {self.model.name}...")
        self.model.fit(self.X, self.y)
        print("Training complete")
        
        return self.model
    
    def evaluate(self, n_splits=5):
        """Cross-validate model performance."""
        if self.X is None:
            raise ValueError("Must load data first using load_data()")
        
        print(f"\nRunning {n_splits}-fold cross-validation...")
        results = cross_validate(
            self.model,
            self.X,
            self.y,
            self.weights,
            self.returns,
            n_splits=n_splits
        )
        
        print(f"\nResults:")
        print(f"  Mean utility: {results['mean_utility']:.6f}")
        print(f"  Std utility:  {results['std_utility']:.6f}")
        
        return results
    
    def get_summary(self):
        """Get pipeline summary."""
        if self.metadata is None:
            return "No data loaded"
        
        summary = {
            'n_samples': self.metadata['n_samples'],
            'n_features_original': self.metadata['n_features'],
            'n_features_final': self.X.shape[1] if self.X is not None else None,
            'pca_applied': self.apply_pca,
            'model': self.model.name,
            'use_ensemble': self.use_ensemble,
        }
        
        if self.apply_pca and self.pca:
            summary['variance_explained'] = self.metadata.get('variance_explained')
        
        return summary


def create_pipeline(data_path='data/train.csv', apply_pca=True, use_ensemble=True, random_state=42):
    """Factory function to create configured pipeline."""
    return Pipeline(data_path, apply_pca=apply_pca, use_ensemble=use_ensemble, random_state=random_state)
