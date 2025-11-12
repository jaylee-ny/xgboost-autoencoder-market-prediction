import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    """Reduce dimensions"""
    
    def __init__(self, variance_threshold=0.95, random_state=42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components = None
        
    def fit(self, X):
        """Fit PCA on training data."""
        X_scaled = self.scaler.fit_transform(X)
        
        pca_temp = PCA(random_state=self.random_state)
        pca_temp.fit(X_scaled)
        
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
        
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        """Transform data to reduced dimensions."""
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        cols = [f'pc_{i}' for i in range(self.n_components)]
        return pd.DataFrame(X_reduced, columns=cols, index=X.index)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
