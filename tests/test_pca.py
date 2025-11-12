import pytest
import pandas as pd
import numpy as np

from jane_street.features.pca import PCAReducer


def test_pca_reduces_dimensions():
    """Test PCA reduces feature count."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))
    
    reducer = PCAReducer(variance_threshold=0.95)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape[1] < X.shape[1]
    assert X_reduced.shape[0] == X.shape[0]


def test_pca_preserves_variance():
    """Test variance threshold is met."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))
    
    reducer = PCAReducer(variance_threshold=0.95)
    reducer.fit(X)
    
    variance_explained = reducer.pca.explained_variance_ratio_.sum()
    assert variance_explained >= 0.95


def test_pca_column_names():
    """Test output has proper column names."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))
    
    reducer = PCAReducer()
    X_reduced = reducer.fit_transform(X)
    
    assert all(col.startswith('pc_') for col in X_reduced.columns)
