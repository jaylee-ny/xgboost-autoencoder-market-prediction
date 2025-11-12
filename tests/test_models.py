import pytest
import pandas as pd
import numpy as np

from jane_street.models.xgboost_model import XGBoostModel


def test_xgboost_trains():
    """Test XGBoost can train on data."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randint(0, 2, 100)
    
    model = XGBoostModel(n_estimators=10)
    model.fit(X, y)
    
    assert model.is_fitted


def test_xgboost_predicts():
    """Test XGBoost makes predictions."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randint(0, 2, 100)
    
    model = XGBoostModel(n_estimators=10)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_xgboost_predict_proba():
    """Test XGBoost returns probabilities."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randint(0, 2, 100)
    
    model = XGBoostModel(n_estimators=10)
    model.fit(X, y)
    
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
