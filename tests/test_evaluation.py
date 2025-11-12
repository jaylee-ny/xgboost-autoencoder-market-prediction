import pytest
import numpy as np
import pandas as pd

from jane_street.evaluation.metrics import (
    calculate_utility,
    calculate_utility_improvement,
    calculate_transaction_costs
)
from jane_street.evaluation.cross_validation import TimeSeriesSplit


def test_calculate_utility():
    """Test utility calculation."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.6, 0.3, 0.7, 0.4, 0.2])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    returns = np.array([0.01, -0.01, 0.02, 0.01, -0.01])
    
    utility = calculate_utility(y_true, y_pred, weights, returns)
    
    assert isinstance(utility, (int, float))


def test_utility_improvement():
    """Test improvement calculation."""
    improvement = calculate_utility_improvement(0.524, 0.587)
    
    assert improvement > 0
    assert abs(improvement - 12.0) < 1.0


def test_transaction_costs():
    """Test transaction cost calculation."""
    predictions = np.array([0.6, 0.3, 0.7, 0.4, 0.2])
    returns = np.array([0.01, -0.01, 0.02, 0.01, -0.01])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    result = calculate_transaction_costs(predictions, returns, weights)
    
    assert 'gross_utility' in result
    assert 'transaction_costs' in result
    assert 'net_utility' in result
    assert result['net_utility'] < result['gross_utility']


def test_time_series_split():
    """Test time series CV splits."""
    X = pd.DataFrame(np.random.randn(100, 5))
    cv = TimeSeriesSplit(n_splits=5)
    
    splits = list(cv.split(X))
    
    assert len(splits) == 5
    
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert max(train_idx) < min(test_idx)
