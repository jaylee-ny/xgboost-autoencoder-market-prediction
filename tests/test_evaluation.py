import pytest
import numpy as np
import pandas as pd

from jane_street.evaluation.metrics import (
    calculate_utility,
    calculate_utility_improvement,
    calculate_transaction_costs
)
from jane_street.evaluation.cross_validation import TimeSeriesSplit


def test_calculate_utility_correctness():
    """
    Test utility calculation with hand-calculated example.
    
    Setup: 5 samples, threshold=0.5
    - Sample 0: pred=0.6 (act=1), ret=0.01, wt=1.0 -> contrib=0.01
    - Sample 1: pred=0.3 (act=0), ret=-0.01, wt=1.0 -> contrib=0
    - Sample 2: pred=0.7 (act=1), ret=0.02, wt=1.0 -> contrib=0.02
    - Sample 3: pred=0.4 (act=0), ret=0.01, wt=1.0 -> contrib=0
    - Sample 4: pred=0.2 (act=0), ret=-0.01, wt=1.0 -> contrib=0
    
    Expected utility: 0.01 + 0.02 = 0.03
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.6, 0.3, 0.7, 0.4, 0.2])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    returns = np.array([0.01, -0.01, 0.02, 0.01, -0.01])
    
    utility = calculate_utility(y_true, y_pred, weights, returns)
    
    expected = 0.01 + 0.02  # Only samples 0 and 2 have pred >= 0.5
    assert np.isclose(utility, expected), f"Expected {expected}, got {utility}"


def test_calculate_utility_all_actions():
    """Test utility when all predictions trigger actions."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.7])
    weights = np.array([1.0, 1.0, 1.0])
    returns = np.array([0.01, 0.02, 0.01])
    
    utility = calculate_utility(y_true, y_pred, weights, returns)
    
    expected = 0.01 + 0.02 + 0.01
    assert np.isclose(utility, expected)


def test_calculate_utility_no_actions():
    """Test utility when no predictions trigger actions."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 1.0, 1.0])
    returns = np.array([0.01, 0.02, 0.01])
    
    utility = calculate_utility(y_true, y_pred, weights, returns)
    
    assert utility == 0.0


def test_utility_improvement():
    """Test improvement calculation."""
    improvement = calculate_utility_improvement(0.524, 0.587)
    
    assert improvement > 0
    assert abs(improvement - 12.0) < 1.0


def test_transaction_costs():
    """Test transaction cost calculation reduces utility."""
    predictions = np.array([0.6, 0.3, 0.7, 0.4, 0.2])
    returns = np.array([0.01, -0.01, 0.02, 0.01, -0.01])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    result = calculate_transaction_costs(predictions, returns, weights, cost_bps=5)
    
    assert 'gross_utility' in result
    assert 'transaction_costs' in result
    assert 'net_utility' in result
    assert result['transaction_costs'] > 0
    assert result['net_utility'] < result['gross_utility']
    assert result['num_trades'] == 2  # Predictions >= 0.5: samples 0, 2


def test_time_series_split():
    """Test time series CV splits maintain chronological order."""
    X = pd.DataFrame(np.random.randn(100, 5))
    cv = TimeSeriesSplit(n_splits=5)
    
    splits = list(cv.split(X))
    
    assert len(splits) == 5
    
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Train must precede test
        assert max(train_idx) < min(test_idx)
