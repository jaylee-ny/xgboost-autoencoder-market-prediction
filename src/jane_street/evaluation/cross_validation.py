import numpy as np
from sklearn.model_selection import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):
    """
    Walk-forward validation for time series.
    Train on past, test on future, with gap to prevent leakage.
    """
    
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits.
        
        Yields:
            train_idx, test_idx for each fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            test_idx = indices[test_start:test_end]
            
            train_end = test_start - self.gap
            train_idx = indices[:train_end]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def cross_validate(model, X, y, weights, returns, n_splits=5):
    """
    Cross-validate model with utility metric.
    
    Args:
        model: Model instance with fit/predict_proba methods
        X: Feature dataframe
        y: Target series
        weights: Sample weights
        returns: Sample returns
        n_splits: Number of CV folds
    
    Returns:
        dict with mean/std utility and individual fold scores
    """
    from .metrics import calculate_utility
    
    cv = TimeSeriesSplit(n_splits=n_splits, gap=1)
    utilities = []
    
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_test = weights.iloc[test_idx]
        r_test = returns.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        
        utility = calculate_utility(y_test, y_pred, w_test, r_test)
        utilities.append(utility)
        
        print(f"Fold {fold_num + 1}/{n_splits}: utility = {utility:.6f}")
    
    return {
        'mean_utility': np.mean(utilities),
        'std_utility': np.std(utilities),
        'utilities': utilities
    }
