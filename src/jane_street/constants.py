"""
Configuration constants for Jane Street market prediction.

Design decisions:
- XGBoost 70% / Autoencoder 30%: Tree models provide stable baseline, neural adds diversity
- PCA 95% variance: Balance between speed and information preservation
- CV gap=1 day: Prevents label leakage in temporal features
"""

# Ensemble weights - XGBoost primary (stable), Autoencoder diversity
XGB_WEIGHT = 0.7
AE_WEIGHT = 0.3

# Decision threshold for binary actions
DECISION_THRESHOLD = 0.5

# PCA configuration
PCA_VARIANCE_THRESHOLD = 0.95
N_PCA_COMPONENTS_MIN = 10

# Time-series cross-validation
CV_N_SPLITS = 5
CV_GAP_DAYS = 1

# Transaction costs
DEFAULT_COST_BPS = 5
COST_BPS_SCENARIOS = [0, 5, 10, 20]

# XGBoost defaults
XGBOOST_DEFAULTS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
}

# Autoencoder defaults
AUTOENCODER_DEFAULTS = {
    'encoding_dim': 32,
    'epochs': 50,
    'batch_size': 1024,
    'learning_rate': 0.001,
}

RANDOM_SEED = 42
