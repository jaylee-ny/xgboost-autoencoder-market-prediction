# Ensemble weights (must sum to 1.0)
XGB_WEIGHT = 0.7  # XGBoost contribution
AE_WEIGHT = 0.3   # Autoencoder contribution

# Feature engineering
PCA_VARIANCE_THRESHOLD = 0.95  # Retain 95% of variance
N_PCA_COMPONENTS_MIN = 10      # Minimum components regardless of variance

# Cross-validation
CV_N_SPLITS = 5           # Number of folds
CV_GAP_DAYS = 1          # Gap between train/test (prevents leakage)

# Transaction costs
DEFAULT_COST_BPS = 5     # 5 basis points per trade
COST_BPS_SCENARIOS = [0, 5, 10, 20]  # For sensitivity analysis

# Random seed for reproducibility
RANDOM_SEED = 42
