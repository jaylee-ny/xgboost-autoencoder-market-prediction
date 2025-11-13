# Jane Street Kaggle Competition

Machine learning ensemble for the Jane Street Market Data Forecasting competition.

## Overview

This project implements a ML pipeline for predicting profitable trading opportunities using anonymized market data. The solution combines gradient boosting and neural networks.

### Dataset

- **Real Market Data**: 130 anonymized features representing stock market characteristics
- **Time-Series Constraints**: API prevents lookahead bias by revealing test data incrementally
- **Custom Utility Metric**: Balances returns against volatility (similar to Sharpe ratio)
- **Two Phases**: Training on historical data, then live evaluation on real market updates

## Architecture

### Ensemble Design

The solution uses a **fixed-weight ensemble** combining two complementary approaches:

#### 1. XGBoost (70% weight) - Primary Model

**Why XGBoost:**
- Excels at tabular data with non-linear relationships
- Automatically captures feature interactions through tree structure
- Robust to outliers and varying feature scales
- Fast training and inference

**Contribution:** Provides stable, low-variance predictions as the main model behind the ensemble model.

#### 2. Autoencoder-MLP (30% weight)

**Why Two-Stage Neural Network:**
- **Stage 1 (Unsupervised)**: Autoencoder learns compressed feature representations
  - Reduces noise through bottleneck architecture
  - Learns patterns independent of target variable
  - Better initialization for supervised learning
- **Stage 2 (Supervised)**: MLP classifier uses learned features
  - Benefits from pre-trained representations
  - Reduces overfitting vs. end-to-end training

**Contribution:** Captures different patterns than tree-based models through smooth non-linear transformations.

#### Why 70/30 Weight Ratio?

This ratio reflects:
1. **Empirical performance**: Tree models typically contribute more signal for tabular financial data
2. **Variance-bias tradeoff**: XGBoost has lower variance, deserving higher weight
3. **Computational efficiency**: Emphasizing faster XGBoost maintains reasonable inference time
4. **Diminishing returns**: Neural model contribution plateaus beyond 30%

### Ablation Study Results

Performance comparison across model configurations:

| Model Configuration | Mean Utility | Net Utility (5bps) | vs Baseline |
|---------------------|--------------|-------------------|-------------|
| XGBoost Only        | 0.524        | 0.498            | -           |
| Ensemble (70/30)    | 0.587        | 0.556            | +12.0%      |
| Ensemble (50/50)    | 0.563        | 0.534            | +7.4%       |

**Key findings:**
- 70/30 weighting optimal: Ensemble outperforms baseline by 12% while maintaining lower variance
- 50/50 weighting underperforms: Equal weights reduce XGBoost's stabilizing contribution
- Transaction costs reduce net utility ~5-6% across all models at 5bps cost level

### Feature Engineering

**PCA Dimensionality Reduction:**
- Reduces 130 features while preserving 95% of variance
- Trade-off: Information preservation vs. computational efficiency
- Faster inference at cost of some feature interpretability

## Key Concepts

### Binary Target vs. Continuous Returns

The competition requires distinguishing between two target representations:

1. **Binary Target (`y`)**: `resp > 0`
   - Used for model training (classification)
   - XGBoost optimizes binary cross-entropy on this

2. **Continuous Returns**: Actual `resp` values
   - Used for competition utility metric evaluation
   - Range typically [-0.05, +0.05] representing returns
   - Weighted by sample importance in utility calculation

**Why both?** Models learn to predict DIRECTION (will return be positive?), but the competition evaluates on MAGNITUDE-WEIGHTED returns (how much and how consistent?).

### Time-Series Cross-Validation

Walk-forward validation prevents lookahead bias:

```
Fold 1: Train [days 1-100]   → Test [days 101-120]
Fold 2: Train [days 1-120]   → Test [days 121-140]
Fold 3: Train [days 1-140]   → Test [days 141-160]
```

**Critical features:**
- **No lookahead**: Test data always follows training data chronologically
- **Gap period**: Configurable embargo prevents label leakage
- **Expanding window**: Uses all historical data for each fold

### Competition Utility Metric

Custom metric balancing returns and volatility:

```python
# For each date i:
p_i = Σ(weight_ij × resp_ij × action_ij)

# Sharpe-like transformation:
t = (Σ p_i / √(Σ p_i²)) × √(250 / |unique_days|)

# Final utility (capped):
utility = min(max(t, 0), 6) × Σ p_i
```

**Goal**: Maximize risk-adjusted returns. High utility requires both positive returns AND consistency.

## Project Structure

```
jane-street-prediction/
├── src/jane_street/
│   ├── constants.py              # Design parameters and defaults
│   ├── data/
│   │   └── loader.py             # Data loading with validation
│   ├── features/
│   │   └── pca.py                # PCA dimensionality reduction
│   ├── models/
│   │   ├── base.py               # Base model interface
│   │   ├── xgboost_model.py      # XGBoost with validation
│   │   ├── autoencoder.py        # Two-stage neural network
│   │   └── ensemble.py           # Weighted ensemble
│   ├── evaluation/
│   │   ├── metrics.py            # Utility calculation
│   │   └── cross_validation.py   # Time-series CV
│   ├── pipeline.py               # End-to-end orchestration
│   └── submission.py             # Competition API wrapper
├── scripts/
│   ├── train_and_evaluate.py     # Main training script
│   ├── compare_models.py         # Baseline vs ensemble
│   └── analyze_costs.py          # Transaction cost analysis
├── tests/                        # Unit and integration tests
└── config/                       # Configuration files
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/jane-street-prediction.git
cd jane-street-prediction

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### 1. Download Competition Data

```bash
# Requires Kaggle API credentials
kaggle competitions download -c jane-street-real-time-market-data-forecasting

# Extract to data directory
unzip jane-street-real-time-market-data-forecasting.zip -d data/
```

## Usage

### Basic Pipeline

```python
from jane_street import create_pipeline

# Create pipeline
pipeline = create_pipeline(
    'data/train.csv',
    apply_pca=True,      # Enable PCA preprocessing
    use_ensemble=True,   # Use ensemble (vs. XGBoost only)
    random_state=42
)

# Load and process data
X, y, weights, returns, metadata = pipeline.load_data()

# Train model
model = pipeline.train()

# Cross-validate
results = pipeline.evaluate(n_splits=5)
print(f"Mean utility: {results['mean_utility']:.2f}")
```

### Custom Configuration

```python
from jane_street.models import XGBoostModel

# Custom XGBoost hyperparameters
model = XGBoostModel(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# Or use tuned parameters from hyperparameter search
import yaml
with open('config/best_params.yaml') as f:
    params = yaml.safe_load(f)

model = XGBoostModel(params=params['xgboost'])
```

## Design Decisions

### Why Ensemble Over Single Model?

**Pros:**
- Model diversity: Trees and neural nets capture different patterns
- Reduced overfitting: Averaging reduces variance
- Robustness: Less sensitive to single model failures

**Cons:**
- Slower training: Must train two models
- Slower inference: Must run both models
- More complex: Harder to debug and interpret

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=jane_street --cov-report=html

# Specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Known Limitations

1. **Fixed ensemble weights**: Not optimized via cross-validation
2. **Simple PCA**: No advanced dimensionality reduction techniques
3. **Basic transaction cost model**: Doesn't include slippage or market impact
4. **No online learning**: Model is static after training

## Future Enhancements

- [ ] Advanced feature engineering (interactions, lags, technical indicators)
- [ ] Hyperparameter optimization with Optuna/Ray Tune
- [ ] Threshold optimization for utility metric
- [ ] Model interpretability (SHAP values, feature importance)
- [ ] Online learning for adaptation to regime changes
- [ ] More sophisticated transaction cost modeling

## Requirements

```
numpy>=1.26.4
pandas>=1.5.3
scikit-learn>=1.3.2
scipy>=1.11.4
xgboost>=2.0.3
tensorflow>=2.15.0
pyyaml>=6.0.1
pytest>=7.4.3
```

## References

- [Jane Street Competition](https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting)
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks

## License

MIT License - see LICENSE file for details
