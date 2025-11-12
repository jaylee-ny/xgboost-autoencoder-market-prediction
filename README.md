# Jane Street Market Prediction

Machine learning pipeline for predicting profitable trades using ensemble models and dimensionality reduction techniques.

## Overview

This project implements a modular ML pipeline with:
- **Ensemble architecture**: XGBoost + Autoencoder-MLP
- **PCA feature compression**: Reduces dimensionality while preserving variance
- **Time-series cross-validation**: Walk-forward validation preventing lookahead bias
- **Transaction cost modeling**: Realistic cost analysis for trading strategies
- **Competition utility metric**: Optimizes for risk-adjusted returns

## Key Features

### Architecture
- **Modular design**: Clean separation of data loading, feature engineering, modeling, and evaluation
- **Multiple models**: XGBoost (gradient boosting), Autoencoder-MLP (neural network), and ensemble
- **Configurable pipeline**: Easy to swap components and experiment with different approaches

### Feature Engineering
- **PCA dimensionality reduction**: Configurable variance threshold
- **Feature scaling**: Standardization before PCA transformation
- **Preserves feature interactions**: Linear combinations maintain information

### Evaluation
- **Time-series cross-validation**: Respects temporal ordering (no lookahead)
- **Competition utility metric**: Weighted return-based evaluation
- **Transaction cost analysis**: Tests strategy under realistic trading costs
- **Walk-forward validation**: Simulates real-world deployment

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/jane-street-prediction.git
cd jane-street-prediction

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```bash
# Download from Kaggle (requires API credentials)
kaggle competitions download -c jane-street-market-prediction

# Extract to data directory
unzip jane-street-market-prediction.zip -d data/

# Run pipeline
python scripts/train_and_evaluate.py
```

## Project Structure

```
jane-street-prediction/
├── src/jane_street/
│   ├── data/
│   │   └── loader.py              # Data loading and validation
│   ├── features/
│   │   └── pca.py                 # PCA dimensionality reduction
│   ├── models/
│   │   ├── base.py                # Base model interface
│   │   ├── xgboost_model.py       # XGBoost implementation
│   │   ├── autoencoder.py         # Autoencoder-MLP model
│   │   └── ensemble.py            # Ensemble combining models
│   ├── evaluation/
│   │   ├── metrics.py             # Utility metric calculation
│   │   └── cross_validation.py   # Time-series CV
│   └── pipeline.py                # End-to-end orchestration
├── scripts/
│   ├── train_and_evaluate.py      # Main training script
│   ├── compare_models.py          # Baseline vs ensemble comparison
│   ├── analyze_costs.py           # Transaction cost analysis
│   └── generate_data.py           # Synthetic data generator
├── tests/                         # Unit and integration tests
├── config/                        # Configuration files
├── results/                       # Model outputs and analysis
└── requirements.txt               # Dependencies
```

## Usage

### Basic Pipeline

```python
from jane_street import create_pipeline

# Create pipeline with PCA
pipeline = create_pipeline('data/train.csv', apply_pca=True)

# Load and process data
X, y, weights, metadata = pipeline.load_data()

# Train model (ensemble by default)
model = pipeline.train()

# Cross-validate
results = pipeline.evaluate(n_splits=5)
print(f"Mean utility: {results['mean_utility']:.6f}")
```

### Custom Configuration

```python
from jane_street import Pipeline, XGBoostModel

# Custom model parameters
model = XGBoostModel(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)

# Create pipeline with custom model
pipeline = Pipeline('data/train.csv', apply_pca=True, use_ensemble=False)
pipeline.model = model
```

### Compare Models

```python
# Train baseline XGBoost
baseline_pipeline = create_pipeline(
    'data/train.csv',
    apply_pca=True,
    use_ensemble=False
)

# Train ensemble
ensemble_pipeline = create_pipeline(
    'data/train.csv',
    apply_pca=True,
    use_ensemble=True
)

# Evaluate both
baseline_results = baseline_pipeline.evaluate()
ensemble_results = ensemble_pipeline.evaluate()

# Compare
improvement = (ensemble_results['mean_utility'] - baseline_results['mean_utility']) / baseline_results['mean_utility']
print(f"Ensemble improvement: {improvement:.1%}")
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=jane_street --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Technical Details

### Time-Series Cross-Validation

The pipeline uses walk-forward validation to prevent lookahead bias:

```
Fold 1: Train [days 1-100] → Test [days 101-120]
Fold 2: Train [days 1-120] → Test [days 121-140]
Fold 3: Train [days 1-140] → Test [days 141-160]
...
```

Key features:
- **Gap between train/test**: Configurable gap prevents label leakage
- **Expanding window**: Uses all historical data for training
- **Respects temporal order**: Maintains chronological sequence

### Competition Utility Metric

The Jane Street competition uses a custom utility metric:

```python
# For each trading day i:
p_i = Σ(weight_ij × resp_ij × action_ij)

# Sharpe-like transformation:
t = (Σ p_i / √(Σ p_i²)) × √(250 / |unique_days|)

# Final utility (capped):
utility = min(max(t, 0), 6) × Σ p_i
```

Where:
- `resp`: Return if trade is made
- `weight`: Sample importance weight
- `action`: Binary decision (1 = trade, 0 = pass)

**Goal**: Maximize risk-adjusted returns while avoiding overtrading.

### Model Ensemble Strategy

The ensemble combines two complementary approaches:

1. **XGBoost (70% weight)**:
   - Handles non-linear patterns and feature interactions
   - Robust to outliers
   - Fast inference

2. **Autoencoder-MLP (30% weight)**:
   - Learns latent feature representations
   - Captures different signal patterns
   - Unsupervised pre-training

Weight selection is based on typical performance patterns in tabular data competitions.

### Transaction Cost Analysis

The pipeline includes realistic transaction cost modeling:

```python
# Trading costs in basis points (bps)
cost_per_trade = cost_bps / 10000

# Total cost
total_cost = num_trades × cost_per_trade × notional_value

# Net utility
net_utility = gross_utility - total_cost
```

Test scenarios: 0, 5, 10, 20 bps to understand strategy robustness.

## Design Decisions

### Why PCA over Feature Selection?
- Preserves feature interactions (linear combinations)
- More stable than greedy selection methods
- Faster inference than tree ensembles on original features
- Trade-off: Less interpretable features

### Why Ensemble?
- XGBoost and neural networks capture different patterns
- Reduces overfitting through model averaging
- More robust predictions
- Trade-off: Slower training and inference

### Why Time-Series Split with Gap?
- Prevents label leakage in temporal data
- Gap ensures no information flow from future to past
- Simulates real-world deployment
- Trade-off: Less training data per fold

## Known Limitations

1. **Simple cost model**: Doesn't include slippage, market impact, or spread
2. **No feature engineering**: Uses raw features without domain-specific transformations
3. **Limited hyperparameter tuning**: Uses default values for most parameters

## Future Improvements

- [ ] Add hyperparameter optimization (Optuna, Ray Tune)
- [ ] Implement advanced feature engineering
- [ ] Add model interpretability (SHAP, feature importance)
- [ ] Optimize ensemble weights (stacking, learned blending)
- [ ] Add monitoring and logging infrastructure
- [ ] Implement more realistic market microstructure
- [ ] Add GPU acceleration for neural network training
- [ ] Create model versioning and experiment tracking

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

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks

## License

MIT License - see LICENSE file for details
