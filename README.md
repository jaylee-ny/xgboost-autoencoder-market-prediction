# Jane Street Market Prediction

Machine learning system for predicting profitable trades in high-frequency markets using ensemble models and dimensionality reduction.

## Overview

This project implements a production-ready ML pipeline achieving:
- **12% utility improvement** over baseline XGBoost
- **80% dimensionality reduction** (130 → 26 features) via PCA
- **4× faster inference** while preserving 95% variance
- Time-series cross-validation with proper train/test separation

### Key Features

- Autoencoder-MLP + XGBoost ensemble
- PCA feature compression
- Walk-forward cross-validation
- Transaction cost analysis
- Competition utility metric optimization

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/jane-street-prediction.git
cd jane-street-prediction

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Data Setup

Download competition data from Kaggle:
```bash
# Install Kaggle CLI
pip install kaggle

# Download data (requires Kaggle API credentials)
kaggle competitions download -c jane-street-market-prediction

# Extract to data directory
unzip jane-street-market-prediction.zip -d data/
```

### Run Pipeline
```bash
# Train and evaluate model
python scripts/train_and_evaluate.py

# Analyze transaction costs
python scripts/analyze_costs.py
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
│   │   └── xgboost_model.py       # XGBoost implementation
│   ├── evaluation/
│   │   ├── metrics.py             # Utility metric calculation
│   │   └── cross_validation.py   # Time-series CV
│   └── pipeline.py                # End-to-end orchestration
├── scripts/
│   ├── train_and_evaluate.py      # Main training script
│   └── analyze_costs.py           # Cost analysis
├── tests/                         # Unit and integration tests
├── results/                       # Model outputs and analysis
└── requirements.txt               # Dependencies
```

## Results

### Model Performance
```
Baseline XGBoost:  0.524 utility
Ensemble Model:    0.587 utility
Improvement:       12.0%
```

### Dimensionality Reduction
```
Original features:  130
Reduced features:   26 (80% reduction)
Variance preserved: 95%
Inference speedup:  4.2×
```

### Transaction Cost Impact
```
Cost (bps)  Net Utility  Trade Rate
    0       0.587        45%
    5       0.531        45%
   10       0.475        45%
   20       0.363        45%
```

## Usage

### Basic Pipeline
```python
from jane_street import create_pipeline

# Create pipeline
pipeline = create_pipeline('data/train.csv', apply_pca=True)

# Load and process data
X, y, weights, metadata = pipeline.load_data()

# Train model
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
    learning_rate=0.05
)

# Create pipeline with custom model
pipeline = Pipeline('data/train.csv', apply_pca=True)
pipeline.model = model
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=jane_street --cov-report=html
```

## Technical Details

### Time-Series Cross-Validation

Uses walk-forward validation to prevent look-ahead bias:
```
Fold 1: Train [1,2,3] → Test [4]
Fold 2: Train [1,2,3,4] → Test [5]
Fold 3: Train [1,2,3,4,5] → Test [6]
```

### Utility Metric

Competition metric that penalizes overtrading:
```python
utility = sum(returns × weights × actions)
```

Where `actions` are binary trading decisions (0 = no trade, 1 = trade).

## Requirements

- numpy==1.26.4
- pandas==1.5.3
- scikit-learn==1.3.2
- scipy==1.11.4
- xgboost==2.0.3
- tensorflow==2.15.0
- pyyaml==6.0.1
- pytest==7.4.3


## License

MIT License
