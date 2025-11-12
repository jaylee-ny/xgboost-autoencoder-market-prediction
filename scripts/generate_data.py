import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_jane_street_data(
    n_samples=10000,
    n_features=130,
    n_days=100,
    random_state=42
):
    """
    Generate synthetic HFT data matching Jane Street competition format.
    
    Args:
        n_samples: Total number of observations
        n_features: Number of features (130 in competition)
        n_days: Number of trading days
        random_state: Random seed
    """
    np.random.seed(random_state)
    
    print(f"Generating {n_samples:,} samples with {n_features} features...")
    
    # Generate dates
    dates = np.random.randint(0, n_days, n_samples)
    
    # Generate features with some correlation structure
    # First 10 features are "base" features
    base_features = np.random.randn(n_samples, 10)
    
    # Remaining features are combinations/transformations
    features = [base_features]
    for i in range(n_features - 10):
        # Mix of independent and correlated features
        if i % 3 == 0:
            # Independent noise
            features.append(np.random.randn(n_samples, 1))
        else:
            # Correlated with base features
            base_idx = i % 10
            correlated = base_features[:, base_idx:base_idx+1] + np.random.randn(n_samples, 1) * 0.5
            features.append(correlated)
    
    X = np.hstack(features)
    
    # Generate realistic returns (small values, normally distributed)
    returns = np.random.normal(0, 0.01, n_samples)
    
    # Generate binary target based on returns with some signal
    # Use weighted combination of features to create signal
    signal = np.dot(X[:, :5], np.array([0.1, -0.05, 0.08, -0.03, 0.06]))
    signal_prob = 1 / (1 + np.exp(-signal))
    resp = np.where(np.random.rand(n_samples) < signal_prob, 
                    np.abs(returns), -np.abs(returns))
    
    # Generate weights (mostly 1.0, some variation)
    weights = np.random.uniform(0.8, 1.2, n_samples)
    
    # Create DataFrame
    data = {
        'date': dates,
        'weight': weights,
        'resp': resp,
    }
    
    # Add feature columns
    for i in range(n_features):
        data[f'feature_{i}'] = X[:, i]
    
    df = pd.DataFrame(data)
    
    # Sort by date (time-series)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def main():
    """Generate and save synthetic data."""
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Generate data
    df = generate_synthetic_jane_street_data(
        n_samples=50000,  # Smaller for faster training
        n_features=130,
        n_days=500,
        random_state=42
    )
    
    # Save to CSV
    output_path = 'data/train.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSynthetic data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData statistics:")
    print(f"  Date range: {df['date'].min()} - {df['date'].max()}")
    print(f"  Unique days: {df['date'].nunique()}")
    print(f"  Mean return: {df['resp'].mean():.6f}")
    print(f"  Return std: {df['resp'].std():.6f}")
    print(f"  Positive returns: {(df['resp'] > 0).sum() / len(df):.1%}")


if __name__ == '__main__':
    main()
