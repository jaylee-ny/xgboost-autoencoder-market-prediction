import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from jane_street import create_pipeline


def main():
    """Run complete training and evaluation pipeline."""
    
    pipeline = create_pipeline(
        data_path='data/train.csv',
        apply_pca=True,
        random_state=42
    )
    
    print("="*60)
    print("Jane Street Market Prediction")
    print("="*60)
    
    print("\n1. Loading data...")
    X, y, weights, metadata = pipeline.load_data()
    print(f"   Loaded {metadata['n_samples']:,} samples")
    print(f"   Features: {metadata['n_features']} -> {X.shape[1]}")
    if 'variance_explained' in metadata:
        print(f"   Variance explained: {metadata['variance_explained']:.1%}")
    
    print("\n2. Training model...")
    model = pipeline.train()
    
    print("\n3. Cross-validating...")
    results = pipeline.evaluate(n_splits=5)
    
    print("\n" + "="*60)
    print(f"Final Mean Utility: {results['mean_utility']:.6f}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nSetup instructions:")
        print("  1. Download data from Kaggle:")
        print("     kaggle competitions download -c jane-street-market-prediction")
        print("  2. Extract to data/ directory")
        print("  3. Run again")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
