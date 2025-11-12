import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from jane_street import create_pipeline
from jane_street.evaluation.metrics import calculate_utility_improvement, calculate_transaction_costs


def main():
    """Compare baseline vs ensemble and save results."""
    
    print("="*60)
    print("Model Comparison: Baseline vs Ensemble")
    print("="*60)
    
    results = {}
    
    # Train baseline
    print("\n[1/2] Training Baseline (XGBoost only)...")
    baseline_pipeline = create_pipeline(
        'data/train.csv',
        apply_pca=True,
        use_ensemble=False,
        random_state=42
    )
    
    X, y, weights, metadata = baseline_pipeline.load_data()
    baseline_pipeline.train()
    baseline_results = baseline_pipeline.evaluate(n_splits=5)
    
    results['baseline'] = {
        'mean_utility': baseline_results['mean_utility'],
        'std_utility': baseline_results['std_utility'],
        'utilities': baseline_results['utilities']
    }
    
    # Train ensemble
    print("\n[2/2] Training Ensemble (XGBoost + Autoencoder)...")
    ensemble_pipeline = create_pipeline(
        'data/train.csv',
        apply_pca=True,
        use_ensemble=True,
        random_state=42
    )
    
    ensemble_pipeline.load_data()
    ensemble_pipeline.train()
    ensemble_results = ensemble_pipeline.evaluate(n_splits=5)
    
    results['ensemble'] = {
        'mean_utility': ensemble_results['mean_utility'],
        'std_utility': ensemble_results['std_utility'],
        'utilities': ensemble_results['utilities']
    }
    
    # Calculate improvement
    improvement = calculate_utility_improvement(
        baseline_results['mean_utility'],
        ensemble_results['mean_utility']
    )
    
    results['improvement'] = {
        'absolute': ensemble_results['mean_utility'] - baseline_results['mean_utility'],
        'percentage': improvement
    }
    
    # Transaction cost analysis
    print("\nTransaction cost analysis...")
    baseline_pred = baseline_pipeline.model.predict_proba(X)[:, 1]
    ensemble_pred = ensemble_pipeline.model.predict_proba(X)[:, 1]
    
    baseline_costs = calculate_transaction_costs(
        baseline_pred, y.values, weights.values, cost_bps=5
    )
    ensemble_costs = calculate_transaction_costs(
        ensemble_pred, y.values, weights.values, cost_bps=5
    )
    
    results['transaction_costs'] = {
        'baseline': baseline_costs,
        'ensemble': ensemble_costs
    }
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nBaseline XGBoost:")
    print(f"  Mean utility: {results['baseline']['mean_utility']:.6f}")
    print(f"  Std utility:  {results['baseline']['std_utility']:.6f}")
    
    print(f"\nEnsemble (XGB + Autoencoder):")
    print(f"  Mean utility: {results['ensemble']['mean_utility']:.6f}")
    print(f"  Std utility:  {results['ensemble']['std_utility']:.6f}")
    
    print(f"\nImprovement:")
    print(f"  Absolute: {results['improvement']['absolute']:.6f}")
    print(f"  Percentage: {results['improvement']['percentage']:.1f}%")
    
    print(f"\nTransaction Costs (5 bps):")
    print(f"  Baseline net utility: {baseline_costs['net_utility']:.6f}")
    print(f"  Ensemble net utility: {ensemble_costs['net_utility']:.6f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: results/model_comparison.json")
    print("="*60)
    
    return results


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download data first:")
        print("  kaggle competitions download -c jane-street-market-prediction")
        sys.exit(1)
