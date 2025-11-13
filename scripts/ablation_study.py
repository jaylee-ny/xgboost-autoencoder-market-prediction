import sys
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from jane_street import create_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Ablation study: Compare XGBoost-only, Autoencoder-only, and Ensemble.
    Test different ensemble weight ratios.
    """
    
    logger.info("="*60)
    logger.info("Ablation Study: Model Comparison")
    logger.info("="*60)
    
    results = {}
    
    # XGBoost only
    logger.info("\n[1/3] Training XGBoost only...")
    xgb_pipeline = create_pipeline(
        'data/train.csv',
        apply_pca=True,
        use_ensemble=False,
        random_state=42
    )
    
    X, y, weights, returns, metadata = xgb_pipeline.load_data()
    xgb_pipeline.train()
    xgb_results = xgb_pipeline.evaluate(n_splits=5, cost_bps=5)
    
    results['xgboost_only'] = {
        'mean_utility': xgb_results['mean_utility'],
        'std_utility': xgb_results['std_utility'],
        'mean_net_utility': xgb_results['mean_net_utility'],
    }
    
    # Ensemble 70/30 (default)
    logger.info("\n[2/3] Training Ensemble (70% XGB / 30% AE)...")
    ensemble_pipeline = create_pipeline(
        'data/train.csv',
        apply_pca=True,
        use_ensemble=True,
        random_state=42
    )
    
    ensemble_pipeline.load_data()
    ensemble_pipeline.train()
    ensemble_results = ensemble_pipeline.evaluate(n_splits=5, cost_bps=5)
    
    results['ensemble_70_30'] = {
        'mean_utility': ensemble_results['mean_utility'],
        'std_utility': ensemble_results['std_utility'],
        'mean_net_utility': ensemble_results['mean_net_utility'],
    }
    
    # Ensemble 50/50
    logger.info("\n[3/3] Training Ensemble (50% XGB / 50% AE)...")
    from jane_street.models.ensemble import Ensemble
    
    ensemble_50_pipeline = create_pipeline(
        'data/train.csv',
        apply_pca=True,
        use_ensemble=False,  # We'll manually set ensemble
        random_state=42
    )
    
    ensemble_50_pipeline.load_data()
    ensemble_50_pipeline.model = Ensemble(xgb_weight=0.5, ae_weight=0.5, random_state=42)
    ensemble_50_pipeline.train()
    ensemble_50_results = ensemble_50_pipeline.evaluate(n_splits=5, cost_bps=5)
    
    results['ensemble_50_50'] = {
        'mean_utility': ensemble_50_results['mean_utility'],
        'std_utility': ensemble_50_results['std_utility'],
        'mean_net_utility': ensemble_50_results['mean_net_utility'],
    }
    
    # Calculate improvements
    baseline = results['xgboost_only']['mean_utility']
    for config, res in results.items():
        if config != 'xgboost_only':
            improvement = ((res['mean_utility'] - baseline) / abs(baseline)) * 100
            res['improvement_pct'] = improvement
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/ablation_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*60)
    logger.info(f"\n{'Model':<25} {'Utility':>12} {'Net (5bps)':>12} {'Improvement':>12}")
    logger.info("-"*60)
    
    for config, res in results.items():
        config_name = config.replace('_', ' ').title()
        improvement_str = f"+{res.get('improvement_pct', 0):.1f}%" if 'improvement_pct' in res else "-"
        logger.info(f"{config_name:<25} {res['mean_utility']:>12.6f} {res['mean_net_utility']:>12.6f} {improvement_str:>12}")
    
    logger.info("="*60)
    logger.info(f"Results saved to: results/ablation_study.json")
    logger.info("="*60)
    
    return results


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.info("\nPlease download data first:")
        logger.info("  kaggle competitions download -c jane-street-market-prediction")
        sys.exit(1)
