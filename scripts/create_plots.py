import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_model_comparison():
    """Create bar chart comparing baseline vs ensemble."""
    
    with open('results/model_comparison.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Utility comparison
    models = ['Baseline\n(XGBoost)', 'Ensemble\n(XGB+AE)']
    utilities = [
        results['baseline']['mean_utility'],
        results['ensemble']['mean_utility']
    ]
    std = [
        results['baseline']['std_utility'],
        results['ensemble']['std_utility']
    ]
    
    colors = ['#3498db', '#2ecc71']
    axes[0].bar(models, utilities, color=colors, alpha=0.7, yerr=std, capsize=5)
    axes[0].set_ylabel('Utility Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (v, s) in enumerate(zip(utilities, std)):
        axes[0].text(i, v + s + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
    
    # CV folds comparison
    fold_nums = list(range(1, len(results['baseline']['utilities']) + 1))
    
    axes[1].plot(fold_nums, results['baseline']['utilities'], 
                 marker='o', label='Baseline', linewidth=2)
    axes[1].plot(fold_nums, results['ensemble']['utilities'], 
                 marker='s', label='Ensemble', linewidth=2)
    axes[1].set_xlabel('CV Fold')
    axes[1].set_ylabel('Utility Score')
    axes[1].set_title('Cross-Validation Fold Performance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/model_comparison.png")
    
    plt.close()


def plot_transaction_costs():
    """Create plot showing transaction cost impact."""
    
    with open('results/model_comparison.json', 'r') as f:
        results = json.load(f)
    
    baseline_tc = results['transaction_costs']['baseline']
    ensemble_tc = results['transaction_costs']['ensemble']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Gross\nUtility', 'Transaction\nCosts', 'Net\nUtility']
    baseline_vals = [
        baseline_tc['gross_utility'],
        -baseline_tc['transaction_costs'],
        baseline_tc['net_utility']
    ]
    ensemble_vals = [
        ensemble_tc['gross_utility'],
        -ensemble_tc['transaction_costs'],
        ensemble_tc['net_utility']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7)
    ax.bar(x + width/2, ensemble_vals, width, label='Ensemble', alpha=0.7)
    
    ax.set_ylabel('Utility')
    ax.set_title('Transaction Cost Impact (5 bps)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/transaction_costs.png', dpi=300, bbox_inches='tight')
    print("Saved: results/transaction_costs.png")
    
    plt.close()


def main():
    """Generate all plots."""
    
    if not Path('results/model_comparison.json').exists():
        print("Error: results/model_comparison.json not found")
        print("Run: python scripts/compare_models.py first")
        return
    
    print("Generating plots...")
    plot_model_comparison()
    plot_transaction_costs()
    print("\nAll plots generated successfully")


if __name__ == '__main__':
    main()
