import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from jane_street import create_pipeline
from jane_street.evaluation.metrics import calculate_transaction_costs


def main():
    """Analyze transaction cost impact."""
    
    print("="*60)
    print("Transaction Cost Analysis")
    print("="*60)
    
    pipeline = create_pipeline('data/train.csv', apply_pca=True)
    
    print("\nLoading and processing data...")
    X, y, weights, metadata = pipeline.load_data()
    
    print("Training model...")
    model = pipeline.train()
    
    print("Generating predictions...")
    predictions = model.predict_proba(X)[:, 1]
    
    print("\n" + "="*60)
    print("Cost Impact Analysis")
    print("="*60)
    
    for cost_bps in [0, 5, 10, 20]:
        result = calculate_transaction_costs(
            predictions,
            pipeline.returns.values,
            weights.values,
            cost_bps=cost_bps
        )
        
        print(f"\nTransaction cost: {cost_bps} bps")
        print(f"  Gross utility: {result['gross_utility']:.6f}")
        print(f"  Transaction costs: {result['transaction_costs']:.6f}")
        print(f"  Net utility: {result['net_utility']:.6f}")
        print(f"  Trade rate: {result['trade_rate']:.1%}")
        print(f"  Cost impact: {(result['transaction_costs']/result['gross_utility'])*100:.1f}%")
    
    print("\n" + "="*60)
    
    with open('results/transaction_cost_analysis.txt', 'w') as f:
        f.write("Transaction Cost Analysis\n")
        f.write("="*60 + "\n\n")
        
        for cost_bps in [0, 5, 10, 20]:
            result = calculate_transaction_costs(
                predictions,
                pipeline.returns.values,
                weights.values,
                cost_bps=cost_bps
            )
            
            f.write(f"Transaction cost: {cost_bps} bps\n")
            f.write(f"  Gross utility: {result['gross_utility']:.6f}\n")
            f.write(f"  Net utility: {result['net_utility']:.6f}\n")
            f.write(f"  Trade rate: {result['trade_rate']:.1%}\n\n")
    
    print("\nResults saved to results/transaction_cost_analysis.txt")


if __name__ == '__main__':
    main()
