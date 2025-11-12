import numpy as np


def calculate_utility(y_true, y_pred, weights, returns):
    """
    Args:
        y_true: Actual binary labels (for interface consistency)
        y_pred: Predicted probabilities
        weights: Sample weights from competition
        returns: Actual returns from competition
        
    Returns:
        Utility score (higher is better)
    """
    actions = (y_pred >= 0.5).astype(int)
    utility = np.sum(returns * weights * actions)
    return utility


def calculate_utility_improvement(baseline_utility, improved_utility):
    if baseline_utility == 0:
        return 100.0 if improved_utility > 0 else 0.0
    
    improvement = (improved_utility - baseline_utility) / abs(baseline_utility) * 100
    return improvement


def calculate_transaction_costs(predictions, returns, weights, cost_bps=5):
    """
    Args:
        predictions: Predicted probabilities
        returns: Actual returns
        weights: Sample weights
        cost_bps: Transaction cost in basis points (default 5 bps)
        
    Returns:
        dict with gross utility, costs, and net utility
    """
    actions = (predictions >= 0.5).astype(int)
    
    gross_utility = np.sum(returns * weights * actions)
    
    num_trades = np.sum(actions)
    positions = np.abs(returns * weights)
    notional_traded = np.sum(positions[actions == 1])
    total_costs = (cost_bps / 10000) * notional_traded
    
    net_utility = gross_utility - total_costs
    
    return {
        'gross_utility': gross_utility,
        'transaction_costs': total_costs,
        'net_utility': net_utility,
        'num_trades': num_trades,
        'trade_rate': num_trades / len(predictions)
    }
