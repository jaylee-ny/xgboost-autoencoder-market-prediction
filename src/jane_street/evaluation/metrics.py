import numpy as np


def calculate_utility(y_true, y_pred, weights, returns, threshold=0.5):
    """
    Formula from competition:
    - p_i = sum(weight * resp * action) for each date i
    - t = (sum(p_i) / sqrt(sum(p_i^2))) * sqrt(250 / n_days)
    - utility = min(max(t, 0), 6) * sum(p_i)
    
    For simplicity, using sum(p_i) as proxy since per-date aggregation requires date column.
    """
    actions = (y_pred >= threshold).astype(int)
    
    # Daily P&L proxy: weighted returns when action = 1
    daily_pnl = returns * weights * actions
    
    # Simplified utility: sum of profitable actions
    # Full implementation would require date grouping for Sharpe calculation
    utility = np.sum(daily_pnl)
    
    return utility


def calculate_utility_improvement(baseline_utility, improved_utility):
    """Percentage improvement over baseline."""
    if baseline_utility == 0:
        return 100.0 if improved_utility > 0 else 0.0
    
    improvement = (improved_utility - baseline_utility) / abs(baseline_utility) * 100
    return improvement


def calculate_transaction_costs(predictions, returns, weights, cost_bps=5, threshold=0.5):
    """
    Net utility after transaction costs.
    
    Args:
        predictions: Predicted probabilities
        returns: Actual returns
        weights: Sample weights
        cost_bps: Transaction cost in basis points
        threshold: Decision threshold
        
    Returns:
        dict with gross utility, costs, and net utility
    """
    actions = (predictions >= threshold).astype(int)
    
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
