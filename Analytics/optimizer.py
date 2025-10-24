from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov: pd.DataFrame, periods_per_year: int = TRADING_DAYS):
    port_return = np.dot(weights, mean_returns) * periods_per_year
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov * periods_per_year, weights)))
    return port_return, port_vol

def neg_sharpe(weights, mean_returns, cov, rf=0.0, periods_per_year: int = TRADING_DAYS):
    pret, pvol = portfolio_performance(weights, mean_returns, cov, periods_per_year)
    if pvol == 0:
        return 1e6
    return -(pret - rf) / pvol

def minimize_volatility(weights, mean_returns, cov, periods_per_year: int = TRADING_DAYS):
    return portfolio_performance(weights, mean_returns, cov, periods_per_year)[1]

def optimize_portfolio(returns: pd.DataFrame, rf_annual: float = 0.0, objective: str = "max_sharpe"):
    returns = returns.dropna(how="any")
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    x0 = np.repeat(1/n, n)

    if objective == "max_sharpe":
        res = minimize(neg_sharpe, x0, args=(mean_returns, cov, rf_annual), method="SLSQP",
                       bounds=bounds, constraints=constraints)
    elif objective == "min_vol":
        res = minimize(minimize_volatility, x0, args=(mean_returns, cov), method="SLSQP",
                       bounds=bounds, constraints=constraints)
    else:
        raise ValueError("objective must be 'max_sharpe' or 'min_vol'")
    weights = pd.Series(res.x, index=mean_returns.index)
    pret, pvol = portfolio_performance(weights.values, mean_returns, cov)
    return {"weights": weights, "annual_return": pret, "annual_vol": pvol, "success": res.success}
