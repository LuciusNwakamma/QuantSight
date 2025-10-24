from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252

def annualize_return(returns, periods_per_year: int = TRADING_DAYS):
    return returns.mean() * periods_per_year

def annualize_volatility(returns, periods_per_year: int = TRADING_DAYS):
    return returns.std(ddof=0) * np.sqrt(periods_per_year)

def _safe_div(num, den):
    """Divide num by den; if denominator is zero (scalar or elementwise), return NaN."""
    if isinstance(den, (pd.Series, pd.DataFrame)):
        den = den.replace(0, np.nan)
        return num / den
    return np.nan if (den == 0 or np.isnan(den)) else num / den

def sharpe_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS):
    excess = returns - rf / periods_per_year
    ann_ret = annualize_return(excess, periods_per_year)
    ann_vol = annualize_volatility(excess, periods_per_year)
    return _safe_div(ann_ret, ann_vol)

def sortino_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS):
    excess = returns - rf / periods_per_year
    downside = excess.copy()
    downside[downside > 0] = 0
    dd_vol = downside.std(ddof=0) * np.sqrt(periods_per_year)
    ann_ret = annualize_return(excess, periods_per_year)
    return _safe_div(ann_ret, dd_vol)

def max_drawdown(prices):
    if isinstance(prices, pd.DataFrame):
        return prices.apply(max_drawdown, axis=0)
    cummax = prices.cummax()
    drawdown = prices / cummax - 1.0
    return drawdown.min()

def var_historic(returns: pd.Series, level: float = 0.95):
    if returns.empty:
        return np.nan
    q = np.quantile(returns, 1 - level)
    return -q

def cvar_historic(returns: pd.Series, level: float = 0.95):
    if returns.empty:
        return np.nan
    cutoff = np.quantile(returns, 1 - level)
    tail_losses = returns[returns <= cutoff]
    return 0.0 if tail_losses.empty else -tail_losses.mean()

def var_parametric(returns: pd.Series, level: float = 0.95, method: str = "normal"):
    mu, sigma = returns.mean(), returns.std(ddof=0)
    if np.isnan(sigma) or sigma == 0:
        return np.nan
    if method == "normal":
        z = norm.ppf(1 - level)
        return -(mu + z * sigma)
    raise NotImplementedError("Only normal method implemented")

def summarize_risk(prices: pd.DataFrame, returns: pd.DataFrame, rf_annual: float = 0.0) -> pd.DataFrame:
    idx = returns.index.intersection(prices.index)
    returns, prices = returns.loc[idx], prices.loc[idx]

    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if r.empty:
            continue
        s = pd.Series({
            "ann_return": float(annualize_return(r)),
            "ann_vol": float(annualize_volatility(r)),
            "sharpe": float(sharpe_ratio(r, rf=rf_annual)),
            "sortino": float(sortino_ratio(r, rf=rf_annual)),
            "max_drawdown": float(max_drawdown(prices[col])),
            "VaR_95_hist": float(var_historic(r, 0.95)),
            "CVaR_95_hist": float(cvar_historic(r, 0.95)),
            "VaR_95_param": float(var_parametric(r, 0.95)),
        }, name=col)
        rows.append(s)
    return pd.DataFrame(rows)
