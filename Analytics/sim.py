import numpy as np
import pandas as pd

TRADING_DAYS = 252

def mc_portfolio_paths(returns: pd.DataFrame, weights: pd.Series, years: int = 1, sims: int = 5000, seed: int | None = 42):
    """Geometric Brownian-ish bootstrap using historical daily returns."""
    rng = np.random.default_rng(seed)
    R = returns.dropna().values  # T x N daily log-returns
    T, N = R.shape
    path_len = years * TRADING_DAYS
    w = weights.values.reshape(1, -1)
    # Sample with replacement from historical joint returns
    samples = R[rng.integers(0, T, size=(sims, path_len)), :]   # sims x path_len x N
    port_log = (samples @ w.T).squeeze(-1)                       # sims x path_len
    # cumulate from 1.0
    cum = np.exp(np.cumsum(port_log, axis=1))
    return cum  # sims x path_len
