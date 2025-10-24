from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/raw/stock_prices.csv")
OUT_PATH = Path("data/processed/returns.csv")

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute asset returns from price data.
    method: "log" for log returns, "pct" for simple percentage returns.
    Returns a DataFrame aligned to prices.index with NaNs dropped.
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "pct":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'pct'")
    return rets.dropna(how="all")

def main():
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    rets = compute_returns(prices, method="log")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rets.to_csv(OUT_PATH)
    print("Saved:", OUT_PATH.resolve(), rets.shape)

if __name__ == "__main__":
    main()
