from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "SPY"]
START = "2015-01-01"
END = None  # up to today

def main():
    outdir = Path("data") / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    # Do NOT group by ticker; keep the standard column layout
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=False,      # keep both Close and Adj Close if available
        group_by="column",      # important: prevents the 'ticker-first' layout
        threads=True,
        interval="1d",
    )

    # Robustly select the adjusted close (fallback to Close if needed)
    if isinstance(raw.columns, pd.MultiIndex):
        # top level names like 'Adj Close', 'Close', etc.
        top0 = raw.columns.get_level_values(0)
        if "Adj Close" in set(top0):
            df = raw["Adj Close"]
        else:
            df = raw["Close"]
    else:
        if "Adj Close" in raw.columns:
            df = raw["Adj Close"]
        else:
            df = raw["Close"]

    df.to_csv(outdir / "stock_prices.csv")
    print("Saved:", (outdir / "stock_prices.csv").resolve(), df.shape)

if __name__ == "__main__":
    main()

