from pathlib import Path
import pandas as pd
from fredapi import Fred

# Put your FRED API key here (get one free at fred.stlouisfed.org)
FRED_API_KEY = "YOUR_API_KEY_HERE"

SERIES = {
    "CPI": "CPIAUCSL",       # Consumer Price Index (All Urban Consumers)
    "UNEMP": "UNRATE",       # Unemployment Rate
    "FEDFUNDS": "FEDFUNDS",  # Federal Funds Rate
    "GDP": "GDP"             # Gross Domestic Product
}

def main():
    fred = Fred(api_key=FRED_API_KEY)
    outdir = Path("data") / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    for name, code in SERIES.items():
        s = fred.get_series(code)
        df[name] = s

    df.index.name = "Date"
    df.to_csv(outdir / "macro_fred.csv")
    print("Saved:", (outdir / "macro_fred.csv").resolve(), df.shape)

if __name__ == "__main__":
    main()
