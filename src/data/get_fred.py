from pathlib import Path
import os
import pandas as pd
from fredapi import Fred

OUT = Path("data") / "raw"
OUT.mkdir(parents=True, exist_ok=True)

# Series IDs:
# DFF = Effective Fed Funds Rate (daily)
# CPIAUCSL = CPI All Urban Consumers (monthly, SA)
# UNRATE = Unemployment Rate (monthly)
SERIES = {
    "DFF": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
}

def main():
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set. In PowerShell: $env:FRED_API_KEY='your_key'")

    fred = Fred(api_key=key)
    frames = []
    for sid, name in SERIES.items():
        s = fred.get_series(sid)      # pandas Series indexed by date
        frames.append(s.rename(name).to_frame())

    df = pd.concat(frames, axis=1).sort_index()
    out_path = OUT / "macro_data.csv"
    df.to_csv(out_path)
    print("Saved:", out_path.resolve(), df.shape)

if __name__ == "__main__":
    main()
