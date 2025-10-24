# app/app.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from analytics.preprocess import compute_returns
from analytics.risk_metrics import summarize_risk
from analytics.optimizer import optimize_portfolio


st.set_page_config(page_title="QuantSight", layout="wide")
st.title("QuantSight: Portfolio Risk & Optimization Dashboard")

#Data loading with cache
raw_path = Path("data/raw/stock_prices.csv")

@st.cache_data(show_spinner=False)
def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df

if not raw_path.exists():
    st.warning("Run `python get_prices.py` first to download price data into data/raw/stock_prices.csv")
    st.stop()

prices_all = load_prices(raw_path)

# Sidebar controls (use unique keys!)
st.sidebar.header("Portfolio Settings")

tickers = st.sidebar.multiselect(
    label="Select Assets",
    options=list(prices_all.columns),
    default=list(prices_all.columns),
    key="tickers_select",
)

min_d = prices_all.index.min().date()
max_d = prices_all.index.max().date()
d1, d2 = st.sidebar.date_input(
    label="Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
    key="date_range",
)

rf = st.sidebar.number_input(
    label="Risk-free rate (annual, e.g., 0.02 for 2%)",
    min_value=0.0,
    max_value=1.0,
    value=0.02,
    step=0.001,
    key="rf_input",
)

# Guard against no selection
if not tickers:
    st.info("Please select at least one asset from the sidebar.")
    st.stop()

# Slice period & compute returns
prices = prices_all.loc[pd.Timestamp(d1):pd.Timestamp(d2), tickers]
returns = compute_returns(prices, method="log")

# Portfolio Overview
st.subheader("Portfolio Overview")
st.line_chart(prices)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = returns.corr()
st.dataframe(corr.style.background_gradient(cmap="coolwarm").format("{:.2f}"))
st.download_button(
    label="Download Correlation Matrix (CSV)",
    data=corr.to_csv().encode(),
    file_name="correlation_matrix.csv",
    mime="text/csv",
    key="dl_corr",
)

# Risk Metrics 
st.subheader("Risk Metrics")
risk_df = summarize_risk(prices, returns, rf_annual=rf)

fmt = {
    "ann_return": "{:.2%}",
    "ann_vol": "{:.2%}",
    "sharpe": "{:.3f}",
    "sortino": "{:.3f}",
    "max_drawdown": "{:.2%}",
    "VaR_95_hist": "{:.2%}",
    "CVaR_95_hist": "{:.2%}",
    "VaR_95_param": "{:.2%}",
}
st.dataframe(risk_df.style.format(fmt))

st.download_button(
    "Download risk metrics (CSV)",
    data=risk_df.to_csv().encode(),
    file_name="risk_metrics.csv",
    mime="text/csv",
    key="dl_risk",
)

# Optimization
st.subheader("Optimization")
col1, col2 = st.columns(2)

with col1:
    res_sharpe = optimize_portfolio(returns, rf_annual=rf, objective="max_sharpe")
    st.markdown("**Max Sharpe Portfolio**")
    st.write(res_sharpe["weights"].to_frame("weight").style.format("{:.2%}"))
    st.json({k: round(v, 6) for k, v in res_sharpe.items() if k in ("annual_return", "annual_vol")})

with col2:
    res_minv = optimize_portfolio(returns, rf_annual=rf, objective="min_vol")
    st.markdown("**Min Volatility Portfolio**")
    st.write(res_minv["weights"].to_frame("weight").style.format("{:.2%}"))
    st.json({k: round(v, 6) for k, v in res_minv.items() if k in ("annual_return", "annual_vol")})

st.caption("Note: Results are educational only and not investment advice.")

