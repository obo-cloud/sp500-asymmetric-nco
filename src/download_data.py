import os
import time
import datetime
import requests
import pandas as pd
import numpy as np

DATA_DIR = r"d:\.TESIS\eksperimen\eksprimen 3\paper_sp500_nco\data"
os.makedirs(DATA_DIR, exist_ok=True)

# 11 S&P 500 Sector ETFs + Benchmarks
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services"
}

BENCHMARKS = ["SPY", "^VIX", "^IRX"]
ALL_TICKERS = list(SECTORS.keys()) + BENCHMARKS

start_dt = datetime.datetime(2018, 7, 1)
end_dt = datetime.datetime(2025, 12, 31)
start_ts = int(start_dt.timestamp())
end_ts = int(end_dt.timestamp())

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

daily_series = {}

print(f"[INFO] Fetching daily historical data from Yahoo Finance API ({start_dt.date()} to {end_dt.date()})...")

for ticker in ALL_TICKERS:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        
        # Check if adjclose exists in indicators
        if "adjclose" in result["indicators"] and result["indicators"]["adjclose"] and "adjclose" in result["indicators"]["adjclose"][0]:
            prices = result["indicators"]["adjclose"][0]["adjclose"]
        else:
            prices = result["indicators"]["quote"][0]["close"]
            
        dates = [datetime.datetime.utcfromtimestamp(ts).date() for ts in timestamps]
        s = pd.Series(prices, index=pd.to_datetime(dates), name=ticker)
        daily_series[ticker] = s
        print(f"  [+] {ticker:6s}: {len(s)} daily rows fetched.")
    except Exception as e:
        print(f"  [-] {ticker:6s}: FAILED ({e})")
    time.sleep(0.5)

# Combine into single DataFrame
df_daily = pd.DataFrame(daily_series)
df_daily = df_daily.sort_index()
df_daily = df_daily.ffill().bfill()

daily_csv = os.path.join(DATA_DIR, "sp500_daily_adjclose.csv")
df_daily.to_csv(daily_csv)
print(f"\n[SUCCESS] Daily prices saved to: {daily_csv} (Shape: {df_daily.shape})")

# Monthly Resampling (Month-End)
try:
    df_monthly_prices = df_daily.resample("ME").last()
except Exception:
    df_monthly_prices = df_daily.resample("M").last()

# Calculate Monthly Returns
df_monthly_returns = df_monthly_prices.pct_change().dropna()

# Sector returns
df_sector_returns = df_monthly_returns[list(SECTORS.keys())]
sector_csv = os.path.join(DATA_DIR, "sp500_sectors_monthly_returns.csv")
df_sector_returns.to_csv(sector_csv, float_format="%.6f")
print(f"[SUCCESS] Sector monthly returns saved to: {sector_csv} (Shape: {df_sector_returns.shape})")

# Macro / Benchmark returns & levels
# For ^VIX and ^IRX, we also save the end-of-month level as well as percentage change
df_macro_prices = df_monthly_prices[BENCHMARKS]
macro_prices_csv = os.path.join(DATA_DIR, "sp500_macro_levels_monthly.csv")
df_macro_prices.to_csv(macro_prices_csv, float_format="%.4f")

macro_returns_csv = os.path.join(DATA_DIR, "sp500_macro_monthly.csv")
df_monthly_returns[BENCHMARKS].to_csv(macro_returns_csv, float_format="%.6f")
print(f"[SUCCESS] Macro data saved to: {macro_prices_csv} & {macro_returns_csv}")

print("\n--- Summary of 11 S&P 500 Sectors Monthly Returns (2018-08 to 2025-12) ---")
summary_stats = pd.DataFrame({
    "Sector": [SECTORS[s] for s in df_sector_returns.columns],
    "Mean (Ann.)": df_sector_returns.mean() * 12 * 100,
    "Std (Ann.)": df_sector_returns.std() * np.sqrt(12) * 100,
    "Min (Monthly %)": df_sector_returns.min() * 100,
    "Max (Monthly %)": df_sector_returns.max() * 100
})
print(summary_stats.to_string())
