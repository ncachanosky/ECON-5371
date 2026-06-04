"""
download_data.py
================
One-time script to download all FRED and Yahoo Finance series used in
ECON-5371: Time Series Analysis and Forecasting.

Run this script once from the project root to populate the data/raw/
directory. All chapters read from these CSV files instead of hitting
live APIs at render time.

Usage
-----
    python download_data.py

Requirements
------------
    pip install pandas pandas_datareader yfinance --break-system-packages

Output
------
    data/raw/
        GDPC1.csv          Real GDP, quarterly, SA (billions 2017 USD)
        ND000334Q.csv      Real GDP, quarterly, NSA (billions 2017 USD)
        CPIAUCSL.csv       CPI All Items, monthly, SA (index 1982-84=100)
        CPIAUCNS.csv       CPI All Items, monthly, NSA (index 1982-84=100)
        PCEPILFE.csv       Core PCE Price Index, monthly, SA (index 2012=100)
        ICSA.csv           Initial Jobless Claims, weekly, NSA (thousands)
        FEDFUNDS.csv       Federal Funds Rate, monthly, percent
        GS10.csv           10-Year Treasury Yield, monthly, percent
        TB3MS.csv          3-Month Treasury Bill Rate, monthly, percent
        INDPRO.csv         Industrial Production Index, monthly, SA (2017=100)
        UNRATE.csv         Unemployment Rate, monthly, SA (percent)
        WTISPLC.csv        WTI Crude Oil Price, monthly, USD per barrel
        SP500.csv          S&P 500 daily closing prices (via Yahoo Finance)

    data/README.md         Documents every series.
"""

import time
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
FRED_START = "1947-01-01"
FRED_END   = "2024-12-31"
SP500_START = "1980-01-01"
SP500_END   = "2024-12-31"
DATA_DIR   = Path("data/raw")

# Timeout and retry settings
TIMEOUT    = 360   # seconds per request
RETRIES    = 5
WAIT       = 20    # seconds between retries

# ── FRED series catalogue ──────────────────────────────────────────────────────
FRED_SERIES = [
    ("GDPC1",     "GDPC1.csv",     "Real GDP, SA",                    "Quarterly", "Billions 2017 USD",  "1,2,3,4,5,6,7,11"),
    ("ND000334Q", "ND000334Q.csv", "Real GDP, NSA",                   "Quarterly", "Billions 2017 USD",  "2,4"),
    ("CPIAUCSL",  "CPIAUCSL.csv",  "CPI All Items, SA",               "Monthly",   "Index 1982-84=100",  "1,4,7"),
    ("CPIAUCNS",  "CPIAUCNS.csv",  "CPI All Items, NSA",              "Monthly",   "Index 1982-84=100",  "6"),
    ("PCEPILFE",  "PCEPILFE.csv",  "Core PCE Price Index, SA",        "Monthly",   "Index 2012=100",     "11"),
    ("ICSA",      "ICSA.csv",      "Initial Jobless Claims, NSA",     "Weekly",    "Thousands",          "3"),
    ("FEDFUNDS",  "FEDFUNDS.csv",  "Federal Funds Rate",              "Monthly",   "Percent",            "3,7"),
    ("GS10",      "GS10.csv",      "10-Year Treasury Yield",          "Monthly",   "Percent",            "8"),
    ("TB3MS",     "TB3MS.csv",     "3-Month Treasury Bill Rate",      "Monthly",   "Percent",            "8"),
    ("INDPRO",    "INDPRO.csv",    "Industrial Production Index, SA", "Monthly",   "Index 2017=100",     "11"),
    ("UNRATE",    "UNRATE.csv",    "Unemployment Rate, SA",           "Monthly",   "Percent",            "11"),
    ("WTISPLC",   "WTISPLC.csv",  "WTI Crude Oil Price",             "Monthly",   "USD per barrel",     "7"),
]

FRED_SERIES = [
    ("ICSA",      "ICSA.csv",      "Initial Jobless Claims, NSA",     "Weekly",    "Thousands",          "3"),
]

def download_fred(series_id, filename):
    """Download a single FRED series with retry logic."""
    import pandas_datareader as pdr
    pdr.fred.FredReader.timeout = TIMEOUT

    path = DATA_DIR / filename
    for attempt in range(1, RETRIES + 1):
        try:
            df = web.DataReader(series_id, "fred", FRED_START, FRED_END)
            df.index.name = "date"
            df.columns    = [series_id]
            df.to_csv(path)
            print(f"  OK  {series_id:<14} → {filename}  ({len(df):,} obs)")
            return
        except Exception as e:
            if attempt < RETRIES:
                print(f"  ... {series_id} attempt {attempt} failed, retrying in {WAIT}s  ({e})")
                time.sleep(WAIT)
            else:
                print(f"  ERR {series_id:<14}  All {RETRIES} attempts failed: {e}")


def download_sp500():
    """Download S&P 500 daily prices from Yahoo Finance."""
    path = DATA_DIR / "SP500.csv"
    try:
        df = yf.download("^GSPC",
                         start=SP500_START,
                         end=SP500_END,
                         auto_adjust=True,
                         progress=False)
        df.index.name = "date"
        df[["Close"]].to_csv(path)
        print(f"  OK  SP500 (^GSPC)     → SP500.csv  ({len(df):,} obs)")
    except Exception as e:
        print(f"  ERR SP500 (^GSPC)     {e}")


def write_readme():
    """Write data/README.md."""
    readme = DATA_DIR.parent / "README.md"
    lines = [
        "# ECON-5371 Data Directory\n",
        "All files in `raw/` were downloaded by `download_data.py` from FRED",
        "and Yahoo Finance. Re-run that script to refresh.\n",
        "## FRED Series\n",
        "| File | Series ID | Description | Frequency | Units | Chapters |",
        "|------|-----------|-------------|-----------|-------|----------|",
    ]
    for sid, fname, desc, freq, units, chs in FRED_SERIES:
        lines.append(f"| {fname} | {sid} | {desc} | {freq} | {units} | {chs} |")
    lines += [
        "",
        "## Yahoo Finance\n",
        "| File | Ticker | Description | Frequency | Chapters |",
        "|------|--------|-------------|-----------|----------|",
        "| SP500.csv | ^GSPC | S&P 500 daily closing price | Daily | 9 |",
        "",
        f"\n_Last updated: {datetime.today().strftime('%Y-%m-%d')}_",
    ]
    readme.write_text("\n".join(lines))
    print(f"\n  README written → {readme}")


def main():
    print("=" * 60)
    print("  ECON-5371 — Centralised Data Download")
    print(f"  FRED sample: {FRED_START} to {FRED_END}")
    print(f"  Timeout: {TIMEOUT}s  |  Retries: {RETRIES}  |  Wait: {WAIT}s")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading FRED series...")
    for sid, fname, *_ in FRED_SERIES:
        download_fred(sid, fname)

    print("\nDownloading Yahoo Finance series...")
    download_sp500()

    write_readme()

    print("\nDone. All files saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()