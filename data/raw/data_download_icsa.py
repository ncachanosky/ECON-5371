"""
download_icsa.py
================
Downloads ICSA (Initial Jobless Claims) in two parts and merges into one CSV.
Run from the project root:
    python download_icsa.py
"""

import time
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader as pdr
from pathlib import Path

pdr.fred.FredReader.timeout = 360

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARTS = [
    ("1967-01-01", "1995-12-31"),
    ("1996-01-01", "2024-12-31"),
]

RETRIES = 5
WAIT    = 20

def fetch_part(start, end):
    for attempt in range(1, RETRIES + 1):
        try:
            df = web.DataReader("ICSA", "fred", start, end)
            print(f"  OK  ICSA {start} to {end}  ({len(df):,} obs)")
            return df
        except Exception as e:
            if attempt < RETRIES:
                print(f"  ... attempt {attempt} failed, retrying in {WAIT}s  ({e})")
                time.sleep(WAIT)
            else:
                raise RuntimeError(f"All {RETRIES} attempts failed for {start}–{end}: {e}")

parts = []
for start, end in PARTS:
    parts.append(fetch_part(start, end))

icsa = pd.concat(parts)
icsa = icsa[~icsa.index.duplicated(keep="first")].sort_index()
icsa.index.name = "date"
icsa.columns    = ["ICSA"]

out = DATA_DIR / "ICSA.csv"
icsa.to_csv(out)
print(f"\n  Merged → {out}  ({len(icsa):,} total obs)")