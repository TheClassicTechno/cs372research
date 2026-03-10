import yfinance as yf
import pandas as pd

ticker = "NVDA"
date = "2022-09-30"

df = yf.download(ticker, start="2022-09-29", end="2022-10-02", progress=False)

if not df.empty:
    # Handle both MultiIndex and single index cases
    if isinstance(df.columns, pd.MultiIndex):
        row = df.loc[pd.Timestamp(date)]
        print("\nMultiIndex Columns and Values:")
        for col in df.columns:
            val = row[col]
            print(f"  {col}: {val:.6f}")
    else:
        row = df.loc[pd.Timestamp(date)]
        print("\nSingle Index Columns and Values:")
        for col in df.columns:
            val = row[col]
            print(f"  {col}: {val:.6f}")
else:
    print("No data found.")
