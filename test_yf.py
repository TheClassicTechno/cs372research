import yfinance as yf
import pandas as pd

yt = yf.Ticker("SLB")
hist = yt.history(start="2023-04-01", end="2023-06-30")
print(f"SLB 2023 Q2 history: {len(hist)} rows")
