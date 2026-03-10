import yfinance as yf
import datetime as dt

q_end = dt.date(2022, 9, 30)
start_long = q_end - dt.timedelta(days=600)
start_short = dt.date(2022, 7, 1)

df_long = yf.download("NVDA", start=start_long.isoformat(), end=q_end.isoformat(), progress=False)
df_short = yf.download("NVDA", start=start_short.isoformat(), end=q_end.isoformat(), progress=False)

print(f"Last Close (Long Range):  {df_long['Close'].iloc[-1].iloc[0] if isinstance(df_long['Close'].iloc[-1], (list, object)) else df_long['Close'].iloc[-1]:.6f}")
print(f"Last Close (Short Range): {df_short['Close'].iloc[-1].iloc[0] if isinstance(df_short['Close'].iloc[-1], (list, object)) else df_short['Close'].iloc[-1]:.6f}")
