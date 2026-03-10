import json
from pathlib import Path

tickers = ["AAPL", "AMD", "MSFT", "NVDA", "SPY"]
quarters = [("2022", "Q3"), ("2022", "Q4")]

for t in tickers:
    print(f"\nTicker: {t}")
    for y, q in quarters:
        p = Path(f"data-pipeline/daily_prices/data/{t}/{y}_{q}.json")
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            bars = data.get("daily_close", [])
            if bars:
                s = bars[0]
                e = bars[-1]
                print(f"  {y} {q}: Start={s['date']} ({s['close']:.4f}) End={e['date']} ({e['close']:.4f}) Count={len(bars)}")
            else:
                print(f"  {y} {q}: Empty bars")
        else:
            print(f"  {y} {q}: Missing file")
