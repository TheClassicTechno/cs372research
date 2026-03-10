import json
from pathlib import Path

tickers = ["AMD", "AMZN", "BAC", "CAT", "CVX", "PG", "RTX", "SLB"]
year = 2023
q = "Q2"

base_dir = Path("data-pipeline/daily_prices/data")

for t in tickers:
    path = base_dir / t / f"{year}_{q}.json"
    if not path.exists():
        print(f"[{t}] MISSING: {path}")
    else:
        with open(path, "r") as f:
            data = json.load(f)
            bars = data.get("daily_close", [])
            print(f"[{t}] FOUND: {len(bars)} daily bars in {path.name}")

# Also check SPY
spy_path = base_dir / "SPY" / f"{year}_{q}.json"
if not spy_path.exists():
    print(f"[SPY] MISSING: {spy_path}")
else:
    with open(spy_path, "r") as f:
        data = json.load(f)
        bars = data.get("daily_close", [])
        print(f"[SPY] FOUND: {len(bars)} daily bars in {spy_path.name}")

