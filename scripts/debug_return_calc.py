import json
from pathlib import Path

# Data from debate_slim_no_macro_019
initial_cash = 100000.0
allocation = {
    "AAPL": 0.3,
    "AMD": 0.05000000000000001,
    "MSFT": 0.3333333333333333,
    "NVDA": 0.10000000000000002,
    "_CASH_": 0.21666666666666667
}

tickers = ["AAPL", "AMD", "MSFT", "NVDA"]
year, q = "2022", "Q4"
p_year, p_q = "2022", "Q3"

daily_prices = {}
for t in tickers:
    p_path = Path(f"data-pipeline/daily_prices/data/{t}/{year}_{q}.json")
    with open(p_path) as f:
        invest_data = json.load(f)["daily_close"]
    
    prior_path = Path(f"data-pipeline/daily_prices/data/{t}/{p_year}_{p_q}.json")
    with open(prior_path) as f:
        prior_close = json.load(f)["daily_close"][-1]
    
    # Prepend
    bars = [prior_close] + invest_data
    daily_prices[t] = bars

# Simulation Logic from build_daily_equity_curve
n_days = 64
ticker_prices = {t: [b["close"] for b in bars] for t, bars in daily_prices.items()}

curve = []
for day_idx in range(n_days):
    day_value = 0.0
    for t, weight in allocation.items():
        if t in ticker_prices:
            prices = ticker_prices[t]
            day_value += weight * initial_cash * (prices[day_idx] / prices[0])
        else:
            day_value += weight * initial_cash
    curve.append(day_value)

print(f"Curve[0]:  {curve[0]:.2f}")
print(f"Curve[-1]: {curve[-1]:.2f}")
ret = ((curve[-1] - curve[0]) / curve[0]) * 100.0
print(f"Daily Metrics Return: {ret:.4f}%")
