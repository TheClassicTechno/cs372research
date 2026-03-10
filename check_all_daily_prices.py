import json
import yaml
from pathlib import Path

# Need to parse all scenario files to build a comprehensive list of all tickers we actually care about
scenario_dir = Path("config/scenarios/baseline_sweep")
all_scenario_tickers = set()

for path in scenario_dir.rglob("*.yaml"):
    with open(path, "r") as f:
        try:
            scenario_config = yaml.safe_load(f)
            if scenario_config and "tickers" in scenario_config:
                all_scenario_tickers.update(scenario_config["tickers"])
        except Exception:
            pass

quarters = [
    "2021_Q4", "2022_Q1", "2022_Q2", "2022_Q3", "2022_Q4", 
    "2023_Q1", "2023_Q2", "2023_Q3", "2023_Q4", 
    "2024_Q1", "2024_Q2", "2024_Q3", "2024_Q4", 
    "2025_Q1"
]

base_dir = Path("data-pipeline/daily_prices/data")

missing = []
for t in all_scenario_tickers:
    if t == "_CASH_": continue
    for q in quarters:
        path = base_dir / t / f"{q}.json"
        if not path.exists():
            missing.append(f"{t} - {q}")

if missing:
    print(f"Found {len(missing)} missing daily price files across {len(all_scenario_tickers)} scenario tickers:")
    for m in missing[:50]:
        print(f"  {m}")
    if len(missing) > 50:
        print(f"  ... and {len(missing) - 50} more")
else:
    print(f"All daily price files present for all {len(all_scenario_tickers)} unique tickers found in baseline_sweep scenarios.")
