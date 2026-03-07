import os
import json
from pathlib import Path
from collections import defaultdict

def list_tickers_by_quarter():
    data_dir = Path("data-pipeline/quarterly_asset_details/data")
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found.")
        return

    quarter_to_tickers = defaultdict(list)

    # Iterate through ticker directories
    for ticker_dir in data_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            # Iterate through JSON files in each ticker directory
            for json_file in ticker_dir.glob("*.json"):
                quarter = json_file.stem  # e.g., "2022_Q4"
                quarter_to_tickers[quarter].append(ticker)

    # Sort quarters and tickers
    sorted_quarters = sorted(quarter_to_tickers.keys())
    
    print("Tickers available per quarter in data-pipeline/quarterly_asset_details/data:")
    print("=" * 80)
    for quarter in sorted_quarters:
        tickers = sorted(quarter_to_tickers[quarter])
        print(f"{quarter}: {', '.join(tickers)}")
        print(f"Total: {len(tickers)}")
        print("-" * 80)

if __name__ == "__main__":
    list_tickers_by_quarter()
