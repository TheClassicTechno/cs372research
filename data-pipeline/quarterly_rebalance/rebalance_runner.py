#!/usr/bin/env python3

"""
REBALANCE RUNNER

Builds:
  - Macro quarter state
  - Asset quarter state

For a specified quarter range.

Example:

  python rebalance_runner.py \
    --start 2024Q4 \
    --end 2025Q4 \
    --tickers AAPL,NVDA,MSFT \
    --output-dir data/

Produces:

data/
  2024_Q4/
    macro.json
    assets.json
  2025_Q1/
    macro.json
    assets.json
  ...
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import List, Tuple

import yaml

from macro_quarter_builder import build_macro_state
from asset_quarter_builder import build_asset_state


# ----------------------------
# Quarter Utilities
# ----------------------------

def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    """
    "2025Q2" -> (2025, "Q2")
    """
    year = int(qstr[:4])
    quarter = qstr[4:]
    if quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError("Quarter must be Q1-Q4")
    return year, quarter


def next_quarter(year: int, quarter: str) -> Tuple[int, str]:
    order = ["Q1", "Q2", "Q3", "Q4"]
    idx = order.index(quarter)
    if idx < 3:
        return year, order[idx + 1]
    else:
        return year + 1, "Q1"


def quarter_range(start: str, end: str):
    """
    Generator from start to end inclusive.
    """
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)

    while True:
        yield y, q
        if (y == end_y) and (q == end_q):
            break
        y, q = next_quarter(y, q)


# ----------------------------
# Runner
# ----------------------------

def run_rebalance(
    start: str,
    end: str,
    tickers: List[str],
    output_dir: Path,
    fred_key: str = None,
    back_years: int = 2,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    for year, quarter in quarter_range(start, end):

        print(f"\n=== Building {year} {quarter} ===")

        # Directory per quarter
        qdir = output_dir / f"{year}_{quarter}"
        qdir.mkdir(parents=True, exist_ok=True)

        # Build Macro
        macro_doc = build_macro_state(
            year=year,
            quarter=quarter,
            fred_key=fred_key,
            back_years=back_years,
        )

        with open(qdir / "macro.json", "w") as f:
            json.dump(macro_doc, f, indent=2)

        # Build Asset State
        asset_doc = build_asset_state(
            year=year,
            quarter=quarter,
            tickers=tickers,
        )

        with open(qdir / "assets.json", "w") as f:
            json.dump(asset_doc, f, indent=2)

        print(f"Saved: {qdir}")


_SUPPORTED_TICKERS_PATH = Path(__file__).resolve().parent.parent / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start quarter (e.g., 2024Q4)")
    p.add_argument("--end", required=True, help="End quarter (e.g., 2025Q4)")
    p.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", default=False,
                   help="Use all tickers from supported_tickers.yaml")
    p.add_argument("--output-dir", default="rebalance_data")
    p.add_argument("--fred-key", default=None)
    p.add_argument("--back-years", type=int, default=2)

    args = p.parse_args()

    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        p.error("either --tickers or --supported is required")

    run_rebalance(
        start=args.start,
        end=args.end,
        tickers=tickers,
        output_dir=Path(args.output_dir),
        fred_key=args.fred_key or os.environ.get("FRED_API_KEY"),
        back_years=args.back_years,
    )


if __name__ == "__main__":
    main()