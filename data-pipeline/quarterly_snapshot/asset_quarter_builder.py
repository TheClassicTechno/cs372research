#!/usr/bin/env python3
"""
Asset Quarter Builder — Per-ticker price features for a quarter.

Fetches price history from Yahoo Finance and computes price, returns,
volatility, drawdown, and moving averages for each ticker as-of quarter end.

Output:
  data-pipeline/quarterly_snapshot/data/assets_{YEAR}_{QUARTER}.json

Examples:

  # Single quarter
  python asset_quarter_builder.py --year 2025 --quarter Q2 --tickers AAPL,NVDA

  # Quarter range with supported tickers
  python asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --supported

  # Custom output path (single quarter only)
  python asset_quarter_builder.py --year 2025 --quarter Q2 --tickers AAPL --out /tmp/assets.json
"""

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data"
_SUPPORTED_TICKERS_PATH = _SCRIPT_DIR.parent / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


def iso(d): return d.isoformat()

def q_end_dates(year):
    return {
        "Q1": dt.date(year, 3, 31),
        "Q2": dt.date(year, 6, 30),
        "Q3": dt.date(year, 9, 30),
        "Q4": dt.date(year, 12, 31),
    }

def annualized_vol(r):
    if len(r.dropna()) < 10: return None
    return float(r.std() * np.sqrt(252))

def max_drawdown(px):
    peak = px.cummax()
    return float((px / peak - 1).min())

def sma(px, w):
    if len(px) < w: return None
    return float(px.rolling(w).mean().iloc[-1])

def next_quarter(year: int, quarter: str) -> Tuple[int, str]:
    order = ["Q1", "Q2", "Q3", "Q4"]
    idx = order.index(quarter)
    if idx < 3:
        return year, order[idx + 1]
    return year + 1, "Q1"

def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    """'2025Q2' -> (2025, 'Q2')"""
    year = int(qstr[:4])
    quarter = qstr[4:]
    if quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError(f"Invalid quarter: {quarter}")
    return year, quarter

def quarter_range(start: str, end: str) -> List[Tuple[int, str]]:
    """Generate list of (year, quarter) from start to end inclusive."""
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)
    result = []
    while True:
        result.append((y, q))
        if y == end_y and q == end_q:
            break
        y, q = next_quarter(y, q)
    return result


def build_asset_state(year, quarter, tickers):

    q_end = q_end_dates(year)[quarter]
    start = q_end - dt.timedelta(days=400)

    out = {
        "schema_version": "asset_state_v1",
        "year": year,
        "quarter": quarter,
        "as_of": iso(q_end),
        "tickers": {}
    }

    print(f"Building asset state for {year} {quarter} (as-of {iso(q_end)})", flush=True)
    print(f"  {len(tickers)} ticker(s): {', '.join(tickers)}", flush=True)

    def _process_ticker(t):
        try:
            df = yf.download(t, start=iso(start), end=iso(q_end), progress=False)
        except Exception as e:
            tqdm.write(f"  [{t}] FAIL — {type(e).__name__}: {e}")
            return t, {"error": str(e)}

        if df.empty:
            tqdm.write(f"  [{t}] WARN — no price data")
            return t, {"error": "no_data"}

        px = df["Close"].dropna()
        px = px[px.index <= pd.to_datetime(q_end)]

        if len(px) < 20:
            tqdm.write(f"  [{t}] WARN — insufficient data ({len(px)} points)")
            return t, {"error": "insufficient_data"}

        px60 = px.iloc[-61:]
        price = float(px.iloc[-1])

        result = {
            "PRICE": price,
            "RET60": float(px60.iloc[-1] / px60.iloc[0] - 1) * 100,
            "VOL60": annualized_vol(np.log(px60).diff()),
            "DD60": max_drawdown(px60),
            "SMA20": sma(px, 20),
            "SMA50": sma(px, 50),
        }
        tqdm.write(f"  [{t}] OK — ${price:.2f}")
        return t, result

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_process_ticker, t): t for t in tickers}
        for future in tqdm(
            as_completed(futures), desc="Fetching prices", unit="ticker", total=len(futures),
        ):
            t = futures[future]
            try:
                ticker_name, ticker_data = future.result()
                out["tickers"][ticker_name] = ticker_data
            except Exception as e:
                tqdm.write(f"  [{t}] FAIL — {type(e).__name__}: {e}")
                out["tickers"][t] = {"error": str(e)}

    ok = sum(1 for v in out["tickers"].values() if "error" not in v)
    print(f"Asset state complete: {ok}/{len(tickers)} tickers OK")
    return out


def main():
    p = argparse.ArgumentParser(description="Build per-ticker asset state for one or more quarters")
    # Single quarter mode
    p.add_argument("--year", type=int, default=None, help="Year (single quarter mode)")
    p.add_argument("--quarter", default=None, choices=["Q1","Q2","Q3","Q4"],
                   help="Quarter (single quarter mode)")
    # Range mode
    p.add_argument("--start", type=str, default=None, help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None, help="End quarter, e.g. 2025Q3")
    # Tickers
    p.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", default=False,
                   help="Use all tickers from supported_tickers.yaml")
    p.add_argument("--out", default=None,
                   help="Output path (single quarter only; default: data/assets_YEAR_QUARTER.json)")
    args = p.parse_args()

    # Resolve quarters
    if args.start and args.end:
        quarters = quarter_range(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    # Resolve tickers
    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        p.error("either --tickers or --supported is required")

    print(f"Building asset state for {len(quarters)} quarter(s), {len(tickers)} ticker(s)", flush=True)

    for year, quarter in tqdm(quarters, desc="Quarters", unit="quarter"):
        doc = build_asset_state(year, quarter, tickers)

        if args.out and len(quarters) == 1:
            out_path = Path(args.out)
        else:
            out_path = _DEFAULT_DATA_DIR / f"assets_{year}_{quarter}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)

        tqdm.write(f"Wrote: {out_path}")

    print(f"Done: {len(quarters)} quarter(s) complete.")


if __name__ == "__main__":
    main()