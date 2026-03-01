#!/usr/bin/env python3
"""
Quarterly Snapshot Builder — Canonical per-quarter data merge.

Merges all upstream pipeline outputs into a single point-in-time-safe
snapshot JSON per quarter, containing all requested tickers.

Inputs:
  - EDGAR/finished_summaries/{TICKER}/{YEAR}/Q{#}.json
  - sentiment/data/sentiment_{YEAR}_Q{#}.json
  - macro/data/macro_{YEAR}_Q{#}.json
  - quarterly_asset_details/data/assets_{YEAR}_Q{#}.json

Output:
  final_snapshots/snapshot_{YEAR}_Q{#}.json

Examples:

  # Quarter range with supported tickers
  python quarterly_quarterly_json.py --start 2024Q4 --end 2025Q3 --supported

  # Single quarter, custom tickers
  python quarterly_quarterly_json.py --year 2025 --quarter Q1 --tickers AAPL,NVDA

Point-in-time safety:
  Every data source is filtered to only include information that was
  publicly available on or before the rebalance_date.  Filing summaries
  use filing_date <= rebalance_date.  No future data is ever included.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# ==========================
# CONFIG — default paths relative to this script
# ==========================

_SCRIPT_DIR = Path(__file__).resolve().parent           # final_snapshots/
_PIPELINE_DIR = _SCRIPT_DIR.parent                      # data-pipeline/
_SUPPORTED_TICKERS_PATH = _PIPELINE_DIR / "supported_tickers.yaml"

DEFAULT_SUMMARIES_DIR = _PIPELINE_DIR / "EDGAR" / "finished_summaries"
DEFAULT_SENTIMENT_DIR = _PIPELINE_DIR / "sentiment" / "data"
DEFAULT_MACRO_DIR = _PIPELINE_DIR / "macro" / "data"
DEFAULT_ASSETS_DIR = _PIPELINE_DIR / "quarterly_asset_details" / "data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


# ==========================
# QUARTER UTILITIES
# ==========================

QUARTER_ENDS = {
    "Q1": (3, 31),
    "Q2": (6, 30),
    "Q3": (9, 30),
    "Q4": (12, 31),
}


def quarter_from_date(d: dt.date) -> Tuple[int, str]:
    for q_label, (month, day) in QUARTER_ENDS.items():
        if d <= dt.date(d.year, month, day):
            return (d.year, q_label)
    return (d.year, "Q4")


def quarter_end_date(year: int, quarter: str) -> dt.date:
    month, day = QUARTER_ENDS[quarter]
    return dt.date(year, month, day)


def next_quarter_ql(year: int, quarter: str) -> Tuple[int, str]:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx < 3:
        return year, labels[idx + 1]
    return year + 1, "Q1"


def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    """'2025Q2' -> (2025, 'Q2')"""
    year = int(qstr[:4])
    q = qstr[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter: {q}")
    return year, q


def quarter_range_list(start: str, end: str) -> List[Tuple[int, str]]:
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)
    result = []
    while True:
        result.append((y, q))
        if y == end_y and q == end_q:
            break
        y, q = next_quarter_ql(y, q)
    return result


# ==========================
# DATA LOADERS
# ==========================

def load_filing_summaries(
    summaries_dir: Path,
    ticker: str,
    rebalance_date: dt.date,
) -> Dict[str, Any]:
    """Load filing summaries for a ticker.

    Directory: summaries_dir/{TICKER}/{YEAR}/Q{#}.json
    Point-in-time: only summaries with filing_date <= rebalance_date.

    Returns:
      {"periodic": <most recent 10-Q/10-K or null>, "event_filings": [...]}
    """
    ticker_dir = summaries_dir / ticker
    result: Dict[str, Any] = {"periodic": None, "event_filings": []}

    if not ticker_dir.exists():
        return result

    all_summaries: List[Dict[str, Any]] = []
    for json_file in ticker_dir.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                summary = json.load(f)
            if isinstance(summary, dict) and "filing_date" in summary:
                all_summaries.append(summary)
        except (json.JSONDecodeError, OSError):
            continue

    cutoff = rebalance_date.isoformat()
    eligible = [s for s in all_summaries if s.get("filing_date", "") <= cutoff]

    periodic_forms = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
    periodic = [s for s in eligible if s.get("form", "") in periodic_forms]
    if periodic:
        periodic.sort(key=lambda s: s.get("filing_date", ""), reverse=True)
        result["periodic"] = periodic[0]

    cutoff_90 = (rebalance_date - dt.timedelta(days=90)).isoformat()
    event_forms = {"8-K", "8-K/A"}
    events = [
        s for s in eligible
        if s.get("form", "") in event_forms
        and s.get("filing_date", "") >= cutoff_90
    ]
    events.sort(key=lambda s: s.get("filing_date", ""), reverse=True)
    result["event_filings"] = events

    return result


def load_sentiment(
    sentiment_dir: Path,
    ticker: str,
    year: int,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    """Load sentiment for a ticker/quarter.

    Tries per-quarter file first, falls back to legacy sentiment_output.json.
    """
    q_num = quarter[1]

    per_quarter_file = sentiment_dir / f"sentiment_{year}_Q{q_num}.json"
    if per_quarter_file.exists():
        try:
            with open(per_quarter_file, "r") as f:
                data = json.load(f)
            return data.get("results", {}).get(ticker)
        except (json.JSONDecodeError, OSError):
            pass

    legacy_file = sentiment_dir / "sentiment_output.json"
    if legacy_file.exists():
        try:
            with open(legacy_file, "r") as f:
                data = json.load(f)
            ticker_data = data.get("results", {}).get(ticker)
            if ticker_data:
                return ticker_data.get(f"{year}Q{q_num}")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def load_macro(
    macro_dir: Path,
    year: int,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    """Load macro regime for a quarter.

    File: macro_{YEAR}_{QUARTER}.json
    """
    macro_file = macro_dir / f"macro_{year}_{quarter}.json"
    if not macro_file.exists():
        return None

    try:
        with open(macro_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    layers = data.get("layers", {})
    regime: Dict[str, Any] = {}

    for layer_key in ("L1", "L4", "L5"):
        layer = layers.get(layer_key, {})
        metrics = layer.get("metrics")
        if metrics:
            label = {"L1": "macro_metrics", "L4": "vol_metrics", "L5": "internals_metrics"}[layer_key]
            regime[label] = metrics

    return regime if regime else None


def load_asset_data(
    assets_dir: Path,
    ticker: str,
    year: int,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    """Load asset features for a ticker/quarter.

    File: assets_{YEAR}_{QUARTER}.json
    """
    asset_file = assets_dir / f"assets_{year}_{quarter}.json"
    if not asset_file.exists():
        return None

    try:
        with open(asset_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    return data.get("tickers", {}).get(ticker)


# ==========================
# LEAKAGE CHECK
# ==========================

def check_leakage(ticker_data: dict, rebalance_date: dt.date) -> List[str]:
    violations = []
    cutoff = rebalance_date.isoformat()

    fs = ticker_data.get("filing_summary", {})
    if isinstance(fs, dict):
        for key in ("periodic", "event_filings"):
            items = fs.get(key)
            if isinstance(items, dict):
                items = [items]
            elif not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                fd = item.get("filing_date", "")
                if fd and fd > cutoff:
                    violations.append(f"filing_date {fd} > {cutoff}")

    return violations


# ==========================
# BUILD ONE QUARTER (all tickers)
# ==========================

def build_quarter_snapshot(
    year: int,
    quarter: str,
    tickers: List[str],
    summaries_dir: Path,
    sentiment_dir: Path,
    macro_dir: Path,
    assets_dir: Path,
) -> Dict[str, Any]:
    """Build a single quarterly snapshot JSON with all tickers."""
    rebal_date = quarter_end_date(year, quarter)

    # Macro is shared across tickers
    macro = load_macro(macro_dir, year, quarter)

    # Per-ticker data
    ticker_data: Dict[str, Any] = {}
    for t in tickers:
        filing_summary = load_filing_summaries(summaries_dir, t, rebal_date)
        sentiment = load_sentiment(sentiment_dir, t, year, quarter)
        asset_features = load_asset_data(assets_dir, t, year, quarter)

        ticker_data[t] = {
            "filing_summary": filing_summary,
            "news_sentiment": sentiment,
            "asset_features": asset_features,
        }

        leaks = check_leakage(ticker_data[t], rebal_date)
        if leaks:
            tqdm.write(f"  [{t}] LEAKAGE {year}{quarter}: {leaks}")

    return {
        "year": year,
        "quarter": quarter,
        "as_of_date": rebal_date.isoformat(),
        "tickers": list(ticker_data.keys()),
        "macro_regime": macro,
        "ticker_data": ticker_data,
    }


# ==========================
# PIPELINE
# ==========================

def run_pipeline(
    tickers: List[str],
    quarters: List[Tuple[int, str]],
    summaries_dir: Path,
    sentiment_dir: Path,
    macro_dir: Path,
    assets_dir: Path,
    output_dir: Path,
) -> None:
    print(f"Building {len(quarters)} snapshot(s) with {len(tickers)} ticker(s)",
          flush=True)

    for year, quarter in tqdm(quarters, desc="Quarters", unit="qtr"):
        doc = build_quarter_snapshot(
            year, quarter, tickers,
            summaries_dir, sentiment_dir, macro_dir, assets_dir,
        )

        out_path = output_dir / f"snapshot_{year}_{quarter}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        tmp.rename(out_path)

        tqdm.write(f"  Wrote: {out_path}")

    print(f"Done: {len(quarters)} quarter(s) complete.", flush=True)


# ==========================
# CLI
# ==========================

def main():
    p = argparse.ArgumentParser(
        description="Build quarterly snapshots for multi-agent trading research",
    )
    p.add_argument("--year", type=int, default=None,
                   help="Year (single quarter mode)")
    p.add_argument("--quarter", default=None, choices=["Q1", "Q2", "Q3", "Q4"],
                   help="Quarter (single quarter mode)")
    p.add_argument("--start", type=str, default=None,
                   help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None,
                   help="End quarter, e.g. 2025Q3")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", default=False,
                   help="Use all tickers from supported_tickers.yaml")
    p.add_argument("--summaries-dir", type=str, default=str(DEFAULT_SUMMARIES_DIR))
    p.add_argument("--sentiment-dir", type=str, default=str(DEFAULT_SENTIMENT_DIR))
    p.add_argument("--macro-dir", type=str, default=str(DEFAULT_MACRO_DIR))
    p.add_argument("--assets-dir", type=str, default=str(DEFAULT_ASSETS_DIR))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = p.parse_args()

    # Resolve quarters
    if args.start and args.end:
        quarters = quarter_range_list(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    # Resolve tickers
    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        p.error("either --tickers or --supported is required")

    run_pipeline(
        tickers=tickers,
        quarters=quarters,
        summaries_dir=Path(args.summaries_dir),
        sentiment_dir=Path(args.sentiment_dir),
        macro_dir=Path(args.macro_dir),
        assets_dir=Path(args.assets_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
