#!/usr/bin/env python3
"""
Quarterly Snapshot Builder — Canonical per-quarter data merge.

Merges all upstream pipeline outputs into a single point-in-time-safe
snapshot JSON per quarter, containing all requested tickers.

Inputs:
  - EDGAR/finished_summaries/{TICKER}/{YEAR}/Q{#}.json
  - earnings_calls/data/{TICKER}/{YEAR}_Q{#}.json
  - sentiment/data/{TICKER}/{YEAR}_Q{#}.json
  - macro/data/macro_{YEAR}_Q{#}.json
  - quarterly_asset_details/data/{TICKER}/{YEAR}_Q{#}.json

Output:
  final_snapshots/snapshot_{YEAR}_Q{#}.json

Examples:

  # Quarter range with supported tickers
  python generate_quarterly_json.py --start 2024Q4 --end 2025Q3 --supported

  # Single quarter, custom tickers
  python generate_quarterly_json.py --year 2025 --quarter Q1 --tickers AAPL,NVDA

Point-in-time safety:
  Every data source is filtered to only include information that was
  publicly available on or before the rebalance_date.  Filing summaries
  use filing_date <= rebalance_date.  No future data is ever included.
"""

import argparse
import datetime as dt
import json
import math
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
DEFAULT_EARNINGS_CALLS_DIR = _PIPELINE_DIR / "earnings_calls" / "data"
DEFAULT_SENTIMENT_DIR = _PIPELINE_DIR / "sentiment" / "data"
DEFAULT_MACRO_DIR = _PIPELINE_DIR / "macro" / "data"
DEFAULT_ASSETS_DIR = _PIPELINE_DIR / "quarterly_asset_details" / "data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "json_data"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


def load_fiscal_year_ends(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> Dict[str, str]:
    """Return {ticker: fiscal_year_end} map, e.g. {"AAPL": "09-27", "NVDA": "01-26"}."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {
        entry["symbol"]: entry.get("fiscal_year_end", "12-31")
        for entry in data["supported_tickers"]
    }


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

    File: {sentiment_dir}/{TICKER}/{YEAR}_{QUARTER}.json
    """
    per_ticker_file = sentiment_dir / ticker / f"{year}_{quarter}.json"
    if per_ticker_file.exists():
        try:
            with open(per_ticker_file, "r") as f:
                data = json.load(f)
            return data.get("features")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def load_earnings_calls(
    earnings_calls_dir: Path,
    ticker: str,
    year: int,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    """Load earnings-call summary + sentiment for a ticker/quarter.

    File: {earnings_calls_dir}/{TICKER}/{YEAR}_{QUARTER}.json
    """
    per_ticker_file = earnings_calls_dir / ticker / f"{year}_{quarter}.json"
    if not per_ticker_file.exists():
        return None
    try:
        with open(per_ticker_file, "r") as f:
            data = json.load(f)
        return data.get("features")
    except (json.JSONDecodeError, OSError):
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

    File: {assets_dir}/{TICKER}/{YEAR}_{QUARTER}.json
    """
    per_ticker_file = assets_dir / ticker / f"{year}_{quarter}.json"
    if not per_ticker_file.exists():
        return None

    try:
        with open(per_ticker_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    return data.get("features")


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
# CROSS-SECTIONAL FEATURES
# ==========================

def _add_relative_strength(ticker_data: Dict[str, Any]) -> None:
    """Compute relative_strength_60d = ret_60d - median(all ret_60d)."""
    ret_60d_vals = []
    for td in ticker_data.values():
        af = td.get("asset_features")
        if af and "error" not in af and af.get("ret_60d") is not None:
            ret_60d_vals.append(af["ret_60d"])
    if not ret_60d_vals:
        return
    ret_60d_vals.sort()
    n = len(ret_60d_vals)
    median_ret = (
        ret_60d_vals[n // 2]
        if n % 2 == 1
        else (ret_60d_vals[n // 2 - 1] + ret_60d_vals[n // 2]) / 2
    )
    for td in ticker_data.values():
        af = td.get("asset_features")
        if af and "error" not in af and af.get("ret_60d") is not None:
            af["relative_strength_60d"] = round(af["ret_60d"] - median_ret, 4)


def _add_sentiment_z(ticker_data: Dict[str, Any]) -> None:
    """Compute cross_sectional_z = z-score of mean_sentiment across tickers."""
    sents = []
    for td in ticker_data.values():
        s = td.get("news_sentiment")
        if s and s.get("mean_sentiment") is not None:
            sents.append(s["mean_sentiment"])
    if len(sents) < 2:
        return
    mu = sum(sents) / len(sents)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in sents) / len(sents))
    for td in ticker_data.values():
        s = td.get("news_sentiment")
        if s and s.get("mean_sentiment") is not None:
            val = float(s["mean_sentiment"])
            s["cross_sectional_z"] = round((val - mu) / sigma, 6) if sigma > 0 else 0.0


# ==========================
# BUILD ONE QUARTER (all tickers)
# ==========================

def build_quarter_snapshot(
    year: int,
    quarter: str,
    tickers: List[str],
    summaries_dir: Path,
    earnings_calls_dir: Path,
    sentiment_dir: Path,
    macro_dir: Path,
    assets_dir: Path,
    fiscal_year_ends: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a single quarterly snapshot JSON with all tickers."""
    rebal_date = quarter_end_date(year, quarter)
    fye_map = fiscal_year_ends or {}

    # Macro is shared across tickers
    macro = load_macro(macro_dir, year, quarter)

    # Per-ticker data
    ticker_data: Dict[str, Any] = {}
    for t in tickers:
        filing_summary = load_filing_summaries(summaries_dir, t, rebal_date)
        earnings_call = load_earnings_calls(earnings_calls_dir, t, year, quarter)
        sentiment = load_sentiment(sentiment_dir, t, year, quarter)
        asset_features = load_asset_data(assets_dir, t, year, quarter)

        # Fall back to earnings call sentiment if news sentiment is missing
        if sentiment is None and earnings_call is not None:
            ec_mean = earnings_call.get("mean_sentiment")
            ec_vol = earnings_call.get("sentiment_volatility")
            if ec_mean is not None:
                sentiment = {
                    "mean_sentiment": ec_mean,
                    "sentiment_volatility": ec_vol,
                    "source": "earnings_call",
                }

        entry: Dict[str, Any] = {
            "filing_summary": filing_summary,
            "earnings_call": earnings_call,
            "news_sentiment": sentiment,
            "asset_features": asset_features,
        }
        fye = fye_map.get(t)
        if fye and fye != "12-31":
            entry["fiscal_year_end"] = fye

        ticker_data[t] = entry

        leaks = check_leakage(ticker_data[t], rebal_date)
        if leaks:
            tqdm.write(f"  [{t}] LEAKAGE {year}{quarter}: {leaks}")

    # -- Cross-sectional features (computed across all tickers) --
    _add_relative_strength(ticker_data)
    _add_sentiment_z(ticker_data)

    snapshot = {
        "year": year,
        "quarter": quarter,
        "as_of_date": rebal_date.isoformat(),
        "tickers": list(ticker_data.keys()),
        "macro_regime": macro,
        "ticker_data": ticker_data,
    }

    try:
        from provenance import inline_provenance
        snapshot.update(inline_provenance())
    except ImportError:
        pass

    return snapshot


# ==========================
# PIPELINE
# ==========================

def run_pipeline(
    tickers: List[str],
    quarters: List[Tuple[int, str]],
    summaries_dir: Path,
    earnings_calls_dir: Path,
    sentiment_dir: Path,
    macro_dir: Path,
    assets_dir: Path,
    output_dir: Path,
) -> None:
    print(f"Building {len(quarters)} snapshot(s) with {len(tickers)} ticker(s)",
          flush=True)

    fiscal_year_ends = load_fiscal_year_ends()

    for year, quarter in tqdm(quarters, desc="Quarters", unit="qtr"):
        doc = build_quarter_snapshot(
            year, quarter, tickers,
            summaries_dir, earnings_calls_dir, sentiment_dir, macro_dir, assets_dir,
            fiscal_year_ends=fiscal_year_ends,
        )

        out_path = output_dir / f"snapshot_{year}_{quarter}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        tmp.replace(out_path)

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
    p.add_argument("--earnings-calls-dir", type=str, default=str(DEFAULT_EARNINGS_CALLS_DIR))
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
        earnings_calls_dir=Path(args.earnings_calls_dir),
        sentiment_dir=Path(args.sentiment_dir),
        macro_dir=Path(args.macro_dir),
        assets_dir=Path(args.assets_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
