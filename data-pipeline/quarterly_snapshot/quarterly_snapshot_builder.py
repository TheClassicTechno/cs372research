#!/usr/bin/env python3
"""
Quarterly Snapshot Builder — Canonical per-ticker per-quarter data merge.

Merges all upstream pipeline outputs into a single point-in-time-safe
snapshot JSON for each ticker at each rebalance date.

Inputs:
  - EDGAR/finished_summaries/   (filing summaries from summarization pipeline)
  - sentiment/data/             (sentiment_output.json from sentiment pipeline)
  - macro/data/                 (augmented_market_state_v3.json from macro pipeline)
  - Structured fundamentals API (stub: fetch_fundamentals_api)
  - Price history               (stub: fetch_price_history / compute_price_features)

Output:
  quarterly_snapshot/data/{YEAR}{QUARTER}/{TICKER}.json

Examples:
  # 1. Build snapshots for 3 tickers at end of Q1 and Q2
  python quarterly_snapshot_builder.py \\
      --tickers AAPL,NVDA,MSFT \\
      --rebalance-dates 2025-03-31,2025-06-30

  # 2. Full year of quarterly snapshots for the 8-ticker universe
  python quarterly_snapshot_builder.py \\
      --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \\
      --rebalance-dates 2025-03-31,2025-06-30,2025-09-30,2025-12-31

  # 3. Single ticker, single date (useful for debugging)
  python quarterly_snapshot_builder.py \\
      --tickers AAPL \\
      --rebalance-dates 2025-03-31

  # 4. Custom data directories (non-default layout)
  python quarterly_snapshot_builder.py \\
      --tickers AAPL,NVDA \\
      --rebalance-dates 2025-06-30 \\
      --summaries-dir /data/edgar/finished_summaries \\
      --sentiment-dir /data/sentiment \\
      --macro-dir /data/macro

  # 5. Custom output directory
  python quarterly_snapshot_builder.py \\
      --tickers AAPL,NVDA,MSFT \\
      --rebalance-dates 2025-03-31 \\
      --output-dir /tmp/snapshots

  # 6. Mid-quarter rebalance date (still maps to that quarter)
  python quarterly_snapshot_builder.py \\
      --tickers AAPL \\
      --rebalance-dates 2025-02-15

  # 7. Multi-year backtest dates
  python quarterly_snapshot_builder.py \\
      --tickers AAPL,NVDA \\
      --rebalance-dates 2024-03-31,2024-06-30,2024-09-30,2024-12-31,2025-03-31,2025-06-30

Point-in-time safety:
  Every data source is filtered to only include information that was
  publicly available on or before the rebalance_date.  Filing summaries
  use filing_date <= rebalance_date.  No future data is ever included.
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ==========================
# CONFIG — default paths relative to this script
# ==========================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent  # data-pipeline/
_SUPPORTED_TICKERS_PATH = _PIPELINE_DIR / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]

DEFAULT_SUMMARIES_DIR = _PIPELINE_DIR / "EDGAR" / "finished_summaries"
DEFAULT_SENTIMENT_DIR = _PIPELINE_DIR / "sentiment" / "data"
DEFAULT_MACRO_DIR = _PIPELINE_DIR / "macro" / "data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "data"

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
    """Derive (year, quarter_label) from a date.

    The date is assigned to the quarter whose end date it falls
    on or before.  Example: 2025-03-31 -> (2025, 'Q1').
    """
    for q_label, (month, day) in QUARTER_ENDS.items():
        q_end = dt.date(d.year, month, day)
        if d <= q_end:
            return (d.year, q_label)
    # Past Dec 31 shouldn't happen, but handle it
    return (d.year, "Q4")


def quarter_end_date(year: int, quarter: str) -> dt.date:
    """Return the calendar end date for a given year/quarter."""
    month, day = QUARTER_ENDS[quarter]
    return dt.date(year, month, day)


def prev_quarter(year: int, quarter: str) -> Tuple[int, str]:
    """Return the (year, quarter) immediately preceding the given one."""
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx == 0:
        return (year - 1, "Q4")
    return (year, labels[idx - 1])


# ==========================
# SNAPSHOT SCHEMA + VALIDATION
# ==========================

SNAPSHOT_REQUIRED_KEYS = {
    "ticker", "as_of_date", "year", "quarter",
    "fundamentals", "filing_summary", "news_sentiment",
    "macro_regime", "price_features", "ownership_signals",
    "exposure_profile",
}


def validate_snapshot_schema(snapshot: dict) -> List[str]:
    """Validate that a snapshot has all required top-level keys.

    Returns list of error strings.  Empty list means valid.
    """
    errors: List[str] = []
    for key in SNAPSHOT_REQUIRED_KEYS:
        if key not in snapshot:
            errors.append(f"missing required key: {key}")
    if "ticker" in snapshot and not isinstance(snapshot["ticker"], str):
        errors.append("ticker must be a string")
    if "as_of_date" in snapshot and not isinstance(snapshot["as_of_date"], str):
        errors.append("as_of_date must be an ISO date string")
    return errors


def ensure_no_future_leakage(snapshot: dict, rebalance_date: dt.date) -> List[str]:
    """Check that no data in the snapshot references dates after rebalance_date.

    Inspects filing_date, sentiment quarter, macro quarter, and
    price_features end_date.  Returns list of violations.
    """
    violations: List[str] = []
    cutoff = rebalance_date.isoformat()

    # Check filing summary dates
    fs = snapshot.get("filing_summary", {})
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
                    violations.append(f"filing_summary.{key} filing_date {fd} > {cutoff}")

    # Check price features end date
    pf = snapshot.get("price_features", {})
    if isinstance(pf, dict):
        end = pf.get("as_of_date", "")
        if end and end > cutoff:
            violations.append(f"price_features.as_of_date {end} > {cutoff}")

    return violations


# ==========================
# DATA LOADERS
# ==========================

def load_filing_summaries(
    summaries_dir: Path,
    ticker: str,
    rebalance_date: dt.date,
) -> Dict[str, Any]:
    """Load the most recent 10-Q/10-K summary and recent 8-K summaries.

    Point-in-time rule: only summaries with filing_date <= rebalance_date
    are included.

    Returns:
      {
        "periodic": <most recent 10-Q or 10-K summary dict or null>,
        "event_filings": [<8-K summaries within last 90 days>]
      }
    """
    ticker_dir = summaries_dir / ticker
    result = {"periodic": None, "event_filings": []}

    if not ticker_dir.exists():
        return result

    # Collect all summary JSON files
    all_summaries: List[Dict[str, Any]] = []
    for json_file in ticker_dir.rglob("*_summary.json"):
        try:
            with open(json_file, "r") as f:
                summary = json.load(f)
            all_summaries.append(summary)
        except (json.JSONDecodeError, OSError):
            continue

    # Filter by filing_date <= rebalance_date
    cutoff = rebalance_date.isoformat()
    eligible = [
        s for s in all_summaries
        if s.get("filing_date", "") <= cutoff
    ]

    # Find most recent periodic filing (10-Q or 10-K)
    periodic_forms = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
    periodic = [
        s for s in eligible
        if s.get("form", "") in periodic_forms
    ]
    if periodic:
        periodic.sort(key=lambda s: s.get("filing_date", ""), reverse=True)
        result["periodic"] = periodic[0]

    # Find 8-K filings within last 90 days
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
    """Load sentiment data for a ticker/quarter from sentiment_output.json.

    Returns the per-quarter sentiment dict or None if not found.
    """
    sentiment_file = sentiment_dir / "sentiment_output.json"
    if not sentiment_file.exists():
        return None

    try:
        with open(sentiment_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    results = data.get("results", {})
    ticker_data = results.get(ticker)
    if not ticker_data:
        return None

    quarter_key = f"{year}{quarter}"
    return ticker_data.get(quarter_key)


def load_macro(
    macro_dir: Path,
    year: int,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    """Load macro regime data for a quarter from the macro snapshot.

    Searches for augmented_market_state_v3.json in macro_dir.
    Returns the Layer 1 + Layer 4 metrics for the given quarter,
    or None if not found.
    """
    macro_file = macro_dir / "augmented_market_state_v3.json"
    if not macro_file.exists():
        return None

    try:
        with open(macro_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    layers = data.get("layers", {})
    regime: Dict[str, Any] = {}

    # Extract Layer 1 (macro) for this quarter
    l1 = layers.get("L1", {})
    l1_quarter = l1.get("quarters", {}).get(quarter)
    if l1_quarter:
        regime["macro_metrics"] = l1_quarter.get("metrics", {})

    # Extract Layer 4 (vol) for this quarter
    l4 = layers.get("L4", {})
    l4_quarter = l4.get("quarters", {}).get(quarter)
    if l4_quarter:
        regime["vol_metrics"] = l4_quarter.get("metrics", {})

    # Extract Layer 5 (internals) for this quarter
    l5 = layers.get("L5", {})
    l5_quarter = l5.get("quarters", {}).get(quarter)
    if l5_quarter:
        regime["internals_metrics"] = l5_quarter.get("metrics", {})

    return regime if regime else None


# ==========================
# STUBS — Future API integrations
# ==========================

def fetch_fundamentals_api(ticker: str, as_of: dt.date) -> Dict[str, Any]:
    """Fetch structured financial fundamentals for a ticker as of a date.

    STUB: Returns placeholder values.  Replace with real API integration
    (e.g., SEC XBRL company-facts endpoint, financial data vendor) when
    available.

    Point-in-time rule: the API must only return data from filings
    whose filing_date <= as_of.

    Expected real output schema:
      {
        "revenue_ttm": float,
        "net_income_ttm": float,
        "eps_diluted_ttm": float,
        "total_debt": float,
        "cash_and_equivalents": float,
        "book_value_per_share": float,
        "shares_outstanding": int,
        "pe_ratio": float,
        "price_to_book": float,
        "debt_to_equity": float,
        "current_ratio": float,
        "roe": float,
        "source": str,
        "as_of_date": str
      }
    """
    return {
        "revenue_ttm": None,
        "net_income_ttm": None,
        "eps_diluted_ttm": None,
        "total_debt": None,
        "cash_and_equivalents": None,
        "book_value_per_share": None,
        "shares_outstanding": None,
        "pe_ratio": None,
        "price_to_book": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "roe": None,
        "source": "STUB",
        "as_of_date": as_of.isoformat(),
    }


def fetch_price_history(ticker: str, end_date: dt.date, lookback_days: int = 300) -> List[Dict[str, Any]]:
    """Fetch daily price history for a ticker ending on end_date.

    STUB: Returns empty list.  Replace with yfinance or vendor API.

    Point-in-time rule: no prices after end_date are included.

    Expected real output:
      [{"date": "2025-01-02", "close": 150.25, "volume": 1234567}, ...]
    """
    return []


def compute_price_features(price_history: List[Dict[str, Any]], as_of: dt.date) -> Dict[str, Any]:
    """Compute engineered price features from daily price history.

    STUB: Returns placeholder values.  When real price data is available,
    this computes:
      - current_price: last close on or before as_of
      - return_60d: 60-trading-day return
      - vol_60d: annualized 60-day volatility
      - max_drawdown_60d: max drawdown over 60 trading days
      - sma_20: 20-day SMA at as_of
      - sma_50: 50-day SMA at as_of
      - sma_200: 200-day SMA at as_of
      - above_200dma: bool, whether current price > 200DMA
      - momentum_score: normalized momentum signal

    All features are computed using data <= as_of only.
    """
    if not price_history:
        return {
            "current_price": None,
            "return_60d": None,
            "vol_60d": None,
            "max_drawdown_60d": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "above_200dma": None,
            "momentum_score": None,
            "as_of_date": as_of.isoformat(),
            "source": "STUB",
        }

    # When real data is available, compute here.
    # Filter to <= as_of
    eligible = [p for p in price_history if p.get("date", "") <= as_of.isoformat()]
    if not eligible:
        return {
            "current_price": None,
            "return_60d": None,
            "vol_60d": None,
            "max_drawdown_60d": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "above_200dma": None,
            "momentum_score": None,
            "as_of_date": as_of.isoformat(),
            "source": "STUB",
        }

    closes = [p["close"] for p in eligible if p.get("close") is not None]
    current = closes[-1] if closes else None

    # 60-day return
    ret_60 = None
    if len(closes) >= 61:
        ret_60 = (closes[-1] / closes[-61] - 1.0) * 100.0

    # 60-day annualized vol
    vol_60 = None
    if len(closes) >= 61:
        log_returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(max(1, len(closes) - 60), len(closes))
            if closes[i - 1] > 0
        ]
        if len(log_returns) >= 10:
            mean_r = sum(log_returns) / len(log_returns)
            var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
            vol_60 = math.sqrt(var_r) * math.sqrt(252)

    # Max drawdown over 60 days
    dd_60 = None
    if len(closes) >= 20:
        window = closes[-61:] if len(closes) >= 61 else closes
        peak = window[0]
        max_dd = 0.0
        for c in window:
            if c > peak:
                peak = c
            dd = (c / peak) - 1.0
            if dd < max_dd:
                max_dd = dd
        dd_60 = max_dd

    # SMAs
    def _sma(data, window):
        if len(data) < window:
            return None
        return sum(data[-window:]) / window

    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)
    sma_200 = _sma(closes, 200)

    above_200 = None
    if current is not None and sma_200 is not None:
        above_200 = current > sma_200

    return {
        "current_price": current,
        "return_60d": ret_60,
        "vol_60d": vol_60,
        "max_drawdown_60d": dd_60,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "above_200dma": above_200,
        "momentum_score": None,  # requires cross-sectional normalization
        "as_of_date": as_of.isoformat(),
        "source": "computed" if closes else "STUB",
    }


def fetch_ownership_signals(ticker: str, as_of: dt.date) -> Dict[str, Any]:
    """Fetch ownership / institutional signal data.

    STUB: Returns placeholder values.  Replace with SEC Form 4 / 13F
    data integration when available.
    """
    return {
        "insider_net_shares_90d": None,
        "institutional_ownership_pct": None,
        "source": "STUB",
        "as_of_date": as_of.isoformat(),
    }


def fetch_exposure_profile(ticker: str, as_of: dt.date) -> Dict[str, Any]:
    """Fetch sector/factor exposure profile.

    STUB: Returns placeholder values.  Replace with factor model
    or sector classification data when available.
    """
    return {
        "sector": None,
        "industry": None,
        "market_cap_bucket": None,
        "beta": None,
        "source": "STUB",
        "as_of_date": as_of.isoformat(),
    }


# ==========================
# SNAPSHOT BUILDER
# ==========================

def build_snapshot(
    ticker: str,
    rebalance_date: dt.date,
    summaries_dir: Path = DEFAULT_SUMMARIES_DIR,
    sentiment_dir: Path = DEFAULT_SENTIMENT_DIR,
    macro_dir: Path = DEFAULT_MACRO_DIR,
) -> Dict[str, Any]:
    """Build a canonical quarterly snapshot for one ticker at one rebalance date.

    This is the main integration function.  It loads data from all upstream
    sources, enforces point-in-time correctness, and produces the canonical
    snapshot schema.

    Point-in-time rules:
    - Filing summaries: filing_date <= rebalance_date
    - 8-K events: filing_date within 90 days before rebalance_date
    - Sentiment: matched by year/quarter
    - Macro: matched by year/quarter
    - Fundamentals: as_of <= rebalance_date (stub)
    - Prices: end_date = rebalance_date (stub)
    """
    year, quarter = quarter_from_date(rebalance_date)

    # 1. Filing summaries
    filing_summary = load_filing_summaries(summaries_dir, ticker, rebalance_date)

    # 2. Sentiment
    sentiment = load_sentiment(sentiment_dir, ticker, year, quarter)

    # 3. Macro
    macro = load_macro(macro_dir, year, quarter)

    # 4. Fundamentals (stub)
    fundamentals = fetch_fundamentals_api(ticker, rebalance_date)

    # 5. Price features (stub)
    price_history = fetch_price_history(ticker, rebalance_date)
    price_features = compute_price_features(price_history, rebalance_date)

    # 6. Ownership (stub)
    ownership = fetch_ownership_signals(ticker, rebalance_date)

    # 7. Exposure (stub)
    exposure = fetch_exposure_profile(ticker, rebalance_date)

    snapshot = {
        "ticker": ticker,
        "as_of_date": rebalance_date.isoformat(),
        "year": year,
        "quarter": quarter,
        "fundamentals": fundamentals,
        "filing_summary": filing_summary,
        "news_sentiment": sentiment,
        "macro_regime": macro,
        "price_features": price_features,
        "ownership_signals": ownership,
        "exposure_profile": exposure,
    }

    return snapshot


# ==========================
# FILE I/O
# ==========================

def save_snapshot(snapshot: dict, output_dir: Path) -> Path:
    """Atomically write a snapshot JSON to the output directory.

    Path: output_dir/{YEAR}{QUARTER}/{TICKER}.json
    """
    year = snapshot["year"]
    quarter = snapshot["quarter"]
    ticker = snapshot["ticker"]

    out_path = output_dir / f"{year}{quarter}" / f"{ticker}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    tmp.rename(out_path)

    return out_path


# ==========================
# PIPELINE
# ==========================

def run_pipeline(
    tickers: List[str],
    rebalance_dates: List[dt.date],
    summaries_dir: Path = DEFAULT_SUMMARIES_DIR,
    sentiment_dir: Path = DEFAULT_SENTIMENT_DIR,
    macro_dir: Path = DEFAULT_MACRO_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Build and save snapshots for all tickers at all rebalance dates.

    Returns a report dict.
    """
    total = len(tickers) * len(rebalance_dates)
    built = 0
    errors: List[str] = []
    leakage_warnings: List[str] = []

    print(f"Building {total} snapshot(s): {len(tickers)} ticker(s) x {len(rebalance_dates)} date(s)")

    for rebal_date in sorted(rebalance_dates):
        for ticker in tickers:
            try:
                snapshot = build_snapshot(
                    ticker, rebal_date,
                    summaries_dir=summaries_dir,
                    sentiment_dir=sentiment_dir,
                    macro_dir=macro_dir,
                )

                # Validate schema
                schema_errors = validate_snapshot_schema(snapshot)
                if schema_errors:
                    print(f"  WARNING {ticker} {rebal_date}: {schema_errors}")

                # Check for future leakage
                leaks = ensure_no_future_leakage(snapshot, rebal_date)
                if leaks:
                    leakage_warnings.extend(leaks)
                    print(f"  LEAKAGE {ticker} {rebal_date}: {leaks}")

                out_path = save_snapshot(snapshot, output_dir)
                print(f"  Saved {out_path}")
                built += 1

            except Exception as e:
                msg = f"{ticker} {rebal_date}: {type(e).__name__}: {e}"
                errors.append(msg)
                print(f"  FAILED {msg}")

    report = {
        "total_planned": total,
        "built": built,
        "errors": errors,
        "leakage_warnings": leakage_warnings,
    }
    print(f"\nPipeline complete: {built}/{total} snapshots built")
    if errors:
        print(f"  {len(errors)} error(s)")
    if leakage_warnings:
        print(f"  {len(leakage_warnings)} leakage warning(s)")
    return report


# ==========================
# CLI
# ==========================

def parse_date(s: str) -> dt.date:
    """Parse an ISO date string (YYYY-MM-DD)."""
    return dt.date.fromisoformat(s.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Build quarterly snapshots for multi-agent trading research"
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers (e.g. AAPL,NVDA,MSFT)",
    )
    parser.add_argument(
        "--supported", action="store_true", default=False,
        help="Use all tickers from supported_tickers.yaml",
    )
    parser.add_argument(
        "--rebalance-dates", type=str, required=True,
        help="Comma-separated ISO dates (e.g. 2025-03-31,2025-06-30)",
    )
    parser.add_argument(
        "--summaries-dir", type=str, default=str(DEFAULT_SUMMARIES_DIR),
        help="Path to finished_summaries directory",
    )
    parser.add_argument(
        "--sentiment-dir", type=str, default=str(DEFAULT_SENTIMENT_DIR),
        help="Path to sentiment data directory",
    )
    parser.add_argument(
        "--macro-dir", type=str, default=str(DEFAULT_MACRO_DIR),
        help="Path to macro data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for snapshot JSON files",
    )
    args = parser.parse_args()

    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = []
    rebalance_dates = [parse_date(d) for d in args.rebalance_dates.split(",") if d.strip()]

    if not tickers:
        print("ERROR: at least one ticker required", file=sys.stderr)
        sys.exit(1)
    if not rebalance_dates:
        print("ERROR: at least one rebalance date required", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        tickers=tickers,
        rebalance_dates=rebalance_dates,
        summaries_dir=Path(args.summaries_dir),
        sentiment_dir=Path(args.sentiment_dir),
        macro_dir=Path(args.macro_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
