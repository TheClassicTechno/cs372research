#!/usr/bin/env python3
"""
Daily Price Builder — Fetch and store daily close prices for investment quarters.

Downloads daily close prices from Yahoo Finance for each ticker in the
investment quarter.  These are NOT fed to agents — they are used purely
for post-hoc evaluation (Sharpe, Sortino, max drawdown, etc.) via
eval/financial.py.

Output per ticker per quarter:
  data-pipeline/daily_prices/data/{TICKER}/{YEAR}_{QUARTER}.json

Also fetches SPY as a benchmark for every quarter.

Examples:

  # Single quarter, specific tickers
  python daily_price_builder.py --invest-quarter 2022Q2 --tickers AAPL,NVDA,JPM

  # Single quarter, all supported tickers
  python daily_price_builder.py --invest-quarter 2022Q2 --supported

  # Multiple quarters
  python daily_price_builder.py --quarters 2021Q4,2022Q2,2022Q3,2023Q1,2023Q3,2024Q1,2025Q1 --supported

  # Dry run (show what would be fetched)
  python daily_price_builder.py --invest-quarter 2022Q2 --supported --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data"
_SUPPORTED_TICKERS_PATH = _SCRIPT_DIR.parent / "supported_tickers.yaml"
_ASSETS_DIR = _SCRIPT_DIR.parent / "quarterly_asset_details" / "data"


# ---------------------------------------------------------------------------
# Quarter helpers (duplicated from asset_quarter_builder to avoid imports)
# ---------------------------------------------------------------------------

def _q_start_date(year: int, quarter: str) -> dt.date:
    """First calendar day of the quarter."""
    return {
        "Q1": dt.date(year, 1, 1),
        "Q2": dt.date(year, 4, 1),
        "Q3": dt.date(year, 7, 1),
        "Q4": dt.date(year, 10, 1),
    }[quarter]


def _q_end_date(year: int, quarter: str) -> dt.date:
    """Last calendar day of the quarter."""
    return {
        "Q1": dt.date(year, 3, 31),
        "Q2": dt.date(year, 6, 30),
        "Q3": dt.date(year, 9, 30),
        "Q4": dt.date(year, 12, 31),
    }[quarter]


def _parse_quarter(qstr: str) -> tuple[int, str]:
    """'2022Q2' -> (2022, 'Q2')."""
    if len(qstr) < 5:
        raise ValueError(f"Invalid quarter string: {qstr!r}")
    year = int(qstr[:4])
    q = qstr[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter: {q}")
    return year, q


def _iso(d: dt.date) -> str:
    return d.isoformat()


# ---------------------------------------------------------------------------
# Ticker loading
# ---------------------------------------------------------------------------

def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list[str]:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


def _load_existing_close(ticker: str, year: int, quarter: str) -> Optional[float]:
    """Load the existing quarterly close price for cross-validation."""
    path = _ASSETS_DIR / ticker / f"{year}_{quarter}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc.get("features", {}).get("close")
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Core: fetch + validate daily prices
# ---------------------------------------------------------------------------

def fetch_daily_prices(
    tickers: list[str],
    year: int,
    quarter: str,
    *,
    include_spy: bool = True,
) -> dict[str, list[dict]]:
    """Download daily close prices for the given quarter.

    Returns {ticker: [{"date": "YYYY-MM-DD", "close": float}, ...]} sorted
    by date ascending.  Also includes SPY if include_spy=True.
    """
    q_start = _q_start_date(year, quarter)
    q_end = _q_end_date(year, quarter)

    # yfinance end date is exclusive, so add 1 day
    dl_end = q_end + dt.timedelta(days=1)

    all_symbols = list(tickers)
    if include_spy and "SPY" not in all_symbols:
        all_symbols.append("SPY")

    print(f"  Downloading {len(all_symbols)} symbols for {year} {quarter} "
          f"({_iso(q_start)} to {_iso(q_end)})...", flush=True)

    try:
        raw = yf.download(
            all_symbols,
            start=_iso(q_start),
            end=_iso(dl_end),
            group_by="ticker",
            progress=False,
        )
    except Exception as e:
        print(f"  FAIL batch download — {type(e).__name__}: {e}", flush=True)
        return {}

    if raw.empty:
        print("  WARNING: empty download result", flush=True)
        return {}

    result: dict[str, list[dict]] = {}
    for sym in all_symbols:
        try:
            if len(all_symbols) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                df = raw[sym]

            px = df["Close"].dropna()
            # Filter to quarter bounds (safety)
            px = px[(px.index >= pd.Timestamp(q_start))
                    & (px.index <= pd.Timestamp(q_end))]

            if len(px) == 0:
                print(f"  [{sym}] WARNING: no data in quarter", flush=True)
                continue

            daily = [
                {"date": idx.strftime("%Y-%m-%d"), "close": round(float(val), 4)}
                for idx, val in px.items()
            ]
            result[sym] = daily

        except Exception as e:
            print(f"  [{sym}] FAIL — {type(e).__name__}: {e}", flush=True)

    return result


def validate_daily_prices(
    daily: list[dict],
    ticker: str,
    year: int,
    quarter: str,
) -> list[str]:
    """Run validation checks on fetched daily prices.

    Returns list of warning strings (empty = all good).
    """
    warnings: list[str] = []
    q_start = _q_start_date(year, quarter)
    q_end = _q_end_date(year, quarter)

    if not daily:
        warnings.append(f"[{ticker}] No daily data")
        return warnings

    dates = [dt.date.fromisoformat(d["date"]) for d in daily]
    first_date = dates[0]
    last_date = dates[-1]

    # Check 1: first trading day within 5 calendar days of quarter start
    if (first_date - q_start).days > 5:
        warnings.append(
            f"[{ticker}] First trading day {first_date} is {(first_date - q_start).days}d "
            f"after quarter start {q_start}"
        )

    # Check 2: last trading day within 5 calendar days of quarter end
    if (q_end - last_date).days > 5:
        warnings.append(
            f"[{ticker}] Last trading day {last_date} is {(q_end - last_date).days}d "
            f"before quarter end {q_end}"
        )

    # Check 3: reasonable number of trading days (expect ~60-66)
    n_days = len(daily)
    if n_days < 50:
        warnings.append(f"[{ticker}] Only {n_days} trading days (expected ~60-66)")
    elif n_days > 70:
        warnings.append(f"[{ticker}] {n_days} trading days (expected ~60-66)")

    # Check 4: no large gaps (> 5 calendar days between consecutive trading days)
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > 5:
            warnings.append(
                f"[{ticker}] {gap}-day gap between {dates[i-1]} and {dates[i]}"
            )

    # Check 5: cross-validate with existing quarterly close price
    existing_close = _load_existing_close(ticker, year, quarter)
    if existing_close is not None:
        last_close = daily[-1]["close"]
        pct_diff = abs(last_close - existing_close) / existing_close
        if pct_diff > 0.03:  # 3% tolerance for adjusted close differences (splits, etc.)
            warnings.append(
                f"[{ticker}] Last daily close {last_close:.2f} differs from "
                f"quarterly close {existing_close:.2f} by {pct_diff:.1%}"
            )

    return warnings


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_daily_prices(
    ticker: str,
    year: int,
    quarter: str,
    daily: list[dict],
    out_dir: Path = _DEFAULT_DATA_DIR,
) -> Path:
    """Write daily price JSON for one ticker/quarter."""
    ticker_dir = out_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    out_path = ticker_dir / f"{year}_{quarter}.json"

    doc = {
        "schema_version": "daily_prices_v1",
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "start_date": daily[0]["date"] if daily else None,
        "end_date": daily[-1]["date"] if daily else None,
        "trading_days": len(daily),
        "daily_close": daily,
    }

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    tmp.rename(out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch and store daily close prices for investment quarters."
    )
    p.add_argument("--invest-quarter", type=str, default=None,
                   help="Single invest quarter (e.g. 2022Q2)")
    p.add_argument("--quarters", type=str, default=None,
                   help="Comma-separated invest quarters (e.g. 2021Q4,2022Q2)")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true",
                   help="Use all tickers from supported_tickers.yaml")
    p.add_argument("--extra-tickers", type=str, default=None,
                   help="Additional tickers beyond supported (e.g. SLB,OXY)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: data-pipeline/daily_prices/data/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be fetched without downloading")
    p.add_argument("--no-spy", action="store_true",
                   help="Skip fetching SPY benchmark data")
    args = p.parse_args()

    # --- Resolve quarters ---
    if args.quarters:
        quarter_strs = [q.strip() for q in args.quarters.split(",")]
    elif args.invest_quarter:
        quarter_strs = [args.invest_quarter]
    else:
        p.error("specify --invest-quarter or --quarters")

    quarters = [_parse_quarter(q) for q in quarter_strs]

    # --- Resolve tickers ---
    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        p.error("specify --tickers or --supported")

    if args.extra_tickers:
        extras = [t.strip().upper() for t in args.extra_tickers.split(",")]
        tickers = list(dict.fromkeys(tickers + extras))  # dedupe, preserve order

    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_DATA_DIR

    # --- Summary ---
    print(f"Daily Price Builder")
    print(f"  Quarters: {', '.join(quarter_strs)}")
    print(f"  Tickers:  {len(tickers)} ({', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''})")
    print(f"  SPY:      {'yes' if not args.no_spy else 'no'}")
    print(f"  Output:   {out_dir}")
    print()

    if args.dry_run:
        total_files = len(quarters) * (len(tickers) + (0 if args.no_spy else 1))
        print(f"  DRY RUN: would fetch {total_files} ticker-quarter files")
        for year, q in quarters:
            print(f"    {year}{q}: {', '.join(tickers)}"
                  f"{', SPY' if not args.no_spy else ''}")
        return

    # --- Fetch and write ---
    total_ok = 0
    total_warn = 0
    total_fail = 0

    for year, quarter in quarters:
        print(f"\n{'='*60}")
        print(f"Quarter: {year} {quarter}")
        print(f"{'='*60}")

        all_daily = fetch_daily_prices(
            tickers, year, quarter,
            include_spy=not args.no_spy,
        )

        # Write each ticker (including SPY)
        all_tickers_to_write = list(tickers)
        if not args.no_spy:
            all_tickers_to_write.append("SPY")

        for t in all_tickers_to_write:
            daily = all_daily.get(t, [])
            if not daily:
                print(f"  [{t}] SKIP — no data", flush=True)
                total_fail += 1
                continue

            # Validate
            warnings = validate_daily_prices(daily, t, year, quarter)
            for w in warnings:
                print(f"  WARN: {w}", flush=True)
                total_warn += 1

            # Write
            path = write_daily_prices(t, year, quarter, daily, out_dir)
            n = len(daily)
            first = daily[0]["date"]
            last = daily[-1]["date"]
            print(f"  [{t}] OK — {n} days ({first} to {last}) → {path.name}",
                  flush=True)
            total_ok += 1

    print(f"\n{'='*60}")
    print(f"Summary: {total_ok} OK, {total_warn} warnings, {total_fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
