#!/usr/bin/env python3
"""Auto-generate snapshot JSONs and memos for a scenario's tickers/quarter.

Called by run_simulation.py before the simulation starts to ensure the
required data files exist.  Wraps stages 6 (snapshot JSON) and 7 (memo)
of the data pipeline.

Usage (standalone):
    python snapshot_builder.py --tickers AAPL,NVDA --invest-quarter 2025Q1

Usage (from run_simulation.py — called via subprocess):
    subprocess.run([sys.executable, "data-pipeline/final_snapshots/snapshot_builder.py",
                    "--tickers", "AAPL,NVDA", "--invest-quarter", "2025Q1"], check=True)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve directories relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent          # final_snapshots/
_PIPELINE_DIR = _SCRIPT_DIR.parent                     # data-pipeline/

# Upstream data directories
_ASSETS_DIR = _PIPELINE_DIR / "quarterly_asset_details" / "data"
_MACRO_DIR = _PIPELINE_DIR / "macro" / "data"
_SUMMARIES_DIR = _PIPELINE_DIR / "EDGAR" / "finished_summaries"
_EARNINGS_CALLS_DIR = _PIPELINE_DIR / "earnings_calls" / "data"
_SENTIMENT_DIR = _PIPELINE_DIR / "sentiment" / "data"

# Output directories
_JSON_DIR = _SCRIPT_DIR / "json_data"
_MEMO_DIR = _SCRIPT_DIR / "memo_data"


def _prev_quarter(year: int, quarter: str) -> tuple[int, str]:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx == 0:
        return year - 1, "Q4"
    return year, labels[idx - 1]


def _parse_invest_quarter(invest_quarter: str) -> tuple[int, str]:
    year = int(invest_quarter[:4])
    q = invest_quarter[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter: {q}")
    return year, q


def _preflight_check(
    tickers: list[str],
    quarters: list[tuple[int, str]],
) -> None:
    """Verify required upstream files exist; raise with actionable message if not."""
    missing_required: list[str] = []
    warnings: list[str] = []

    for year, quarter in quarters:
        # Macro — required (shared across tickers)
        macro_file = _MACRO_DIR / f"macro_{year}_{quarter}.json"
        if not macro_file.exists():
            missing_required.append(str(macro_file.relative_to(_PIPELINE_DIR)))

        for ticker in tickers:
            # Asset details — required (has prices)
            asset_file = _ASSETS_DIR / ticker / f"{year}_{quarter}.json"
            if not asset_file.exists():
                missing_required.append(str(asset_file.relative_to(_PIPELINE_DIR)))

            # EDGAR summaries — warn if empty
            edgar_dir = _SUMMARIES_DIR / ticker
            if not edgar_dir.exists() or not any(edgar_dir.rglob("*.json")):
                warnings.append(f"EDGAR/finished_summaries/{ticker}/ — no filing summaries (filings may lag)")

            # Sentiment — warn if missing (optional)
            sentiment_file = _SENTIMENT_DIR / ticker / f"{year}_{quarter}.json"
            if not sentiment_file.exists():
                warnings.append(f"sentiment/data/{ticker}/{year}_{quarter}.json — missing (optional)")

    if warnings:
        # Group warnings by category for compact display
        sentiment_missing: dict[str, list[str]] = {}  # quarter -> [tickers]
        edgar_missing: list[str] = []
        other: list[str] = []
        for w in warnings:
            if "missing (optional)" in w:
                # Extract ticker and quarter from path like "sentiment/data/AAPL/2022_Q2.json"
                parts = w.split("/")
                if len(parts) >= 4:
                    ticker = parts[2]
                    qtr = parts[3].replace(".json — missing (optional)", "")
                    sentiment_missing.setdefault(qtr, []).append(ticker)
                else:
                    other.append(w)
            elif "no filing summaries" in w:
                ticker = w.split("/")[1] if "/" in w else w
                edgar_missing.append(ticker)
            else:
                other.append(w)
        if sentiment_missing:
            for qtr in sorted(sentiment_missing):
                tickers_list = sorted(sentiment_missing[qtr])
                print(
                    f"  WARNING: sentiment data missing (optional) for {qtr}: "
                    f"{', '.join(tickers_list)} ({len(tickers_list)} tickers)",
                    file=sys.stderr,
                )
        if edgar_missing:
            print(
                f"  WARNING: no EDGAR filing summaries for: "
                f"{', '.join(sorted(edgar_missing))} (filings may lag)",
                file=sys.stderr,
            )
        for w in other:
            print(f"  WARNING: {w}", file=sys.stderr)

    if missing_required:
        # Build helpful remediation commands
        q_set = {f"{y}Q{q[1:]}" for y, q in quarters}
        q_sorted = sorted(q_set)
        ticker_str = ",".join(sorted(set(tickers)))

        msg_lines = [
            "Missing upstream data required for snapshot generation.",
            "",
            "Required files not found:",
        ]
        for f in missing_required:
            msg_lines.append(f"  {f}")
        msg_lines.append("")
        msg_lines.append("Run the data pipeline to generate missing data:")
        msg_lines.append("  cd data-pipeline")
        msg_lines.append(
            f"  python quarterly_asset_details/asset_quarter_builder.py"
            f" --start {q_sorted[0]} --end {q_sorted[-1]} --tickers {ticker_str}"
        )
        msg_lines.append(
            f"  python macro/macro_quarter_builder.py"
            f" --start {q_sorted[0]} --end {q_sorted[-1]}"
        )
        raise FileNotFoundError("\n".join(msg_lines))


def ensure_snapshots(
    tickers: list[str],
    invest_quarter: str,
) -> None:
    """Build snapshot JSONs and memos for the invest quarter and its prior quarter.

    Stages:
        1. Pre-flight check — verify upstream data exists.
        2. Stage 6 — build quarterly snapshot JSONs via build_quarter_snapshot().
        3. Stage 7 — build filtered memos via build_memo().
    """
    inv_year, inv_q = _parse_invest_quarter(invest_quarter)
    prior_year, prior_q = _prev_quarter(inv_year, inv_q)
    quarters = [(prior_year, prior_q), (inv_year, inv_q)]

    # --- Pre-flight ---
    _preflight_check(tickers, quarters)

    # --- Stage 6: snapshot JSONs ---
    from generate_quarterly_json import (
        build_quarter_snapshot,
        load_fiscal_year_ends,
    )

    fiscal_year_ends = load_fiscal_year_ends()

    for year, quarter in quarters:
        print(f"  Building snapshot {year}_{quarter} for {len(tickers)} tickers...")
        snapshot = build_quarter_snapshot(
            year, quarter, tickers,
            summaries_dir=_SUMMARIES_DIR,
            earnings_calls_dir=_EARNINGS_CALLS_DIR,
            sentiment_dir=_SENTIMENT_DIR,
            macro_dir=_MACRO_DIR,
            assets_dir=_ASSETS_DIR,
            fiscal_year_ends=fiscal_year_ends,
        )

        out_path = _JSON_DIR / f"snapshot_{year}_{quarter}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        tmp.replace(out_path)
        print(f"  Wrote: {out_path}")

    # --- Stage 7: memos ---
    from generate_quarterly_memo import build_memo

    for year, quarter in quarters:
        snapshot_path = _JSON_DIR / f"snapshot_{year}_{quarter}.json"
        with open(snapshot_path, "r") as f:
            doc = json.load(f)

        memo = build_memo(doc, filter_tickers=tickers)

        out_path = _MEMO_DIR / f"memo_{year}_{quarter}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(memo)
        print(f"  Wrote: {out_path}")

    print(f"  Snapshot generation complete for {invest_quarter}.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Auto-generate snapshot JSONs and memos for a scenario.",
    )
    p.add_argument("--tickers", required=True, type=str,
                   help="Comma-separated tickers (e.g. AAPL,NVDA,JPM)")
    p.add_argument("--invest-quarter", required=True, type=str,
                   help="Invest quarter (e.g. 2025Q1)")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ensure_snapshots(tickers=tickers, invest_quarter=args.invest_quarter)


if __name__ == "__main__":
    main()
