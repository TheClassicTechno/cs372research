#!/usr/bin/env python3
"""Run data pipeline stages 1-4 for all historical scenario gaps.

Stage 1: EDGAR raw filings (MS, SLB only)
Stage 2: EDGAR filing summarization (requires ANTHROPIC_API_KEY)
Stage 3: Macro economic data
Stage 4: Asset feature details

Usage:
    python data-pipeline/run_historical_stages_1_4.py
"""

import os
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

# ── Stage definitions ────────────────────────────────────────────────────

STAGE_1_EDGAR_RAW = [
    {"tickers": "MS,SLB", "years": "2021", "quarters": "Q3"},
    {"tickers": "MS",     "years": "2022", "quarters": "Q1"},
    {"tickers": "MS,SLB", "years": "2022", "quarters": "Q2,Q4"},
    {"tickers": "MS,SLB", "years": "2023", "quarters": "Q2"},
]

STAGE_2_EDGAR_SUMMARIES = ["MS", "SLB"]

STAGE_3_MACRO_QUARTERS = ["2021Q3", "2022Q2", "2022Q4", "2023Q2", "2023Q4"]

STAGE_4_ASSETS = [
    ("2021Q3", "AAPL,AMD,AMZN,BAC,CAT,COP,COST,CVX,DE,GS,JNJ,JPM,MS,MSFT,NKE,NVDA,PG,SLB,TSLA,UNH,WMT,XOM"),
    ("2022Q2", "AAPL,AMD,AMZN,BAC,CAT,COP,COST,CVX,DAL,GS,JNJ,JPM,MS,MSFT,NVDA,PG,RTX,SLB,TSLA,UNH,WMT,XOM"),
    ("2022Q4", "AAPL,AMD,AMZN,BAC,CAT,COP,COST,CVX,GS,JNJ,JPM,MS,MSFT,NKE,NVDA,PG,RTX,SLB,TSLA,UNH,WMT,XOM"),
    ("2023Q2", "AAPL,AMD,AMZN,BAC,CAT,COP,COST,CVX,GS,JNJ,JPM,MS,MSFT,NKE,NVDA,PG,RTX,SLB,TSLA,UNH,WMT,XOM"),
    ("2023Q4", "AAPL,AMD,AMZN,AVGO,BAC,CAT,COP,COST,CVX,GS,JPM,LLY,MSFT,NKE,NVDA,PG,RTX,TSLA,UNH,WMT,XOM"),
]


def run(cmd: list[str], label: str) -> None:
    """Run a subprocess, printing the label and failing fast on error."""
    print(f"  {label}")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PIPELINE_DIR)
    if result.returncode != 0:
        print(f"\n  ERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    # Pre-flight
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set (required for Stage 2).")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    py = sys.executable
    print("=" * 50)
    print(" Data Pipeline: Historical Stages 1-4")
    print("=" * 50)
    print(f"  Python:    {py}")
    print(f"  Directory: {PIPELINE_DIR}")
    print()

    # ── Stage 1: EDGAR raw filings ───────────────────────────────────
    print("=" * 50)
    print(" Stage 1: EDGAR Raw Filings")
    print("=" * 50)
    for job in STAGE_1_EDGAR_RAW:
        run(
            [py, "EDGAR/get_sec_data.py",
             "--tickers", job["tickers"],
             "--years", job["years"],
             "--quarters", job["quarters"]],
            f"EDGAR raw — {job['tickers']} {job['years']} {job['quarters']}",
        )
    print("  Stage 1 complete.\n")

    # ── Stage 2: EDGAR filing summarization ──────────────────────────
    print("=" * 50)
    print(" Stage 2: EDGAR Filing Summarization")
    print("=" * 50)
    for ticker in STAGE_2_EDGAR_SUMMARIES:
        run(
            [py, "EDGAR/filing_summarization_pipeline.py",
             "--ticker", ticker, "--force"],
            f"Summarize {ticker}",
        )
    print("  Stage 2 complete.\n")

    # ── Stage 3: Macro economic data ─────────────────────────────────
    print("=" * 50)
    print(" Stage 3: Macro Economic Data")
    print("=" * 50)
    for q in STAGE_3_MACRO_QUARTERS:
        run(
            [py, "macro/macro_quarter_builder.py",
             "--start", q, "--end", q],
            f"Macro — {q}",
        )
    print("  Stage 3 complete.\n")

    # ── Stage 4: Asset feature details ───────────────────────────────
    print("=" * 50)
    print(" Stage 4: Asset Feature Details")
    print("=" * 50)
    for quarter, tickers in STAGE_4_ASSETS:
        n = len(tickers.split(","))
        run(
            [py, "quarterly_asset_details/asset_quarter_builder.py",
             "--start", quarter, "--end", quarter,
             "--tickers", tickers],
            f"Assets — {quarter} ({n} tickers)",
        )
    print("  Stage 4 complete.\n")

    # ── Done ─────────────────────────────────────────────────────────
    print("=" * 50)
    print(" All stages 1-4 complete!")
    print()
    print(" Next steps (stages 6-7):")
    print(f"   {py} final_snapshots/generate_quarterly_json.py ...")
    print(f"   {py} final_snapshots/generate_quarterly_memo.py ...")
    print("=" * 50)


if __name__ == "__main__":
    main()
