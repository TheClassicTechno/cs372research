"""Backfill dashboard metric artifacts for all historical runs.

Usage:
    python tools/dashboard/backfill_dashboard_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dashboard import run_scanner

RUNS_BASE = Path("logging/runs")


def backfill() -> None:
    experiments = run_scanner.list_experiments(RUNS_BASE)
    total = 0
    for exp in experiments:
        if exp == "test":
            continue  # Skip test fixture
        runs = run_scanner.list_runs(RUNS_BASE, exp)
        for run_info in runs:
            run_id = run_info["run_id"]
            print(f"Backfilling {exp}/{run_id}...")
            try:
                run_scanner.compute_debate_impact(RUNS_BASE, exp, run_id)
                run_scanner.compute_collapse_diagnostics(RUNS_BASE, exp, run_id)
                run_scanner.get_run_detail(RUNS_BASE, exp, run_id)
                total += 1
            except Exception as e:
                print(f"  ERROR: {e}")
    print(f"Done. Backfilled {total} runs.")


if __name__ == "__main__":
    backfill()
