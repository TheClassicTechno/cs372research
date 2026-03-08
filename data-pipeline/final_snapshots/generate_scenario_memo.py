#!/usr/bin/env python3
"""Generate a scenario-specific memo from a scenario config YAML.

Reads the scenario config to get invest_quarter and tickers, loads the
corresponding snapshot JSON, and writes a filtered memo to
final_snapshots/scenario_memos/.

The memo filename includes the scenario name and quarter:
  scenario_memos/{scenario_name}_memo_{YEAR}_{QUARTER}.txt

Usage:
  python generate_scenario_memo.py --scenario config/scenarios/2023Q2_higher_for_longer_insanity_enriched.yaml

  # Explicit path to scenario (absolute or relative)
  python generate_scenario_memo.py --scenario /path/to/scenario.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_JSON_DIR = _SCRIPT_DIR / "json_data"
_SCENARIO_MEMO_DIR = _SCRIPT_DIR / "scenario_memos"


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate scenario-specific memo from a scenario config YAML.",
    )
    p.add_argument(
        "--scenario", required=True, type=str,
        help="Path to scenario config YAML (e.g. config/scenarios/2023Q2_higher_for_longer.yaml)",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(_SCENARIO_MEMO_DIR),
        help=f"Output directory (default: {_SCENARIO_MEMO_DIR})",
    )
    args = p.parse_args()

    scenario_path = Path(args.scenario).resolve()
    if not scenario_path.exists():
        print(f"ERROR: scenario config not found: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    with open(scenario_path, "r") as f:
        config = yaml.safe_load(f)

    invest_quarter = config.get("invest_quarter")
    tickers = config.get("tickers", [])
    if not invest_quarter:
        print("ERROR: scenario config missing 'invest_quarter'", file=sys.stderr)
        sys.exit(1)
    if not tickers:
        print("ERROR: scenario config missing 'tickers'", file=sys.stderr)
        sys.exit(1)

    scenario_name = scenario_path.stem  # e.g. "2023Q2_higher_for_longer_insanity_enriched"
    macro_context = config.get("macro_context")

    inv_year, inv_q = _parse_invest_quarter(invest_quarter)
    prior_year, prior_q = _prev_quarter(inv_year, inv_q)
    quarters = [(prior_year, prior_q), (inv_year, inv_q)]

    from generate_quarterly_memo import build_memo

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year, quarter in quarters:
        snapshot_path = _JSON_DIR / f"snapshot_{year}_{quarter}.json"
        if not snapshot_path.exists():
            print(f"  SKIP: {snapshot_path} not found", file=sys.stderr)
            continue

        with open(snapshot_path, "r") as f:
            doc = json.load(f)

        memo = build_memo(doc, filter_tickers=tickers, macro_context=macro_context)

        out_path = output_dir / f"{scenario_name}_memo_{year}_{quarter}.txt"
        with open(out_path, "w") as f:
            f.write(memo)
        print(f"  Wrote: {out_path} ({len(tickers)} tickers)")

    print("Done.")


if __name__ == "__main__":
    main()
