#!/usr/bin/env python3
"""Validate snapshot data for every scenario in config/scenarios/.

For each scenario YAML:
  1. Extract invest_quarter and tickers
  2. Compute the data quarter (prior quarter — what agents actually see)
  3. Generate the snapshot JSON (stage 6) for that quarter + tickers
  4. Generate the memo (stage 7) for that quarter + tickers
  5. Run validate_snapshot.py against that quarter
  6. Additionally check that every scenario ticker is present in the snapshot

Snapshots are overwritten each run, so scenarios are processed one at a time.

Usage:
    python data-pipeline/validate_scenarios.py
    python data-pipeline/validate_scenarios.py --verbose
    python data-pipeline/validate_scenarios.py --scenario 2022Q1_inflation_shock
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent          # data-pipeline/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # cs372research/
_SCENARIOS_DIR = _PROJECT_ROOT / "config" / "scenarios"
_SNAPSHOT_DIR = _SCRIPT_DIR / "final_snapshots" / "json_data"

_GENERATE_JSON = _SCRIPT_DIR / "final_snapshots" / "generate_quarterly_json.py"
_GENERATE_MEMO = _SCRIPT_DIR / "final_snapshots" / "generate_quarterly_memo.py"
_VALIDATE_SNAPSHOT = _SCRIPT_DIR / "validate_snapshot.py"


# ── Quarter helpers ──────────────────────────────────────────────────────────

def prev_quarter(year: int, quarter: str) -> Tuple[int, str]:
    """Return the quarter before the given one."""
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx > 0:
        return year, labels[idx - 1]
    return year - 1, "Q4"


def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    year = int(qstr[:4])
    q = qstr[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter: {q}")
    return year, q


# ── Core logic ───────────────────────────────────────────────────────────────

def load_scenario(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_snapshot(year: int, quarter: str, tickers: list[str]) -> bool:
    """Run generate_quarterly_json.py. Returns True on success."""
    qstr = f"{year}{quarter}"
    ticker_str = ",".join(tickers)
    cmd = [
        sys.executable, str(_GENERATE_JSON),
        "--start", qstr, "--end", qstr,
        "--tickers", ticker_str,
    ]
    result = subprocess.run(cmd, cwd=_SCRIPT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    GENERATE JSON FAILED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"      {line}")
        return False
    return True


def generate_memo(year: int, quarter: str, tickers: list[str]) -> bool:
    """Run generate_quarterly_memo.py. Returns True on success."""
    qstr = f"{year}{quarter}"
    ticker_str = ",".join(tickers)
    cmd = [
        sys.executable, str(_GENERATE_MEMO),
        "--start", qstr, "--end", qstr,
        "--tickers", ticker_str,
    ]
    result = subprocess.run(cmd, cwd=_SCRIPT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    GENERATE MEMO FAILED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"      {line}")
        return False
    return True


def validate_snapshot(year: int, quarter: str, tickers: list[str], verbose: bool) -> int:
    """Run validate_snapshot.py with scenario-specific tickers. Returns exit code."""
    cmd = [
        sys.executable, str(_VALIDATE_SNAPSHOT),
        "--year", str(year), "--quarter", quarter,
        "--tickers", ",".join(tickers),
    ]
    if verbose:
        cmd.append("--verbose")
    result = subprocess.run(cmd, cwd=_SCRIPT_DIR)
    return result.returncode


def check_tickers_present(year: int, quarter: str, tickers: list[str]) -> list[str]:
    """Check that all scenario tickers appear in the snapshot. Returns missing tickers."""
    snapshot_path = _SNAPSHOT_DIR / f"snapshot_{year}_{quarter}.json"
    if not snapshot_path.exists():
        return tickers  # all missing

    with open(snapshot_path) as f:
        doc = json.load(f)

    # Tickers live in doc["tickers"] (list) and doc["ticker_data"] (dict)
    snapshot_tickers = set(doc.get("tickers", []))
    snapshot_tickers.update(doc.get("ticker_data", {}).keys())

    return [t for t in tickers if t not in snapshot_tickers]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate snapshots for all scenarios"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Pass --verbose to validate_snapshot.py"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Validate a single scenario by stem name (e.g. 2022Q1_inflation_shock)"
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip snapshot/memo generation; only validate existing snapshots"
    )
    args = parser.parse_args()

    # Discover scenario files
    if args.scenario:
        target = _SCENARIOS_DIR / f"{args.scenario}.yaml"
        if not target.exists():
            print(f"ERROR: scenario not found: {target}")
            sys.exit(1)
        scenario_files = [target]
    else:
        scenario_files = sorted(_SCENARIOS_DIR.glob("*.yaml"))

    if not scenario_files:
        print(f"No scenario files found in {_SCENARIOS_DIR}")
        sys.exit(1)

    print("=" * 64)
    print(" Scenario Validation Pipeline")
    print(f" Scenarios: {len(scenario_files)}")
    print("=" * 64)

    results: list[dict] = []

    for sf in scenario_files:
        scenario = load_scenario(sf)
        invest_q = scenario.get("invest_quarter")
        tickers = scenario.get("tickers", [])

        if not invest_q or not tickers:
            print(f"\n  SKIP: {sf.stem} — missing invest_quarter or tickers")
            results.append({"name": sf.stem, "status": "SKIP"})
            continue

        year, quarter = parse_quarter_string(invest_q)
        data_year, data_quarter = prev_quarter(year, quarter)
        data_qstr = f"{data_year}{data_quarter}"

        print(f"\n{'─' * 64}")
        print(f"  Scenario:       {sf.stem}")
        print(f"  Invest quarter: {invest_q}")
        print(f"  Data quarter:   {data_qstr}")
        print(f"  Tickers:        {len(tickers)} — {', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}")
        print(f"{'─' * 64}")

        scenario_ok = True

        # Step 1: Generate snapshot JSON
        if not args.skip_generate:
            print(f"  [1/4] Generating snapshot JSON for {data_qstr}...")
            if not generate_snapshot(data_year, data_quarter, tickers):
                scenario_ok = False
                results.append({"name": sf.stem, "status": "FAIL", "reason": "snapshot generation failed"})
                continue

            # Step 2: Generate memo
            print(f"  [2/4] Generating memo for {data_qstr}...")
            if not generate_memo(data_year, data_quarter, tickers):
                scenario_ok = False
                results.append({"name": sf.stem, "status": "FAIL", "reason": "memo generation failed"})
                continue
        else:
            print("  [1/4] Skipping snapshot generation (--skip-generate)")
            print("  [2/4] Skipping memo generation (--skip-generate)")

        # Step 3: Check scenario tickers are in snapshot
        print(f"  [3/4] Checking ticker presence in snapshot...")
        missing = check_tickers_present(data_year, data_quarter, tickers)
        if missing:
            print(f"    WARN: {len(missing)} ticker(s) missing from snapshot: {', '.join(missing)}")

        # Step 4: Validate snapshot
        print(f"  [4/4] Validating snapshot {data_qstr}...")
        exit_code = validate_snapshot(data_year, data_quarter, tickers, args.verbose)
        if exit_code != 0:
            scenario_ok = False

        status = "PASS" if scenario_ok and not missing else (
            "WARN" if scenario_ok and missing else "FAIL"
        )
        results.append({
            "name": sf.stem,
            "status": status,
            "missing_tickers": missing if missing else [],
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(" Summary")
    print(f"{'=' * 64}")

    passes = sum(1 for r in results if r["status"] == "PASS")
    warns = sum(1 for r in results if r["status"] == "WARN")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    skips = sum(1 for r in results if r["status"] == "SKIP")

    for r in results:
        marker = {"PASS": "+", "WARN": "~", "FAIL": "x", "SKIP": "-"}[r["status"]]
        line = f"  [{marker}] {r['name']:45s} {r['status']}"
        if r.get("reason"):
            line += f"  ({r['reason']})"
        elif r.get("missing_tickers"):
            line += f"  (missing: {', '.join(r['missing_tickers'])})"
        print(line)

    print(f"\n  {passes} passed, {warns} warnings, {fails} failed, {skips} skipped")
    print(f"{'=' * 64}")

    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
