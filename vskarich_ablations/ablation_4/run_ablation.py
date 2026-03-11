#!/usr/bin/env python3
"""Run ablation 4 — Intervention stress test.

Single debate config x single scenario to verify both intervention paths
(JS collapse post_revision, reasoning quality post_crit) fire correctly.

Scans:
    vskarich_ablations/ablation_4/debate_configs/*.yaml
    vskarich_ablations/ablation_4/scenario_configs/*.yaml

Usage:
    python vskarich_ablations/ablation_4/run_ablation.py
    python vskarich_ablations/ablation_4/run_ablation.py --reset
    python vskarich_ablations/ablation_4/run_ablation.py --retries 1
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from tqdm import tqdm

ABLATION_DIR = Path(__file__).resolve().parent
DEBATE_DIR = ABLATION_DIR / "debate_configs"
SCENARIO_DIR = ABLATION_DIR / "scenario_configs"
STATUS_FILE = ABLATION_DIR / "ablation_status.json"
MANIFEST_FILE = ABLATION_DIR / "ablation_manifest.json"
REPO_ROOT = ABLATION_DIR.parent.parent


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def _load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_status(status: dict) -> None:
    STATUS_FILE.write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_key(debate_stem: str, scenario_stem: str) -> str:
    return f"{debate_stem}  x  {scenario_stem}"


# ---------------------------------------------------------------------------
# Experiment manifest
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_manifest(
    debate_configs: list[Path],
    scenario_configs: list[Path],
    total: int,
    workers: int,
    retries: int,
) -> None:
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "workers": workers,
        "retries": retries,
        "debate_configs": [f.name for f in debate_configs],
        "scenario_configs": [f.name for f in scenario_configs],
        "total_runs": total,
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m"
    return f"{int(seconds)}s"


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def _execute_run(
    debate_path: str,
    scenario_path: str,
    retries: int,
    quiet: bool = False,
) -> dict:
    """Run a single ablation pair with retry logic."""
    debate_name = Path(debate_path).stem
    scenario_name = Path(scenario_path).stem

    cmd = [
        sys.executable, "run_simulation.py",
        "--agents", debate_path,
        "--scenario", scenario_path,
    ]

    last_exit = -1
    attempts = retries + 1
    elapsed = 0

    for attempt in range(1, attempts + 1):
        try:
            start = time.time()
            result = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
            )
            elapsed = time.time() - start
            last_exit = result.returncode

            if result.returncode == 0:
                return {
                    "key": _run_key(debate_name, scenario_name),
                    "ok": True,
                    "exit_code": 0,
                    "elapsed_s": round(elapsed, 1),
                    "debate_config": Path(debate_path).name,
                    "scenario_config": Path(scenario_path).name,
                    "attempts": attempt,
                }

            if attempt < attempts:
                tqdm.write(
                    f"  Attempt {attempt}/{attempts} failed (exit {result.returncode}), retrying..."
                )

        except Exception as exc:
            elapsed = time.time() - start if 'start' in dir() else 0
            tqdm.write(
                f"  ERROR during attempt {attempt}/{attempts}: {exc}\n"
                f"  {traceback.format_exc()}"
                f"  Continuing to next experiment..."
            )
            last_exit = -1
            if attempt < attempts:
                tqdm.write(f"  Retrying...")

    return {
        "key": _run_key(debate_name, scenario_name),
        "ok": False,
        "exit_code": last_exit,
        "elapsed_s": round(elapsed, 1),
        "debate_config": Path(debate_path).name,
        "scenario_config": Path(scenario_path).name,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation 4 — intervention stress test.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe ablation_status.json and start fresh.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, choices=[1, 2, 3, 4],
        help="Number of parallel workers (1-4; default: 1 = serial).",
    )
    parser.add_argument(
        "--retries", type=int, default=0,
        help="Number of retries per failed run (default: 0).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-run simulation output (keep only progress bar).",
    )
    args = parser.parse_args()

    if args.reset and STATUS_FILE.exists():
        STATUS_FILE.unlink()
        print("Status file reset.\n")

    debate_configs = sorted(DEBATE_DIR.glob("*.yaml"))
    scenario_configs = sorted(SCENARIO_DIR.glob("*.yaml"))

    if not debate_configs:
        print(f"No debate configs found in {DEBATE_DIR}")
        sys.exit(1)
    if not scenario_configs:
        print(f"No scenario configs found in {SCENARIO_DIR}")
        sys.exit(1)

    pairs = list(product(debate_configs, scenario_configs))
    total = len(pairs)

    status = _load_status()

    remaining = [
        (d, s) for d, s in pairs
        if status.get(_run_key(d.stem, s.stem), {}).get("result") != "done"
    ]
    already_done = total - len(remaining)

    print(f"Ablation 4 — Intervention Stress Test")
    print(f"  {len(debate_configs)} debate config(s) x "
          f"{len(scenario_configs)} scenario config(s) = {total} total")
    if already_done:
        print(f"  {already_done} already done, {len(remaining)} remaining")
    print(f"  workers={args.workers}, retries={args.retries}\n")

    if not remaining:
        print("All runs complete. Use --reset to start fresh.")
        sys.exit(0)

    _write_manifest(debate_configs, scenario_configs, total, args.workers, args.retries)

    session_results: list[dict] = []
    elapsed_times: list[float] = []

    if args.workers <= 1:
        pbar = tqdm(total=len(remaining), desc="Ablation 4", unit="run")

        for debate, scenario in remaining:
            pbar.set_postfix_str(f"{debate.stem[:30]}...", refresh=True)

            run_result = _execute_run(str(debate), str(scenario), args.retries, quiet=args.quiet)

            status[run_result["key"]] = {
                "result": "done" if run_result["ok"] else "failed",
                "exit_code": run_result["exit_code"],
                "elapsed_s": run_result["elapsed_s"],
                "debate_config": run_result["debate_config"],
                "scenario_config": run_result["scenario_config"],
            }
            _save_status(status)

            session_results.append(run_result)
            elapsed_times.append(run_result["elapsed_s"])
            pbar.update(1)

            avg = sum(elapsed_times) / len(elapsed_times)
            runs_left = len(remaining) - len(session_results)
            eta = _format_eta(avg * runs_left)
            pbar.set_postfix_str(f"ETA {eta}", refresh=True)

            if not run_result["ok"]:
                tqdm.write(
                    f"  WARNING: {run_result['key']} failed "
                    f"(exit {run_result['exit_code']})"
                )

        pbar.close()

    else:
        pbar = tqdm(total=len(remaining), desc="Ablation 4", unit="run")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_key = {}
            for debate, scenario in remaining:
                future = executor.submit(
                    _execute_run, str(debate), str(scenario), args.retries, args.quiet,
                )
                future_to_key[future] = _run_key(debate.stem, scenario.stem)

            for future in as_completed(future_to_key):
                try:
                    run_result = future.result()
                except Exception as exc:
                    key = future_to_key[future]
                    tqdm.write(f"  ERROR: {key} raised {exc}")
                    tqdm.write(traceback.format_exc())
                    run_result = {
                        "key": key,
                        "ok": False,
                        "exit_code": -1,
                        "elapsed_s": 0,
                        "debate_config": "",
                        "scenario_config": "",
                    }

                status[run_result["key"]] = {
                    "result": "done" if run_result["ok"] else "failed",
                    "exit_code": run_result["exit_code"],
                    "elapsed_s": run_result["elapsed_s"],
                    "debate_config": run_result["debate_config"],
                    "scenario_config": run_result["scenario_config"],
                }
                _save_status(status)

                session_results.append(run_result)
                elapsed_times.append(run_result["elapsed_s"])
                pbar.update(1)

                avg = sum(elapsed_times) / len(elapsed_times)
                runs_left = len(remaining) - len(session_results)
                eta = _format_eta(avg * runs_left)
                pbar.set_postfix_str(f"ETA {eta}", refresh=True)

                if not run_result["ok"]:
                    tqdm.write(
                        f"  WARNING: {run_result['key']} failed "
                        f"(exit {run_result['exit_code']})"
                    )

        pbar.close()

    # Session summary
    passed = sum(1 for r in session_results if r["ok"])
    failed = len(session_results) - passed
    total_done = sum(1 for v in status.values() if v.get("result") == "done")
    total_time = sum(r["elapsed_s"] for r in session_results)

    print("\n" + "=" * 70)
    print(f"SESSION: {passed} passed, {failed} failed, {_format_eta(total_time)}")
    print(f"OVERALL: {total_done}/{total} done")
    print("=" * 70)

    if failed:
        print("\nFailed this session:")
        for r in session_results:
            if not r["ok"]:
                print(f"  - {r['key']}")
        print("\nRe-run to retry failed runs.")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
