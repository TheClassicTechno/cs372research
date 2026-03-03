"""CLI entrypoint for the PID stability-first ablation suite.

Usage:
    python -m scripts.ablation.main [options]

Examples:
    # Full stability sweep (mock, ~130+ runs × 3 replicates)
    python -m scripts.ablation.main --mock --seed 42

    # Dry run — print matrix only
    python -m scripts.ablation.main --dry-run

    # Random gain sampling only
    python -m scripts.ablation.main --groups random_gain_samples --seed 42 --mock

    # Quad-scenario run
    python -m scripts.ablation.main --groups gains --scenario-set quad --mock

    # Replicated run (5x each), specific groups
    python -m scripts.ablation.main --groups gains,dynamics --replicates 5 --mock --seed 42

    # Real API, single group, conservative
    python -m scripts.ablation.main --groups gains --max-workers 2 --delay 3

    # Anthropic models (auto max_workers=1)
    python -m scripts.ablation.main --groups models --seed 42

    # Quarterly episodes (4 debates per config, using real memo data)
    python -m scripts.ablation.main --quarterly --mock --seed 42

    # Quarterly with custom dataset path
    python -m scripts.ablation.main --quarterly --dataset-path data-pipeline/final_snapshots --mock
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from scripts.ablation.config import (
    ALL_GROUPS,
    DEFAULT_DATASET_PATH,
    DEFAULT_MEMO_FORMAT,
    DEFAULT_REPLICATES,
    SWEEP_GROUPS,
)
from scripts.ablation.io import build_suite_summary, write_results
from scripts.ablation.matrix import count_runs, generate_run_matrix
from scripts.ablation.quarterly_runner import run_quarterly_ablation
from scripts.ablation.runner import run_single_ablation
from scripts.ablation.visualize import generate_plots

logger = logging.getLogger("ablation")

SCENARIO_SETS = {
    "single": ["neutral"],
    "triple": ["bullish", "neutral", "riskoff"],
    "quad": ["bullish", "neutral", "riskoff", "conflicted"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PID stability-first ablation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print run matrix without executing.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM (no API calls).",
    )
    parser.add_argument(
        "--groups", type=str, default=None,
        help="Comma-separated sweep groups to run (default: all).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=3,
        help="Max parallel debates (default: 3).",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Inter-run delay in seconds (default: 2).",
    )
    parser.add_argument(
        "--force-parallel", action="store_true",
        help="Override auto max_workers=1 for Anthropic models.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_REPLICATES,
        help=f"Replicates per config (default: {DEFAULT_REPLICATES}).",
    )
    parser.add_argument(
        "--scenario-set", type=str, default="single",
        choices=list(SCENARIO_SETS.keys()),
        help="Scenario set: single (neutral), triple (+bullish/riskoff), quad (+conflicted).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="ablation_results",
        help="Base output directory (default: ablation_results).",
    )
    parser.add_argument(
        "--list-groups", action="store_true",
        help="List available sweep groups and exit.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip visualization generation.",
    )
    parser.add_argument(
        "--quarterly", action="store_true",
        help="Run 4 quarterly episodes (2025Q1-Q4) per config instead of single.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
        help=f"Path to final_snapshots/ for real quarterly data (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--memo-format", type=str, default=DEFAULT_MEMO_FORMAT,
        choices=["text", "json"],
        help=f"Memo format for quarterly data loader (default: {DEFAULT_MEMO_FORMAT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_groups:
        print("Available sweep groups:")
        for name in ALL_GROUPS:
            entries = SWEEP_GROUPS.get(name, [])
            count = len(entries) if entries else "(runtime-generated)"
            print(f"  {name:30s} {count} configs")
        sys.exit(0)

    # Set seeds
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass

    # Resolve groups
    groups = args.groups.split(",") if args.groups else None
    if groups:
        unknown = set(groups) - set(ALL_GROUPS)
        if unknown:
            print(f"ERROR: Unknown groups: {unknown}")
            print(f"Available: {ALL_GROUPS}")
            sys.exit(1)

    scenarios = SCENARIO_SETS[args.scenario_set]

    # Generate matrix
    matrix = generate_run_matrix(
        groups=groups,
        scenarios=scenarios,
        replicates=args.replicates,
        seed=args.seed,
    )

    total = len(matrix)
    est_count = count_runs(groups, scenarios, args.replicates)

    mode_label = "QUARTERLY (4 episodes/config)" if args.quarterly else "SINGLE"

    print(f"\n{'='*60}")
    print(f"  PID ABLATION SUITE")
    print(f"{'='*60}")
    print(f"  Mode: {mode_label}")
    print(f"  Groups: {groups or 'all'}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Replicates: {args.replicates}")
    print(f"  Total runs: {total}")
    print(f"  Mock: {args.mock}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Seed: {args.seed}")
    if args.quarterly:
        print(f"  Dataset path: {args.dataset_path}")
        print(f"  Memo format: {args.memo_format}")
    print(f"{'='*60}\n")

    if args.dry_run:
        _print_matrix(matrix)
        print(f"\n  Total: {total} runs (dry run — no execution)")
        sys.exit(0)

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)

    # Run ablation suite
    results = _run_suite(
        matrix=matrix,
        mock=args.mock,
        trace_dir=trace_dir,
        max_workers=args.max_workers,
        delay=args.delay,
        force_parallel=args.force_parallel,
        quarterly=args.quarterly,
        dataset_path=args.dataset_path,
        memo_format=args.memo_format,
    )

    # Write results
    cli_args = vars(args)
    write_results(results, output_dir, cli_args)

    # Generate plots
    if not args.no_plots:
        plots = generate_plots(results, output_dir)
        if plots:
            logger.info("Generated %d plots", len(plots))

    # Print summary
    summary = build_suite_summary(results)
    print(f"\n{summary}")

    print(f"\n  Output: {output_dir}")
    print(f"  Summary CSV: {output_dir / 'summary.csv'}")
    if not args.no_plots:
        print(f"  Plots: {output_dir / 'plots'}")


def _run_suite(
    matrix: list[dict],
    mock: bool,
    trace_dir: Path,
    max_workers: int,
    delay: float,
    force_parallel: bool,
    quarterly: bool = False,
    dataset_path: str = DEFAULT_DATASET_PATH,
    memo_format: str = DEFAULT_MEMO_FORMAT,
) -> list[dict]:
    """Execute the full ablation matrix with parallelization and rate limiting."""
    total = len(matrix)
    results: list[dict] = []

    # Provider-safe parallelism: auto-throttle for Anthropic
    has_anthropic = any(
        r.get("model_name", "").startswith("claude") for r in matrix
    )
    effective_workers = max_workers
    effective_delay = delay
    if has_anthropic and not force_parallel:
        effective_workers = min(effective_workers, 1)
        effective_delay = max(effective_delay, 5.0)
        logger.info(
            "Anthropic models detected — throttling to %d worker(s), %.1fs delay",
            effective_workers, effective_delay,
        )

    completed_count = 0

    # Select runner function based on mode
    def _execute_one(run_config: dict) -> dict:
        if quarterly:
            return run_quarterly_ablation(
                run_config, mock=mock, trace_dir=trace_dir,
                dataset_path=dataset_path, memo_format=memo_format,
            )
        return run_single_ablation(run_config, mock=mock, trace_dir=trace_dir)

    if effective_workers <= 1:
        # Sequential execution
        for i, run_config in enumerate(matrix):
            if i > 0:
                time.sleep(effective_delay)
            logger.info("[%d/%d] Starting %s", i + 1, total, run_config["run_id"])
            result = _execute_one(run_config)
            results.append(result)
            completed_count += 1
            _print_progress(completed_count, total)
    else:
        # Parallel execution with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {}
            for i, run_config in enumerate(matrix):
                future = executor.submit(
                    _delayed_run, run_config, mock, trace_dir,
                    effective_delay * i / effective_workers,
                    quarterly, dataset_path, memo_format,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "run_id": matrix[idx].get("run_id", f"run_{idx}"),
                        "status": f"executor_error: {exc}",
                    }
                results.append(result)
                completed_count += 1
                _print_progress(completed_count, total)

    # Sort results by run_id for deterministic output
    results.sort(key=lambda r: r.get("run_id", ""))
    return results


def _delayed_run(
    run_config: dict, mock: bool, trace_dir: Path, delay: float,
    quarterly: bool = False, dataset_path: str = DEFAULT_DATASET_PATH,
    memo_format: str = DEFAULT_MEMO_FORMAT,
) -> dict:
    """Run a single ablation with initial delay for rate limiting."""
    if delay > 0:
        time.sleep(delay)
    if quarterly:
        return run_quarterly_ablation(
            run_config, mock=mock, trace_dir=trace_dir,
            dataset_path=dataset_path, memo_format=memo_format,
        )
    return run_single_ablation(run_config, mock=mock, trace_dir=trace_dir)


def _print_progress(done: int, total: int) -> None:
    """Print progress bar to stderr."""
    pct = 100 * done / total if total else 100
    bar_len = 30
    filled = int(bar_len * done / total) if total else bar_len
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%)", end="", flush=True, file=sys.stderr)
    if done == total:
        print(file=sys.stderr)


def _print_matrix(matrix: list[dict]) -> None:
    """Print the run matrix in a readable format."""
    print(f"\n  {'run_id':60s} {'group':25s} {'param':15s} {'value':20s} {'scenario':10s} rep")
    print(f"  {'-'*60} {'-'*25} {'-'*15} {'-'*20} {'-'*10} ---")
    for r in matrix:
        print(
            f"  {r['run_id']:60s} {r['group']:25s} {r['param']:15s} "
            f"{str(r['value']):20s} {r['scenario']:10s} {r['replicate']}"
        )


if __name__ == "__main__":
    main()
