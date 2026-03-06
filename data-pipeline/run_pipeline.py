#!/usr/bin/env python3
"""
Pipeline orchestrator - runs the 6 data-gathering stages in sequence.

Usage:
    # Add a new ticker and run full pipeline
    python run_pipeline.py --ticker NFLX --sector "Communication Services" \
        --description "Netflix -- Streaming" --start 2024Q4 --end 2025Q3

    # Re-run pipeline for all existing supported tickers
    python run_pipeline.py --start 2024Q4 --end 2025Q3

    # Preview commands without executing
    python run_pipeline.py --ticker NFLX --sector Tech \
        --description "..." --start 2024Q4 --end 2025Q3 --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from provenance import PipelineRun

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
YAML_PATH = PIPELINE_DIR / "supported_tickers.yaml"

STAGE_SCRIPTS = {
    1: PIPELINE_DIR / "EDGAR" / "get_sec_data.py",
    2: PIPELINE_DIR / "EDGAR" / "filing_summarization_pipeline.py",
    3: PIPELINE_DIR / "earnings_calls" / "get_earring_calls_summary.py",
    4: PIPELINE_DIR / "macro" / "macro_quarter_builder.py",
    5: PIPELINE_DIR / "quarterly_asset_details" / "asset_quarter_builder.py",
    6: PIPELINE_DIR / "sentiment" / "sentiment.py",
}

STAGE_NAMES = {
    1: "EDGAR download",
    2: "Filing summarization",
    3: "Earnings call summary",
    4: "Macro data",
    5: "Asset features",
    6: "Sentiment scoring",
}

# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------


def parse_quarter(s: str) -> tuple[int, int]:
    """Parse '2024Q4' -> (2024, 4)."""
    s = s.strip().upper()
    if len(s) != 6 or s[4] != "Q":
        raise ValueError(f"Invalid quarter format '{s}', expected YYYYQ# (e.g. 2024Q4)")
    year = int(s[:4])
    q = int(s[5])
    if q < 1 or q > 4:
        raise ValueError(f"Quarter must be 1-4, got {q}")
    return year, q


def quarter_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return list of (year, quarter) tuples from start to end inclusive."""
    sy, sq = parse_quarter(start)
    ey, eq = parse_quarter(end)
    quarters = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        quarters.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


def quarters_to_years_and_quarters(quarters: list[tuple[int, int]]) -> tuple[str, str]:
    """Convert quarter list to EDGAR-style --years and --quarters args.

    Returns (years_csv, quarters_csv) - the cross-product over-fetches
    slightly but EDGAR skips existing files.
    """
    years = sorted({y for y, _ in quarters})
    qs = sorted({q for _, q in quarters})
    return ",".join(str(y) for y in years), ",".join(f"Q{q}" for q in qs)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_yaml() -> dict:
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ticker_exists(data: dict, ticker: str) -> bool:
    return any(
        t["symbol"].upper() == ticker.upper()
        for t in data.get("supported_tickers", [])
    )


def add_ticker_to_yaml(ticker: str, sector: str, description: str, dry_run: bool):
    data = load_yaml()
    if ticker_exists(data, ticker):
        print(f"  {ticker} already in supported_tickers.yaml - skipping YAML update")
        return
    new_entry = {"symbol": ticker, "sector": sector, "description": description}
    data.setdefault("supported_tickers", []).append(new_entry)
    if dry_run:
        print("  [dry-run] Would add to supported_tickers.yaml:")
        print(f"    - symbol: {ticker}")
        print(f"      sector: {sector}")
        print(f"      description: {description}")
        return
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Added {ticker} to supported_tickers.yaml")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight(args) -> dict[str, str | None]:
    """Check env vars and scripts exist. Returns resolved API keys."""
    print("Pre-flight checks")
    print("-" * 40)

    ok = True

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    finnhub_key = os.environ.get("FINNHUB_KEY")
    fred_key = os.environ.get("FRED_API_KEY")

    if not anthropic_key:
        print("  FAIL  ANTHROPIC_API_KEY not set (required for Stage 2/3)")
        ok = False
    else:
        print("  OK    ANTHROPIC_API_KEY")

    if not hf_token:
        print("  FAIL  HF_TOKEN not set (required for Stage 3)")
        ok = False
    else:
        print("  OK    HF_TOKEN")

    if not finnhub_key:
        print("  FAIL  FINNHUB_KEY not set (required for Stage 6)")
        ok = False
    else:
        print("  OK    FINNHUB_KEY")

    if not fred_key:
        print("  WARN  FRED_API_KEY not set (Stage 4 will run with nulls)")
    else:
        print("  OK    FRED_API_KEY")

    for stage, script in STAGE_SCRIPTS.items():
        if not script.exists():
            print(f"  FAIL  Stage {stage} script missing: {script.relative_to(PIPELINE_DIR)}")
            ok = False

    if not ok and not args.dry_run:
        print("\nPre-flight failed. Fix the issues above and re-run.")
        sys.exit(1)

    print()
    return {
        "anthropic": anthropic_key,
        "hf": hf_token,
        "finnhub": finnhub_key,
        "fred": fred_key,
    }


# ---------------------------------------------------------------------------
# Macro skip logic
# ---------------------------------------------------------------------------


def all_macro_files_exist(quarters: list[tuple[int, int]]) -> bool:
    macro_dir = PIPELINE_DIR / "macro" / "data"
    for year, q in quarters:
        path = macro_dir / f"macro_{year}_Q{q}.json"
        if not path.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------


def build_commands(
    args,
    keys: dict[str, str | None],
    quarters: list[tuple[int, int]],
) -> list[tuple[int, str, list[str] | None]]:
    """Build the command list for all 6 data-gathering stages.

    Returns list of (stage_num, stage_name, argv).
    """
    years_csv, quarters_csv = quarters_to_years_and_quarters(quarters)
    new_ticker = args.ticker
    start, end = args.start, args.end

    stages: list[tuple[int, str, list[str] | None]] = []

    # Stage 1 - EDGAR download
    cmd = [sys.executable, str(STAGE_SCRIPTS[1])]
    if new_ticker:
        cmd += ["--tickers", new_ticker]
    else:
        cmd += ["--supported"]
    cmd += ["--years", years_csv, "--quarters", quarters_csv, "--parallel"]
    stages.append((1, STAGE_NAMES[1], cmd))

    # Stage 2 - Filing summarization
    cmd = [sys.executable, str(STAGE_SCRIPTS[2])]
    if new_ticker:
        cmd += ["--ticker", new_ticker]
    else:
        cmd += ["--supported"]
    cmd += ["--parallel"]
    if args.force:
        cmd += ["--force"]
    stages.append((2, STAGE_NAMES[2], cmd))

    # Stage 3 - Earnings calls (HF transcripts + Claude + FinBERT)
    cmd = [sys.executable, str(STAGE_SCRIPTS[3])]
    cmd += ["--start", start, "--end", end]
    if new_ticker:
        cmd += ["--tickers", new_ticker]
    else:
        cmd += ["--supported"]
    if keys["hf"]:
        cmd += ["--hf-token", keys["hf"]]
    if keys["anthropic"]:
        cmd += ["--api-key", keys["anthropic"]]
    if args.force:
        cmd += ["--force"]
    stages.append((3, STAGE_NAMES[3], cmd))

    # Stage 4 - Macro data (skip if all files exist, unless --force)
    if not args.force and all_macro_files_exist(quarters):
        stages.append((4, STAGE_NAMES[4], None))
    else:
        cmd = [sys.executable, str(STAGE_SCRIPTS[4])]
        cmd += ["--start", start, "--end", end]
        if keys["fred"]:
            cmd += ["--fred-key", keys["fred"]]
        stages.append((4, STAGE_NAMES[4], cmd))

    # Stage 5 - Asset features
    cmd = [sys.executable, str(STAGE_SCRIPTS[5])]
    cmd += ["--start", start, "--end", end]
    if new_ticker:
        cmd += ["--tickers", new_ticker]
    else:
        cmd += ["--supported"]
    stages.append((5, STAGE_NAMES[5], cmd))

    # Stage 6 - Sentiment
    cmd = [sys.executable, str(STAGE_SCRIPTS[6])]
    cmd += ["--start", start, "--end", end]
    if new_ticker:
        cmd += ["--tickers", new_ticker]
    else:
        cmd += ["--supported"]
    if keys["finnhub"]:
        cmd += ["--api-key", keys["finnhub"]]
    cmd += ["--workers", "4"]
    if args.force:
        cmd += ["--force"]
    stages.append((6, STAGE_NAMES[6], cmd))

    return stages


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def run_stages(stages, dry_run: bool, pipeline_run=None) -> list[tuple[int, str, str, float]]:
    """Execute stages sequentially. Returns results list."""
    results: list[tuple[int, str, str, float]] = []

    for stage_num, name, cmd in stages:
        if cmd is None:
            print(f"\nStage {stage_num}: {name}")
            print("  Skipped - all output files already exist")
            results.append((stage_num, name, "SKIP", 0.0))
            if pipeline_run:
                pipeline_run.record_stage(
                    f"{stage_num}_{name.lower().replace(' ', '_')}",
                    {"script": STAGE_NAMES.get(stage_num, name), "status": "skipped"},
                )
            continue

        print(f"\nStage {stage_num}: {name}")
        display_cmd = " ".join(cmd)
        print(f"  $ {display_cmd}")

        if dry_run:
            results.append((stage_num, name, "DRY-RUN", 0.0))
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else "FAIL"
        results.append((stage_num, name, status, elapsed))

        if pipeline_run:
            stage_key = f"{stage_num}_{name.lower().replace(' ', '_')}"
            script_rel = str(STAGE_SCRIPTS.get(stage_num, ""))
            pipeline_run.record_stage(
                stage_key,
                {
                    "script": script_rel,
                    "status": "completed" if status == "OK" else "failed",
                    "duration_seconds": round(elapsed, 1),
                },
            )

        if result.returncode != 0:
            print(f"\n  Stage {stage_num} failed (exit code {result.returncode})")
            print("  Pipeline stopped.")
            return results

    return results


def print_summary(results: list[tuple[int, str, str, float]]):
    print("\n" + "=" * 56)
    print("Pipeline Summary")
    print("=" * 56)
    print(f"  {'Stage':<8}{'Name':<26}{'Status':<10}{'Time'}")
    print(f"  {'-'*6:<8}{'-'*24:<26}{'-'*6:<10}{'-'*8}")
    for stage_num, name, status, elapsed in results:
        time_str = format_duration(elapsed) if elapsed > 0 else "-"
        print(f"  {stage_num:<8}{name:<26}{status:<10}{time_str}")
    print()

    failed = [r for r in results if r[2] == "FAIL"]
    if failed:
        print(f"  Pipeline FAILED at Stage {failed[0][0]}.")
    elif all(r[2] in ("OK", "SKIP") for r in results):
        total = sum(r[3] for r in results)
        print(f"  All stages complete. Total time: {format_duration(total)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run the data-gathering pipeline (6 stages).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker", help="New ticker to add (omit to run for all supported)")
    parser.add_argument("--sector", help="Sector for new ticker (required with --ticker)")
    parser.add_argument("--description", help="Description for new ticker (required with --ticker)")
    parser.add_argument("--start", required=True, help="Start quarter, e.g. 2024Q4")
    parser.add_argument("--end", required=True, help="End quarter, e.g. 2025Q3")
    parser.add_argument("--force", action="store_true", help="Re-run all stages, ignoring cached output")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    if args.ticker and (not args.sector or not args.description):
        parser.error("--sector and --description are required when --ticker is specified")

    if args.ticker:
        args.ticker = args.ticker.upper()

    quarters = quarter_range(args.start, args.end)
    if not quarters:
        parser.error(f"Empty quarter range: {args.start} to {args.end}")

    print(f"Pipeline: {args.start} -> {args.end} ({len(quarters)} quarters)")
    if args.ticker:
        print(f"New ticker: {args.ticker} ({args.sector})")
    else:
        print("Running for all supported tickers")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    keys = preflight(args)

    if args.ticker:
        print("Updating supported_tickers.yaml")
        print("-" * 40)
        add_ticker_to_yaml(args.ticker, args.sector, args.description, args.dry_run)
        print()

    stages = build_commands(args, keys, quarters)

    if args.ticker:
        resolved_tickers = [args.ticker]
    else:
        data = load_yaml()
        resolved_tickers = [t["symbol"] for t in data.get("supported_tickers", [])]

    pipeline_run = PipelineRun(
        args=args,
        tickers=resolved_tickers,
        quarters=quarters,
        pipeline_dir=PIPELINE_DIR,
    )

    if not args.dry_run:
        pipeline_run.start()
        print(f"Provenance: run_id={pipeline_run.run_id}")
        print(f"  Manifest: {pipeline_run.manifest_path.relative_to(PIPELINE_DIR)}")
        print()

    try:
        results = run_stages(stages, args.dry_run, pipeline_run)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        if not args.dry_run:
            pipeline_run.finalize("interrupted")
        sys.exit(1)

    if not args.dry_run:
        final_status = "failed" if any(r[2] == "FAIL" for r in results) else "completed"
        manifest_path = pipeline_run.finalize(final_status)
        print(f"\nProvenance manifest: {manifest_path.relative_to(PIPELINE_DIR)}")

    print_summary(results)

    if any(r[2] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
