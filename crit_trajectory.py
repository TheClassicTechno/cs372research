#!/usr/bin/env python3
"""
Per-round CRIT trajectory analysis across classification groups.

Reads run metadata from /tmp/crit_analysis.csv, then loads per-round
crit_scores.json from the logging directory tree to build round-level
trajectories for each classification (baseline, enriched, enriched_intense).
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

CSV_PATH = "/tmp/crit_analysis.csv"
LOGGING_ROOT = Path(
    "/Users/vskarich/Desktop/important_docs/repos/cs372research/logging/runs"
)

# Canonical pillar names used in output
PILLAR_DISPLAY = {
    "logical_validity": "Logical Validity",
    "evidential_support": "Evidential Support",
    "alternative_consideration": "Alternative Consid.",
    "causal_alignment": "Causal Alignment",
}
PILLAR_CANONICAL = list(PILLAR_DISPLAY.keys())

# Mapping from raw JSON keys to canonical names.
# Older format: IC, ES, TA, CI.  Newer format: LV, ES, AC, CA.
RAW_TO_CANONICAL = {
    "IC": "logical_validity",
    "LV": "logical_validity",
    "ES": "evidential_support",
    "TA": "alternative_consideration",
    "AC": "alternative_consideration",
    "CI": "causal_alignment",
    "CA": "causal_alignment",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_round_scores(experiment: str, run_id: str) -> list[dict]:
    """
    Load all per-round crit_scores.json for a given run.
    Returns a list of dicts with keys: round, rho_bar, IC, ES, TA, CI.
    """
    run_dir = LOGGING_ROOT / experiment / run_id / "rounds"
    if not run_dir.is_dir():
        return []

    results = []
    for round_dir in sorted(run_dir.iterdir()):
        score_file = round_dir / "metrics" / "crit_scores.json"
        if not score_file.is_file():
            continue
        try:
            data = json.loads(score_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        round_num = data.get("round")
        rho_bar = data.get("rho_bar")
        if round_num is None or rho_bar is None:
            continue

        # Average pillar scores across all agents, normalising key variants
        pillar_totals = defaultdict(list)
        agents = data.get("agent_scores", {})
        for agent_name, agent_data in agents.items():
            pillars = agent_data.get("pillar_scores", {})
            for raw_key, val in pillars.items():
                canonical = RAW_TO_CANONICAL.get(raw_key)
                if canonical is not None and val is not None:
                    pillar_totals[canonical].append(val)

        pillar_means = {}
        for pk in PILLAR_CANONICAL:
            vals = pillar_totals[pk]
            pillar_means[pk] = sum(vals) / len(vals) if vals else None

        results.append({
            "round": int(round_num),
            "rho_bar": float(rho_bar),
            **pillar_means,
        })

    return results


def fmt(val, width=8):
    """Format a float or None for table display."""
    if val is None:
        return "-".center(width)
    return f"{val:.4f}".rjust(width)


def fmt_delta(val, width=8):
    """Format a signed delta value."""
    if val is None:
        return "-".center(width)
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}".rjust(width)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # 1. Read CSV to get experiment / run_id / classification mapping
    runs = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append({
                "experiment": row["experiment"],
                "run_id": row["run_id"],
                "classification": row["classification"],
            })

    print(f"Loaded {len(runs)} runs from CSV\n")

    # 2. Collect per-round data grouped by classification
    # Structure: class_round_data[classification][round_num] -> list of dicts
    class_round_data = defaultdict(lambda: defaultdict(list))
    runs_loaded = 0
    rounds_loaded = 0

    for run in runs:
        round_scores = load_round_scores(run["experiment"], run["run_id"])
        if not round_scores:
            continue
        runs_loaded += 1
        for rs in round_scores:
            class_round_data[run["classification"]][rs["round"]].append(rs)
            rounds_loaded += 1

    print(f"Loaded per-round data: {runs_loaded} runs, {rounds_loaded} round-level observations\n")

    # Classification display order
    CLASS_ORDER = ["baseline", "enriched", "enriched_intense"]
    CLASS_LABELS = {
        "baseline": "Baseline",
        "enriched": "Enriched",
        "enriched_intense": "Enriched Intense",
    }

    # Determine the max round across all classifications
    all_rounds = set()
    for cls in CLASS_ORDER:
        if cls in class_round_data:
            all_rounds.update(class_round_data[cls].keys())
    if not all_rounds:
        print("ERROR: No per-round data found. Check that logging/runs directories exist.")
        sys.exit(1)
    max_round = max(all_rounds)

    # ── 3. Compute aggregated stats ──────────────────────────────────────

    # agg[classification][round] = {rho_bar: mean, IC: mean, ..., n: count}
    agg = {}
    for cls in CLASS_ORDER:
        agg[cls] = {}
        for rd in range(1, max_round + 1):
            observations = class_round_data[cls].get(rd, [])
            n = len(observations)
            if n == 0:
                agg[cls][rd] = None  # No data for this round
                continue
            rho_mean = sum(o["rho_bar"] for o in observations) / n
            pillar_aggs = {}
            for pk in PILLAR_CANONICAL:
                vals = [o[pk] for o in observations if o[pk] is not None]
                pillar_aggs[pk] = sum(vals) / len(vals) if vals else None
            agg[cls][rd] = {"rho_bar": rho_mean, **pillar_aggs, "n": n}

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 1: rho_bar trajectory by classification and round
    # ══════════════════════════════════════════════════════════════════════

    sep = "=" * 90
    print(sep)
    print("  TABLE 1: Mean rho_bar Trajectory by Round")
    print(sep)

    # Header
    hdr = f"{'Round':>6}"
    for cls in CLASS_ORDER:
        label = CLASS_LABELS[cls]
        hdr += f"  {label:>18} {'(n)':>5}"
    print(hdr)
    print("-" * 90)

    for rd in range(1, max_round + 1):
        row = f"{rd:>6}"
        for cls in CLASS_ORDER:
            entry = agg[cls].get(rd)
            if entry is None:
                row += f"  {'--':>18} {'--':>5}"
            else:
                row += f"  {entry['rho_bar']:>18.4f} {entry['n']:>5}"
        print(row)

    print()

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 2: Round-over-round delta for rho_bar
    # ══════════════════════════════════════════════════════════════════════

    print(sep)
    print("  TABLE 2: Round-over-Round Delta (rho_bar)")
    print(sep)

    hdr = f"{'Round':>6}"
    for cls in CLASS_ORDER:
        label = CLASS_LABELS[cls]
        hdr += f"  {label:>18}"
    print(hdr)
    print("-" * 90)

    for rd in range(1, max_round + 1):
        row = f"{rd:>6}"
        for cls in CLASS_ORDER:
            curr = agg[cls].get(rd)
            prev = agg[cls].get(rd - 1) if rd > 1 else None
            if curr is None:
                row += f"  {'--':>18}"
            elif prev is None:
                # First round or no previous data: show absolute value as "start"
                row += f"  {'(start)':>18}" if rd == 1 else f"  {'(no prev)':>18}"
            else:
                delta = curr["rho_bar"] - prev["rho_bar"]
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta:>17.4f}"
        print(row)

    print()

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 3: Per-pillar trajectories
    # ══════════════════════════════════════════════════════════════════════

    for pk, pk_label in PILLAR_DISPLAY.items():
        print(sep)
        print(f"  TABLE 3-{pk}: {pk_label} Trajectory by Round")
        print(sep)

        hdr = f"{'Round':>6}"
        for cls in CLASS_ORDER:
            label = CLASS_LABELS[cls]
            hdr += f"  {label:>18} {'(n)':>5}"
        print(hdr)
        print("-" * 90)

        for rd in range(1, max_round + 1):
            row = f"{rd:>6}"
            for cls in CLASS_ORDER:
                entry = agg[cls].get(rd)
                if entry is None or entry.get(pk) is None:
                    row += f"  {'--':>18} {'--':>5}"
                else:
                    row += f"  {entry[pk]:>18.4f} {entry['n']:>5}"
            print(row)

        print()

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 4: Starting Point & Convergence Analysis
    # ══════════════════════════════════════════════════════════════════════

    print(sep)
    print("  TABLE 4: Starting Point & Convergence Analysis")
    print(sep)
    print()

    for cls in CLASS_ORDER:
        label = CLASS_LABELS[cls]

        # Find first and last round with data
        valid_rounds = [rd for rd in range(1, max_round + 1) if agg[cls].get(rd) is not None]
        if not valid_rounds:
            print(f"  {label}: NO DATA")
            print()
            continue

        first_rd = min(valid_rounds)
        last_rd = max(valid_rounds)
        first = agg[cls][first_rd]
        last = agg[cls][last_rd]

        # Count how many runs have data at various rounds to show sample attrition
        round_counts = [(rd, agg[cls][rd]["n"]) for rd in valid_rounds]

        print(f"  {label}")
        print(f"  {'─' * 60}")
        print(f"    Rounds with data: {', '.join(str(r) for r in valid_rounds)}")
        print(f"    Sample sizes by round: {', '.join(f'R{r}={n}' for r, n in round_counts)}")
        print()
        print(f"    {'Metric':<28} {'Round 1':>10} {'Last (R{})'.format(last_rd):>10} {'Delta':>10}")
        print(f"    {'─' * 58}")

        # rho_bar
        r1_rho = first["rho_bar"]
        rl_rho = last["rho_bar"]
        d_rho = rl_rho - r1_rho
        print(f"    {'rho_bar':<28} {r1_rho:>10.4f} {rl_rho:>10.4f} {fmt_delta(d_rho, 10)}")

        # Each pillar
        for pk, pk_label in PILLAR_DISPLAY.items():
            r1_val = first.get(pk)
            rl_val = last.get(pk)
            if r1_val is not None and rl_val is not None:
                delta = rl_val - r1_val
                print(f"    {pk_label:<28} {r1_val:>10.4f} {rl_val:>10.4f} {fmt_delta(delta, 10)}")
            else:
                print(f"    {pk_label:<28} {'--':>10} {'--':>10} {'--':>10}")

        print()

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 5: Head-to-head -- Starting Point vs Improvement
    # ══════════════════════════════════════════════════════════════════════

    print(sep)
    print("  TABLE 5: Starting Point (R1) vs Trajectory Gain (R1 -> Last)")
    print(sep)
    print()
    print(f"  {'Classification':<20} {'R1 rho_bar':>12} {'Last rho_bar':>14} {'Total Delta':>13} {'Rounds':>8}")
    print(f"  {'─' * 67}")

    for cls in CLASS_ORDER:
        label = CLASS_LABELS[cls]
        valid_rounds = [rd for rd in range(1, max_round + 1) if agg[cls].get(rd) is not None]
        if not valid_rounds:
            print(f"  {label:<20} {'--':>12} {'--':>14} {'--':>13} {'--':>8}")
            continue
        first_rd = min(valid_rounds)
        last_rd = max(valid_rounds)
        r1 = agg[cls][first_rd]["rho_bar"]
        rl = agg[cls][last_rd]["rho_bar"]
        delta = rl - r1
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {r1:>12.4f} {rl:>14.4f} {sign}{delta:>12.4f} {f'{first_rd}-{last_rd}':>8}")

    print()

    # ══════════════════════════════════════════════════════════════════════
    # INTERPRETATION
    # ══════════════════════════════════════════════════════════════════════

    print(sep)
    print("  INTERPRETATION NOTES")
    print(sep)
    print()

    # Check: do enriched start higher?
    r1_vals = {}
    for cls in CLASS_ORDER:
        valid_rounds = [rd for rd in range(1, max_round + 1) if agg[cls].get(rd) is not None]
        if valid_rounds:
            r1_vals[cls] = agg[cls][min(valid_rounds)]["rho_bar"]

    if "baseline" in r1_vals and "enriched" in r1_vals:
        diff = r1_vals["enriched"] - r1_vals["baseline"]
        pct = (diff / r1_vals["baseline"]) * 100 if r1_vals["baseline"] != 0 else 0
        direction = "higher" if diff > 0 else "lower"
        print(f"  * Enriched starts {direction} than baseline at Round 1:")
        print(f"    Enriched R1 = {r1_vals['enriched']:.4f} vs Baseline R1 = {r1_vals['baseline']:.4f}")
        print(f"    Difference = {diff:+.4f} ({pct:+.1f}%)")
        print()

    if "baseline" in r1_vals and "enriched_intense" in r1_vals:
        diff = r1_vals["enriched_intense"] - r1_vals["baseline"]
        pct = (diff / r1_vals["baseline"]) * 100 if r1_vals["baseline"] != 0 else 0
        direction = "higher" if diff > 0 else "lower"
        print(f"  * Enriched Intense starts {direction} than baseline at Round 1:")
        print(f"    Enriched Intense R1 = {r1_vals['enriched_intense']:.4f} vs Baseline R1 = {r1_vals['baseline']:.4f}")
        print(f"    Difference = {diff:+.4f} ({pct:+.1f}%)")
        print()

    # Check: does enriched converge faster? (fewer rounds to reach peak)
    for cls in CLASS_ORDER:
        label = CLASS_LABELS[cls]
        valid_rounds = [rd for rd in range(1, max_round + 1) if agg[cls].get(rd) is not None]
        if len(valid_rounds) < 2:
            continue
        rho_series = [(rd, agg[cls][rd]["rho_bar"]) for rd in valid_rounds]
        peak_rd, peak_val = max(rho_series, key=lambda x: x[1])
        print(f"  * {label} peak rho_bar = {peak_val:.4f} at Round {peak_rd}")

    print()
    print(sep)
    print("  Analysis complete.")
    print(sep)


if __name__ == "__main__":
    main()
