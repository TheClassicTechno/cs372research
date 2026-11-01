#!/usr/bin/env python3
"""
Paper Table Verification & Scenario-Level Uncertainty
=====================================================
Audits the two financial results tables in the manuscript against the saved
run artifacts under ``results/scenario_runs/``, and adds the uncertainty
quantification the manuscript currently lacks.

Why this exists
---------------
CLAUDE.md 42 requires every numerical claim to trace:

    paper claim -> table -> saved result -> evaluation script
                -> experiment configuration -> raw output

``scripts/compare_multiple_configs.py`` already regenerates the point estimates
and paired-t significance markers for those tables, so this script *reuses* its
loader rather than reimplementing it. What it adds:

  1. Bootstrap 95% CIs resampled at the **scenario** level. The manuscript
     reports SEM only. The independent unit here is the market scenario
     (checklist 14/17) -- not the agent turn, not the trading day -- because
     the 35 scenarios are the things that were independently sampled.
  2. Paired bootstrap CIs on the *difference* against a reference column,
     which is the quantity the manuscript's claims are actually about
     ("debate reduces drawdown relative to single agent").
  3. Config provenance: the SHA-256 prefix recorded in each tracking CSV is
     checked against every surviving YAML under config/. A run whose config
     no longer exists is reported as such rather than silently accepted.

Deterministic: fixed seed, no network, no API calls, no model inference.
Reads only files already in the repository.

Usage
-----
    python3 analysis/verify_paper_tables.py

Output
------
    analysis/verify_paper_tables_report.txt   human-readable audit
    analysis/verify_paper_tables.csv          machine-readable grid
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the existing loader so the audit reads results exactly the way the
# table-generating script does. Duplicating it would let the two drift apart,
# which would defeat the purpose of an audit.
from compare_multiple_configs import load_scenario_metrics  # noqa: E402

OUTPUT_REPORT = REPO_ROOT / "analysis" / "verify_paper_tables_report.txt"
OUTPUT_CSV = REPO_ROOT / "analysis" / "verify_paper_tables.csv"

N_BOOTSTRAP = 10_000
SEED = 20260829
CI_LEVEL = 95.0

# Rows of the manuscript tables -> keys inside summary.json's episode summary.
METRICS = [
    ("daily_metrics_excess_return_pct", "Excess Return % (v SPY)", 2),
    ("daily_metrics_annualized_sharpe", "Sharpe", 2),
    ("daily_metrics_annualized_volatility", "Volatility", 2),
    ("daily_metrics_max_drawdown_pct", "Max Drawdown %", 2),
    ("daily_metrics_annualized_sortino", "Sortino", 2),
    ("daily_metrics_total_return_pct", "Total Return %", 2),
]

# Each entry mirrors one table in neuripsworkshoppaper.tex. The reference
# column is the baseline the manuscript's comparative claims are made against.
TABLES = [
    {
        "label": "tab:prompts",
        "caption": "Prompt tier baselines (main text)",
        "reference": "A",
        "columns": [
            ("A", "Single Agent, Basic", "results_tracking_single_agent_slim.csv"),
            ("B", "3 Roles Mean, Basic", "results_tracking_vote_slim_no_macro.csv"),
            ("C", "3 Roles Mean, Causal",
             "results_tracking_vote_causal_no_macro_causal_out_fix_scenario.csv"),
            ("D", "3 Roles Mean, Enriched", "results_tracking_vote_no_macro_enriched.csv"),
        ],
    },
    {
        "label": "tab:debate_rca",
        "caption": "Causal debate baselines & RCA (appendix)",
        "reference": "A",
        "columns": [
            ("A", "3 Roles Mean, Causal",
             "results_tracking_vote_causal_no_macro_causal_out_fix_scenario.csv"),
            ("B", "3 Roles Debate, Causal",
             "results_tracking_debate_1_round_no_macro_causal_out.csv"),
            ("C", "3 Roles Debate, Causal + RCA",
             "results_tracking_debate_1_round_no_macro_causal_out_rca_revise.csv"),
        ],
    },
]


def config_hash(path: Path) -> str:
    """Match scripts/run_scenario_list.py:get_file_hash -- sha256, 16 hex chars."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def surviving_config_hashes() -> dict[str, str]:
    """Hash every YAML under config/ so recorded run hashes can be resolved."""
    table: dict[str, str] = {}
    config_dir = REPO_ROOT / "config"
    if not config_dir.is_dir():
        return table
    for path in sorted(config_dir.rglob("*.yaml")):
        table.setdefault(config_hash(path), str(path.relative_to(REPO_ROOT)))
    return table


def recorded_config_hash(csv_path: Path) -> str | None:
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            value = row.get("config_hash")
            if value:
                return value
    return None


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling scenarios with replacement."""
    if len(values) < 2:
        raise ValueError(f"need >=2 scenarios to bootstrap, got {len(values)}")
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    lo = (100.0 - CI_LEVEL) / 2.0
    return float(np.percentile(means, lo)), float(np.percentile(means, 100.0 - lo))


def audit_table(spec: dict, hashes: dict[str, str], rng: np.random.Generator,
                lines: list[str], rows: list[dict]) -> None:
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"{spec['label']}  --  {spec['caption']}")
    lines.append("=" * 78)

    loaded: dict[str, dict] = {}
    for letter, paper_label, csv_name in spec["columns"]:
        csv_path = REPO_ROOT / csv_name
        if not csv_path.exists():
            # Fail loudly. A silently-skipped column would make the audit lie.
            raise FileNotFoundError(f"tracking CSV missing: {csv_path}")
        metrics, _order = load_scenario_metrics(csv_path)
        if not metrics:
            raise RuntimeError(f"no scenario metrics loaded from {csv_path}")
        loaded[letter] = {
            "paper_label": paper_label,
            "csv": csv_name,
            "metrics": metrics,
        }

    # Paired statistics require the same scenarios in every column.
    common = set.intersection(*(set(v["metrics"]) for v in loaded.values()))
    common_sorted = sorted(common)
    lines.append(f"Scenarios common to all columns: {len(common_sorted)}")

    lines.append("")
    lines.append("Config provenance (recorded run hash -> surviving YAML):")
    for letter, _paper_label, csv_name in spec["columns"]:
        rec = recorded_config_hash(REPO_ROOT / csv_name)
        resolved = hashes.get(rec) if rec else None
        status = resolved if resolved else "*** NO SURVIVING CONFIG MATCHES THIS HASH ***"
        lines.append(f"  [{letter}] {rec}  {status}")

    for key, row_label, precision in METRICS:
        lines.append("")
        lines.append(f"--- {row_label} ---")
        ref = spec["reference"]
        ref_vals = np.array([loaded[ref]["metrics"][s][key] for s in common_sorted])

        for letter, _paper_label, csv_name in spec["columns"]:
            vals = np.array([loaded[letter]["metrics"][s][key] for s in common_sorted])
            mean = float(vals.mean())
            sem = float(stats.sem(vals))
            lo, hi = bootstrap_ci(vals, rng)

            record = {
                "table": spec["label"],
                "column": letter,
                "config": loaded[letter]["paper_label"],
                "metric": row_label,
                "n_scenarios": len(common_sorted),
                "mean": round(mean, 4),
                "sem": round(sem, 4),
                "boot_ci_low": round(lo, 4),
                "boot_ci_high": round(hi, 4),
            }

            note = ""
            if letter != ref:
                diff = vals - ref_vals  # paired: same scenario, both columns
                d_mean = float(diff.mean())
                d_lo, d_hi = bootstrap_ci(diff, rng)
                t_stat, p_val = stats.ttest_rel(vals, ref_vals)
                # Cohen's d for paired samples.
                d_eff = d_mean / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
                record.update({
                    "paired_vs": ref,
                    "diff_mean": round(d_mean, 4),
                    "diff_ci_low": round(d_lo, 4),
                    "diff_ci_high": round(d_hi, 4),
                    "paired_t": round(float(t_stat), 4),
                    "paired_p": round(float(p_val), 4),
                    "cohens_d": round(float(d_eff), 4),
                })
                crosses_zero = d_lo <= 0.0 <= d_hi
                note = (f"   | vs {ref}: {d_mean:+.{precision}f} "
                        f"[{d_lo:+.{precision}f}, {d_hi:+.{precision}f}] "
                        f"p={p_val:.3f} d={d_eff:+.2f}"
                        f"{'  (CI crosses 0)' if crosses_zero else '  (CI excludes 0)'}")

            rows.append(record)
            lines.append(
                f"  [{letter}] {mean:>8.{precision}f}  SEM {sem:.{precision}f}  "
                f"boot{CI_LEVEL:.0f}% [{lo:.{precision}f}, {hi:.{precision}f}]{note}"
            )


def main() -> int:
    rng = np.random.default_rng(SEED)
    hashes = surviving_config_hashes()

    lines: list[str] = [
        "PAPER TABLE VERIFICATION -- scenario-level uncertainty & provenance",
        "=" * 78,
        f"Bootstrap resamples : {N_BOOTSTRAP:,}",
        f"Seed                : {SEED}",
        f"CI level            : {CI_LEVEL:.0f}% (percentile bootstrap)",
        "Independent unit    : market scenario (checklist 14/17)",
        f"Surviving config YAMLs hashed: {len(hashes)}",
        "",
        "Point estimates and paired-t markers are produced by",
        "scripts/compare_multiple_configs.py; this script reuses that loader and",
        "adds bootstrap CIs plus config-provenance resolution.",
    ]
    rows: list[dict] = []

    for spec in TABLES:
        audit_table(spec, hashes, rng, lines, rows)

    lines.append("")
    lines.append("=" * 78)
    lines.append("NOT VERIFIABLE FROM THIS REPOSITORY")
    lines.append("=" * 78)
    lines.append(
        "The CRIT reasoning scores underlying tab:correlation_reasoning (n=210),\n"
        "the CRIT pillar tables, and the rho=0.72->0.84 prompt-tier gain are NOT\n"
        "reproducible here: no per-round CRIT log corpus is present. The only\n"
        "crit_scores.json files in the tree are unit-test fixtures under tests/.\n"
        "Those numbers are author-verified only and are reported as such."
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n")

    fieldnames = [
        "table", "column", "config", "metric", "n_scenarios",
        "mean", "sem", "boot_ci_low", "boot_ci_high",
        "paired_vs", "diff_mean", "diff_ci_low", "diff_ci_high",
        "paired_t", "paired_p", "cohens_d",
    ]
    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print("\n".join(lines))
    print(f"\nWrote {OUTPUT_REPORT.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
