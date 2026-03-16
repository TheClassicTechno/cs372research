#!/usr/bin/env python3
"""Print paired t-test results for ablations 7, 8, and 10.

Outputs two tables:
  Table 1 — Summary: mean ± SEM for each condition and metric
  Table 2 — Paired t-tests: diff, 95% CI, t, p, Cohen's d

Usage:
    python analysis/ablation_paired_ttests.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "logging" / "runs"

ABLATIONS = [
    ("Ablation 7",  "vskarich_ablation_7",  "2-ag."),
    ("Ablation 8",  "vskarich_ablation_8",  "3-ag."),
    ("Ablation 10", "vskarich_ablation_10", "2-ag."),
]

# (json key, display name, column width)
METRICS = [
    ("daily_metrics_total_return_pct",  "Return %",        14),
    ("daily_metrics_annualized_sharpe", "Sharpe",          14),
    ("daily_metrics_annualized_sortino","Sortino",         14),
    ("daily_metrics_max_drawdown_pct",  "Max DD %",        14),
    ("daily_metrics_excess_return_pct", "Excess vs SPY %", 16),
]


def load_runs(experiment: str) -> list[dict]:
    exp_dir = RUNS_DIR / experiment
    if not exp_dir.exists():
        return []
    runs = []
    for run_dir in sorted(exp_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        fin_path = run_dir / "_dashboard" / "financial_metrics.json"
        if not manifest_path.exists() or not fin_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        fin = json.loads(fin_path.read_text())
        config_paths = manifest.get("config_paths", [])
        debate_config = config_paths[0] if config_paths else ""
        is_baseline = "baseline" in Path(debate_config).stem.lower()
        scenario = Path(config_paths[1]).stem if len(config_paths) > 1 else "unknown"
        runs.append({
            "condition": "baseline" if is_baseline else "treatment",
            "scenario": scenario,
            "metrics": fin,
        })
    return runs


def pair_runs(runs: list[dict]) -> tuple[list[str], dict, dict]:
    bmap, tmap = {}, {}
    for r in runs:
        if r["condition"] == "baseline":
            bmap[r["scenario"]] = r["metrics"]
        else:
            tmap[r["scenario"]] = r["metrics"]
    shared = sorted(set(bmap) & set(tmap))
    return shared, bmap, tmap


def fmt_mean_sem(vals: np.ndarray) -> str:
    return f"{np.mean(vals):+.2f} \u00b1 {stats.sem(vals):.2f}"


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    if p < 0.10:  return ".  "
    return "   "


def main() -> None:
    # Gather all data first
    ablation_data = []
    for label, experiment, agent_note in ABLATIONS:
        runs = load_runs(experiment)
        if not runs:
            continue
        shared, bmap, tmap = pair_runs(runs)
        n_base = sum(1 for r in runs if r["condition"] == "baseline")
        n_treat = sum(1 for r in runs if r["condition"] == "treatment")
        ablation_data.append({
            "label": label, "agent_note": agent_note,
            "shared": shared, "bmap": bmap, "tmap": tmap,
            "n_base": n_base, "n_treat": n_treat,
        })

    # ── Table 1: Summary (mean ± SEM) ─────────────────────────────────────
    print()
    print("TABLE 1: DESCRIPTIVE STATISTICS  (mean \u00b1 SEM)")
    print("=" * 110)

    # Header
    config_w = 34
    n_w = 4
    hdr = f"  {'Configuration':<{config_w}} {'N':>{n_w}}"
    for _, name, w in METRICS:
        hdr += f"  {name:>{w}}"
    print(hdr)
    print("  " + "-" * 106)

    for ad in ablation_data:
        shared = ad["shared"]
        if len(shared) < 3:
            continue

        for condition, cond_label in [("baseline", "Base"), ("treatment", "Intervention")]:
            m = ad["bmap"] if condition == "baseline" else ad["tmap"]
            n = ad["n_base"] if condition == "baseline" else ad["n_treat"]
            config_str = f"{ad['label']} {cond_label} ({ad['agent_note']})"

            row = f"  {config_str:<{config_w}} {n:>{n_w}}"
            for key, _, w in METRICS:
                vals = np.array([m[s][key] for s in shared], dtype=float)
                cell = fmt_mean_sem(vals)
                row += f"  {cell:>{w}}"
            print(row)

        # Blank line between ablations
        print()

    # ── Table 2: Paired t-tests ────────────────────────────────────────────
    print()
    print("TABLE 2: PAIRED T-TESTS  (treatment \u2212 baseline)")
    print("Sig codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    print("=" * 120)

    hdr2 = f"  {'Ablation':<14} {'Metric':<17} {'Diff':>8} {'95% CI':>21} {'t':>7} {'p':>8} {'Sig':>3} {'d':>6}"
    print(hdr2)
    print("  " + "-" * 116)

    for ad in ablation_data:
        shared = ad["shared"]
        if len(shared) < 3:
            continue
        for key, name, _ in METRICS:
            b = np.array([ad["bmap"][s][key] for s in shared], dtype=float)
            t = np.array([ad["tmap"][s][key] for s in shared], dtype=float)
            mask = ~(np.isnan(b) | np.isnan(t))
            b, t = b[mask], t[mask]
            if len(b) < 3:
                continue

            diffs = t - b
            n = len(diffs)
            mean_d = np.mean(diffs)
            std_d = np.std(diffs, ddof=1)
            se = std_d / np.sqrt(n)
            t_stat, p_val = stats.ttest_rel(t, b)
            ci_half = stats.t.ppf(0.975, df=n - 1) * se
            d_val = mean_d / std_d if std_d > 0 else 0.0
            ci_str = f"[{mean_d - ci_half:+.3f}, {mean_d + ci_half:+.3f}]"
            stars = sig_stars(p_val)

            print(
                f"  {ad['label']:<14} {name:<17} {mean_d:+8.3f} {ci_str:>21} "
                f"{t_stat:+7.3f} {p_val:8.4f} {stars} {d_val:+6.3f}"
            )
        print()


if __name__ == "__main__":
    main()
