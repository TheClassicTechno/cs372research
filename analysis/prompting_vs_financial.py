#!/usr/bin/env python3
"""Correlate prompt quality tier (standard → enriched → enriched_intense)
with CRIT reasoning scores AND financial performance.

Loads ablation 1 runs directly from the logging directory, classifies
each run by enrichment tier extracted from agent_profiles in manifest.json.

Usage:
    python analysis/prompting_vs_financial.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "logging" / "runs"
OUTPUT = REPO_ROOT / "analysis" / "prompting_vs_financial_report.txt"

TIERS = ["standard", "light", "intense_light", "intense"]
TIER_LABELS = {
    "standard": "Standard",
    "light": "Light",
    "intense_light": "Intense Light",
    "intense": "Intense",
}


def classify_enrichment(agent_profiles: dict) -> str | None:
    """Extract enrichment tier from agent profile names.

    Profile names like 'macro_enriched_intense' → 'intense',
    'risk_enriched_intense_light' → 'intense_light',
    'macro_enriched_light' → 'light',
    'macro_standard' or 'macro_enriched' → 'standard'.
    """
    for name in agent_profiles.values():
        name = str(name).lower()
        if "intense_light" in name:
            return "intense_light"
        if "intense" in name:
            return "intense"
        if "light" in name:
            return "light"
    # If enriched but no intensity qualifier, or standard
    return "standard"


def main() -> None:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    w("=" * 95)
    w("  PROMPT QUALITY vs REASONING vs FINANCIAL PERFORMANCE")
    w("  Data source: Ablation 1 (245 runs, 4 enrichment tiers)")
    w("=" * 95)
    w()

    # Load all ablation 1 runs
    exp_dir = RUNS_DIR / "vskarich_ablation_1"
    records = []
    missing = 0

    for run_dir in sorted(exp_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        fin_path = run_dir / "_dashboard" / "financial_metrics.json"
        crit_path = run_dir / "final" / "pid_crit_all_rounds.json"

        if not manifest_path.exists() or not fin_path.exists():
            missing += 1
            continue

        manifest = json.loads(manifest_path.read_text())
        fin = json.loads(fin_path.read_text())

        profiles = manifest.get("agent_profiles", {})
        tier = classify_enrichment(profiles)

        # Load CRIT scores (list of rounds; rho_bar is nested under "crit")
        rho_bar = None
        if crit_path.exists():
            crit = json.loads(crit_path.read_text())
            if isinstance(crit, list) and crit:
                last_round = crit[-1]
                rho_bar = last_round.get("crit", {}).get("rho_bar")
            elif isinstance(crit, dict):
                rho_bar = crit.get("rho_bar")

        records.append({
            "tier": tier,
            "run_id": run_dir.name,
            "rho_bar": rho_bar,
            "sharpe": fin.get("daily_metrics_annualized_sharpe"),
            "total_return": fin.get("daily_metrics_total_return_pct"),
            "sortino": fin.get("daily_metrics_annualized_sortino"),
            "max_dd": fin.get("daily_metrics_max_drawdown_pct"),
            "excess": fin.get("daily_metrics_excess_return_pct"),
        })

    w(f"  Records with financial data: {len(records)}")
    w(f"  Missing: {missing}")
    w()

    # Count by tier
    tier_counts = defaultdict(int)
    for r in records:
        tier_counts[r["tier"]] += 1
    for t in TIERS:
        w(f"    {TIER_LABELS.get(t, t):<22}: {tier_counts[t]}")
    w()

    if len(records) < 10:
        w("  Insufficient data for analysis.")
        OUTPUT.write_text("\n".join(lines))
        return

    # ── Table 1: Descriptive stats by tier ─────────────────────────────────
    w("=" * 95)
    w("  TABLE 1: DESCRIPTIVE STATISTICS BY PROMPT QUALITY TIER  (mean ± SEM)")
    w("=" * 95)
    w()

    metrics = [
        ("rho_bar",      "rho_bar"),
        ("sharpe",       "Sharpe"),
        ("total_return", "Return %"),
        ("sortino",      "Sortino"),
        ("max_dd",       "Max DD %"),
        ("excess",       "Excess vs SPY %"),
    ]

    hdr = f"  {'Tier':<22} {'N':>4}"
    for _, label in metrics:
        hdr += f"  {label:>16}"
    w(hdr)
    w("  " + "-" * 91)

    tier_arrays = {}
    for t in TIERS:
        tier_recs = [r for r in records if r["tier"] == t]
        n = len(tier_recs)
        if n == 0:
            continue
        tier_arrays[t] = {}
        row_str = f"  {TIER_LABELS[t]:<22} {n:>4}"
        for key, _ in metrics:
            vals = [r[key] for r in tier_recs if r[key] is not None]
            arr = np.array(vals, dtype=float)
            tier_arrays[t][key] = arr
            if len(arr) > 0:
                row_str += f"  {np.mean(arr):>+8.3f} ± {stats.sem(arr):.3f}"
            else:
                row_str += f"  {'--':>16}"
        w(row_str)

    w()

    # ── Table 2: Does better prompting → better CRIT? ─────────────────────
    w("=" * 95)
    w("  TABLE 2: PAIRWISE COMPARISONS — PROMPT TIER EFFECTS")
    w("  Independent samples t-test (Welch's)")
    w("=" * 95)
    w()

    comparisons = [
        ("standard", "light"),
        ("standard", "intense_light"),
        ("standard", "intense"),
        ("light", "intense"),
        ("intense_light", "intense"),
    ]

    w(f"  {'Comparison':<35} {'Metric':<17} {'Diff':>8} {'t':>7} {'p':>8} {'Sig':>3}")
    w("  " + "-" * 82)

    for t1, t2 in comparisons:
        if t1 not in tier_arrays or t2 not in tier_arrays:
            continue
        label = f"{TIER_LABELS[t1]} vs {TIER_LABELS[t2]}"
        for key, mname in metrics:
            a1 = tier_arrays[t1].get(key, np.array([]))
            a2 = tier_arrays[t2].get(key, np.array([]))
            if len(a1) < 3 or len(a2) < 3:
                continue
            t_stat, p_val = stats.ttest_ind(a1, a2, equal_var=False)
            diff = np.mean(a2) - np.mean(a1)
            sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else ".  " if p_val < 0.10 else "   "
            w(f"  {label:<35} {mname:<17} {diff:+8.3f} {t_stat:+7.3f} {p_val:8.4f} {sig}")
        w()

    # ── Table 3: Correlation rho_bar vs financial within each tier ─────────
    w("=" * 95)
    w("  TABLE 3: WITHIN-TIER CORRELATIONS (rho_bar vs financial)")
    w("=" * 95)
    w()

    w(f"  {'Tier':<22} {'Metric':<17} {'r':>7} {'p':>8} {'n':>5}")
    w("  " + "-" * 62)

    for t in TIERS:
        if t not in tier_arrays:
            continue
        rho = tier_arrays[t].get("rho_bar", np.array([]))
        for key, mname in [("sharpe", "Sharpe"), ("total_return", "Return %")]:
            fin_arr = tier_arrays[t].get(key, np.array([]))
            if len(rho) < 5 or len(fin_arr) < 5:
                continue
            # Align lengths (both from same tier records)
            min_n = min(len(rho), len(fin_arr))
            r_val, p_val = stats.pearsonr(rho[:min_n], fin_arr[:min_n])
            w(f"  {TIER_LABELS[t]:<22} {mname:<17} {r_val:+7.3f} {p_val:8.4f} {min_n:>5}")
    w()

    # ── Table 4: Pooled correlation ────────────────────────────────────────
    w("=" * 95)
    w("  TABLE 4: POOLED CORRELATION (all tiers combined)")
    w("=" * 95)
    w()

    all_rho = np.array([r["rho_bar"] for r in records if r["rho_bar"] is not None], dtype=float)
    for key, mname in [("sharpe", "Sharpe"), ("total_return", "Return %"), ("sortino", "Sortino")]:
        all_fin = np.array([r[key] for r in records if r[key] is not None and r["rho_bar"] is not None], dtype=float)
        min_n = min(len(all_rho), len(all_fin))
        if min_n < 5:
            continue
        r_p, p_p = stats.pearsonr(all_rho[:min_n], all_fin[:min_n])
        r_s, p_s = stats.spearmanr(all_rho[:min_n], all_fin[:min_n])
        w(f"  rho_bar vs {mname:<17}  Pearson r={r_p:+.3f} (p={p_p:.4f})  Spearman r={r_s:+.3f} (p={p_s:.4f})  n={min_n}")
    w()

    # ── Table 5: Does the EW benchmark gap change by tier? ────────────────
    w("=" * 95)
    w("  TABLE 5: SUMMARY — DOES BETTER PROMPTING IMPROVE FINANCIAL OUTCOMES?")
    w("=" * 95)
    w()

    # Ordinal correlation: tier rank (0,1,2) vs financial metrics
    tier_rank = {"standard": 0, "light": 1, "intense_light": 2, "intense": 3}
    ranks = np.array([tier_rank[r["tier"]] for r in records], dtype=float)

    for key, mname in metrics:
        vals = np.array([r[key] if r[key] is not None else np.nan for r in records], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        r_s, p_s = stats.spearmanr(ranks[mask], vals[mask])
        sig = "*" if p_s < 0.05 else "." if p_s < 0.10 else ""
        w(f"  Tier rank vs {mname:<17}  Spearman r={r_s:+.3f}  p={p_s:.4f} {sig}")

    w()
    w("  Positive r = higher tier → higher metric value")
    w("  Tier rank: Baseline=0, Enriched=1, Enriched Intense=2")
    w()

    report = "\n".join(lines)
    OUTPUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
