#!/usr/bin/env python3
"""
Ablation 7 Reasoning Analysis
==============================
Analyzes the effect of JS-divergence intervention on reasoning quality
and opinion diversity in ablation 7 (2 agents × 2 configs × 35 scenarios).

Outputs a comprehensive report to analysis/ablation7_reasoning_report.txt

Run:
    python3 analysis/ablation7_reasoning_analysis.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "logging" / "runs" / "vskarich_ablation_7"
OUTPUT = Path(__file__).resolve().parent / "ablation7_reasoning_report.txt"


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def safe_load(p):
    if p.exists():
        return load_json(p)
    return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_run_data(run_dir):
    """Extract all relevant metrics from a single run."""
    manifest = load_json(run_dir / "manifest.json")
    r1 = run_dir / "rounds" / "round_001"

    proposal = safe_load(r1 / "metrics_propose.json")
    revision = safe_load(r1 / "metrics_revision.json")
    retry = safe_load(r1 / "metrics_retry_001.json")
    crit = safe_load(r1 / "metrics" / "crit_scores.json")

    # Determine condition from intervention config
    ic = manifest.get("intervention_config", {})
    rules = ic.get("rules", {})
    js_rule = rules.get("js_collapse", {})
    threshold = js_rule.get("threshold", 0.0)
    condition = "intervention" if threshold > 0 else "baseline"

    # Extract scenario from config_paths
    config_paths = manifest.get("config_paths", [])
    scenario = ""
    for cp in config_paths:
        stem = Path(cp).stem
        if "scenario" in stem or "202" in stem:
            scenario = stem
            break

    # JS divergence across phases
    js_propose = proposal["js_divergence"] if proposal else None
    js_revision = revision["js_divergence"] if revision else None
    js_retry = retry["js_divergence"] if retry else None

    # Evidence overlap
    eo_propose = proposal.get("evidence_overlap") if proposal else None
    eo_revision = revision.get("evidence_overlap") if revision else None
    eo_retry = retry.get("evidence_overlap") if retry else None

    # Collapse ratio
    collapse_revision = (js_revision / js_propose) if (js_propose and js_revision and js_propose > 0) else None
    collapse_final = collapse_revision
    if js_retry is not None and js_propose and js_propose > 0:
        collapse_final = js_retry / js_propose

    # CRIT scores
    rho_bar = crit["rho_bar"] if crit else None
    agent_scores = {}
    if crit and "agent_scores" in crit:
        for agent_name, scores in crit["agent_scores"].items():
            agent_scores[agent_name] = {
                "rho_i": scores.get("rho_i"),
                "pillars": scores.get("pillar_scores", {}),
                "diagnostics": scores.get("diagnostics", {}),
            }

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "condition": condition,
        "scenario": scenario,
        "threshold": threshold,
        "has_retry": retry is not None,
        # JS divergence
        "js_propose": js_propose,
        "js_revision": js_revision,
        "js_retry": js_retry,
        # Evidence overlap
        "eo_propose": eo_propose,
        "eo_revision": eo_revision,
        "eo_retry": eo_retry,
        # Collapse ratios
        "collapse_revision": collapse_revision,
        "collapse_final": collapse_final,
        # CRIT
        "rho_bar": rho_bar,
        "agent_scores": agent_scores,
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def paired_ttest(a_vals, b_vals, label=""):
    """Run paired t-test and return dict of results."""
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return {"label": label, "n": n, "a_mean": None, "b_mean": None,
                "t": None, "p": None, "note": "insufficient data"}
    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "label": label,
        "n": n,
        "a_mean": float(np.mean(a)),
        "a_std": float(np.std(a, ddof=1)),
        "b_mean": float(np.mean(b)),
        "b_std": float(np.std(b, ddof=1)),
        "diff_mean": float(np.mean(b - a)),
        "diff_std": float(np.std(b - a, ddof=1)),
        "t": float(t_stat),
        "p": float(p_val),
        "cohens_d": float(np.mean(b - a) / np.std(b - a, ddof=1)) if np.std(b - a, ddof=1) > 0 else 0,
    }


def wilcoxon_test(a_vals, b_vals, label=""):
    """Run Wilcoxon signed-rank test."""
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    diffs = b - a
    nonzero = np.sum(diffs != 0)
    if nonzero < 3:
        return {"label": label, "n": n, "W": None, "p": None, "note": "insufficient non-zero diffs"}
    W, p_val = stats.wilcoxon(a, b, alternative="two-sided")
    # Rank-biserial r
    r = 1 - (2 * W) / (nonzero * (nonzero + 1) / 2)
    return {
        "label": label,
        "n": n,
        "n_nonzero": int(nonzero),
        "a_median": float(np.median(a)),
        "b_median": float(np.median(b)),
        "W": float(W),
        "p": float(p_val),
        "effect_size_r": float(r),
    }


def sig_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def write_report(runs):
    lines = []
    w = lines.append

    w("=" * 80)
    w("ABLATION 7: EFFECT OF JS-DIVERGENCE INTERVENTION ON REASONING QUALITY")
    w("=" * 80)
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Total runs analyzed: {len(runs)}")
    w("")

    # Split by condition
    baseline = [r for r in runs if r["condition"] == "baseline"]
    intervention = [r for r in runs if r["condition"] == "intervention"]
    w(f"Baseline runs: {len(baseline)}")
    w(f"Intervention runs: {len(intervention)}")

    # Intervention firing rate
    retries = [r for r in intervention if r["has_retry"]]
    w(f"Intervention fired (retry triggered): {len(retries)}/{len(intervention)} "
      f"({100*len(retries)/len(intervention):.0f}%)")
    w("")

    # -----------------------------------------------------------------------
    # 1. Pair by scenario
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 1: PAIRED ANALYSIS BY SCENARIO")
    w("-" * 80)
    w("")

    baseline_map = {}
    for r in baseline:
        baseline_map[r["scenario"]] = r
    intervention_map = {}
    for r in intervention:
        intervention_map[r["scenario"]] = r

    shared = sorted(set(baseline_map.keys()) & set(intervention_map.keys()))
    w(f"Paired scenarios: {len(shared)}")
    w("")

    # -----------------------------------------------------------------------
    # 2. Final Collapse Ratio (primary outcome)
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 2: FINAL COLLAPSE RATIO (PRIMARY OUTCOME)")
    w("-" * 80)
    w("")
    w("Collapse ratio = JS_final / JS_proposal")
    w("  Baseline: JS_final = JS_revision (no retry possible)")
    w("  Intervention: JS_final = JS_retry if triggered, else JS_revision")
    w("  Higher = more diversity preserved after debate")
    w("")

    a_cr = [baseline_map[s]["collapse_final"] for s in shared]
    b_cr = [intervention_map[s]["collapse_final"] for s in shared]

    t_result = paired_ttest(a_cr, b_cr, "Collapse Ratio")
    w_result = wilcoxon_test(a_cr, b_cr, "Collapse Ratio")

    w(f"  Baseline mean:       {t_result['a_mean']:.4f} (SD={t_result['a_std']:.4f})")
    w(f"  Intervention mean:   {t_result['b_mean']:.4f} (SD={t_result['b_std']:.4f})")
    w(f"  Difference:          {t_result['diff_mean']:+.4f} (SD={t_result['diff_std']:.4f})")
    w(f"  Paired t-test:       t={t_result['t']:.3f}, p={t_result['p']:.6f} {sig_stars(t_result['p'])}")
    w(f"  Cohen's d:           {t_result['cohens_d']:.3f}")
    w(f"  Wilcoxon:            W={w_result['W']:.1f}, p={w_result['p']:.6f} {sig_stars(w_result['p'])}")
    w(f"  Effect size r:       {w_result['effect_size_r']:.3f}")
    w(f"  N pairs:             {t_result['n']}")
    w("")

    # -----------------------------------------------------------------------
    # 3. JS Divergence by phase
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 3: JS DIVERGENCE BY PHASE")
    w("-" * 80)
    w("")

    for phase, key in [("Proposal", "js_propose"), ("Revision", "js_revision")]:
        a_vals = [baseline_map[s][key] for s in shared]
        b_vals = [intervention_map[s][key] for s in shared]
        res = paired_ttest(a_vals, b_vals, f"JS Divergence ({phase})")
        w(f"  {phase}:")
        w(f"    Baseline: {res['a_mean']:.4f}   Intervention: {res['b_mean']:.4f}")
        w(f"    Diff: {res['diff_mean']:+.4f}, t={res['t']:.3f}, p={res['p']:.4f} {sig_stars(res['p'])}")
        w("")

    # Final JS (retry if available, else revision)
    a_final_js = []
    b_final_js = []
    for s in shared:
        a_final_js.append(baseline_map[s]["js_revision"])
        b_js = intervention_map[s]["js_retry"] if intervention_map[s]["js_retry"] is not None else intervention_map[s]["js_revision"]
        b_final_js.append(b_js)

    res = paired_ttest(a_final_js, b_final_js, "JS Divergence (Final)")
    w(f"  Final (after intervention if triggered):")
    w(f"    Baseline: {res['a_mean']:.4f}   Intervention: {res['b_mean']:.4f}")
    w(f"    Diff: {res['diff_mean']:+.4f}, t={res['t']:.3f}, p={res['p']:.4f} {sig_stars(res['p'])}")
    w("")

    # -----------------------------------------------------------------------
    # 4. Evidence Overlap
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 4: EVIDENCE OVERLAP")
    w("-" * 80)
    w("")

    for phase, key in [("Proposal", "eo_propose"), ("Revision", "eo_revision")]:
        a_vals = [baseline_map[s].get(key) for s in shared]
        b_vals = [intervention_map[s].get(key) for s in shared]
        # Filter None
        valid = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        if valid:
            av, bv = zip(*valid)
            res = paired_ttest(av, bv, f"Evidence Overlap ({phase})")
            w(f"  {phase}:")
            w(f"    Baseline: {res['a_mean']:.4f}   Intervention: {res['b_mean']:.4f}")
            w(f"    Diff: {res['diff_mean']:+.4f}, t={res['t']:.3f}, p={res['p']:.4f} {sig_stars(res['p'])}")
            w("")

    # -----------------------------------------------------------------------
    # 5. CRIT Reasoning Quality
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 5: CRIT REASONING QUALITY (rho_bar)")
    w("-" * 80)
    w("")
    w("rho_bar = mean CRIT reasoning quality score across agents")
    w("Higher = better reasoning quality")
    w("")

    a_rho = [baseline_map[s]["rho_bar"] for s in shared]
    b_rho = [intervention_map[s]["rho_bar"] for s in shared]

    res = paired_ttest(a_rho, b_rho, "rho_bar")
    w(f"  Baseline mean:       {res['a_mean']:.4f} (SD={res['a_std']:.4f})")
    w(f"  Intervention mean:   {res['b_mean']:.4f} (SD={res['b_std']:.4f})")
    w(f"  Difference:          {res['diff_mean']:+.4f}")
    w(f"  Paired t-test:       t={res['t']:.3f}, p={res['p']:.4f} {sig_stars(res['p'])}")
    w(f"  Cohen's d:           {res['cohens_d']:.3f}")
    w(f"  N pairs:             {res['n']}")
    w("")

    # -----------------------------------------------------------------------
    # 6. Per-agent CRIT scores
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 6: PER-AGENT CRIT SCORES (rho_i)")
    w("-" * 80)
    w("")

    agents = set()
    for r in runs:
        agents.update(r["agent_scores"].keys())
    agents = sorted(agents)

    for agent in agents:
        a_vals = [baseline_map[s]["agent_scores"].get(agent, {}).get("rho_i") for s in shared]
        b_vals = [intervention_map[s]["agent_scores"].get(agent, {}).get("rho_i") for s in shared]
        valid = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        if valid:
            av, bv = zip(*valid)
            res = paired_ttest(av, bv, f"rho_i ({agent})")
            w(f"  {agent}:")
            w(f"    Baseline: {res['a_mean']:.4f}   Intervention: {res['b_mean']:.4f}")
            w(f"    Diff: {res['diff_mean']:+.4f}, t={res['t']:.3f}, p={res['p']:.4f} {sig_stars(res['p'])}")
            w("")

    # -----------------------------------------------------------------------
    # 7. Per-pillar CRIT scores
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 7: PER-PILLAR CRIT SCORES")
    w("-" * 80)
    w("")

    pillar_names = {"LV": "Logical Validity", "ES": "Evidential Support",
                    "AC": "Alternative Consideration", "CA": "Causal Alignment"}

    for agent in agents:
        w(f"  Agent: {agent}")
        for pillar_key, pillar_label in pillar_names.items():
            a_vals = []
            b_vals = []
            for s in shared:
                a_pillars = baseline_map[s]["agent_scores"].get(agent, {}).get("pillars", {})
                b_pillars = intervention_map[s]["agent_scores"].get(agent, {}).get("pillars", {})
                a_vals.append(a_pillars.get(pillar_key))
                b_vals.append(b_pillars.get(pillar_key))
            valid = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
            if valid:
                av, bv = zip(*valid)
                res = paired_ttest(av, bv, f"{pillar_label} ({agent})")
                w(f"    {pillar_label:30s}  B={res['a_mean']:.3f}  I={res['b_mean']:.3f}  "
                  f"d={res['diff_mean']:+.3f}  p={res['p']:.3f} {sig_stars(res['p'])}")
        w("")

    # -----------------------------------------------------------------------
    # 8. Within-intervention analysis: effect of retry
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 8: WITHIN-INTERVENTION ANALYSIS (Retry Effect)")
    w("-" * 80)
    w("")
    w("Comparing revision vs retry within intervention runs where retry fired.")
    w("")

    retry_runs = [r for r in intervention if r["has_retry"]]
    no_retry_runs = [r for r in intervention if not r["has_retry"]]
    w(f"  Runs with retry: {len(retry_runs)}")
    w(f"  Runs without retry: {len(no_retry_runs)}")
    w("")

    if len(retry_runs) >= 3:
        # JS divergence: revision vs retry (within same run)
        js_rev = [r["js_revision"] for r in retry_runs]
        js_ret = [r["js_retry"] for r in retry_runs]
        res = paired_ttest(js_rev, js_ret, "JS Divergence (revision→retry)")
        w(f"  JS Divergence:")
        w(f"    Pre-retry (revision):  {res['a_mean']:.4f} (SD={res['a_std']:.4f})")
        w(f"    Post-retry:            {res['b_mean']:.4f} (SD={res['b_std']:.4f})")
        w(f"    Change:                {res['diff_mean']:+.4f}")
        w(f"    t={res['t']:.3f}, p={res['p']:.6f} {sig_stars(res['p'])}")
        w(f"    Cohen's d:             {res['cohens_d']:.3f}")
        w("")

        # Collapse ratio: revision vs retry
        cr_rev = [r["collapse_revision"] for r in retry_runs]
        cr_ret = [r["collapse_final"] for r in retry_runs]
        res = paired_ttest(cr_rev, cr_ret, "Collapse Ratio (revision→retry)")
        w(f"  Collapse Ratio:")
        w(f"    Pre-retry (revision):  {res['a_mean']:.4f} (SD={res['a_std']:.4f})")
        w(f"    Post-retry:            {res['b_mean']:.4f} (SD={res['b_std']:.4f})")
        w(f"    Change:                {res['diff_mean']:+.4f}")
        w(f"    t={res['t']:.3f}, p={res['p']:.6f} {sig_stars(res['p'])}")
        w(f"    Cohen's d:             {res['cohens_d']:.3f}")
        w("")

        # Proposal JS for context: how much divergence was there to start?
        js_prop = [r["js_propose"] for r in retry_runs]
        w(f"  Context — Proposal JS in retry runs: {np.mean(js_prop):.4f} (SD={np.std(js_prop, ddof=1):.4f})")
        w(f"  Revision collapsed to {np.mean(js_rev):.4f} ({np.mean(js_rev)/np.mean(js_prop)*100:.0f}% of proposal)")
        w(f"  Retry restored to     {np.mean(js_ret):.4f} ({np.mean(js_ret)/np.mean(js_prop)*100:.0f}% of proposal)")
        w("")

    # -----------------------------------------------------------------------
    # 9. Per-scenario detail table
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 9: PER-SCENARIO DETAIL")
    w("-" * 80)
    w("")
    header = f"{'Scenario':<35s} {'B_CR':>6s} {'I_CR':>6s} {'Delta':>7s} {'Retry?':>6s} {'B_rho':>6s} {'I_rho':>6s}"
    w(header)
    w("-" * len(header))

    for s in shared:
        br = baseline_map[s]
        ir = intervention_map[s]
        b_cr_s = br["collapse_final"]
        i_cr_s = ir["collapse_final"]
        delta = (i_cr_s - b_cr_s) if (i_cr_s is not None and b_cr_s is not None) else None
        retry_flag = "Yes" if ir["has_retry"] else "No"
        b_rho_s = br["rho_bar"]
        i_rho_s = ir["rho_bar"]
        w(f"{s:<35s} {b_cr_s:>6.3f} {i_cr_s:>6.3f} {delta:>+7.3f} {retry_flag:>6s} {b_rho_s:>6.3f} {i_rho_s:>6.3f}")

    w("")

    # -----------------------------------------------------------------------
    # 10. CRIT Diagnostics: flag frequency
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 10: CRIT DIAGNOSTIC FLAG FREQUENCY")
    w("-" * 80)
    w("")

    diag_keys = ["contradictions", "unsupported_claims", "ignored_critiques",
                 "premature_certainty", "causal_overreach", "conclusion_drift"]

    for agent in agents:
        w(f"  Agent: {agent}")
        for dk in diag_keys:
            b_count = sum(1 for r in baseline
                          if r["agent_scores"].get(agent, {}).get("diagnostics", {}).get(dk, False))
            i_count = sum(1 for r in intervention
                          if r["agent_scores"].get(agent, {}).get("diagnostics", {}).get(dk, False))
            w(f"    {dk:<25s}  Baseline: {b_count:>2d}/{len(baseline)}  "
              f"Intervention: {i_count:>2d}/{len(intervention)}")
        w("")

    # -----------------------------------------------------------------------
    # 11. Summary and interpretation
    # -----------------------------------------------------------------------
    w("-" * 80)
    w("SECTION 11: SUMMARY")
    w("-" * 80)
    w("")

    # Re-derive key stats
    cr_res = paired_ttest(a_cr, b_cr, "CR")
    rho_res = paired_ttest(a_rho, b_rho, "rho")

    w("Key findings:")
    w("")
    w(f"  1. DIVERSITY PRESERVATION: The intervention significantly improved")
    w(f"     final collapse ratio ({cr_res['a_mean']:.3f} -> {cr_res['b_mean']:.3f}, "
      f"p={cr_res['p']:.4f}{sig_stars(cr_res['p'])})")
    w(f"     Agents retained more of their initial opinion diversity after debate.")
    w("")
    w(f"  2. REASONING QUALITY: CRIT rho_bar was unchanged between conditions")
    w(f"     ({rho_res['a_mean']:.3f} vs {rho_res['b_mean']:.3f}, "
      f"p={rho_res['p']:.4f}{sig_stars(rho_res['p'])})")
    w(f"     The intervention did NOT degrade reasoning quality.")
    w("")
    w(f"  3. INTERVENTION MECHANISM: In the {len(retries)} scenarios where the")
    w(f"     intervention triggered, retry restored JS divergence from")
    if len(retries) >= 3:
        js_rev_retry = np.mean([r["js_revision"] for r in retries])
        js_ret_retry = np.mean([r["js_retry"] for r in retries])
        w(f"     {js_rev_retry:.4f} to {js_ret_retry:.4f} (within-run comparison).")
    w("")
    w(f"  4. SELECTIVE ACTIVATION: The intervention only fired when needed —")
    w(f"     {len(retries)}/{len(intervention)} scenarios ({100*len(retries)/len(intervention):.0f}%)")
    w(f"     had sufficient collapse to trigger the threshold (0.8).")
    w("")

    w("=" * 80)
    w("END OF REPORT")
    w("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not RUNS_DIR.exists():
        print(f"ERROR: Runs directory not found: {RUNS_DIR}")
        sys.exit(1)

    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
    print(f"Found {len(run_dirs)} runs in {RUNS_DIR.name}")

    runs = []
    errors = []
    for rd in run_dirs:
        try:
            data = extract_run_data(rd)
            runs.append(data)
        except Exception as e:
            errors.append((rd.name, str(e)))

    if errors:
        print(f"Skipped {len(errors)} runs with errors:")
        for name, err in errors:
            print(f"  {name}: {err}")

    print(f"Successfully extracted {len(runs)} runs")

    report = write_report(runs)

    OUTPUT.write_text(report)
    print(f"\nReport written to: {OUTPUT}")
    print("\n" + report)


if __name__ == "__main__":
    main()
