#!/usr/bin/env python3
"""
CRIT-Financial Correlation Analysis -- Ablation 1
===================================================
Examines whether CRIT reasoning scores (rho_bar, per-agent rho) correlate
with financial performance (Sharpe ratio, total return) for ablation 1.

Ablation 1 design:
  - 10 agent-pair configs (m_r, m_t, m_v, r_t, r_v, t_v, no_macro, no_risk,
    no_technical, no_value)
  - 4 enrichment styles (intense_light, intense, light, standard)
  - 2 PID tuning modes (aggressive, conservative)
  - 4 scenarios (2 quarters x with/without constraints)
  - NO baseline/treatment pairing => no delta analysis
  - Instead: pooled, per-condition, and per-agent correlations + OLS

Output: analysis/crit_financial_correlation_abl1_report.txt
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "logging" / "runs"
ABL_DIR = RUNS_ROOT / "vskarich_ablation_1"
OUTPUT_PATH = REPO_ROOT / "analysis" / "crit_financial_correlation_abl1_report.txt"


# ---------------------------------------------------------------------------
# Config-name parser
# ---------------------------------------------------------------------------

def parse_debate_config(config_basename: str) -> dict:
    """
    Parse an ablation-1 debate config filename into its factors.

    Example: debate_m_r_intense_light_pid_aggressive_r2_t0.3.yaml
      -> agent_pair='m_r', enrichment='intense_light', pid_type='aggressive'

    Example: debate_no_macro_standard_pid_conservative_r2_t0.3.yaml
      -> agent_pair='no_macro', enrichment='standard', pid_type='conservative'
    """
    name = config_basename.replace(".yaml", "").replace(".yml", "")
    name = name.replace("debate_", "")

    # PID type
    if "pid_aggressive" in name:
        pid_type = "aggressive"
    elif "pid_conservative" in name:
        pid_type = "conservative"
    else:
        pid_type = "unknown"

    # Enrichment style (check longest match first)
    if "_intense_light_" in name:
        enrichment = "intense_light"
    elif "_intense_" in name:
        enrichment = "intense"
    elif "_light_" in name:
        enrichment = "light"
    elif "_standard_" in name:
        enrichment = "standard"
    else:
        enrichment = "unknown"

    # Agent pair: everything before the enrichment keyword
    # For 'no_X' patterns: no_macro, no_risk, no_technical, no_value
    # For pair patterns: m_r, m_t, m_v, r_t, r_v, t_v
    enr_key = enrichment.split("_")[0]  # 'intense', 'light', or 'standard'
    idx = name.index(f"_{enr_key}")
    agent_pair = name[:idx]

    return {
        "agent_pair": agent_pair,
        "enrichment": enrichment,
        "pid_type": pid_type,
    }


def parse_scenario_name(scenario_basename: str) -> dict:
    """Parse scenario config filename into quarter and constraint flag."""
    name = scenario_basename.replace(".yaml", "").replace(".yml", "")
    has_constraints = "WITH_constraints" in name
    # Extract quarter identifier
    if "2022Q2" in name:
        quarter = "2022Q2"
    elif "2023Q2" in name:
        quarter = "2023Q2"
    else:
        quarter = name  # fallback
    return {
        "scenario": name,
        "quarter": quarter,
        "has_constraints": has_constraints,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_runs():
    """Load all completed ablation-1 runs that have both financial and CRIT data."""
    rows = []

    if not ABL_DIR.exists():
        print(f"  [ERROR] Ablation dir not found: {ABL_DIR}")
        return pd.DataFrame()

    for run_dir in sorted(ABL_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        fin_path = run_dir / "_dashboard" / "financial_metrics.json"
        crit_path = run_dir / "final" / "pid_crit_all_rounds.json"
        manifest_path = run_dir / "manifest.json"

        if not (fin_path.exists() and crit_path.exists() and manifest_path.exists()):
            continue

        try:
            fin = json.loads(fin_path.read_text())
            crit_rounds = json.loads(crit_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Skipping {run_dir.name}: {e}")
            continue

        # Financial metrics
        sharpe = fin.get("daily_metrics_annualized_sharpe")
        total_return = fin.get("daily_metrics_total_return_pct")
        if sharpe is None or total_return is None:
            continue

        # CRIT: use the last round
        last_round = crit_rounds[-1]
        crit_data = last_round.get("crit", {})
        rho_bar = crit_data.get("rho_bar")
        if rho_bar is None:
            continue

        # Per-agent rhos
        agents_data = crit_data.get("agents", {})
        rho_i = {name: adata["rho_i"] for name, adata in agents_data.items()
                 if "rho_i" in adata}

        # Parse config factors
        config_paths = manifest.get("config_paths", ["", ""])
        debate_cfg = os.path.basename(config_paths[0]) if len(config_paths) > 0 else ""
        scenario_cfg = os.path.basename(config_paths[1]) if len(config_paths) > 1 else ""

        parsed_debate = parse_debate_config(debate_cfg) if debate_cfg else {}
        parsed_scenario = parse_scenario_name(scenario_cfg) if scenario_cfg else {}

        # Roles from manifest
        roles = tuple(sorted(manifest.get("roles", [])))

        row = {
            "run_id": run_dir.name,
            "agent_pair": parsed_debate.get("agent_pair", ""),
            "enrichment": parsed_debate.get("enrichment", ""),
            "pid_type": parsed_debate.get("pid_type", ""),
            "scenario": parsed_scenario.get("scenario", ""),
            "quarter": parsed_scenario.get("quarter", ""),
            "has_constraints": parsed_scenario.get("has_constraints", None),
            "roles": roles,
            "rho_bar": rho_bar,
            "sharpe": sharpe,
            "total_return": total_return,
        }
        # Add per-agent rhos
        for agent_name, rho_val in rho_i.items():
            row[f"rho_{agent_name}"] = rho_val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------

def corr_report(x, y, label_x, label_y):
    """Compute Pearson and Spearman correlations, return formatted string."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)
    if n < 5:
        return f"  {label_x} vs {label_y}: n={n} (too few observations)\n"

    r_p, p_p = stats.pearsonr(x_clean, y_clean)
    r_s, p_s = stats.spearmanr(x_clean, y_clean)
    lines = []
    lines.append(f"  {label_x} vs {label_y}  (n={n})")
    lines.append(f"    Pearson  r={r_p:+.4f}  p={p_p:.4f}{'  *' if p_p < 0.05 else ''}")
    lines.append(f"    Spearman r={r_s:+.4f}  p={p_s:.4f}{'  *' if p_s < 0.05 else ''}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# OLS regression (manual, no statsmodels)
# ---------------------------------------------------------------------------

def manual_ols(y, X, feature_names):
    """
    OLS regression via normal equations: beta = (X'X)^-1 X'y
    Returns formatted report string.
    """
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = X[mask]
    y_clean = y[mask]
    n, k = X_clean.shape

    if n < k + 2:
        return "  OLS: too few observations\n"

    # Add intercept
    ones = np.ones((n, 1))
    X_full = np.hstack([ones, X_clean])
    names = ["(intercept)"] + list(feature_names)

    try:
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
        return "  OLS: singular matrix, cannot compute\n"

    beta = XtX_inv @ (X_full.T @ y_clean)
    y_hat = X_full @ beta
    residuals = y_clean - y_hat
    sse = residuals @ residuals
    sst = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1.0 - sse / sst if sst > 0 else float("nan")
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - len(names)) if n > len(names) else float("nan")

    # Standard errors
    mse = sse / (n - len(names)) if n > len(names) else float("nan")
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - len(names))

    # F-statistic (model vs intercept-only)
    ssr = sst - sse
    df_model = len(names) - 1
    df_resid = n - len(names)
    if df_model > 0 and df_resid > 0 and sse > 0:
        f_stat = (ssr / df_model) / (sse / df_resid)
        f_pvalue = stats.f.sf(f_stat, df_model, df_resid)
    else:
        f_stat = float("nan")
        f_pvalue = float("nan")

    lines = []
    lines.append(f"  n={n}, R²={r_squared:.4f}, Adj-R²={adj_r_squared:.4f}")
    lines.append(f"  F({df_model},{df_resid})={f_stat:.3f}, p={f_pvalue:.4f}")
    lines.append(f"  {'Variable':<25s} {'Coef':>10s} {'SE':>10s} {'t':>8s} {'p':>8s}")
    lines.append(f"  {'-'*63}")
    for i, name in enumerate(names):
        sig = " *" if p_values[i] < 0.05 else ""
        lines.append(
            f"  {name:<25s} {beta[i]:>10.4f} {se[i]:>10.4f} {t_stats[i]:>8.3f} {p_values[i]:>8.4f}{sig}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("Loading ablation 1 data...")
    df = load_all_runs()
    if df.empty:
        print("ERROR: No data loaded.")
        sys.exit(1)
    print(f"Loaded {len(df)} runs")

    report_lines = []

    def section(title):
        report_lines.append("")
        report_lines.append("=" * 72)
        report_lines.append(title)
        report_lines.append("=" * 72)

    def subsection(title):
        report_lines.append("")
        report_lines.append(f"--- {title} ---")

    def add(text):
        report_lines.append(text)

    # -------------------------------------------------------------------
    section("CRIT-FINANCIAL CORRELATION ANALYSIS -- ABLATION 1")
    # -------------------------------------------------------------------
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Total runs loaded: {len(df)}")
    add("")
    add("Ablation 1 design (no baseline/treatment pairing):")
    add(f"  Agent pairs:       {sorted(df['agent_pair'].unique())}")
    add(f"  Enrichment styles: {sorted(df['enrichment'].unique())}")
    add(f"  PID types:         {sorted(df['pid_type'].unique())}")
    add(f"  Quarters:          {sorted(df['quarter'].unique())}")
    add(f"  Constraint flag:   {sorted(df['has_constraints'].unique())}")
    add("")

    # Counts per factor
    add("Runs per agent pair:")
    for ap in sorted(df["agent_pair"].unique()):
        add(f"  {ap:>15s}: {len(df[df['agent_pair'] == ap]):>4d}")
    add("")
    add("Runs per enrichment:")
    for e in sorted(df["enrichment"].unique()):
        add(f"  {e:>15s}: {len(df[df['enrichment'] == e]):>4d}")
    add("")
    add("Runs per PID type:")
    for p in sorted(df["pid_type"].unique()):
        add(f"  {p:>15s}: {len(df[df['pid_type'] == p]):>4d}")
    add("")
    add("Runs per scenario:")
    for s in sorted(df["scenario"].unique()):
        add(f"  {len(df[df['scenario'] == s]):>4d}  {s}")
    add("")

    add("Descriptive statistics:")
    for col in ["rho_bar", "sharpe", "total_return"]:
        vals = df[col]
        add(f"  {col:>15s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}")

    # -------------------------------------------------------------------
    section("1. POOLED CORRELATIONS (all 245 runs)")
    # -------------------------------------------------------------------
    add(corr_report(df["rho_bar"].values, df["sharpe"].values, "rho_bar", "Sharpe"))
    add(corr_report(df["rho_bar"].values, df["total_return"].values, "rho_bar", "TotalReturn"))

    # -------------------------------------------------------------------
    section("2. PER-AGENT-PAIR CORRELATIONS")
    # -------------------------------------------------------------------
    for ap in sorted(df["agent_pair"].unique()):
        subsection(f"Agent pair: {ap}")
        sub = df[df["agent_pair"] == ap]
        add(f"  (n={len(sub)} runs)")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -------------------------------------------------------------------
    section("3. PER-ENRICHMENT-STYLE CORRELATIONS")
    # -------------------------------------------------------------------
    for e in sorted(df["enrichment"].unique()):
        subsection(f"Enrichment: {e}")
        sub = df[df["enrichment"] == e]
        add(f"  (n={len(sub)} runs)")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -------------------------------------------------------------------
    section("4. PER-PID-TYPE CORRELATIONS")
    # -------------------------------------------------------------------
    for p in sorted(df["pid_type"].unique()):
        subsection(f"PID type: {p}")
        sub = df[df["pid_type"] == p]
        add(f"  (n={len(sub)} runs)")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -------------------------------------------------------------------
    section("5. PER-QUARTER CORRELATIONS")
    # -------------------------------------------------------------------
    for q in sorted(df["quarter"].unique()):
        subsection(f"Quarter: {q}")
        sub = df[df["quarter"] == q]
        add(f"  (n={len(sub)} runs)")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # Constraints vs no constraints
    subsection("With constraints vs without")
    for c in [True, False]:
        label = "WITH constraints" if c else "NO constraints"
        sub = df[df["has_constraints"] == c]
        add(f"  {label} (n={len(sub)}):")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -------------------------------------------------------------------
    section("6. PER-AGENT RHO CORRELATIONS")
    # -------------------------------------------------------------------
    rho_cols = [c for c in df.columns if c.startswith("rho_") and c != "rho_bar"]
    agent_names = [c.replace("rho_", "") for c in rho_cols]
    add(f"Agents found across dataset: {', '.join(sorted(set(agent_names)))}")
    add("")

    for agent in sorted(set(agent_names)):
        col = f"rho_{agent}"
        subsection(f"Agent: {agent}")
        sub = df.dropna(subset=[col])
        if len(sub) < 5:
            add(f"  n={len(sub)} (too few observations)")
            continue
        add(f"  n={len(sub)} runs with rho_{agent}")
        add(corr_report(sub[col].values, sub["sharpe"].values, f"rho_{agent}", "Sharpe"))
        add(corr_report(sub[col].values, sub["total_return"].values, f"rho_{agent}", "TotalReturn"))

    # Comparison table: all agents side-by-side
    subsection("Agent rho comparison table (Pearson r with Sharpe)")
    header = f"  {'Agent':<15s} {'r':>8s} {'p':>8s} {'n':>5s}"
    add(header)
    add(f"  {'-'*38}")
    for agent in sorted(set(agent_names)):
        col = f"rho_{agent}"
        sub = df.dropna(subset=[col])
        if len(sub) >= 5:
            x = sub[col].values
            y = sub["sharpe"].values
            mask = np.isfinite(x) & np.isfinite(y)
            r, p = stats.pearsonr(x[mask], y[mask])
            add(f"  {agent:<15s} {r:>+8.4f} {p:>8.4f} {int(mask.sum()):>5d}")
        else:
            add(f"  {agent:<15s}     (too few data)")

    add("")
    subsection("Agent rho comparison table (Pearson r with TotalReturn)")
    header = f"  {'Agent':<15s} {'r':>8s} {'p':>8s} {'n':>5s}"
    add(header)
    add(f"  {'-'*38}")
    for agent in sorted(set(agent_names)):
        col = f"rho_{agent}"
        sub = df.dropna(subset=[col])
        if len(sub) >= 5:
            x = sub[col].values
            y = sub["total_return"].values
            mask = np.isfinite(x) & np.isfinite(y)
            r, p = stats.pearsonr(x[mask], y[mask])
            add(f"  {agent:<15s} {r:>+8.4f} {p:>8.4f} {int(mask.sum()):>5d}")
        else:
            add(f"  {agent:<15s}     (too few data)")

    # -------------------------------------------------------------------
    section("7. OLS REGRESSION: Sharpe ~ rho_bar + factor dummies")
    # -------------------------------------------------------------------
    add("Controls for agent-pair, enrichment, PID type, and quarter effects.")
    add("")

    # ------- Model A: Sharpe ~ rho_bar + agent_pair dummies -------
    subsection("Model A: Sharpe ~ rho_bar + agent_pair dummies")
    agent_pairs_sorted = sorted(df["agent_pair"].unique())
    ref_ap = agent_pairs_sorted[0]
    feature_names_a = ["rho_bar"]
    X_cols_a = [df["rho_bar"].values]
    for ap in agent_pairs_sorted[1:]:
        dummy = (df["agent_pair"] == ap).astype(float).values
        X_cols_a.append(dummy)
        feature_names_a.append(f"ap_{ap}")
    X_a = np.column_stack(X_cols_a)
    add(f"  Reference agent pair: {ref_ap}")
    add(manual_ols(df["sharpe"].values, X_a, feature_names_a))

    subsection("Model A': TotalReturn ~ rho_bar + agent_pair dummies")
    add(f"  Reference agent pair: {ref_ap}")
    add(manual_ols(df["total_return"].values, X_a, feature_names_a))

    # ------- Model B: Sharpe ~ rho_bar + enrichment + pid + quarter -------
    subsection("Model B: Sharpe ~ rho_bar + enrichment + pid_type + quarter dummies")
    feature_names_b = ["rho_bar"]
    X_cols_b = [df["rho_bar"].values]

    enrichments_sorted = sorted(df["enrichment"].unique())
    ref_enr = enrichments_sorted[0]
    for e in enrichments_sorted[1:]:
        dummy = (df["enrichment"] == e).astype(float).values
        X_cols_b.append(dummy)
        feature_names_b.append(f"enr_{e}")

    pid_types_sorted = sorted(df["pid_type"].unique())
    ref_pid = pid_types_sorted[0]
    for p in pid_types_sorted[1:]:
        dummy = (df["pid_type"] == p).astype(float).values
        X_cols_b.append(dummy)
        feature_names_b.append(f"pid_{p}")

    quarters_sorted = sorted(df["quarter"].unique())
    ref_q = quarters_sorted[0]
    for q in quarters_sorted[1:]:
        dummy = (df["quarter"] == q).astype(float).values
        X_cols_b.append(dummy)
        feature_names_b.append(f"qtr_{q}")

    # Constraint dummy
    constraint_dummy = df["has_constraints"].astype(float).values
    X_cols_b.append(constraint_dummy)
    feature_names_b.append("has_constraints")

    X_b = np.column_stack(X_cols_b)
    add(f"  Reference: enrichment={ref_enr}, pid={ref_pid}, quarter={ref_q}, constraints=False")
    add(manual_ols(df["sharpe"].values, X_b, feature_names_b))

    subsection("Model B': TotalReturn ~ rho_bar + enrichment + pid_type + quarter dummies")
    add(f"  Reference: enrichment={ref_enr}, pid={ref_pid}, quarter={ref_q}, constraints=False")
    add(manual_ols(df["total_return"].values, X_b, feature_names_b))

    # ------- Model C: Full model with all factors -------
    subsection("Model C: Sharpe ~ rho_bar + agent_pair + enrichment + pid + quarter + constraints")
    feature_names_c = ["rho_bar"]
    X_cols_c = [df["rho_bar"].values]

    for ap in agent_pairs_sorted[1:]:
        dummy = (df["agent_pair"] == ap).astype(float).values
        X_cols_c.append(dummy)
        feature_names_c.append(f"ap_{ap}")

    for e in enrichments_sorted[1:]:
        dummy = (df["enrichment"] == e).astype(float).values
        X_cols_c.append(dummy)
        feature_names_c.append(f"enr_{e}")

    for p in pid_types_sorted[1:]:
        dummy = (df["pid_type"] == p).astype(float).values
        X_cols_c.append(dummy)
        feature_names_c.append(f"pid_{p}")

    for q in quarters_sorted[1:]:
        dummy = (df["quarter"] == q).astype(float).values
        X_cols_c.append(dummy)
        feature_names_c.append(f"qtr_{q}")

    X_cols_c.append(constraint_dummy)
    feature_names_c.append("has_constraints")

    X_c = np.column_stack(X_cols_c)
    add(f"  Reference: ap={ref_ap}, enr={ref_enr}, pid={ref_pid}, qtr={ref_q}, constraints=False")
    add(manual_ols(df["sharpe"].values, X_c, feature_names_c))

    subsection("Model C': TotalReturn ~ rho_bar + all factor dummies")
    add(f"  Reference: ap={ref_ap}, enr={ref_enr}, pid={ref_pid}, qtr={ref_q}, constraints=False")
    add(manual_ols(df["total_return"].values, X_c, feature_names_c))

    # -------------------------------------------------------------------
    section("8. CROSS-TABULATION: Mean rho_bar and Sharpe by factors")
    # -------------------------------------------------------------------
    subsection("Mean rho_bar and Sharpe by agent pair")
    header = f"  {'Agent Pair':<18s} {'n':>5s} {'rho_bar':>10s} {'Sharpe':>10s} {'Return':>10s}"
    add(header)
    add(f"  {'-'*55}")
    for ap in sorted(df["agent_pair"].unique()):
        sub = df[df["agent_pair"] == ap]
        add(f"  {ap:<18s} {len(sub):>5d} {sub['rho_bar'].mean():>10.4f} "
            f"{sub['sharpe'].mean():>10.4f} {sub['total_return'].mean():>10.4f}")

    subsection("Mean rho_bar and Sharpe by enrichment")
    header = f"  {'Enrichment':<18s} {'n':>5s} {'rho_bar':>10s} {'Sharpe':>10s} {'Return':>10s}"
    add(header)
    add(f"  {'-'*55}")
    for e in sorted(df["enrichment"].unique()):
        sub = df[df["enrichment"] == e]
        add(f"  {e:<18s} {len(sub):>5d} {sub['rho_bar'].mean():>10.4f} "
            f"{sub['sharpe'].mean():>10.4f} {sub['total_return'].mean():>10.4f}")

    subsection("Mean rho_bar and Sharpe by PID type")
    header = f"  {'PID Type':<18s} {'n':>5s} {'rho_bar':>10s} {'Sharpe':>10s} {'Return':>10s}"
    add(header)
    add(f"  {'-'*55}")
    for p in sorted(df["pid_type"].unique()):
        sub = df[df["pid_type"] == p]
        add(f"  {p:<18s} {len(sub):>5d} {sub['rho_bar'].mean():>10.4f} "
            f"{sub['sharpe'].mean():>10.4f} {sub['total_return'].mean():>10.4f}")

    subsection("Mean rho_bar and Sharpe by quarter")
    header = f"  {'Quarter':<18s} {'n':>5s} {'rho_bar':>10s} {'Sharpe':>10s} {'Return':>10s}"
    add(header)
    add(f"  {'-'*55}")
    for q in sorted(df["quarter"].unique()):
        sub = df[df["quarter"] == q]
        add(f"  {q:<18s} {len(sub):>5d} {sub['rho_bar'].mean():>10.4f} "
            f"{sub['sharpe'].mean():>10.4f} {sub['total_return'].mean():>10.4f}")

    subsection("Mean rho_bar and Sharpe by constraints")
    header = f"  {'Constraints':<18s} {'n':>5s} {'rho_bar':>10s} {'Sharpe':>10s} {'Return':>10s}"
    add(header)
    add(f"  {'-'*55}")
    for c in [True, False]:
        label = "YES" if c else "NO"
        sub = df[df["has_constraints"] == c]
        add(f"  {label:<18s} {len(sub):>5d} {sub['rho_bar'].mean():>10.4f} "
            f"{sub['sharpe'].mean():>10.4f} {sub['total_return'].mean():>10.4f}")

    # -------------------------------------------------------------------
    section("9. SUMMARY AND INTERPRETATION")
    # -------------------------------------------------------------------

    n_total = len(df)
    r_pool_sharpe, p_pool_sharpe = stats.pearsonr(df["rho_bar"].values, df["sharpe"].values)
    r_pool_return, p_pool_return = stats.pearsonr(df["rho_bar"].values, df["total_return"].values)
    rs_pool_sharpe, ps_pool_sharpe = stats.spearmanr(df["rho_bar"].values, df["sharpe"].values)
    rs_pool_return, ps_pool_return = stats.spearmanr(df["rho_bar"].values, df["total_return"].values)

    add(f"Total observations: {n_total}")
    add(f"Pooled rho_bar-Sharpe:      Pearson r={r_pool_sharpe:+.4f} (p={p_pool_sharpe:.4f}), "
        f"Spearman r={rs_pool_sharpe:+.4f} (p={ps_pool_sharpe:.4f})")
    add(f"Pooled rho_bar-TotalReturn: Pearson r={r_pool_return:+.4f} (p={p_pool_return:.4f}), "
        f"Spearman r={rs_pool_return:+.4f} (p={ps_pool_return:.4f})")
    add("")

    if p_pool_sharpe < 0.05:
        add("=> Statistically significant pooled correlation between CRIT scores and Sharpe.")
    elif p_pool_sharpe < 0.10:
        add("=> Marginally significant pooled correlation between CRIT scores and Sharpe (p < 0.10).")
    else:
        add("=> No statistically significant pooled correlation between CRIT scores and Sharpe.")

    if p_pool_return < 0.05:
        add("=> Statistically significant pooled correlation between CRIT scores and TotalReturn.")
    elif p_pool_return < 0.10:
        add("=> Marginally significant pooled correlation between CRIT scores and TotalReturn (p < 0.10).")
    else:
        add("=> No statistically significant pooled correlation between CRIT scores and TotalReturn.")

    add("")
    add("Design note: Ablation 1 has no baseline/treatment pairing, so delta")
    add("analysis is not applicable. Instead, we rely on pooled correlations,")
    add("per-condition correlations, and OLS with factor dummies to control for")
    add("the factorial design (agent pair, enrichment, PID type, quarter,")
    add("constraints).")
    add("")
    add("Note: Correlations marked with * are significant at p < 0.05.")
    add("All p-values are two-tailed.")

    # -------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------
    report_text = "\n".join(report_lines) + "\n"

    # Print to stdout
    print(report_text)

    # Write to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report_text)
    print(f"\n[Report written to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
