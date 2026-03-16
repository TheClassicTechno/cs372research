#!/usr/bin/env python3
"""
CRIT-Financial Correlation Analysis
====================================
Examines whether CRIT reasoning scores (rho_bar, per-agent rho) correlate
with financial performance (Sharpe ratio, total return) across ablations 7, 8, 10.

Output: analysis/crit_financial_correlation_report.txt
"""

import json
import os
import sys
import glob
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
ABLATIONS = [7, 8, 10]
OUTPUT_PATH = REPO_ROOT / "analysis" / "crit_financial_correlation_report.txt"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_runs():
    """Load all completed runs across ablations that have both financial and CRIT data."""
    rows = []
    for abl in ABLATIONS:
        abl_dir = RUNS_ROOT / f"vskarich_ablation_{abl}"
        if not abl_dir.exists():
            print(f"  [WARN] Ablation dir not found: {abl_dir}")
            continue

        for run_dir in sorted(abl_dir.iterdir()):
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

            # CRIT: use the last round (there is typically only 1)
            last_round = crit_rounds[-1]
            crit_data = last_round.get("crit", {})
            rho_bar = crit_data.get("rho_bar")
            if rho_bar is None:
                continue

            # Per-agent rhos
            agents_data = crit_data.get("agents", {})
            rho_i = {name: adata["rho_i"] for name, adata in agents_data.items()}

            # Condition: baseline vs treatment
            config_path_0 = manifest.get("config_paths", [""])[0]
            condition = "baseline" if "baseline" in os.path.basename(config_path_0).lower() else "treatment"

            # Scenario: extract from config_paths[1]
            scenario_path = manifest.get("config_paths", ["", ""])[1]
            scenario = os.path.basename(scenario_path).replace(".yaml", "").replace(".yml", "")

            row = {
                "ablation": abl,
                "run_id": run_dir.name,
                "condition": condition,
                "scenario": scenario,
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
    print("Loading data...")
    df = load_all_runs()
    print(f"Loaded {len(df)} runs across ablations {sorted(df['ablation'].unique())}")

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

    # -----------------------------------------------------------------------
    section("CRIT-FINANCIAL CORRELATION ANALYSIS")
    # -----------------------------------------------------------------------
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Total runs loaded: {len(df)}")
    for abl in sorted(df["ablation"].unique()):
        sub = df[df["ablation"] == abl]
        nb = len(sub[sub["condition"] == "baseline"])
        nt = len(sub[sub["condition"] == "treatment"])
        add(f"  Ablation {abl}: {len(sub)} runs (baseline={nb}, treatment={nt})")

    add("")
    add("Descriptive statistics:")
    for col in ["rho_bar", "sharpe", "total_return"]:
        vals = df[col]
        add(f"  {col:>15s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}")

    # -----------------------------------------------------------------------
    section("1. POOLED CORRELATIONS (all runs)")
    # -----------------------------------------------------------------------
    add(corr_report(df["rho_bar"].values, df["sharpe"].values, "rho_bar", "Sharpe"))
    add(corr_report(df["rho_bar"].values, df["total_return"].values, "rho_bar", "TotalReturn"))

    # -----------------------------------------------------------------------
    section("2. WITHIN-ABLATION CORRELATIONS")
    # -----------------------------------------------------------------------
    for abl in sorted(df["ablation"].unique()):
        subsection(f"Ablation {abl}")
        sub = df[df["ablation"] == abl]
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -----------------------------------------------------------------------
    section("3. WITHIN-CONDITION CORRELATIONS")
    # -----------------------------------------------------------------------
    for cond in ["baseline", "treatment"]:
        subsection(f"Condition: {cond}")
        sub = df[df["condition"] == cond]
        add(f"  (n={len(sub)} runs)")
        add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
        add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # Also within-ablation x within-condition
    subsection("Within ablation x condition")
    for abl in sorted(df["ablation"].unique()):
        for cond in ["baseline", "treatment"]:
            sub = df[(df["ablation"] == abl) & (df["condition"] == cond)]
            if len(sub) < 5:
                add(f"  Abl {abl}, {cond}: n={len(sub)} (too few)")
                continue
            add(f"  Abl {abl}, {cond} (n={len(sub)}):")
            add(corr_report(sub["rho_bar"].values, sub["sharpe"].values, "rho_bar", "Sharpe"))
            add(corr_report(sub["rho_bar"].values, sub["total_return"].values, "rho_bar", "TotalReturn"))

    # -----------------------------------------------------------------------
    section("4. WITHIN-SCENARIO PAIRED DIFFERENCES (delta analysis)")
    # -----------------------------------------------------------------------
    add("For each scenario appearing in both baseline and treatment within an ablation,")
    add("compute delta_rho = treatment.rho_bar - baseline.rho_bar")
    add("         delta_sharpe = treatment.sharpe - baseline.sharpe")
    add("         delta_return = treatment.total_return - baseline.total_return")
    add("Then correlate the deltas.")
    add("")

    delta_rows = []
    for abl in sorted(df["ablation"].unique()):
        sub = df[df["ablation"] == abl]
        baseline = sub[sub["condition"] == "baseline"].set_index("scenario")
        treatment = sub[sub["condition"] == "treatment"].set_index("scenario")
        common = baseline.index.intersection(treatment.index)
        for sc in common:
            b = baseline.loc[sc]
            t = treatment.loc[sc]
            # Handle duplicate scenarios (multiple runs per scenario)
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            if isinstance(t, pd.DataFrame):
                t = t.iloc[0]
            delta_rows.append({
                "ablation": abl,
                "scenario": sc,
                "delta_rho": t["rho_bar"] - b["rho_bar"],
                "delta_sharpe": t["sharpe"] - b["sharpe"],
                "delta_return": t["total_return"] - b["total_return"],
            })

    if delta_rows:
        ddf = pd.DataFrame(delta_rows)
        add(f"Paired scenarios found: {len(ddf)}")
        for abl in sorted(ddf["ablation"].unique()):
            add(f"  Ablation {abl}: {len(ddf[ddf['ablation'] == abl])} pairs")
        add("")

        subsection("Pooled deltas")
        add(f"  delta_rho:    mean={ddf['delta_rho'].mean():+.4f}  std={ddf['delta_rho'].std():.4f}")
        add(f"  delta_sharpe: mean={ddf['delta_sharpe'].mean():+.4f}  std={ddf['delta_sharpe'].std():.4f}")
        add(f"  delta_return: mean={ddf['delta_return'].mean():+.4f}  std={ddf['delta_return'].std():.4f}")
        add("")
        add(corr_report(ddf["delta_rho"].values, ddf["delta_sharpe"].values, "delta_rho", "delta_Sharpe"))
        add(corr_report(ddf["delta_rho"].values, ddf["delta_return"].values, "delta_rho", "delta_Return"))

        for abl in sorted(ddf["ablation"].unique()):
            subsection(f"Deltas within ablation {abl}")
            dsub = ddf[ddf["ablation"] == abl]
            add(f"  n={len(dsub)} pairs")
            add(f"  delta_rho:    mean={dsub['delta_rho'].mean():+.4f}  std={dsub['delta_rho'].std():.4f}")
            add(f"  delta_sharpe: mean={dsub['delta_sharpe'].mean():+.4f}  std={dsub['delta_sharpe'].std():.4f}")
            add(f"  delta_return: mean={dsub['delta_return'].mean():+.4f}  std={dsub['delta_return'].std():.4f}")
            add("")
            add(corr_report(dsub["delta_rho"].values, dsub["delta_sharpe"].values, "delta_rho", "delta_Sharpe"))
            add(corr_report(dsub["delta_rho"].values, dsub["delta_return"].values, "delta_rho", "delta_Return"))
    else:
        add("  No paired scenarios found.")

    # -----------------------------------------------------------------------
    section("5. PER-AGENT RHO CORRELATIONS")
    # -----------------------------------------------------------------------
    # Identify all agent rho columns
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

    # -----------------------------------------------------------------------
    section("6. OLS REGRESSION: Sharpe ~ rho_bar + ablation dummies")
    # -----------------------------------------------------------------------
    add("Controls for ablation-level effects using dummy variables.")
    add("Reference ablation: the lowest-numbered ablation.\n")

    # Build design matrix
    ablation_vals = sorted(df["ablation"].unique())
    ref_abl = ablation_vals[0]
    feature_names = ["rho_bar"]
    X_cols = [df["rho_bar"].values]
    for abl in ablation_vals[1:]:
        dummy = (df["ablation"] == abl).astype(float).values
        X_cols.append(dummy)
        feature_names.append(f"abl_{abl}_dummy")

    X = np.column_stack(X_cols)
    y = df["sharpe"].values

    subsection(f"Dependent variable: Sharpe (ref ablation={ref_abl})")
    add(manual_ols(y, X, feature_names))

    subsection(f"Dependent variable: TotalReturn (ref ablation={ref_abl})")
    y2 = df["total_return"].values
    add(manual_ols(y2, X, feature_names))

    # Also with condition dummy
    subsection("Extended model: Sharpe ~ rho_bar + ablation dummies + condition dummy")
    cond_dummy = (df["condition"] == "treatment").astype(float).values
    X_ext = np.column_stack([X, cond_dummy])
    ext_names = feature_names + ["treatment_dummy"]
    add(manual_ols(y, X_ext, ext_names))

    subsection("Extended model: TotalReturn ~ rho_bar + ablation dummies + condition dummy")
    add(manual_ols(y2, X_ext, ext_names))

    # -----------------------------------------------------------------------
    section("7. SUMMARY AND INTERPRETATION")
    # -----------------------------------------------------------------------

    # Compute key numbers for summary
    n_total = len(df)
    r_pool_sharpe, p_pool_sharpe = stats.pearsonr(df["rho_bar"].values, df["sharpe"].values)
    r_pool_return, p_pool_return = stats.pearsonr(df["rho_bar"].values, df["total_return"].values)
    rs_pool_sharpe, ps_pool_sharpe = stats.spearmanr(df["rho_bar"].values, df["sharpe"].values)

    add(f"Total observations: {n_total}")
    add(f"Pooled rho_bar-Sharpe:      Pearson r={r_pool_sharpe:+.4f} (p={p_pool_sharpe:.4f}), "
        f"Spearman r={rs_pool_sharpe:+.4f} (p={ps_pool_sharpe:.4f})")
    add(f"Pooled rho_bar-TotalReturn: Pearson r={r_pool_return:+.4f} (p={p_pool_return:.4f})")
    add("")

    if p_pool_sharpe < 0.05:
        add("=> Statistically significant pooled correlation between CRIT scores and Sharpe.")
    elif p_pool_sharpe < 0.10:
        add("=> Marginally significant pooled correlation between CRIT scores and Sharpe (p < 0.10).")
    else:
        add("=> No statistically significant pooled correlation between CRIT scores and Sharpe.")

    if delta_rows:
        ddf_all = pd.DataFrame(delta_rows)
        r_d, p_d = stats.pearsonr(ddf_all["delta_rho"].values, ddf_all["delta_sharpe"].values)
        add(f"\nDelta analysis (paired within-scenario): r={r_d:+.4f} (p={p_d:.4f})")
        if p_d < 0.05:
            add("=> Changes in CRIT quality significantly predict changes in financial performance.")
        else:
            add("=> Changes in CRIT quality do NOT significantly predict changes in financial performance.")

    add("")
    add("Note: Correlations marked with * are significant at p < 0.05.")
    add("All p-values are two-tailed.")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    report_text = "\n".join(report_lines) + "\n"

    # Print to stdout
    print(report_text)

    # Write to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report_text)
    print(f"\n[Report written to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
