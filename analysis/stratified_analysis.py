#!/usr/bin/env python3
"""
Stratified CRIT–Financial Correlation Analysis
===============================================
Professor recommendation #3: repeat the core CRIT–performance correlation
analysis separately for each agent role (macro, risk, technical) and
separately for each market regime (inflation_shock, recession, rates_stress,
tech_rally, neutral).

This tests whether reasoning quality is informative only for certain roles
or under certain market conditions — rather than being washed out in a
pooled average.

Output
------
  analysis/stratified_analysis_report.txt  — human-readable text report
  analysis/stratified_analysis_grid.csv    — machine-readable role×regime grid

Usage
-----
  python analysis/stratified_analysis.py [--runs-root PATH] [--experiments EXP1 EXP2 ...]

The script defaults to looking in logging/runs/ and searching all subdirs.
Pass --runs-root to point at a different root (e.g. results_juli/).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "logging" / "runs"
REGIME_MANIFEST_PATH = REPO_ROOT / "config" / "regime_manifest.yaml"
OUTPUT_REPORT = REPO_ROOT / "analysis" / "stratified_analysis_report.txt"
OUTPUT_GRID_CSV = REPO_ROOT / "analysis" / "stratified_analysis_grid.csv"

# Minimum observations required before we report a correlation.
# Cells below this threshold are flagged as "insufficient data".
MIN_OBS_WARN = 5    # warn but still compute
MIN_OBS_SKIP = 3    # skip entirely — result would be meaningless

# CRIT pillars stored in crit_scores.json
PILLARS = ("LV", "ES", "AC", "CA")
PILLAR_NAMES = {
    "LV": "logical_validity",
    "ES": "evidential_support",
    "AC": "alternative_consideration",
    "CA": "causal_alignment",
}

# ===========================================================================
# Part A — Data loading
# ===========================================================================

def load_regime_manifest(path: Path = REGIME_MANIFEST_PATH) -> dict[str, Any]:
    """Load regime_manifest.yaml and return the parsed dict.

    Returns the raw YAML dict with keys 'patterns' and 'quarter_fallback'.
    Raises FileNotFoundError if the manifest does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Regime manifest not found at {path}. "
            "Did you run from the repo root with config/regime_manifest.yaml present?"
        )
    with path.open() as fh:
        return yaml.safe_load(fh)


def lookup_regime(scenario_name: str, invest_quarter: str, manifest_data: dict) -> str:
    """Resolve a regime label for a run.

    Strategy (first match wins):
      1. Check scenario_name against each pattern in manifest_data['patterns']
         using case-insensitive substring matching.
      2. Fall back to invest_quarter lookup in manifest_data['quarter_fallback'].
      3. Return "unknown" if neither matches.

    Args:
        scenario_name:   Basename of the scenario YAML without extension,
                         e.g. "2022Q1_inflation_shock".
        invest_quarter:  The invest_quarter string from manifest.json,
                         e.g. "2022-09-30" or "2022Q3".
        manifest_data:   Parsed regime_manifest.yaml dict.

    Returns:
        A regime label string.
    """
    name_lower = scenario_name.lower()

    # Pass 1: pattern matching on scenario filename
    for entry in manifest_data.get("patterns", []):
        if entry["pattern"].lower() in name_lower:
            return entry["regime"]

    # Pass 2: quarter-end date fallback
    fallback = manifest_data.get("quarter_fallback", {})
    # invest_quarter may be "2022-09-30" or "2022Q3" — normalise to YYYY-MM-DD
    normalised = _normalise_quarter(invest_quarter)
    if normalised in fallback:
        return fallback[normalised]

    return "unknown"


def _normalise_quarter(invest_quarter: str) -> str:
    """Convert "2022Q3" or "2022-09-30" to ISO quarter-end date "2022-09-30".

    Quarter-end dates: Q1=03-31, Q2=06-30, Q3=09-30, Q4=12-31.
    If the string is already YYYY-MM-DD, return it unchanged.
    """
    if not invest_quarter:
        return ""
    s = invest_quarter.strip()
    # Already looks like an ISO date
    if len(s) == 10 and s[4] == "-":
        return s
    # Compact form like "2022Q3"
    if len(s) == 6 and "Q" in s.upper():
        year, q = s[:4], s[5]
        ends = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}
        return f"{year}-{ends.get(q, '12-31')}"
    return s


def extract_scenario_name(manifest: dict) -> str:
    """Pull the scenario name out of a manifest dict.

    Tries config_paths[1] first (the convention used by ablation runs),
    then falls back to the invest_quarter string if no path is available.

    Returns the basename without extension, e.g. "2022Q1_inflation_shock".
    """
    config_paths = manifest.get("config_paths", [])
    if len(config_paths) >= 2 and config_paths[1]:
        return Path(config_paths[1]).stem
    # Some manifests only store the scenario via invest_quarter
    iq = manifest.get("invest_quarter", "")
    return _normalise_quarter(str(iq)) if iq else "unknown_scenario"


def _load_last_round_crit(run_dir: Path) -> dict | None:
    """Load CRIT scores from the last completed debate round.

    Prefers the aggregated file final/pid_crit_all_rounds.json.
    Falls back to scanning rounds/round_NNN/metrics/crit_scores.json files
    and picking the highest-numbered round.

    Returns a dict with keys:
        rho_bar     — float, aggregate quality score
        agent_scores — dict: {role: {rho_i, pillar_scores: {LV,ES,AC,CA}}}
    or None if no CRIT data is found.
    """
    # --- Try aggregated file first ---
    agg_path = run_dir / "final" / "pid_crit_all_rounds.json"
    if agg_path.exists():
        try:
            rounds_data = json.loads(agg_path.read_text())
            if rounds_data:
                last = rounds_data[-1]
                crit = last.get("crit", {})
                if crit:
                    return _normalise_crit_block(crit)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass  # fall through to per-round scan

    # --- Fall back: per-round crit_scores.json files ---
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return None

    round_dirs = sorted(
        d for d in rounds_dir.iterdir()
        if d.is_dir() and d.name.startswith("round_")
    )
    if not round_dirs:
        return None

    for rdir in reversed(round_dirs):
        crit_path = rdir / "metrics" / "crit_scores.json"
        if crit_path.exists():
            try:
                raw = json.loads(crit_path.read_text())
                return _normalise_crit_block_from_scores_file(raw)
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def _normalise_crit_block(crit: dict) -> dict:
    """Normalise the 'crit' sub-dict from pid_crit_all_rounds.json.

    The aggregated file stores per-agent data under crit['agents'][role],
    with pillars under 'pillars' (not 'pillar_scores').
    Normalise to: {rho_bar, agent_scores: {role: {rho_i, pillar_scores: {LV,ES,AC,CA}}}}.
    """
    agent_scores: dict[str, dict] = {}
    for role, adata in crit.get("agents", {}).items():
        pillar_scores = adata.get("pillars") or adata.get("pillar_scores") or {}
        agent_scores[role] = {
            "rho_i": adata.get("rho_i"),
            "pillar_scores": pillar_scores,
        }
    return {
        "rho_bar": crit.get("rho_bar"),
        "agent_scores": agent_scores,
    }


def _normalise_crit_block_from_scores_file(raw: dict) -> dict:
    """Normalise a crit_scores.json file dict.

    That file stores per-agent data under raw['agent_scores'][role],
    with pillars under 'pillar_scores'.
    """
    agent_scores: dict[str, dict] = {}
    for role, adata in raw.get("agent_scores", {}).items():
        agent_scores[role] = {
            "rho_i": adata.get("rho_i"),
            "pillar_scores": adata.get("pillar_scores") or {},
        }
    return {
        "rho_bar": raw.get("rho_bar"),
        "agent_scores": agent_scores,
    }


def load_financial_metrics(run_dir: Path) -> dict | None:
    """Load financial performance metrics for a run.

    Looks for run_dir/_dashboard/financial_metrics.json.
    Returns dict with at minimum:
        daily_metrics_annualized_sharpe  — float
        daily_metrics_total_return_pct   — float
        (optionally) daily_metrics_max_drawdown_pct — float
    Returns None if the file is missing or unreadable.
    """
    fin_path = run_dir / "_dashboard" / "financial_metrics.json"
    if not fin_path.exists():
        return None
    try:
        data = json.loads(fin_path.read_text())
        if (
            data.get("daily_metrics_annualized_sharpe") is None
            or data.get("daily_metrics_total_return_pct") is None
        ):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def load_run_as_agent_rows(
    run_dir: Path,
    manifest_data: dict,
) -> list[dict]:
    """Convert one run directory into a list of per-agent rows.

    Each row represents one agent in one run and contains:
        run_id, experiment, scenario, regime, role,
        rho_i, LV, ES, AC, CA,
        rho_bar,                   # aggregate for the run (same across roles)
        sharpe, total_return,      # portfolio outcomes (same across roles)
        drawdown

    Returns an empty list if the run is missing financial or CRIT data.
    This is expected for fixture/mock runs — callers should handle gracefully.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    fin = load_financial_metrics(run_dir)
    if fin is None:
        return []

    crit = _load_last_round_crit(run_dir)
    if crit is None or not crit.get("agent_scores"):
        return []

    scenario_name = extract_scenario_name(manifest)
    invest_quarter = str(manifest.get("invest_quarter", ""))
    regime = lookup_regime(scenario_name, invest_quarter, manifest_data)

    sharpe = fin["daily_metrics_annualized_sharpe"]
    total_return = fin["daily_metrics_total_return_pct"]
    drawdown = fin.get("daily_metrics_max_drawdown_pct")

    rows: list[dict] = []
    for role, adata in crit["agent_scores"].items():
        ps = adata.get("pillar_scores") or {}
        row = {
            "run_id": run_dir.name,
            "experiment": run_dir.parent.name,
            "scenario": scenario_name,
            "regime": regime,
            "role": role,
            "rho_i": adata.get("rho_i"),
            "LV": ps.get("LV"),
            "ES": ps.get("ES"),
            "AC": ps.get("AC"),
            "CA": ps.get("CA"),
            "rho_bar": crit.get("rho_bar"),
            "sharpe": sharpe,
            "total_return": total_return,
            "drawdown": drawdown,
        }
        rows.append(row)

    return rows


def load_all_agent_rows(
    runs_root: Path,
    experiments: list[str] | None = None,
    manifest_data: dict | None = None,
) -> pd.DataFrame:
    """Walk runs_root and collect per-agent rows from every completed run.

    Args:
        runs_root:      Root directory containing experiment subdirs.
        experiments:    If given, only load these experiment names.
                        If None, load all subdirs of runs_root.
        manifest_data:  Parsed regime_manifest.yaml. Loaded from default
                        path if not supplied.

    Returns:
        A DataFrame with columns: run_id, experiment, scenario, regime,
        role, rho_i, LV, ES, AC, CA, rho_bar, sharpe, total_return, drawdown.
        Empty DataFrame if no valid runs are found.
    """
    if manifest_data is None:
        manifest_data = load_regime_manifest()

    if not runs_root.exists():
        print(f"  [WARN] runs_root does not exist: {runs_root}", file=sys.stderr)
        return pd.DataFrame()

    all_rows: list[dict] = []

    exp_dirs = (
        [runs_root / e for e in experiments]
        if experiments
        else sorted(p for p in runs_root.iterdir() if p.is_dir())
    )

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            print(f"  [WARN] Experiment dir not found: {exp_dir}", file=sys.stderr)
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            rows = load_run_as_agent_rows(run_dir, manifest_data)
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Cast numeric columns; coerce bad values to NaN
    for col in ("rho_i", "LV", "ES", "AC", "CA", "rho_bar", "sharpe", "total_return", "drawdown"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ===========================================================================
# Part B — Per-role analysis
# ===========================================================================

def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Pearson and Spearman r for two arrays, return a result dict.

    Returns dict with keys: n, pearson_r, pearson_p, spearman_r, spearman_p.
    Returns None when n < MIN_OBS_SKIP.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    n = int(mask.sum())
    if n < MIN_OBS_SKIP:
        return {"n": n, "pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "insufficient": True}

    pr, pp = stats.pearsonr(x_c, y_c)
    sr, sp = stats.spearmanr(x_c, y_c)
    return {
        "n": n,
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 4),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 4),
        "insufficient": n < MIN_OBS_WARN,
    }


def corr_by_role(df: pd.DataFrame, outcome: str = "sharpe") -> dict[str, dict]:
    """For each agent role, compute rho_i correlation with `outcome`.

    Args:
        df:      Agent-level DataFrame from load_all_agent_rows().
        outcome: Column to use as the dependent variable ("sharpe" or
                 "total_return").

    Returns:
        {role: _corr_pair result dict}
    """
    results: dict[str, dict] = {}
    for role in sorted(df["role"].dropna().unique()):
        sub = df[df["role"] == role]
        results[role] = _corr_pair(
            sub["rho_i"].values,
            sub[outcome].values,
        )
    return results


def pillar_corr_by_role(df: pd.DataFrame, outcome: str = "sharpe") -> dict[str, dict[str, dict]]:
    """For each role, compute per-pillar (LV/ES/AC/CA) correlation with `outcome`.

    Returns:
        {role: {pillar: _corr_pair result dict}}
    """
    results: dict[str, dict[str, dict]] = {}
    for role in sorted(df["role"].dropna().unique()):
        sub = df[df["role"] == role]
        results[role] = {}
        for pillar in PILLARS:
            if pillar not in sub.columns:
                continue
            results[role][pillar] = _corr_pair(
                sub[pillar].values,
                sub[outcome].values,
            )
    return results


# ===========================================================================
# Part C — Per-regime analysis
# ===========================================================================

def corr_by_regime(df: pd.DataFrame, outcome: str = "sharpe") -> dict[str, dict]:
    """For each market regime, compute rho_bar correlation with `outcome`.

    Uses rho_bar (aggregate) rather than per-role rho_i so the regime slice
    is independent of which roles happen to be in a given run.

    Returns:
        {regime: _corr_pair result dict}
    """
    results: dict[str, dict] = {}
    for regime in sorted(df["regime"].dropna().unique()):
        # Deduplicate: one row per run per regime (avoid counting the same
        # financial outcome multiple times because we have one row per role).
        sub = df[df["regime"] == regime].drop_duplicates(subset=["run_id"])
        results[regime] = _corr_pair(
            sub["rho_bar"].values,
            sub[outcome].values,
        )
    return results


# ===========================================================================
# Part D — Cross-stratification [role × regime]
# ===========================================================================

def corr_grid(df: pd.DataFrame, outcome: str = "sharpe") -> dict[str, dict[str, dict]]:
    """Build a 2-D correlation table: rows = roles, columns = regimes.

    Each cell is _corr_pair(rho_i for that role in that regime, outcome).

    Returns:
        {role: {regime: _corr_pair result dict}}
    """
    roles = sorted(df["role"].dropna().unique())
    regimes = sorted(df["regime"].dropna().unique())
    grid: dict[str, dict[str, dict]] = {}
    for role in roles:
        grid[role] = {}
        for regime in regimes:
            sub = df[(df["role"] == role) & (df["regime"] == regime)]
            grid[role][regime] = _corr_pair(
                sub["rho_i"].values,
                sub[outcome].values,
            )
    return grid


# ===========================================================================
# Part E — Formatting & output
# ===========================================================================

def _fmt_corr(result: dict, show_p: bool = True) -> str:
    """Format a _corr_pair result as a compact string for tables.

    Examples:
        "+0.42* (n=12)"   — significant
        "+0.12  (n=8)"    — not significant
        "n<3"             — skipped
        "n=4 warn"        — computed but flagged
    """
    if result.get("insufficient") and result.get("pearson_r") is None:
        return f"n={result['n']} (skip)"
    r = result.get("pearson_r")
    p = result.get("pearson_p")
    n = result.get("n", 0)
    if r is None:
        return f"n={n} (skip)"
    sig = " *" if (p is not None and p < 0.05) else "  "
    warn = " !" if result.get("insufficient") else ""
    return f"{r:+.2f}{sig}(n={n}){warn}"


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _subsection(title: str) -> str:
    return f"\n--- {title} ---"


def format_role_table(
    role_corr: dict[str, dict],
    pillar_corr: dict[str, dict[str, dict]],
    outcome: str,
) -> str:
    """Format per-role correlation table as a text block."""
    lines = [_subsection(f"Per-role rho_i vs {outcome}")]
    header = f"  {'Role':<14} {'Pearson r':>10} {'p':>8} {'Spearman r':>11} {'n':>5}"
    lines.append(header)
    lines.append("  " + "-" * 52)
    for role, res in sorted(role_corr.items()):
        if res.get("pearson_r") is None:
            lines.append(f"  {role:<14} (insufficient data, n={res['n']})")
            continue
        sig = " *" if (res["pearson_p"] or 1.0) < 0.05 else ""
        warn = " [n<5 warn]" if res.get("insufficient") else ""
        lines.append(
            f"  {role:<14} {res['pearson_r']:>+10.4f} {res['pearson_p']:>8.4f}"
            f" {res['spearman_r']:>+11.4f} {res['n']:>5}{sig}{warn}"
        )

    # Pillar breakdown
    lines.append(_subsection(f"Per-role pillar (LV/ES/AC/CA) vs {outcome} — Pearson r"))
    pillar_header = f"  {'Role':<12}" + "".join(f" {p:>12}" for p in PILLARS)
    lines.append(pillar_header)
    lines.append("  " + "-" * (12 + 13 * len(PILLARS)))
    all_roles = sorted(pillar_corr.keys())
    for role in all_roles:
        row = f"  {role:<12}"
        for pillar in PILLARS:
            res = pillar_corr.get(role, {}).get(pillar, {})
            row += f" {_fmt_corr(res):>12}"
        lines.append(row)
    lines.append("")
    lines.append("  Legend: * = p<0.05,  ! = n<5 (interpret cautiously)")
    return "\n".join(lines)


def format_regime_table(regime_corr: dict[str, dict], outcome: str) -> str:
    """Format per-regime rho_bar correlation table."""
    lines = [_subsection(f"Per-regime rho_bar vs {outcome}  (one row per run, deduplicated)")]
    header = f"  {'Regime':<20} {'Pearson r':>10} {'p':>8} {'Spearman r':>11} {'n':>5}"
    lines.append(header)
    lines.append("  " + "-" * 58)
    for regime, res in sorted(regime_corr.items()):
        if res.get("pearson_r") is None:
            lines.append(f"  {regime:<20} (insufficient data, n={res['n']})")
            continue
        sig = " *" if (res["pearson_p"] or 1.0) < 0.05 else ""
        warn = " [n<5 warn]" if res.get("insufficient") else ""
        lines.append(
            f"  {regime:<20} {res['pearson_r']:>+10.4f} {res['pearson_p']:>8.4f}"
            f" {res['spearman_r']:>+11.4f} {res['n']:>5}{sig}{warn}"
        )
    return "\n".join(lines)


def format_grid_table(grid: dict[str, dict[str, dict]], outcome: str) -> str:
    """Format the role × regime correlation heatmap as an ASCII table.

    Each cell shows: "+r.rr*(n=N)" or "n=N (skip)".
    """
    roles = sorted(grid.keys())
    # Collect all regimes that appear in the grid
    regimes = sorted({r for role_dict in grid.values() for r in role_dict})

    col_w = 16
    lines = [_subsection(f"Cross-stratification grid: rho_i vs {outcome}  [role × regime]")]
    lines.append("  (Pearson r; * = p<0.05; ! = n<5)")
    # Header row
    header = f"  {'Role':<12}" + "".join(f" {reg:<{col_w}}" for reg in regimes)
    lines.append(header)
    lines.append("  " + "-" * (12 + (col_w + 1) * len(regimes)))
    for role in roles:
        row = f"  {role:<12}"
        for regime in regimes:
            cell = grid.get(role, {}).get(regime, {"n": 0, "pearson_r": None})
            row += f" {_fmt_corr(cell):<{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def write_grid_csv(
    grid: dict[str, dict[str, dict]],
    outcome: str,
    path: Path,
) -> None:
    """Write the role×regime grid to a CSV file for downstream use."""
    roles = sorted(grid.keys())
    regimes = sorted({r for role_dict in grid.values() for r in role_dict})

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        # Header: outcome, role, regime, pearson_r, pearson_p, spearman_r, spearman_p, n
        writer.writerow(["outcome", "role", "regime",
                         "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n"])
        for role in roles:
            for regime in regimes:
                res = grid.get(role, {}).get(regime, {})
                writer.writerow([
                    outcome, role, regime,
                    res.get("pearson_r", ""),
                    res.get("pearson_p", ""),
                    res.get("spearman_r", ""),
                    res.get("spearman_p", ""),
                    res.get("n", 0),
                ])


# ===========================================================================
# Main report
# ===========================================================================

def _describe(df: pd.DataFrame, col: str) -> str:
    v = df[col].dropna()
    if v.empty:
        return "(no data)"
    return (
        f"mean={v.mean():.4f}  std={v.std():.4f}  "
        f"min={v.min():.4f}  max={v.max():.4f}  n={len(v)}"
    )


def build_report(df: pd.DataFrame) -> list[str]:
    """Build the full text report and return it as a list of lines."""
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    # ------------------------------------------------------------------
    add(_section("STRATIFIED CRIT–FINANCIAL CORRELATION ANALYSIS"))
    add(f"Loaded {len(df)} agent-role rows from "
        f"{df['run_id'].nunique()} unique runs.")
    add("")
    add("Runs per experiment:")
    for exp, cnt in df.groupby("experiment")["run_id"].nunique().items():
        add(f"  {exp}: {cnt} runs")
    add("")
    add("Runs per regime:")
    for regime, cnt in df.groupby("regime")["run_id"].nunique().items():
        add(f"  {regime}: {cnt} runs")
    add("")
    add("Agent-rows per role:")
    for role, cnt in df["role"].value_counts().items():
        add(f"  {role}: {cnt} rows")
    add("")
    add("Descriptive statistics:")
    for col in ("rho_i", "rho_bar", "sharpe", "total_return"):
        add(f"  {col:>14s}: {_describe(df, col)}")

    # ------------------------------------------------------------------
    for outcome in ("sharpe", "total_return"):
        add(_section(f"ROLE STRATIFICATION  (outcome = {outcome})"))
        role_corr = corr_by_role(df, outcome)
        pillar_corr = pillar_corr_by_role(df, outcome)
        add(format_role_table(role_corr, pillar_corr, outcome))

    # ------------------------------------------------------------------
    for outcome in ("sharpe", "total_return"):
        add(_section(f"REGIME STRATIFICATION  (outcome = {outcome})"))
        regime_corr = corr_by_regime(df, outcome)
        add(format_regime_table(regime_corr, outcome))

    # ------------------------------------------------------------------
    for outcome in ("sharpe", "total_return"):
        add(_section(f"CROSS-STRATIFICATION GRID  (outcome = {outcome})"))
        grid = corr_grid(df, outcome)
        add(format_grid_table(grid, outcome))

    # ------------------------------------------------------------------
    add(_section("INTERPRETATION GUIDE"))
    add(
        "Each cell shows Pearson r between rho_i (agent CRIT score) and the\n"
        "portfolio outcome, within the stratum defined by that cell's role and\n"
        "regime.  * = p<0.05.  ! = n<5 (compute but do not claim significance).\n"
        "\n"
        "Key questions to ask of this table:\n"
        "  1. Is any role consistently correlated with performance?\n"
        "     (Row with consistent sign/magnitude across regimes)\n"
        "  2. Is any regime where CRIT aligns with performance?\n"
        "     (Column with consistent sign/magnitude across roles)\n"
        "  3. Are there cells with opposite signs?\n"
        "     (CRIT helps in some contexts, hurts in others)\n"
        "\n"
        "Note: financial outcomes are portfolio-level (same across roles in a run).\n"
        "Per-role CRIT predicts portfolio outcome — not per-role contribution.\n"
        "Interpret carefully: high n and consistent sign across regimes is the\n"
        "strongest evidence of a role-specific CRIT–performance relationship."
    )
    add("")
    add("* = p < 0.05 (two-tailed).  All correlations are observational.")

    return lines


def main(runs_root: Path = DEFAULT_RUNS_ROOT, experiments: list[str] | None = None) -> None:
    print("Loading regime manifest...")
    try:
        manifest_data = load_regime_manifest()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading agent rows from {runs_root} ...")
    df = load_all_agent_rows(runs_root, experiments, manifest_data)

    if df.empty:
        print(
            "\nNo data loaded. Possible reasons:\n"
            "  - The runs_root directory does not exist or has no run_ subdirs.\n"
            "  - Run dirs are missing _dashboard/financial_metrics.json.\n"
            "  - Run dirs are missing CRIT data.\n"
            "\n"
            f"runs_root checked: {runs_root}\n"
            "Try: python analysis/stratified_analysis.py --runs-root results_juli/"
        )
        sys.exit(0)

    print(f"Loaded {len(df)} agent rows from {df['run_id'].nunique()} runs.")

    lines = build_report(df)
    report_text = "\n".join(lines) + "\n"

    print(report_text)

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report_text)
    print(f"\n[Report written to {OUTPUT_REPORT}]")

    # Write CSV grid for both outcomes
    for outcome in ("sharpe", "total_return"):
        grid = corr_grid(df, outcome)
        csv_path = OUTPUT_GRID_CSV.with_name(
            OUTPUT_GRID_CSV.stem + f"_{outcome}" + OUTPUT_GRID_CSV.suffix
        )
        write_grid_csv(grid, outcome, csv_path)
        print(f"[Grid CSV written to {csv_path}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified CRIT–financial correlation analysis (Rec #3)"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Root directory of run logs (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment names to include (default: all subdirs of runs-root)",
    )
    args = parser.parse_args()
    main(runs_root=args.runs_root, experiments=args.experiments)
