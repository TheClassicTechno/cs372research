#!/usr/bin/env python3
"""
Qualitative Failure Analysis — Professor Recommendation #5
===========================================================
Identifies and formats case studies from the run logs across three categories:

  1. High-CRIT / low-return  — reasoning scored well but portfolio underperformed
  2. Low-CRIT  / high-return — reasoning scored poorly but portfolio did well
  3. Intervention cases      — runs where the JSD/causal intervention helped or hurt

For each case the script extracts:
  - CRIT scores (aggregate + per-pillar per role) and CRIT explanations
  - Reasoning excerpts (proposal → critique → revision chain)
  - Final portfolio allocation
  - Financial outcome (Sharpe, return, drawdown)
  - A structured template with [HUMAN ANALYSIS NEEDED] markers

The output is a report file ready to paste into the paper's qualitative section.
A human must fill in the narrative sections — those require judgment the script
cannot provide.

Output
------
  analysis/failure_analysis_report.txt

Usage
-----
  python analysis/failure_analysis.py [--runs-root PATH] [--n N]

  --n   number of cases to select per category (default: 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse data loading from stratified_analysis
from analysis.stratified_analysis import (
    load_all_agent_rows,
    load_regime_manifest,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "logging" / "runs"
OUTPUT_REPORT = REPO_ROOT / "analysis" / "failure_analysis_report.txt"

# How many chars of reasoning text to include per excerpt in the report.
# Long enough to see the argument; short enough to stay readable.
EXCERPT_MAX_CHARS = 600

# Quartile thresholds used to define "high" and "low" performance/quality.
# 0.25 = bottom quartile, 0.75 = top quartile.
QUARTILE_HIGH = 0.75
QUARTILE_LOW  = 0.25


# ===========================================================================
# Part A — Case selection
# ===========================================================================

def build_run_level_df(
    agent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the per-agent-row DataFrame to one row per run.

    The financial outcomes (sharpe, total_return, drawdown) and rho_bar are
    the same for every agent in a run, so we just take the first row per run.
    We also carry experiment, scenario, regime, and condition.

    Args:
        agent_df: Output of load_all_agent_rows() — one row per agent per run.

    Returns:
        DataFrame with one row per run_id.
    """
    keep = ["run_id", "experiment", "scenario", "regime",
            "rho_bar", "sharpe", "total_return", "drawdown"]
    available = [c for c in keep if c in agent_df.columns]
    run_df = (
        agent_df[available]
        .drop_duplicates(subset=["run_id"])
        .reset_index(drop=True)
    )
    for col in ("rho_bar", "sharpe", "total_return", "drawdown"):
        if col in run_df.columns:
            run_df[col] = pd.to_numeric(run_df[col], errors="coerce")
    return run_df


def select_quadrant_cases(
    run_df: pd.DataFrame,
    n: int = 3,
) -> dict[str, list[str]]:
    """Select run_ids falling into each CRIT × performance quadrant.

    Quadrant definitions (using quartile thresholds):
      high_crit_low_return  — rho_bar >= 75th pct  AND  sharpe <= 25th pct
      low_crit_high_return  — rho_bar <= 25th pct  AND  sharpe >= 75th pct
      high_crit_high_return — rho_bar >= 75th pct  AND  sharpe >= 75th pct
        (reference cases: where CRIT and performance align positively)

    Returns at most `n` run_ids per quadrant, sorted by most extreme values first.
    Returns empty lists for quadrants with fewer candidates than MIN_OBS_SKIP.

    Args:
        run_df: One row per run, must have columns rho_bar and sharpe.
        n:      Maximum cases to return per quadrant.

    Returns:
        Dict mapping quadrant label → list of run_id strings.
    """
    if run_df.empty or "rho_bar" not in run_df.columns or "sharpe" not in run_df.columns:
        return {
            "high_crit_low_return": [],
            "low_crit_high_return": [],
            "high_crit_high_return": [],
        }
    clean = run_df.dropna(subset=["rho_bar", "sharpe"])
    if clean.empty:
        return {
            "high_crit_low_return": [],
            "low_crit_high_return": [],
            "high_crit_high_return": [],
        }

    rho_hi  = clean["rho_bar"].quantile(QUARTILE_HIGH)
    rho_lo  = clean["rho_bar"].quantile(QUARTILE_LOW)
    srp_hi  = clean["sharpe"].quantile(QUARTILE_HIGH)
    srp_lo  = clean["sharpe"].quantile(QUARTILE_LOW)

    def _pick(mask: pd.Series, sort_col: str, ascending: bool) -> list[str]:
        subset = clean[mask].sort_values(sort_col, ascending=ascending)
        return subset["run_id"].head(n).tolist()

    return {
        # High CRIT, low return — the interesting failure mode
        "high_crit_low_return": _pick(
            (clean["rho_bar"] >= rho_hi) & (clean["sharpe"] <= srp_lo),
            sort_col="sharpe",
            ascending=True,   # worst Sharpe first
        ),
        # Low CRIT, high return — where bad reasoning still won
        "low_crit_high_return": _pick(
            (clean["rho_bar"] <= rho_lo) & (clean["sharpe"] >= srp_hi),
            sort_col="sharpe",
            ascending=False,  # best Sharpe first
        ),
        # Reference: CRIT and performance both high
        "high_crit_high_return": _pick(
            (clean["rho_bar"] >= rho_hi) & (clean["sharpe"] >= srp_hi),
            sort_col="rho_bar",
            ascending=False,
        ),
    }


def select_intervention_cases(
    runs_root: Path,
    run_df: pd.DataFrame,
    n: int = 2,
) -> dict[str, list[str]]:
    """Identify intervention-enabled runs and split into helped vs hurt.

    A run is "intervention-enabled" when its manifest.json contains
    intervention_config.enabled == true.

    "Helped" = intervention run with sharpe > median sharpe of non-intervention
               runs on the same scenario.
    "Hurt"   = intervention run with sharpe < median sharpe of non-intervention
               runs on the same scenario.

    When no paired baseline exists for a run, that run is skipped.

    Args:
        runs_root: Root of all run directories.
        run_df:    One-row-per-run DataFrame with scenario and sharpe.
        n:         Max cases per category.

    Returns:
        Dict with keys "intervention_helped" and "intervention_hurt".
    """
    helped: list[tuple[float, str]] = []
    hurt:   list[tuple[float, str]] = []

    for _, row in run_df.iterrows():
        run_id = row["run_id"]
        scenario = row.get("scenario", "")
        sharpe = row.get("sharpe")
        if pd.isna(sharpe):
            continue

        # Find the run directory
        run_dir = _find_run_dir(runs_root, run_id)
        if run_dir is None:
            continue

        manifest = _load_json_safe(run_dir / "manifest.json")
        if manifest is None:
            continue

        intervention_cfg = manifest.get("intervention_config", {})
        if not intervention_cfg.get("enabled", False):
            continue

        # Compare against non-intervention runs on the same scenario
        same_scenario = run_df[run_df["scenario"] == scenario]
        baseline_runs = []
        for _, brow in same_scenario.iterrows():
            if brow["run_id"] == run_id:
                continue
            bdir = _find_run_dir(runs_root, brow["run_id"])
            if bdir is None:
                continue
            bman = _load_json_safe(bdir / "manifest.json")
            if bman is None:
                continue
            if not bman.get("intervention_config", {}).get("enabled", False):
                baseline_runs.append(brow["sharpe"])

        if not baseline_runs:
            continue

        baseline_median = float(np.median(baseline_runs))
        delta = sharpe - baseline_median
        if delta > 0:
            helped.append((delta, run_id))
        else:
            hurt.append((delta, run_id))

    helped.sort(key=lambda x: -x[0])
    hurt.sort(key=lambda x: x[0])

    return {
        "intervention_helped": [r for _, r in helped[:n]],
        "intervention_hurt":   [r for _, r in hurt[:n]],
    }


def _find_run_dir(runs_root: Path, run_id: str) -> Path | None:
    """Search runs_root for a directory named run_id under any experiment subdir."""
    for exp_dir in runs_root.iterdir():
        if not exp_dir.is_dir():
            continue
        candidate = exp_dir / run_id
        if candidate.is_dir():
            return candidate
    return None


# ===========================================================================
# Part B — Trace loading
# ===========================================================================

def _load_json_safe(path: Path) -> dict | list | None:
    """Load a JSON file, returning None on any error."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def truncate_text(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Truncate text to max_chars, appending '...[truncated]' if cut.

    Tries to cut at a sentence boundary ('. ') to avoid mid-sentence breaks.
    Falls back to hard truncation if no boundary found.
    """
    if not text or len(text) <= max_chars:
        return text
    cutoff = text.rfind(". ", 0, max_chars)
    if cutoff > max_chars // 2:
        return text[: cutoff + 1] + " ...[truncated]"
    return text[:max_chars] + "...[truncated]"


def load_round_trace(run_dir: Path, round_num: int) -> dict:
    """Load all artifacts for one debate round.

    Returns a dict with:
      round          — int
      proposals      — {role: {text, portfolio}}
      critiques      — {role: [critique dicts]}
      revisions      — {role: {text, portfolio}}
      crit_scores    — raw crit_scores.json dict (or {})
      js_divergence  — float or None
    """
    round_dir = run_dir / "rounds" / f"round_{round_num:03d}"
    if not round_dir.exists():
        return {}

    trace: dict[str, Any] = {"round": round_num, "proposals": {}, "critiques": {}, "revisions": {}}

    # Proposals
    for role_dir in sorted((round_dir / "proposals").iterdir()) if (round_dir / "proposals").exists() else []:
        role = role_dir.name
        trace["proposals"][role] = {
            "text": _read_text_safe(role_dir / "response.txt"),
            "portfolio": _load_json_safe(role_dir / "portfolio.json"),
        }

    # Critiques
    for role_dir in sorted((round_dir / "critiques").iterdir()) if (round_dir / "critiques").exists() else []:
        role = role_dir.name
        raw = _load_json_safe(role_dir / "response.json")
        trace["critiques"][role] = raw.get("critiques", []) if isinstance(raw, dict) else []

    # Revisions
    for role_dir in sorted((round_dir / "revisions").iterdir()) if (round_dir / "revisions").exists() else []:
        role = role_dir.name
        trace["revisions"][role] = {
            "text": _read_text_safe(role_dir / "response.txt"),
            "portfolio": _load_json_safe(role_dir / "portfolio.json"),
        }

    # Metrics
    metrics_dir = round_dir / "metrics"
    trace["crit_scores"] = _load_json_safe(metrics_dir / "crit_scores.json") or {}
    js_raw = _load_json_safe(metrics_dir / "js_divergence.json")
    trace["js_divergence"] = js_raw.get("js_divergence") if isinstance(js_raw, dict) else None

    return trace


def load_run_full_trace(run_dir: Path) -> dict:
    """Load everything from a run directory needed for a case study.

    Returns:
      manifest           — manifest.json dict
      financial          — _dashboard/financial_metrics.json dict
      rounds             — list of round trace dicts (load_round_trace)
      final_portfolio    — final/final_portfolio.json dict
      pid_crit_all       — final/pid_crit_all_rounds.json list
    """
    trace: dict[str, Any] = {
        "manifest":        _load_json_safe(run_dir / "manifest.json") or {},
        "financial":       _load_json_safe(run_dir / "_dashboard" / "financial_metrics.json") or {},
        "final_portfolio": _load_json_safe(run_dir / "final" / "final_portfolio.json") or {},
        "pid_crit_all":    _load_json_safe(run_dir / "final" / "pid_crit_all_rounds.json") or [],
        "rounds":          [],
    }

    rounds_dir = run_dir / "rounds"
    if rounds_dir.exists():
        for rdir in sorted(rounds_dir.iterdir()):
            if rdir.is_dir() and rdir.name.startswith("round_"):
                try:
                    rnum = int(rdir.name.split("_")[1])
                    trace["rounds"].append(load_round_trace(run_dir, rnum))
                except (ValueError, IndexError):
                    pass

    return trace


def _read_text_safe(path: Path) -> str:
    """Read a text file, returning empty string on any error."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ===========================================================================
# Part C — Case formatting
# ===========================================================================

def _fmt_section(title: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}"


def _fmt_divider() -> str:
    return "-" * 80


def format_crit_scores(crit_data: dict) -> str:
    """Format CRIT scores from a crit_scores.json dict into a readable block.

    Covers: rho_bar, per-role rho_i, per-pillar scores, CRIT explanations.
    """
    lines = []
    rho_bar = crit_data.get("rho_bar", "N/A")
    lines.append(f"  Aggregate rho_bar: {rho_bar}")

    agent_scores = crit_data.get("agent_scores", {})
    for role, adata in sorted(agent_scores.items()):
        ps = adata.get("pillar_scores", {})
        pillar_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(ps.items()) if v is not None)
        lines.append(f"  {role:<12} rho_i={adata.get('rho_i', '?'):.3f}  {pillar_str}")

    # CRIT explanations — the most useful qualitative signal
    lines.append("")
    lines.append("  CRIT Explanations (why each pillar scored as it did):")
    for role, adata in sorted(agent_scores.items()):
        explanations = adata.get("explanations", {})
        if not explanations:
            continue
        lines.append(f"    [{role}]")
        for pillar_key, text in explanations.items():
            label = pillar_key.replace("_", " ").title()
            lines.append(f"      {label}: {truncate_text(str(text), 200)}")

    return "\n".join(lines)


def format_portfolio(portfolio: dict) -> str:
    """Format a portfolio allocation dict into a sorted readable string."""
    if not portfolio:
        return "  (no portfolio data)"
    sorted_items = sorted(portfolio.items(), key=lambda x: -float(x[1] or 0))
    parts = [f"{ticker}: {weight:.1%}" for ticker, weight in sorted_items if float(weight or 0) > 0]
    return "  " + "  |  ".join(parts)


def format_reasoning_excerpt(round_trace: dict, role: str, phase: str = "revision") -> str:
    """Extract and truncate one agent's reasoning text from a round trace.

    Args:
        round_trace: Output of load_round_trace().
        role:        Agent role to extract (e.g. "macro").
        phase:       "proposal", "revision", or "critique".

    Returns:
        Truncated text string, or a note if not found.
    """
    if phase == "critique":
        critiques = round_trace.get("critiques", {}).get(role, [])
        if not critiques:
            return "  (no critique found)"
        c = critiques[0]
        lines = [
            f"  Target: {c.get('target_role', '?')}  Claim: {c.get('target_claim', '?')}",
            f"  Objection: {truncate_text(c.get('objection', ''), 300)}",
            f"  Suggested: {truncate_text(c.get('suggested_adjustment', ''), 200)}",
            f"  Evidence:  {' '.join(c.get('counter_evidence', []))}",
        ]
        return "\n".join(lines)

    phase_data = round_trace.get(f"{phase}s", {}).get(role, {})
    text = phase_data.get("text", "") if isinstance(phase_data, dict) else ""
    if not text:
        return f"  (no {phase} text found for {role})"
    return "  " + truncate_text(text).replace("\n", "\n  ")


def format_case_study(
    case_type: str,
    run_id: str,
    run_row: pd.Series,
    full_trace: dict,
) -> str:
    """Format one complete case study for the failure analysis report.

    Produces a structured block with CRIT scores, reasoning excerpts,
    portfolio allocation, financial outcome, and human-analysis markers.

    Args:
        case_type:  Label such as "high_crit_low_return".
        run_id:     The run identifier string.
        run_row:    One row from the run-level DataFrame.
        full_trace: Output of load_run_full_trace().

    Returns:
        Multi-line string ready for the report.
    """
    manifest = full_trace.get("manifest", {})
    financial = full_trace.get("financial", {})
    final_portfolio = full_trace.get("final_portfolio", {})
    rounds = full_trace.get("rounds", [])
    pid_crit_all = full_trace.get("pid_crit_all", [])

    experiment = manifest.get("experiment_name", run_row.get("experiment", "?"))
    scenario   = run_row.get("scenario", manifest.get("config_paths", ["", "unknown"])[-1])
    regime     = run_row.get("regime", "?")
    roles      = manifest.get("roles", [])
    invest_q   = manifest.get("invest_quarter", "?")

    sharpe       = financial.get("daily_metrics_annualized_sharpe", run_row.get("sharpe"))
    total_return = financial.get("daily_metrics_total_return_pct", run_row.get("total_return"))
    drawdown     = financial.get("daily_metrics_max_drawdown_pct", run_row.get("drawdown"))

    # Grab CRIT data from last round
    last_crit: dict = {}
    if pid_crit_all:
        last_crit = pid_crit_all[-1].get("crit", {})
    elif rounds:
        last_crit = rounds[-1].get("crit_scores", {})

    # Human-readable case type label
    type_labels = {
        "high_crit_low_return":  "HIGH CRIT / LOW RETURN  (reasoning looked good — portfolio didn't)",
        "low_crit_high_return":  "LOW CRIT / HIGH RETURN  (reasoning looked poor — portfolio did well)",
        "high_crit_high_return": "HIGH CRIT / HIGH RETURN  (reference: alignment case)",
        "intervention_helped":   "INTERVENTION HELPED  (JSD/causal intervention improved outcome)",
        "intervention_hurt":     "INTERVENTION HURT    (JSD/causal intervention worsened outcome)",
    }
    label = type_labels.get(case_type, case_type.upper())

    lines = [_fmt_section(f"CASE STUDY: {label}")]
    lines.append(f"  Run:           {experiment} / {run_id}")
    lines.append(f"  Scenario:      {scenario}   Regime: {regime}")
    lines.append(f"  Invest quarter:{invest_q}")
    lines.append(f"  Roles:         {', '.join(roles) if roles else 'see manifest'}")

    # -- CRIT scores --
    lines.append("")
    lines.append("CRIT REASONING QUALITY")
    if last_crit:
        lines.append(format_crit_scores(last_crit))
    else:
        lines.append("  (no CRIT data found — check pid_crit_all_rounds.json)")

    # -- Financial outcome --
    lines.append("")
    lines.append("FINANCIAL OUTCOME")
    def _fmt_val(v): return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
    lines.append(f"  Sharpe ratio:  {_fmt_val(sharpe)}")
    lines.append(f"  Total return:  {_fmt_val(total_return)}%")
    lines.append(f"  Max drawdown:  {_fmt_val(drawdown)}%")

    # -- Final portfolio --
    lines.append("")
    lines.append("FINAL PORTFOLIO ALLOCATION")
    lines.append(format_portfolio(final_portfolio))

    # -- Reasoning excerpts (use last round, all roles) --
    if rounds:
        last_round = rounds[-1]
        lines.append("")
        lines.append(f"REASONING EXCERPTS  (round {last_round.get('round', '?')}, revision phase)")
        for role in sorted(last_round.get("revisions", {}).keys()):
            lines.append(f"  [{role}]")
            lines.append(format_reasoning_excerpt(last_round, role, "revision"))
            lines.append("")

        # One critique highlight (most informative for case analysis)
        for role in sorted(last_round.get("critiques", {}).keys()):
            critiques = last_round.get("critiques", {}).get(role, [])
            if critiques:
                lines.append(f"CRITIQUE HIGHLIGHT  [{role} critiquing others]")
                lines.append(format_reasoning_excerpt(last_round, role, "critique"))
                break   # one is enough

    # -- Human analysis section --
    lines.append("")
    lines.append(_fmt_divider())
    lines.append("[HUMAN ANALYSIS NEEDED — paste this section into the paper]")
    lines.append("")
    if case_type == "high_crit_low_return":
        lines.append("Key questions to answer:")
        lines.append("  1. What specific portfolio decision had high CRIT but was wrong?")
        lines.append("     (e.g. sector bet, sizing decision, ignored macro signal)")
        lines.append("  2. What did CRIT measure that was real but irrelevant to returns?")
        lines.append("     (e.g. internal consistency of a wrong macro thesis)")
        lines.append("  3. Was there a reasoning failure CRIT missed that a human analyst would catch?")
        lines.append("     (e.g. causal overreach, regime misidentification)")
    elif case_type == "low_crit_high_return":
        lines.append("Key questions to answer:")
        lines.append("  1. What made this portfolio profitable despite weak reasoning scores?")
        lines.append("     (e.g. lucky macro call, mean-reversion, low-CRIT agent happened to be right)")
        lines.append("  2. Did the low CRIT score correctly identify a real reasoning flaw?")
        lines.append("  3. Is this evidence that CRIT measures the wrong thing, or just noise?")
    elif case_type in ("intervention_helped", "intervention_hurt"):
        lines.append("Key questions to answer:")
        lines.append("  1. Which specific intervention fired? (JSD collapse / reasoning quality?)")
        lines.append("  2. What did the intervention change in the agent's reasoning?")
        lines.append("     (compare pre/post revision text)")
        lines.append("  3. Can you trace the portfolio change that caused the outcome difference?")
    lines.append("")
    lines.append("Narrative (write 3-5 sentences):")
    lines.append("  [INSERT ANALYSIS HERE]")
    lines.append(_fmt_divider())

    return "\n".join(lines)


# ===========================================================================
# Part D — Report assembly
# ===========================================================================

def build_report(
    quadrant_cases: dict[str, list[str]],
    intervention_cases: dict[str, list[str]],
    runs_root: Path,
    run_df: pd.DataFrame,
    agent_df: pd.DataFrame,
) -> list[str]:
    """Assemble the full failure analysis report.

    Iterates over all selected cases, loads their traces, and formats
    each into a case study block.

    Args:
        quadrant_cases:     Output of select_quadrant_cases().
        intervention_cases: Output of select_intervention_cases().
        runs_root:          Root directory of all run logs.
        run_df:             One-row-per-run DataFrame.
        agent_df:           Per-agent-row DataFrame (for per-run percentile context).

    Returns:
        List of lines forming the complete report.
    """
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("=" * 80)
    add("QUALITATIVE FAILURE ANALYSIS  —  Professor Recommendation #5")
    add("=" * 80)
    add(f"Total runs in dataset:  {len(run_df)}")
    add("")
    add("Case selection thresholds:")
    if not run_df.empty and "rho_bar" in run_df.columns:
        add(f"  High CRIT threshold (rho_bar >= {QUARTILE_HIGH:.0%}): "
            f"{run_df['rho_bar'].quantile(QUARTILE_HIGH):.3f}")
        add(f"  Low  CRIT threshold (rho_bar <= {QUARTILE_LOW:.0%}): "
            f"{run_df['rho_bar'].quantile(QUARTILE_LOW):.3f}")
        add(f"  High Sharpe threshold (sharpe >= {QUARTILE_HIGH:.0%}): "
            f"{run_df['sharpe'].quantile(QUARTILE_HIGH):.3f}")
        add(f"  Low  Sharpe threshold (sharpe <= {QUARTILE_LOW:.0%}): "
            f"{run_df['sharpe'].quantile(QUARTILE_LOW):.3f}")
    add("")

    all_cases = {**quadrant_cases, **intervention_cases}
    total_selected = sum(len(v) for v in all_cases.values())
    add(f"Cases selected: {total_selected} total")
    for case_type, run_ids in all_cases.items():
        add(f"  {case_type}: {len(run_ids)} cases  — {run_ids}")
    add("")
    add("NOTE: [HUMAN ANALYSIS NEEDED] sections require human judgment.")
    add("The script extracts data and formats the template; a human must write")
    add("the narrative explaining WHY the CRIT metric aligned or diverged.")

    # Precompute sharpe percentiles across all runs for context
    sharpe_vals = run_df["sharpe"].dropna().values
    rho_vals = run_df["rho_bar"].dropna().values

    for case_type, run_ids in all_cases.items():
        if not run_ids:
            add(_fmt_section(f"{case_type.upper()} — no cases found"))
            add("  Not enough runs in this quadrant given the current dataset size.")
            add("  Add more runs or lower --n before re-running.")
            continue

        for run_id in run_ids:
            row_matches = run_df[run_df["run_id"] == run_id]
            if row_matches.empty:
                continue
            run_row = row_matches.iloc[0]

            # Find and load the run directory
            run_dir = _find_run_dir(runs_root, run_id)
            if run_dir is None:
                add(f"\n[WARN] Could not find directory for run_id={run_id}")
                continue

            full_trace = load_run_full_trace(run_dir)
            case_text = format_case_study(case_type, run_id, run_row, full_trace)
            add(case_text)

            # Add percentile context
            sharpe = run_row.get("sharpe")
            rho_bar = run_row.get("rho_bar")
            if sharpe is not None and not np.isnan(sharpe) and len(sharpe_vals):
                pct = int((sharpe_vals < sharpe).mean() * 100)
                add(f"  [Context] Sharpe percentile among all runs: {pct}th")
            if rho_bar is not None and not np.isnan(rho_bar) and len(rho_vals):
                pct = int((rho_vals < rho_bar).mean() * 100)
                add(f"  [Context] rho_bar percentile among all runs: {pct}th")

    return lines


def main(runs_root: Path = DEFAULT_RUNS_ROOT, n: int = 3) -> None:
    print("Loading regime manifest...")
    try:
        manifest_data = load_regime_manifest()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading agent rows from {runs_root} ...")
    agent_df = load_all_agent_rows(runs_root, manifest_data=manifest_data)

    if agent_df.empty:
        print(
            "\nNo data loaded. Make sure runs have _dashboard/financial_metrics.json.\n"
            f"runs_root checked: {runs_root}\n"
            "Try: python analysis/failure_analysis.py --runs-root results_juli/"
        )
        sys.exit(0)

    print(f"Loaded {len(agent_df)} agent rows from {agent_df['run_id'].nunique()} runs.")

    run_df = build_run_level_df(agent_df)

    print("Selecting quadrant cases...")
    quadrant_cases = select_quadrant_cases(run_df, n=n)
    for cat, ids in quadrant_cases.items():
        print(f"  {cat}: {len(ids)} cases")

    print("Selecting intervention cases...")
    intervention_cases = select_intervention_cases(runs_root, run_df, n=n)
    for cat, ids in intervention_cases.items():
        print(f"  {cat}: {len(ids)} cases")

    print("Building report...")
    lines = build_report(quadrant_cases, intervention_cases, runs_root, run_df, agent_df)
    report_text = "\n".join(lines) + "\n"

    print(report_text)

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report_text)
    print(f"\n[Report written to {OUTPUT_REPORT}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qualitative failure analysis case studies (Rec #5)"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Root directory of run logs (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Max cases to select per category (default: 3)",
    )
    args = parser.parse_args()
    main(runs_root=args.runs_root, n=args.n)
