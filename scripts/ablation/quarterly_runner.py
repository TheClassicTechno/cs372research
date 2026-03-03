"""Quarterly ablation runner: 4 episodes (2025Q1-Q4) per config.

Loads real memo data via load_memo_cases + build_observation when available.
Falls back to synthetic scenario observations when quarterly data is missing.
Each quarter gets a fresh PID controller state via MultiAgentRunner.run().
Metrics are extracted per-quarter then aggregated across the 4 episodes.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from scripts.ablation.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MEMO_FORMAT,
    INVEST_QUARTERS,
)
from scripts.ablation.metrics import extract_metrics
from scripts.ablation.runner import (
    _save_trace,
    build_debate_config,
    build_observation,
    stability_precheck,
)

logger = logging.getLogger(__name__)


def _load_quarterly_observation(
    quarter: str,
    tickers: list[str],
    dataset_path: str,
    memo_format: str,
) -> Any | None:
    """Try to load real memo data for a quarter. Returns Observation or None."""
    try:
        from simulation.feature_engineering import build_observation as fe_build_obs
        from simulation.memo_loader import load_memo_cases

        cases = load_memo_cases(
            dataset_path=dataset_path,
            invest_quarter=quarter,
            memo_format=memo_format,
            tickers=tickers,
        )
        if not cases:
            return None
        # First case is the decision case (agents debate on this)
        return fe_build_obs(cases[0])
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("No quarterly data for %s: %s", quarter, exc)
        return None
    except Exception as exc:
        logger.warning("Error loading quarterly data for %s: %s", quarter, exc)
        return None


def _synthetic_quarterly_observations(
    tickers: list[str],
    scenario: str,
) -> dict[str, Any]:
    """Build synthetic observations keyed by quarter, using the scenario fallback."""
    from scripts.ablation.config import build_scenario_observations

    scenarios = build_scenario_observations(tickers)
    obs = scenarios.get(scenario, scenarios.get("neutral"))

    # Same observation for all 4 quarters (synthetic fallback)
    return {q: obs for q in INVEST_QUARTERS}


def run_quarterly_ablation(
    run_config: dict,
    mock: bool = False,
    trace_dir: Path | None = None,
    dataset_path: str = DEFAULT_DATASET_PATH,
    memo_format: str = DEFAULT_MEMO_FORMAT,
) -> dict[str, Any]:
    """Execute one ablation config across 4 quarterly episodes.

    For each quarter in 2025Q1-Q4:
      1. Load real memo data (or fall back to synthetic)
      2. Run a full debate with fresh PID state
      3. Extract per-quarter metrics

    Then aggregate metrics across the 4 episodes.

    Args:
        run_config: Merged params dict with metadata (run_id, group, etc.)
        mock: Use mock LLM (no API calls)
        trace_dir: Directory to save trace JSONs (if provided)
        dataset_path: Path to final_snapshots/ for real quarterly data
        memo_format: "text" or "json" for memo loader

    Returns:
        Flat dict with config params + aggregated metrics + per-quarter breakdown.
    """
    run_id = run_config["run_id"]
    scenario = run_config.get("scenario", "neutral")
    tickers = run_config.get("tickers", ["AAPL", "NVDA", "MSFT", "GOOG", "JPM"])

    # --- Build result skeleton with config columns ---
    result: dict[str, Any] = {
        "run_id": run_id,
        "group": run_config.get("group", ""),
        "param": run_config.get("param", ""),
        "value": run_config.get("value", ""),
        "scenario": scenario,
        "replicate": run_config.get("replicate", 0),
        "model_name": run_config.get("model_name", "gpt-4o-mini"),
        "temperature": run_config.get("temperature", 0.3),
        "num_tickers": len(tickers),
        "num_agents": len(run_config.get("roles", [])),
        "Kp": run_config.get("Kp"),
        "Ki": run_config.get("Ki"),
        "Kd": run_config.get("Kd"),
        "rho_star": run_config.get("rho_star"),
        "gamma_beta": run_config.get("gamma_beta"),
        "delta_js": run_config.get("delta_js"),
        "delta_s": run_config.get("delta_s"),
        "delta_beta": run_config.get("delta_beta"),
        "epsilon": run_config.get("epsilon"),
        "mu": run_config.get("mu"),
        "initial_beta": run_config.get("initial_beta"),
        "pid_propose": run_config.get("pid_propose"),
        "pid_critique": run_config.get("pid_critique"),
        "pid_revise": run_config.get("pid_revise"),
        "enable_adversarial": run_config.get("enable_adversarial"),
        "num_episodes": len(INVEST_QUARTERS),
    }

    # --- Step 1: Stability pre-check ---
    stable, non_osc = stability_precheck(run_config)
    if not stable:
        result["status"] = "skipped_unstable"
        result["stability_check"] = False
        result["non_oscillation_check"] = non_osc
        logger.info("[%s] SKIPPED — fails stability pre-check", run_id)
        return result

    # --- Step 2: Build debate config ---
    try:
        config = build_debate_config(run_config, mock=mock)
    except Exception as exc:
        result["status"] = f"config_error: {exc}"
        result["stability_check"] = stable
        result["non_oscillation_check"] = non_osc
        logger.error("[%s] Config build error: %s", run_id, exc)
        return result

    # --- Step 3: Load observations for all 4 quarters ---
    quarterly_obs: dict[str, Any] = {}
    using_real_data = False

    for quarter in INVEST_QUARTERS:
        obs = _load_quarterly_observation(quarter, tickers, dataset_path, memo_format)
        if obs is not None:
            quarterly_obs[quarter] = obs
            using_real_data = True

    if not quarterly_obs:
        # Fall back to synthetic for all quarters
        quarterly_obs = _synthetic_quarterly_observations(tickers, scenario)
        logger.info("[%s] Using synthetic observations (no quarterly data found)", run_id)
    elif len(quarterly_obs) < len(INVEST_QUARTERS):
        # Partial data — fill missing quarters with synthetic
        synthetic = _synthetic_quarterly_observations(tickers, scenario)
        for q in INVEST_QUARTERS:
            if q not in quarterly_obs:
                quarterly_obs[q] = synthetic[q]
                logger.info("[%s] Quarter %s: synthetic fallback", run_id, q)

    result["data_source"] = "real" if using_real_data else "synthetic"

    # --- Step 4: Run 4 quarterly debates ---
    from multi_agent.runner import MultiAgentRunner

    runner = MultiAgentRunner(config)
    episode_metrics: list[dict] = []
    episode_traces: list[Any] = []
    total_elapsed = 0.0
    episodes_completed = 0

    for quarter in INVEST_QUARTERS:
        obs = quarterly_obs[quarter]
        t0 = time.monotonic()

        try:
            action, trace = runner.run(obs)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            total_elapsed += elapsed
            logger.warning(
                "[%s] Quarter %s failed after %.1fs: %s",
                run_id, quarter, elapsed, exc,
            )
            continue

        elapsed = time.monotonic() - t0
        total_elapsed += elapsed

        try:
            qm = extract_metrics(trace, run_config, stable, non_osc)
            qm["quarter"] = quarter
            qm["quarter_elapsed"] = round(elapsed, 2)
            episode_metrics.append(qm)
            episode_traces.append(trace)
            episodes_completed += 1
        except Exception as exc:
            logger.warning("[%s] Quarter %s metrics error: %s", run_id, quarter, exc)

        # Save per-quarter trace
        if trace_dir is not None:
            _save_trace(trace_dir, f"{run_id}_{quarter}", trace)

    if not episode_metrics:
        result["status"] = "all_episodes_failed"
        result["stability_check"] = stable
        result["non_oscillation_check"] = non_osc
        result["elapsed_seconds"] = round(total_elapsed, 2)
        return result

    # --- Step 5: Aggregate across quarterly episodes ---
    agg = _aggregate_episodes(episode_metrics)
    result.update(agg)
    result["status"] = "completed"
    result["episodes_completed"] = episodes_completed
    result["elapsed_seconds"] = round(total_elapsed, 2)
    result["stability_check"] = stable
    result["non_oscillation_check"] = non_osc

    logger.info(
        "[%s] DONE %d/4 episodes in %.1fs — ρ=%.3f β=%.3f %s [%s]",
        run_id, episodes_completed, total_elapsed,
        result.get("final_rho_bar_mean", 0),
        result.get("final_beta_mean", 0),
        "STABLE" if result.get("behavioral_stable_rate", 0) >= 0.75 else "unstable",
        result.get("data_source", "?"),
    )
    return result


def _aggregate_episodes(episodes: list[dict]) -> dict[str, Any]:
    """Aggregate per-quarter metrics into cross-episode summary.

    For numeric metrics: compute mean and std across episodes.
    For boolean metrics: compute rate (fraction True).
    For categorical: mode.
    """
    import math
    from collections import Counter

    n = len(episodes)
    agg: dict[str, Any] = {}

    # --- Numeric metrics: mean + std across episodes ---
    numeric_keys = [
        "final_rho_bar", "mean_rho_bar", "final_beta", "beta_range",
        "steady_state_error", "beta_overshoot", "settling_round",
        "rho_variance", "mean_JS", "rho_sign_change_count",
        "rho_contraction_rate", "empirical_kappa",
        "paranoia_rate", "realignment_rate", "net_effect",
        "healthy_persistence", "regression_rate", "chaotic_escape_rate",
        "transition_entropy", "escalation_count", "stuck_rounds",
        "sycophancy_count", "rounds_used",
        "experimental_allocation_entropy", "experimental_concentration_index",
        "experimental_num_active_positions", "experimental_max_weight",
    ]

    for key in numeric_keys:
        vals = [e[key] for e in episodes if e.get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            agg[f"{key}_mean"] = round(mean, 6)
            if len(vals) >= 2:
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                agg[f"{key}_std"] = round(math.sqrt(var), 6)
            else:
                agg[f"{key}_std"] = 0.0
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None

    # --- Boolean metrics: rate across episodes ---
    bool_keys = [
        "beta_oscillation_flag", "rho_oscillation_flag", "rho_limit_cycle_flag",
        "JS_monotonicity_flag", "converged_single", "convergence_window_met",
        "control_stable", "behavioral_stable",
    ]
    for key in bool_keys:
        vals = [1 if e.get(key) else 0 for e in episodes]
        agg[f"{key}_rate"] = sum(vals) / n if n else 0

    # --- Quadrant distribution: mean across episodes ---
    for quad in ["stuck", "chaotic", "converged", "healthy"]:
        key = f"quadrant_{quad}_pct"
        vals = [e.get(key, 0) for e in episodes]
        agg[f"{key}_mean"] = sum(vals) / n if n else 0

    # --- Dominant quadrant: mode across episodes ---
    dom_quads = [e.get("dominant_quadrant", "") for e in episodes]
    if dom_quads:
        agg["dominant_quadrant"] = Counter(dom_quads).most_common(1)[0][0]
    else:
        agg["dominant_quadrant"] = ""

    # --- Stochastic regime (same for all episodes) ---
    agg["stochastic_regime"] = episodes[0].get("stochastic_regime", False) if episodes else False

    # --- Per-quarter breakdown (compact) ---
    agg["per_quarter_rho"] = [
        round(e.get("final_rho_bar", 0) or 0, 4) for e in episodes
    ]
    agg["per_quarter_beta"] = [
        round(e.get("final_beta", 0) or 0, 4) for e in episodes
    ]
    agg["per_quarter_quadrant"] = [
        e.get("dominant_quadrant", "") for e in episodes
    ]
    agg["per_quarter_converged"] = [
        bool(e.get("convergence_window_met")) for e in episodes
    ]

    return agg
