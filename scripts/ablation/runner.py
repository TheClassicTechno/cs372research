"""Single ablation run executor.

Builds DebateConfig from flat params, runs stability pre-check,
executes one debate via MultiAgentRunner, and extracts metrics.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from scripts.ablation.config import build_scenario_observations
from scripts.ablation.metrics import extract_metrics

logger = logging.getLogger(__name__)


def stability_precheck(params: dict) -> tuple[bool, bool]:
    """Run stability and non-oscillation pre-checks.

    Returns (stable, non_oscillatory) booleans.
    """
    from eval.PID.stability import check_non_oscillation, check_stability
    from eval.PID.types import PIDGains

    gains = PIDGains(
        Kp=params.get("Kp", 0.15),
        Ki=params.get("Ki", 0.01),
        Kd=params.get("Kd", 0.03),
    )
    kwargs = dict(
        gains=gains,
        T_max=params.get("max_rounds", 10),
        gamma_beta=params.get("gamma_beta", 0.9),
        rho_star=params.get("rho_star", 1.0),
        mu=params.get("mu", 0.0),
    )
    stable = check_stability(**kwargs)
    non_osc = check_non_oscillation(**kwargs)
    return stable, non_osc


def build_debate_config(params: dict, mock: bool = False) -> Any:
    """Build a DebateConfig from flat ablation params.

    Constructs PIDConfig directly and passes it via pid_config field.
    """
    from eval.PID.types import PIDConfig, PIDGains
    from multi_agent.config import AgentRole, DebateConfig

    pid_config = PIDConfig(
        gains=PIDGains(
            Kp=params.get("Kp", 0.15),
            Ki=params.get("Ki", 0.01),
            Kd=params.get("Kd", 0.03),
        ),
        rho_star=params.get("rho_star", 0.8),
        gamma_beta=params.get("gamma_beta", 0.9),
        mu=params.get("mu", 1.0),
        delta_s=params.get("delta_s", 0.05),
        T_max=params.get("max_rounds", 10),
        epsilon=params.get("epsilon", 0.01),
        delta_js=params.get("delta_js", 0.05),
        delta_beta=params.get("delta_beta", 0.1),
    )

    role_map = {r.value: r for r in AgentRole}
    roles = [role_map[r] for r in params.get("roles", ["macro", "value", "risk", "technical"])]

    return DebateConfig(
        roles=roles,
        max_rounds=params.get("max_rounds", 10),
        agreeableness=params.get("agreeableness", 0.3),
        enable_adversarial=params.get("enable_adversarial", False),
        model_name=params.get("model_name", "gpt-4o-mini"),
        temperature=params.get("temperature", 0.3),
        mock=mock,
        parallel_agents=False,
        verbose=False,
        pid_config=pid_config,
        initial_beta=params.get("initial_beta", 0.5),
        pid_propose=params.get("pid_propose", False),
        pid_critique=params.get("pid_critique", True),
        pid_revise=params.get("pid_revise", True),
        pid_log_metrics=True,
    )


def build_observation(params: dict, scenario: str) -> Any:
    """Build the synthetic Observation for this run's ticker universe and scenario."""
    tickers = params.get("tickers", ["AAPL", "NVDA", "MSFT", "GOOG", "JPM"])
    scenarios = build_scenario_observations(tickers)
    if scenario not in scenarios:
        logger.warning("Unknown scenario '%s', falling back to 'neutral'", scenario)
        scenario = "neutral"
    return scenarios[scenario]


def run_single_ablation(
    run_config: dict,
    mock: bool = False,
    trace_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute a single ablation run and return metrics.

    Args:
        run_config: Merged params dict with metadata (run_id, group, etc.)
        mock: Use mock LLM (no API calls)
        trace_dir: Directory to save trace JSON (if provided)

    Returns:
        Flat dict with config params + all extracted metrics + status.
    """
    run_id = run_config["run_id"]
    scenario = run_config.get("scenario", "neutral")

    result: dict[str, Any] = {
        "run_id": run_id,
        "group": run_config.get("group", ""),
        "param": run_config.get("param", ""),
        "value": run_config.get("value", ""),
        "scenario": scenario,
        "replicate": run_config.get("replicate", 0),
        "model_name": run_config.get("model_name", "gpt-4o-mini"),
        "temperature": run_config.get("temperature", 0.3),
        "num_tickers": len(run_config.get("tickers", [])),
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
    }

    # Step 1: Stability pre-check
    stable, non_osc = stability_precheck(run_config)
    if not stable:
        result["status"] = "skipped_unstable"
        result["stability_check"] = False
        result["non_oscillation_check"] = non_osc
        logger.info("[%s] SKIPPED — fails stability pre-check", run_id)
        return result

    # Step 2: Build config and observation
    try:
        config = build_debate_config(run_config, mock=mock)
        observation = build_observation(run_config, scenario)
    except Exception as exc:
        result["status"] = f"config_error: {exc}"
        result["stability_check"] = stable
        result["non_oscillation_check"] = non_osc
        logger.error("[%s] Config build error: %s", run_id, exc)
        return result

    # Step 3: Run debate
    from multi_agent.runner import MultiAgentRunner

    runner = MultiAgentRunner(config)
    t0 = time.monotonic()
    try:
        action, trace = runner.run(observation)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        result["status"] = f"runtime_error: {exc}"
        result["stability_check"] = stable
        result["non_oscillation_check"] = non_osc
        result["elapsed_seconds"] = round(elapsed, 2)
        logger.error("[%s] Runtime error after %.1fs: %s\n%s",
                     run_id, elapsed, exc, traceback.format_exc())
        return result
    elapsed = time.monotonic() - t0

    # Step 4: Extract metrics
    try:
        metrics = extract_metrics(trace, run_config, stable, non_osc)
        result.update(metrics)
    except Exception as exc:
        result["status"] = f"metrics_error: {exc}"
        result["stability_check"] = stable
        result["non_oscillation_check"] = non_osc
        result["elapsed_seconds"] = round(elapsed, 2)
        logger.error("[%s] Metrics extraction error: %s", run_id, exc)
        return result

    result["status"] = "completed"
    result["elapsed_seconds"] = round(elapsed, 2)

    # Step 5: Save trace JSON
    if trace_dir is not None:
        _save_trace(trace_dir, run_id, trace)

    logger.info(
        "[%s] DONE in %.1fs — ρ=%.3f β=%.3f quad=%s %s",
        run_id, elapsed,
        result.get("final_rho_bar", 0),
        result.get("final_beta", 0),
        result.get("dominant_quadrant", "?"),
        "STABLE" if result.get("behavioral_stable") else "unstable",
    )
    return result


def _save_trace(trace_dir: Path, run_id: str, trace: Any) -> None:
    """Save AgentTrace as JSON to trace directory."""
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{run_id}.json"
        path.write_text(trace.model_dump_json(indent=2))
    except Exception as exc:
        logger.warning("Failed to save trace for %s: %s", run_id, exc)
