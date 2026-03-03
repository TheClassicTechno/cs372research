"""Run matrix generation: expand sweep groups into a flat list of run configs.

Each run config is a dict merging BASELINE with group-specific overrides,
plus metadata (run_id, group, param, value, scenario, replicate).
"""

from __future__ import annotations

import random
from typing import Any

from scripts.ablation.config import (
    AGENT_LABELS,
    BASELINE,
    HIGH_GAIN_STRESS_LABELS,
    HIGH_MU_STRESS_LABELS,
    HIGH_RHO_STRESS_LABELS,
    INTERACTION_LABELS,
    NUM_RANDOM_GAIN_SAMPLES,
    PHASE_LABELS,
    SWEEP_GROUPS,
    TICKER_LABELS,
)

# Groups that use named labels (index-matched)
_LABELED_GROUPS: dict[str, list[str]] = {
    "interactions": INTERACTION_LABELS,
    "phases": PHASE_LABELS,
    "tickers": TICKER_LABELS,
    "agents": AGENT_LABELS,
    "high_gain_stress": HIGH_GAIN_STRESS_LABELS,
    "high_mu_stress": HIGH_MU_STRESS_LABELS,
    "high_rho_star_stress": HIGH_RHO_STRESS_LABELS,
}


def _make_run_label(group: str, overrides: dict, index: int) -> tuple[str, str]:
    """Generate (param_name, value_label) for a run.

    For single-param sweeps: ("Kp", "0.30")
    For labeled groups: ("config", "aggressive_p")
    For models: ("model", "gpt-4o")
    """
    if group in _LABELED_GROUPS:
        labels = _LABELED_GROUPS[group]
        label = labels[index] if index < len(labels) else f"config_{index}"
        return "config", label

    if group == "models":
        return "model", overrides.get("model_name", f"model_{index}")

    if group == "random_gain_samples":
        return "gains", f"random_{index}"

    # Single-param or multi-param override
    changed = {k: v for k, v in overrides.items() if BASELINE.get(k) != v}
    if len(changed) == 1:
        k, v = next(iter(changed.items()))
        return k, f"{v}"
    elif changed:
        parts = [f"{k}-{v}" for k, v in sorted(changed.items())]
        return "multi", "_".join(parts)
    else:
        return "param", f"baseline_{index}"


def _generate_random_gain_samples(seed: int | None) -> list[dict]:
    """Generate random (Kp, Ki, Kd) triples for nonlinear interaction detection."""
    rng = random.Random(seed)
    samples = []
    for _ in range(NUM_RANDOM_GAIN_SAMPLES):
        samples.append({
            "Kp": round(rng.uniform(0.0, 0.4), 4),
            "Ki": round(rng.uniform(0.0, 0.1), 4),
            "Kd": round(rng.uniform(0.0, 0.2), 4),
        })
    return samples


def generate_run_matrix(
    groups: list[str] | None = None,
    scenarios: list[str] | None = None,
    replicates: int = 1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Expand sweep groups into a flat list of run configurations.

    Args:
        groups: Which sweep groups to include. None = all.
        scenarios: Scenario names to run each config against. None = ["neutral"].
        replicates: Number of times to repeat each config.
        seed: RNG seed for random_gain_samples reproducibility.

    Returns:
        List of run config dicts, each with merged params + metadata keys:
        run_id, group, param, value, scenario, replicate.
    """
    if scenarios is None:
        scenarios = ["neutral"]

    active_groups = groups if groups else list(SWEEP_GROUPS.keys())

    # Populate random gain samples
    random_samples = _generate_random_gain_samples(seed)

    matrix: list[dict[str, Any]] = []

    # Always include baseline as first run
    for scenario in scenarios:
        for rep in range(replicates):
            run = dict(BASELINE)
            run["run_id"] = _run_id("baseline", "baseline", "baseline", scenario, rep)
            run["group"] = "baseline"
            run["param"] = "baseline"
            run["value"] = "baseline"
            run["scenario"] = scenario
            run["replicate"] = rep
            matrix.append(run)

    for group_name in active_groups:
        if group_name == "random_gain_samples":
            entries = random_samples
        else:
            entries = SWEEP_GROUPS.get(group_name, [])

        for idx, overrides in enumerate(entries):
            param_name, value_label = _make_run_label(group_name, overrides, idx)

            for scenario in scenarios:
                for rep in range(replicates):
                    run = dict(BASELINE)
                    run.update(overrides)
                    run["run_id"] = _run_id(
                        group_name, param_name, value_label, scenario, rep,
                    )
                    run["group"] = group_name
                    run["param"] = param_name
                    run["value"] = value_label
                    run["scenario"] = scenario
                    run["replicate"] = rep
                    matrix.append(run)

    return matrix


def _run_id(
    group: str, param: str, value: str, scenario: str, replicate: int,
) -> str:
    """Generate a human-readable run_id."""
    base = f"{group}_{param}-{value}"
    if scenario != "neutral":
        base += f"_{scenario}"
    if replicate > 0:
        base += f"_r{replicate}"
    # Sanitize for filesystem
    return base.replace("/", "_").replace(" ", "_")


def count_runs(
    groups: list[str] | None = None,
    scenarios: list[str] | None = None,
    replicates: int = 1,
) -> int:
    """Count total runs without generating the full matrix."""
    if scenarios is None:
        scenarios = ["neutral"]
    active_groups = groups if groups else list(SWEEP_GROUPS.keys())

    n_scenarios = len(scenarios)
    n_configs = 1  # baseline

    for group_name in active_groups:
        if group_name == "random_gain_samples":
            n_configs += NUM_RANDOM_GAIN_SAMPLES
        else:
            n_configs += len(SWEEP_GROUPS.get(group_name, []))

    return n_configs * n_scenarios * replicates
