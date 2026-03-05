"""Shared fixtures for ablation test suite."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.ablation.config import BASELINE


# ---------------------------------------------------------------------------
# Minimal stub types that mirror the real models enough for metrics testing.
# Using these avoids importing the full multi_agent stack in pure-unit tests.
# ---------------------------------------------------------------------------


def _make_round_metrics(round_index: int, rho_bar: float, js: float, ov: float):
    """Create a stub RoundMetrics-like object."""
    return SimpleNamespace(
        round_index=round_index,
        rho_bar=rho_bar,
        js_divergence=js,
        ov_overlap=ov,
    )


def _make_pid_event(
    round_index: int,
    rho_bar: float,
    js: float,
    ov: float,
    beta_new: float,
    e_t: float,
    u_t: float,
    s_t: int,
    quadrant: str,
):
    """Create a stub PIDEvent-like object."""
    return SimpleNamespace(
        round_index=round_index,
        metrics=_make_round_metrics(round_index, rho_bar, js, ov),
        pid_step={
            "beta_new": beta_new,
            "e_t": e_t,
            "u_t": u_t,
            "s_t": s_t,
            "quadrant": quadrant,
        },
        controller_output=SimpleNamespace(action="continue"),
    )


def _make_trace(pid_events, allocation=None):
    """Create a stub AgentTrace-like object."""
    action = SimpleNamespace(
        allocation=allocation,
        orders=[],
        justification="test",
        confidence=0.7,
        claims=[],
    )
    return SimpleNamespace(
        pid_events=pid_events,
        action=action,
        observation_timestamp="2025-03-15T10:00:00Z",
        architecture="debate",
    )


@pytest.fixture
def baseline_params():
    """Return a copy of BASELINE with metadata added."""
    params = dict(BASELINE)
    params.update({
        "run_id": "test_baseline",
        "group": "baseline",
        "param": "baseline",
        "value": "baseline",
        "scenario": "neutral",
        "replicate": 0,
    })
    return params


@pytest.fixture
def converging_trace():
    """A trace where rho converges toward rho_star=0.8 with healthy quadrants.

    5 rounds: rho goes 0.5 -> 0.6 -> 0.7 -> 0.78 -> 0.81
    JS goes from high to near-zero (converges).
    Beta stays near 0.5 (stable).
    """
    events = [
        _make_pid_event(0, 0.50, 0.15, 0.4, 0.50, 0.30, 0.04, 0, "stuck"),
        _make_pid_event(1, 0.60, 0.10, 0.5, 0.48, 0.20, 0.03, 0, "healthy"),
        _make_pid_event(2, 0.70, 0.04, 0.6, 0.46, 0.10, 0.02, 0, "healthy"),
        _make_pid_event(3, 0.78, 0.005, 0.7, 0.45, 0.02, 0.01, 0, "converged"),
        _make_pid_event(4, 0.81, 0.003, 0.8, 0.45, -0.01, -0.001, 0, "converged"),
    ]
    return _make_trace(events)


@pytest.fixture
def oscillating_trace():
    """A trace where rho and beta oscillate (unstable).

    10 rounds of alternating high/low rho with large beta swings.
    """
    events = []
    for i in range(10):
        rho = 0.8 if i % 2 == 0 else 0.3
        beta = 0.7 if i % 2 == 0 else 0.2
        quadrant = "chaotic" if i % 2 == 0 else "stuck"
        events.append(
            _make_pid_event(
                i, rho, 0.12, 0.4, beta,
                e_t=0.0 if i % 2 == 0 else 0.5,
                u_t=0.5 if i % 2 == 0 else -0.5,
                s_t=0,
                quadrant=quadrant,
            )
        )
    return _make_trace(events)


@pytest.fixture
def stuck_trace():
    """A trace where the system is stuck — low rho, no convergence."""
    events = [
        _make_pid_event(i, 0.3, 0.20, 0.3, 0.5, 0.5, 0.08, 0, "stuck")
        for i in range(6)
    ]
    return _make_trace(events)


@pytest.fixture
def empty_trace():
    """A trace with zero PID events (degenerate case)."""
    return _make_trace([])


@pytest.fixture
def sample_completed_results():
    """A list of completed result dicts for IO/summary testing."""
    base = {
        "status": "completed",
        "group": "gains",
        "param": "Kp",
        "scenario": "neutral",
        "Kp": 0.15,
        "Ki": 0.01,
        "Kd": 0.03,
        "rho_star": 0.8,
        "final_rho_bar": 0.75,
        "mean_rho_bar": 0.65,
        "final_beta": 0.45,
        "beta_range": 0.1,
        "steady_state_error": 0.05,
        "beta_overshoot": 0.08,
        "settling_round": 3,
        "rho_variance": 0.02,
        "mean_JS": 0.05,
        "empirical_kappa": 0.85,
        "beta_oscillation_flag": False,
        "rho_oscillation_flag": False,
        "rho_limit_cycle_flag": False,
        "JS_monotonicity_flag": False,
        "converged_single": True,
        "convergence_window_met": True,
        "control_stable": True,
        "behavioral_stable": True,
        "stochastic_regime": False,
        "paranoia_rate": 0.1,
        "realignment_rate": 0.3,
        "net_effect": 0.2,
        "quadrant_stuck_pct": 0.1,
        "quadrant_chaotic_pct": 0.1,
        "quadrant_converged_pct": 0.4,
        "quadrant_healthy_pct": 0.4,
        "dominant_quadrant": "healthy",
        "elapsed_seconds": 2.5,
    }
    results = []
    for i in range(5):
        r = dict(base)
        r["run_id"] = f"gains_Kp-{0.05 * i}"
        r["value"] = str(0.05 * i)
        r["Kp"] = 0.05 * i
        r["final_rho_bar"] = 0.6 + 0.05 * i
        r["replicate"] = 0
        results.append(r)
    return results


@pytest.fixture
def replicate_results():
    """Three replicate results for aggregation testing."""
    base = {
        "status": "completed",
        "run_id": "gains_Kp-0.15",
        "group": "gains",
        "param": "Kp",
        "value": "0.15",
        "scenario": "neutral",
        "model_name": "gpt-4o-mini",
        "temperature": 0.3,
        "Kp": 0.15,
        "Ki": 0.01,
        "Kd": 0.03,
        "rho_star": 0.8,
        "final_rho_bar": 0.75,
        "mean_rho_bar": 0.65,
        "final_beta": 0.45,
        "beta_range": 0.1,
        "steady_state_error": 0.05,
        "beta_overshoot": 0.08,
        "rounds_used": 5,
        "sycophancy_count": 0,
        "settling_round": 3,
        "rho_variance": 0.02,
        "mean_JS": 0.05,
        "empirical_kappa": 0.85,
        "rho_sign_change_count": 1,
        "rho_contraction_rate": 0.85,
        "beta_oscillation_flag": False,
        "rho_oscillation_flag": False,
        "rho_limit_cycle_flag": False,
        "JS_monotonicity_flag": False,
        "converged_single": True,
        "convergence_window_met": True,
        "control_stable": True,
        "behavioral_stable": True,
        "stochastic_regime": False,
        "paranoia_rate": 0.1,
        "realignment_rate": 0.3,
        "net_effect": 0.2,
        "quadrant_stuck_pct": 0.1,
        "quadrant_chaotic_pct": 0.1,
        "quadrant_converged_pct": 0.4,
        "quadrant_healthy_pct": 0.4,
        "escalation_count": 0,
        "stuck_rounds": 1,
        "elapsed_seconds": 2.5,
    }
    results = []
    for i in range(3):
        r = dict(base)
        r["replicate"] = i
        r["final_rho_bar"] = 0.73 + 0.02 * i  # 0.73, 0.75, 0.77
        r["final_beta"] = 0.44 + 0.01 * i
        results.append(r)
    return results
