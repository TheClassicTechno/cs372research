"""Metrics extraction for PID ablation runs.

Computes control-theoretic stability diagnostics, behavioral stability metrics,
quadrant transition analysis, paranoia/correction rates, and optional portfolio
metrics from an AgentTrace with PID events.

Two stability categories are tracked:
  - control_stable: gain pre-check passes + no β oscillation
  - behavioral_stable: no ρ oscillation + convergence window met
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from scripts.ablation.config import (
    CONVERGENCE_WINDOW,
    EPSILON_BAND,
    HIGH_CORRECTION_THRESHOLD,
    OSCILLATION_K,
    STOCHASTIC_REGIME_THRESHOLD,
)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division returning default when b is zero."""
    return a / b if b != 0 else default


def _sign_changes(values: list[float]) -> int:
    """Count sign changes in a sequence of deltas."""
    if len(values) < 2:
        return 0
    changes = 0
    for i in range(1, len(values)):
        delta_prev = values[i - 1]
        delta_curr = values[i]
        if delta_prev * delta_curr < 0:
            changes += 1
    return changes


def extract_metrics(
    trace: Any,
    params: dict,
    stability_check: bool,
    non_oscillation_check: bool,
) -> dict:
    """Extract all metrics from a completed ablation run.

    Args:
        trace: AgentTrace from MultiAgentRunner.run()
        params: The run's parameter dict (includes rho_star, initial_beta, etc.)
        stability_check: Result of check_stability() pre-check
        non_oscillation_check: Result of check_non_oscillation() pre-check

    Returns:
        Flat dict of all metric values for CSV output.
    """
    pid_events = trace.pid_events or []
    action = trace.action
    rho_star = params.get("rho_star", 0.8)
    initial_beta = params.get("initial_beta", 0.5)
    epsilon = params.get("epsilon", 0.01)
    temperature = params.get("temperature", 0.3)

    # Extract time series from PID events
    rho_series: list[float] = []
    beta_series: list[float] = []
    js_series: list[float] = []
    ov_series: list[float] = []
    error_series: list[float] = []
    ut_series: list[float] = []
    quadrants: list[str] = []
    syco_flags: list[int] = []

    for event in pid_events:
        rho_series.append(event.metrics.rho_bar)
        js_series.append(event.metrics.js_divergence)
        ov_series.append(event.metrics.ov_overlap)
        beta_series.append(event.pid_step["beta_new"])
        error_series.append(event.pid_step["e_t"])
        ut_series.append(event.pid_step["u_t"])
        quadrants.append(event.pid_step["quadrant"])
        syco_flags.append(event.pid_step["s_t"])

    n_rounds = len(pid_events)
    m: dict[str, Any] = {}

    # =========================================================================
    # SECTION A: Core PID metrics
    # =========================================================================
    m["rounds_used"] = n_rounds
    m["final_rho_bar"] = rho_series[-1] if rho_series else None
    m["mean_rho_bar"] = sum(rho_series) / n_rounds if n_rounds else None
    m["final_beta"] = beta_series[-1] if beta_series else None
    m["beta_range"] = (max(beta_series) - min(beta_series)) if beta_series else None
    m["sycophancy_count"] = sum(syco_flags)

    # =========================================================================
    # SECTION B: Quadrant distribution
    # =========================================================================
    quad_counts = Counter(quadrants)
    total_q = max(n_rounds, 1)
    m["quadrant_stuck_pct"] = quad_counts.get("stuck", 0) / total_q
    m["quadrant_chaotic_pct"] = quad_counts.get("chaotic", 0) / total_q
    m["quadrant_converged_pct"] = quad_counts.get("converged", 0) / total_q
    m["quadrant_healthy_pct"] = quad_counts.get("healthy", 0) / total_q
    m["dominant_quadrant"] = quad_counts.most_common(1)[0][0] if quadrants else ""

    # =========================================================================
    # SECTION C: β stability diagnostics
    # =========================================================================
    m["steady_state_error"] = (rho_star - rho_series[-1]) if rho_series else None
    m["beta_overshoot"] = (max(beta_series) - initial_beta) if beta_series else None

    # Settling round: first t where |β_t - β_final| < EPSILON_BAND and stays
    beta_final = beta_series[-1] if beta_series else initial_beta
    settling = None
    if beta_series:
        for t in range(n_rounds):
            if all(abs(beta_series[j] - beta_final) < EPSILON_BAND
                   for j in range(t, n_rounds)):
                settling = t
                break
    m["settling_round"] = settling

    # β oscillation: sign changes in (β_t - β_{t-1})
    beta_deltas = [beta_series[i] - beta_series[i - 1]
                   for i in range(1, len(beta_series))]
    beta_sign_changes = _sign_changes(beta_deltas)
    m["beta_oscillation_flag"] = beta_sign_changes > OSCILLATION_K

    # =========================================================================
    # SECTION D: ρ dynamics (controlled variable stability)
    # =========================================================================
    m["rho_variance"] = _variance(rho_series) if rho_series else None
    m["mean_JS"] = sum(js_series) / n_rounds if n_rounds else None

    # ρ sign changes in (ρ_t - ρ_{t-1})
    rho_deltas = [rho_series[i] - rho_series[i - 1]
                  for i in range(1, len(rho_series))]
    rho_sign_changes = _sign_changes(rho_deltas)
    m["rho_sign_change_count"] = rho_sign_changes
    m["rho_oscillation_flag"] = rho_sign_changes > OSCILLATION_K

    # ρ limit cycle: periodicity in last N rounds
    m["rho_limit_cycle_flag"] = _detect_limit_cycle(rho_series)

    # ρ contraction rate: mean(|ρ_{t+1} - ρ*| / |ρ_t - ρ*|)
    m["rho_contraction_rate"] = _contraction_rate(rho_series, rho_star)

    # JS monotonicity: True if JS increases after previously decreasing
    m["JS_monotonicity_flag"] = _js_non_monotonic(js_series)

    # =========================================================================
    # SECTION E: Convergence (refined)
    # =========================================================================
    # Legacy: any single round
    m["converged_single"] = any(js < epsilon for js in js_series)

    # Window: W consecutive rounds
    conv_round = _convergence_window_round(js_series, epsilon, CONVERGENCE_WINDOW)
    m["convergence_window_met"] = conv_round is not None
    m["convergence_round"] = conv_round

    # =========================================================================
    # SECTION F: Quadrant transition matrix
    # =========================================================================
    trans = _quadrant_transition_metrics(quadrants)
    m["healthy_persistence"] = trans["healthy_persistence"]
    m["regression_rate"] = trans["regression_rate"]
    m["chaotic_escape_rate"] = trans["chaotic_escape_rate"]
    m["transition_entropy"] = trans["transition_entropy"]

    # =========================================================================
    # SECTION G: Paranoia / correction metrics
    # =========================================================================
    para = _paranoia_metrics(rho_series, rho_star)
    m["paranoia_rate"] = para["paranoia_rate"]
    m["realignment_rate"] = para["realignment_rate"]
    m["sycophancy_ratio"] = para["sycophancy_ratio"]
    m["net_effect"] = para["net_effect"]

    # =========================================================================
    # SECTION H: Stability labels
    # =========================================================================
    m["stability_check"] = stability_check
    m["non_oscillation_check"] = non_oscillation_check
    m["control_stable"] = stability_check and not m["beta_oscillation_flag"]
    m["behavioral_stable"] = (
        not m["rho_oscillation_flag"] and m["convergence_window_met"]
    )
    m["stochastic_regime"] = temperature >= STOCHASTIC_REGIME_THRESHOLD

    # =========================================================================
    # SECTION I: Escalation metrics
    # =========================================================================
    m["escalation_count"] = sum(1 for u in ut_series if abs(u) > HIGH_CORRECTION_THRESHOLD)
    m["stuck_rounds"] = quad_counts.get("stuck", 0)
    m["high_correction_rounds"] = m["escalation_count"]

    # =========================================================================
    # SECTION J: Empirical contraction estimate (κ)
    # =========================================================================
    m["empirical_kappa"] = _contraction_rate(rho_series, rho_star)

    # =========================================================================
    # SECTION K: Optional portfolio metrics (secondary)
    # =========================================================================
    alloc = getattr(action, "allocation", None) if action else None
    if alloc and isinstance(alloc, dict) and alloc:
        weights = [w for w in alloc.values() if w > 0]
        m["experimental_allocation_entropy"] = _entropy(weights)
        m["experimental_concentration_index"] = max(weights) if weights else None
        m["experimental_num_active_positions"] = len(weights)
        m["experimental_max_weight"] = max(weights) if weights else None
    else:
        m["experimental_allocation_entropy"] = None
        m["experimental_concentration_index"] = None
        m["experimental_num_active_positions"] = None
        m["experimental_max_weight"] = None

    return m


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _variance(values: list[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _entropy(weights: list[float]) -> float:
    """Shannon entropy: -sum(w_i * log(w_i)) for positive weights."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    h = 0.0
    for w in weights:
        if w > 0:
            p = w / total
            h -= p * math.log(p)
    return h


def _contraction_rate(rho_series: list[float], rho_star: float) -> float | None:
    """Empirical contraction rate κ.

    κ = mean(|ρ_{t+1} - ρ*| / |ρ_t - ρ*|) over rounds where |ρ_t - ρ*| > 0.

    κ < 1 consistently → empirical contraction (connects to O(log(1/ε)) termination).
    """
    if len(rho_series) < 2:
        return None
    ratios = []
    for i in range(len(rho_series) - 1):
        denom = abs(rho_series[i] - rho_star)
        if denom > 1e-8:
            numer = abs(rho_series[i + 1] - rho_star)
            ratios.append(numer / denom)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _detect_limit_cycle(series: list[float], min_period: int = 2, max_period: int = 4) -> bool:
    """Detect periodicity in the last N rounds of a time series.

    Checks if the tail of the series repeats with period P in [min_period, max_period].
    Uses simple autocorrelation-like check: compare series[-k] with series[-k-P].
    """
    n = len(series)
    if n < 2 * max_period:
        return False
    for period in range(min_period, max_period + 1):
        tail_len = min(2 * period, n)
        tail = series[-tail_len:]
        if len(tail) < 2 * period:
            continue
        match = True
        for i in range(period):
            if abs(tail[i] - tail[i + period]) > 0.05:
                match = False
                break
        if match:
            return True
    return False


def _js_non_monotonic(js_series: list[float]) -> bool:
    """True if JS increases after previously decreasing (oscillatory convergence)."""
    if len(js_series) < 3:
        return False
    was_decreasing = False
    for i in range(1, len(js_series)):
        delta = js_series[i] - js_series[i - 1]
        if delta < -1e-8:
            was_decreasing = True
        elif delta > 1e-8 and was_decreasing:
            return True
    return False


def _convergence_window_round(
    js_series: list[float], epsilon: float, window: int
) -> int | None:
    """First round where JS < epsilon for `window` consecutive rounds."""
    if len(js_series) < window:
        return None
    for start in range(len(js_series) - window + 1):
        if all(js_series[start + k] < epsilon for k in range(window)):
            return start
    return None


def _quadrant_transition_metrics(quadrants: list[str]) -> dict[str, float | None]:
    """Compute quadrant transition matrix metrics.

    P(Q_{t+1} | Q_t) based metrics:
    - healthy_persistence: P(Healthy → Healthy)
    - regression_rate: P(Healthy → Stuck or Chaotic)
    - chaotic_escape_rate: P(Chaotic → Healthy or Converged)
    - transition_entropy: Shannon entropy of the transition distribution
    """
    if len(quadrants) < 2:
        return {
            "healthy_persistence": None,
            "regression_rate": None,
            "chaotic_escape_rate": None,
            "transition_entropy": None,
        }

    # Build transition counts: {from_state: {to_state: count}}
    trans: dict[str, dict[str, int]] = {}
    for i in range(len(quadrants) - 1):
        src = quadrants[i]
        dst = quadrants[i + 1]
        if src not in trans:
            trans[src] = {}
        trans[src][dst] = trans[src].get(dst, 0) + 1

    def p_transition(src: str, dst_set: set[str]) -> float | None:
        """P(dst in dst_set | src)"""
        if src not in trans:
            return None
        total = sum(trans[src].values())
        if total == 0:
            return None
        hits = sum(trans[src].get(d, 0) for d in dst_set)
        return hits / total

    healthy_persist = p_transition("healthy", {"healthy"})
    regression = p_transition("healthy", {"stuck", "chaotic"})
    chaotic_escape = p_transition("chaotic", {"healthy", "converged"})

    # Transition entropy across all observed transitions
    all_transitions: list[float] = []
    for src, dsts in trans.items():
        total = sum(dsts.values())
        for count in dsts.values():
            if count > 0:
                p = count / total
                all_transitions.append(p)

    t_entropy = _entropy(all_transitions) if all_transitions else 0.0

    return {
        "healthy_persistence": healthy_persist,
        "regression_rate": regression,
        "chaotic_escape_rate": chaotic_escape,
        "transition_entropy": t_entropy,
    }


def _paranoia_metrics(rho_series: list[float], rho_star: float) -> dict[str, float | None]:
    """Compute paranoia and realignment rates from ρ trajectory.

    For each round transition:
    - T→F (paranoia): ρ_t >= ρ* but ρ_{t+1} < ρ* (good → bad after PID push)
    - F→T (realignment): ρ_t < ρ* but ρ_{t+1} >= ρ* (bad → good after PID push)

    paranoia_rate = P(T→F | previously correct)
    realignment_rate = P(F→T | previously wrong)
    sycophancy_ratio = paranoia_rate / realignment_rate (lower = better)
    net_effect = realignment_rate - paranoia_rate (positive = net beneficial)
    """
    if len(rho_series) < 2:
        return {
            "paranoia_rate": None,
            "realignment_rate": None,
            "sycophancy_ratio": None,
            "net_effect": None,
        }

    t_to_f = 0  # paranoia events
    f_to_t = 0  # realignment events
    was_good = 0  # count of rounds where ρ >= ρ* (and we have a next round)
    was_bad = 0   # count of rounds where ρ < ρ*

    for i in range(len(rho_series) - 1):
        curr_good = rho_series[i] >= rho_star
        next_good = rho_series[i + 1] >= rho_star
        if curr_good:
            was_good += 1
            if not next_good:
                t_to_f += 1
        else:
            was_bad += 1
            if next_good:
                f_to_t += 1

    paranoia = _safe_div(t_to_f, was_good)
    realignment = _safe_div(f_to_t, was_bad)
    syco_ratio = _safe_div(paranoia, realignment) if realignment > 0 else None
    net = realignment - paranoia

    return {
        "paranoia_rate": paranoia,
        "realignment_rate": realignment,
        "sycophancy_ratio": syco_ratio,
        "net_effect": net,
    }


# =============================================================================
# REPLICATE AGGREGATION
# =============================================================================


def aggregate_replicates(results: list[dict]) -> dict:
    """Aggregate metrics across replicates for the same config.

    Computes mean and std for numeric metrics across replicates.
    Returns a single row with _mean and _std suffixes.
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    agg: dict[str, Any] = {}
    # Copy config columns from first result
    config_keys = {
        "run_id", "group", "param", "value", "model_name", "temperature",
        "num_tickers", "num_agents", "Kp", "Ki", "Kd", "rho_star",
        "gamma_beta", "delta_js", "delta_s", "delta_beta", "epsilon", "mu",
        "initial_beta",
        "enable_adversarial", "scenario", "status", "stability_check",
        "non_oscillation_check", "stochastic_regime",
    }
    for k in config_keys:
        if k in results[0]:
            agg[k] = results[0][k]

    # Numeric columns to aggregate
    numeric_keys = [
        "rounds_used", "final_rho_bar", "mean_rho_bar", "final_beta",
        "beta_range", "sycophancy_count", "quadrant_stuck_pct",
        "quadrant_chaotic_pct", "quadrant_converged_pct", "quadrant_healthy_pct",
        "steady_state_error", "beta_overshoot", "settling_round",
        "rho_variance", "mean_JS", "rho_sign_change_count",
        "rho_contraction_rate", "empirical_kappa",
        "paranoia_rate", "realignment_rate", "net_effect",
        "healthy_persistence", "regression_rate", "chaotic_escape_rate",
        "transition_entropy", "escalation_count", "stuck_rounds",
        "experimental_allocation_entropy", "experimental_concentration_index",
        "experimental_num_active_positions", "experimental_max_weight",
        "elapsed_seconds",
    ]

    for k in numeric_keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            agg[f"{k}_mean"] = sum(vals) / len(vals)
            agg[f"{k}_std"] = _std(vals)
        else:
            agg[f"{k}_mean"] = None
            agg[f"{k}_std"] = None

    # Boolean columns: rate
    bool_keys = [
        "beta_oscillation_flag", "rho_oscillation_flag", "rho_limit_cycle_flag",
        "JS_monotonicity_flag", "converged_single", "convergence_window_met",
        "control_stable", "behavioral_stable",
    ]
    for k in bool_keys:
        vals = [1 if r.get(k) else 0 for r in results]
        agg[f"{k}_rate"] = sum(vals) / len(vals)

    # Cross-replicate variance metrics
    rho_finals = [r["final_rho_bar"] for r in results if r.get("final_rho_bar") is not None]
    beta_finals = [r["final_beta"] for r in results if r.get("final_beta") is not None]
    conv_met = [1 if r.get("convergence_window_met") else 0 for r in results]

    agg["rho_std_across_replicates"] = _std(rho_finals) if rho_finals else None
    agg["beta_std_across_replicates"] = _std(beta_finals) if beta_finals else None
    agg["convergence_rate_across_replicates"] = (
        sum(conv_met) / len(conv_met) if conv_met else None
    )

    agg["num_replicates"] = len(results)
    return agg


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
