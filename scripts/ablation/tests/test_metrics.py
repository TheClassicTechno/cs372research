"""Tests for scripts/ablation/metrics.py — metric extraction and aggregation."""

from __future__ import annotations

import math

import pytest

from scripts.ablation.config import BASELINE, CONVERGENCE_WINDOW, EPSILON_BAND, OSCILLATION_K
from scripts.ablation.metrics import (
    _contraction_rate,
    _convergence_window_round,
    _detect_limit_cycle,
    _entropy,
    _js_non_monotonic,
    _paranoia_metrics,
    _quadrant_transition_metrics,
    _safe_div,
    _sign_changes,
    _std,
    _variance,
    aggregate_replicates,
    extract_metrics,
)


# ===========================================================================
# Helper functions
# ===========================================================================


class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 2) == 5.0

    def test_zero_denominator(self):
        assert _safe_div(10, 0) == 0.0

    def test_zero_denominator_custom_default(self):
        assert _safe_div(10, 0, -1.0) == -1.0


class TestSignChanges:
    def test_empty_list(self):
        assert _sign_changes([]) == 0

    def test_single_element(self):
        assert _sign_changes([1.0]) == 0

    def test_no_changes(self):
        assert _sign_changes([1.0, 2.0, 3.0]) == 0

    def test_alternating_signs(self):
        assert _sign_changes([1.0, -1.0, 1.0, -1.0]) == 3

    def test_one_change(self):
        assert _sign_changes([1.0, 2.0, -1.0]) == 1

    def test_zeros_dont_trigger_changes(self):
        assert _sign_changes([1.0, 0.0, -1.0]) == 0


class TestVariance:
    def test_empty_list(self):
        assert _variance([]) == 0.0

    def test_single_element(self):
        assert _variance([5.0]) == 0.0

    def test_equal_elements(self):
        assert _variance([3.0, 3.0, 3.0]) == 0.0

    def test_known_variance(self):
        # Population variance of [1, 2, 3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        result = _variance([1.0, 2.0, 3.0])
        assert abs(result - 2 / 3) < 1e-10


class TestEntropy:
    def test_empty_weights(self):
        assert _entropy([]) == 0.0

    def test_single_weight(self):
        # H = -1 * log(1) = 0
        assert _entropy([1.0]) == 0.0

    def test_uniform_two_weights(self):
        # H = -2 * (0.5 * log(0.5)) = log(2) ≈ 0.693
        h = _entropy([1.0, 1.0])
        assert abs(h - math.log(2)) < 1e-10

    def test_all_zeros(self):
        assert _entropy([0.0, 0.0]) == 0.0

    def test_skewed_weights(self):
        # Very skewed → low entropy
        h = _entropy([100.0, 0.001])
        assert h < 0.1


class TestStd:
    def test_empty(self):
        assert _std([]) == 0.0

    def test_single(self):
        assert _std([5.0]) == 0.0

    def test_known_std(self):
        # Sample std of [2, 4, 4, 4, 5, 5, 7, 9]
        vals = [2, 4, 4, 4, 5, 5, 7, 9]
        result = _std(vals)
        expected = math.sqrt(sum((v - 5) ** 2 for v in vals) / 7)
        assert abs(result - expected) < 1e-10


# ===========================================================================
# Contraction rate
# ===========================================================================


class TestContractionRate:
    def test_short_series(self):
        assert _contraction_rate([0.5], 0.8) is None

    def test_contracting_series(self):
        # Each step moves 50% closer to rho_star=1.0
        series = [0.0, 0.5, 0.75, 0.875]
        kappa = _contraction_rate(series, 1.0)
        assert kappa is not None
        assert abs(kappa - 0.5) < 0.01

    def test_diverging_series(self):
        # Moving away from rho_star
        series = [0.8, 0.7, 0.5, 0.2]
        kappa = _contraction_rate(series, 0.8)
        assert kappa is not None
        assert kappa > 1.0

    def test_at_target_series(self):
        # All at rho_star → denom < 1e-8, no ratios
        series = [0.8, 0.8, 0.8]
        kappa = _contraction_rate(series, 0.8)
        assert kappa is None


# ===========================================================================
# Limit cycle detection
# ===========================================================================


class TestDetectLimitCycle:
    def test_short_series(self):
        assert _detect_limit_cycle([0.5, 0.6]) is False

    def test_no_cycle(self):
        series = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert _detect_limit_cycle(series) is False

    def test_period_2_cycle(self):
        # Alternating 0.3 and 0.7 for 8+ rounds
        series = [0.3, 0.7] * 5
        assert _detect_limit_cycle(series) is True

    def test_period_3_cycle(self):
        series = [0.2, 0.5, 0.8] * 4
        assert _detect_limit_cycle(series) is True


# ===========================================================================
# JS non-monotonic
# ===========================================================================


class TestJsNonMonotonic:
    def test_short_series(self):
        assert _js_non_monotonic([0.1, 0.05]) is False

    def test_monotonically_decreasing(self):
        assert _js_non_monotonic([0.5, 0.3, 0.1, 0.05]) is False

    def test_increase_after_decrease(self):
        assert _js_non_monotonic([0.5, 0.3, 0.1, 0.4]) is True

    def test_always_increasing(self):
        assert _js_non_monotonic([0.1, 0.2, 0.3, 0.4]) is False

    def test_flat_then_increase(self):
        # Flat is not "decreasing", so no oscillation detection
        assert _js_non_monotonic([0.3, 0.3, 0.3, 0.4]) is False


# ===========================================================================
# Convergence window
# ===========================================================================


class TestConvergenceWindowRound:
    def test_no_convergence(self):
        js = [0.1, 0.2, 0.15, 0.12]
        assert _convergence_window_round(js, 0.01, 2) is None

    def test_single_round_not_enough(self):
        js = [0.1, 0.005, 0.1, 0.1]
        assert _convergence_window_round(js, 0.01, 2) is None

    def test_two_consecutive_rounds(self):
        js = [0.1, 0.005, 0.003, 0.1]
        result = _convergence_window_round(js, 0.01, 2)
        assert result == 1  # Starts at index 1

    def test_at_end_of_series(self):
        js = [0.2, 0.1, 0.005, 0.003]
        result = _convergence_window_round(js, 0.01, 2)
        assert result == 2

    def test_series_too_short_for_window(self):
        js = [0.005]
        assert _convergence_window_round(js, 0.01, 2) is None

    def test_window_of_one(self):
        js = [0.1, 0.005, 0.2]
        result = _convergence_window_round(js, 0.01, 1)
        assert result == 1


# ===========================================================================
# Quadrant transition metrics
# ===========================================================================


class TestQuadrantTransitionMetrics:
    def test_single_quadrant(self):
        result = _quadrant_transition_metrics(["healthy"])
        assert result["healthy_persistence"] is None

    def test_all_healthy(self):
        result = _quadrant_transition_metrics(["healthy"] * 5)
        assert result["healthy_persistence"] == 1.0

    def test_healthy_to_stuck_regression(self):
        quads = ["healthy", "stuck", "stuck"]
        result = _quadrant_transition_metrics(quads)
        assert result["regression_rate"] == 1.0

    def test_chaotic_escape(self):
        quads = ["chaotic", "healthy"]
        result = _quadrant_transition_metrics(quads)
        assert result["chaotic_escape_rate"] == 1.0

    def test_chaotic_no_escape(self):
        quads = ["chaotic", "chaotic", "chaotic"]
        result = _quadrant_transition_metrics(quads)
        assert result["chaotic_escape_rate"] == 0.0

    def test_transition_entropy_zero_for_single_state(self):
        quads = ["stuck"] * 5
        result = _quadrant_transition_metrics(quads)
        assert result["transition_entropy"] == 0.0

    def test_mixed_transitions_positive_entropy(self):
        quads = ["stuck", "chaotic", "healthy", "converged", "stuck"]
        result = _quadrant_transition_metrics(quads)
        assert result["transition_entropy"] > 0


# ===========================================================================
# Paranoia metrics
# ===========================================================================


class TestParanoiaMetrics:
    def test_short_series(self):
        result = _paranoia_metrics([0.5], 0.8)
        assert result["paranoia_rate"] is None

    def test_always_above_target(self):
        result = _paranoia_metrics([0.9, 0.85, 0.88], 0.8)
        assert result["paranoia_rate"] == 0.0
        assert result["realignment_rate"] is not None

    def test_paranoia_event(self):
        # Above target then drops below
        result = _paranoia_metrics([0.85, 0.7], 0.8)
        assert result["paranoia_rate"] == 1.0

    def test_realignment_event(self):
        # Below target then rises above
        result = _paranoia_metrics([0.7, 0.85], 0.8)
        assert result["realignment_rate"] == 1.0

    def test_net_effect_positive(self):
        # Mostly below target, frequent realignment, rare paranoia
        # [0.5, 0.85, 0.85, 0.5, 0.85] with rho_star=0.8
        # 0→1: bad→good (realign), 1→2: good→good (stay), 2→3: good→bad (paranoia), 3→4: bad→good (realign)
        # was_bad=2, f_to_t=2 → realignment=1.0
        # was_good=2, t_to_f=1 → paranoia=0.5
        # net_effect = 1.0 - 0.5 = 0.5
        result = _paranoia_metrics([0.5, 0.85, 0.85, 0.5, 0.85], 0.8)
        assert result["net_effect"] is not None
        assert result["net_effect"] > 0

    def test_net_effect_negative(self):
        # Mostly above target, frequent paranoia, rare realignment
        # [0.85, 0.5, 0.85, 0.85, 0.5] with rho_star=0.8
        # 0→1: good→bad (paranoia), 1→2: bad→good (realign), 2→3: good→good (stay), 3→4: good→bad (paranoia)
        # was_good=3, t_to_f=2 → paranoia=2/3
        # was_bad=1, f_to_t=1 → realignment=1.0
        # net_effect = 1.0 - 2/3 = 0.333... > 0
        # Need a case where paranoia_rate > realignment_rate
        # [0.85, 0.5, 0.5, 0.85, 0.5] with rho_star=0.8
        # 0→1: good→bad (paranoia), 1→2: bad→bad (stay), 2→3: bad→good (realign), 3→4: good→bad (paranoia)
        # was_good=2, t_to_f=2 → paranoia=1.0
        # was_bad=2, f_to_t=1 → realignment=0.5
        # net = 0.5 - 1.0 = -0.5
        result = _paranoia_metrics([0.85, 0.5, 0.5, 0.85, 0.5], 0.8)
        assert result["net_effect"] is not None
        assert result["net_effect"] < 0


# ===========================================================================
# extract_metrics — integration
# ===========================================================================


class TestExtractMetrics:
    def test_converging_trace_basics(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        assert m["rounds_used"] == 5
        assert m["final_rho_bar"] == pytest.approx(0.81, abs=0.01)
        assert m["final_beta"] == pytest.approx(0.45, abs=0.01)
        assert "status" not in m  # status set by runner, not metrics

    def test_converging_trace_stability(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        assert m["stability_check"] is True
        assert m["non_oscillation_check"] is True
        assert m["control_stable"] is True  # stable check + no beta oscillation

    def test_converging_trace_convergence(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        assert m["converged_single"] is True
        assert m["convergence_window_met"] is True

    def test_converging_trace_quadrant_distribution(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        quad_sum = (
            m["quadrant_stuck_pct"] + m["quadrant_chaotic_pct"]
            + m["quadrant_converged_pct"] + m["quadrant_healthy_pct"]
        )
        assert abs(quad_sum - 1.0) < 0.01

    def test_converging_trace_steady_state_error(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        # rho_star=0.8, final_rho=0.81 → error ≈ -0.01
        assert m["steady_state_error"] is not None
        assert abs(m["steady_state_error"]) < 0.1

    def test_oscillating_trace_flags(self, oscillating_trace, baseline_params):
        m = extract_metrics(oscillating_trace, baseline_params, True, True)
        # 10 rounds of alternating rho/beta → many sign changes
        assert m["beta_oscillation_flag"] is True
        assert m["rho_oscillation_flag"] is True
        assert m["control_stable"] is False  # beta oscillation → not control stable

    def test_oscillating_trace_behavioral_unstable(self, oscillating_trace, baseline_params):
        m = extract_metrics(oscillating_trace, baseline_params, True, True)
        assert m["behavioral_stable"] is False

    def test_stuck_trace_high_stuck_pct(self, stuck_trace, baseline_params):
        m = extract_metrics(stuck_trace, baseline_params, True, True)
        assert m["quadrant_stuck_pct"] == 1.0
        assert m["dominant_quadrant"] == "stuck"
        assert m["stuck_rounds"] == 6

    def test_empty_trace_handles_gracefully(self, empty_trace, baseline_params):
        m = extract_metrics(empty_trace, baseline_params, True, True)
        assert m["rounds_used"] == 0
        assert m["final_rho_bar"] is None
        assert m["final_beta"] is None
        assert m["beta_oscillation_flag"] is False

    def test_sycophancy_count(self, baseline_params):
        """Test sycophancy count from s_t flags."""
        from scripts.ablation.tests.conftest import _make_pid_event, _make_trace

        events = [
            _make_pid_event(0, 0.5, 0.1, 0.4, 0.5, 0.3, 0.04, 1, "stuck"),
            _make_pid_event(1, 0.6, 0.08, 0.5, 0.48, 0.2, 0.03, 1, "healthy"),
            _make_pid_event(2, 0.7, 0.05, 0.6, 0.46, 0.1, 0.02, 0, "healthy"),
        ]
        trace = _make_trace(events)
        m = extract_metrics(trace, baseline_params, True, True)
        assert m["sycophancy_count"] == 2

    def test_stochastic_regime_flag(self, converging_trace):
        """High temperature → stochastic regime flag set."""
        params = dict(BASELINE)
        params.update({
            "run_id": "test", "group": "temp", "param": "temperature",
            "value": "1.0", "scenario": "neutral", "replicate": 0,
            "temperature": 1.0,
        })
        m = extract_metrics(converging_trace, params, True, True)
        assert m["stochastic_regime"] is True

    def test_low_temp_not_stochastic(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        assert m["stochastic_regime"] is False

    def test_escalation_count(self, baseline_params):
        """Large |u_t| triggers escalation count."""
        from scripts.ablation.tests.conftest import _make_pid_event, _make_trace

        events = [
            _make_pid_event(0, 0.5, 0.1, 0.4, 0.5, 0.3, 0.5, 0, "stuck"),
            _make_pid_event(1, 0.6, 0.08, 0.5, 0.48, 0.2, 0.1, 0, "healthy"),
            _make_pid_event(2, 0.7, 0.05, 0.6, 0.46, 0.1, -0.4, 0, "healthy"),
        ]
        trace = _make_trace(events)
        m = extract_metrics(trace, baseline_params, True, True)
        # |0.5| > 0.3 and |-0.4| > 0.3 → 2 escalations
        assert m["escalation_count"] == 2

    def test_portfolio_metrics_with_allocation(self, baseline_params):
        """Portfolio metrics extracted when allocation present."""
        from scripts.ablation.tests.conftest import _make_pid_event, _make_trace

        events = [
            _make_pid_event(0, 0.7, 0.05, 0.5, 0.5, 0.1, 0.01, 0, "healthy"),
        ]
        alloc = {"AAPL": 0.3, "NVDA": 0.3, "MSFT": 0.2, "GOOG": 0.1, "JPM": 0.1}
        trace = _make_trace(events, allocation=alloc)
        m = extract_metrics(trace, baseline_params, True, True)
        assert m["experimental_allocation_entropy"] is not None
        assert m["experimental_allocation_entropy"] > 0
        assert m["experimental_concentration_index"] == 0.3
        assert m["experimental_num_active_positions"] == 5
        assert m["experimental_max_weight"] == 0.3

    def test_portfolio_metrics_without_allocation(self, converging_trace, baseline_params):
        m = extract_metrics(converging_trace, baseline_params, True, True)
        assert m["experimental_allocation_entropy"] is None
        assert m["experimental_concentration_index"] is None

    def test_all_metric_keys_present(self, converging_trace, baseline_params):
        """Verify all expected metric keys are in the output."""
        m = extract_metrics(converging_trace, baseline_params, True, True)
        expected_keys = [
            "rounds_used", "final_rho_bar", "mean_rho_bar", "final_beta",
            "beta_range", "sycophancy_count",
            "quadrant_stuck_pct", "quadrant_chaotic_pct",
            "quadrant_converged_pct", "quadrant_healthy_pct", "dominant_quadrant",
            "steady_state_error", "beta_overshoot", "settling_round",
            "beta_oscillation_flag",
            "rho_variance", "mean_JS", "rho_sign_change_count",
            "rho_oscillation_flag", "rho_limit_cycle_flag", "rho_contraction_rate",
            "JS_monotonicity_flag",
            "converged_single", "convergence_window_met", "convergence_round",
            "healthy_persistence", "regression_rate", "chaotic_escape_rate",
            "transition_entropy",
            "paranoia_rate", "realignment_rate", "sycophancy_ratio", "net_effect",
            "stability_check", "non_oscillation_check",
            "control_stable", "behavioral_stable", "stochastic_regime",
            "escalation_count", "stuck_rounds", "high_correction_rounds",
            "empirical_kappa",
            "experimental_allocation_entropy", "experimental_concentration_index",
            "experimental_num_active_positions", "experimental_max_weight",
        ]
        for key in expected_keys:
            assert key in m, f"Missing metric key: {key}"


# ===========================================================================
# aggregate_replicates
# ===========================================================================


class TestAggregateReplicates:
    def test_empty_list(self):
        assert aggregate_replicates([]) == {}

    def test_single_result_returned_as_is(self):
        r = {"run_id": "test", "final_rho_bar": 0.75}
        assert aggregate_replicates([r]) == r

    def test_numeric_mean_computed(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        # 0.73, 0.75, 0.77 → mean=0.75
        assert agg["final_rho_bar_mean"] == pytest.approx(0.75, abs=0.001)

    def test_numeric_std_computed(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        assert agg["final_rho_bar_std"] is not None
        assert agg["final_rho_bar_std"] > 0

    def test_boolean_rate_computed(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        # All have control_stable=True → rate=1.0
        assert agg["control_stable_rate"] == 1.0

    def test_cross_replicate_variance(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        assert "rho_std_across_replicates" in agg
        assert agg["rho_std_across_replicates"] > 0

    def test_num_replicates(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        assert agg["num_replicates"] == 3

    def test_config_columns_preserved(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        assert agg["group"] == "gains"
        assert agg["Kp"] == 0.15

    def test_convergence_rate_across_replicates(self, replicate_results):
        agg = aggregate_replicates(replicate_results)
        assert agg["convergence_rate_across_replicates"] == 1.0  # All converged
