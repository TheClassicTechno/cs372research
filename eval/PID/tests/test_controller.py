"""Tests for eval.PID.controller — error computation, PID output, stateful controller."""

import pytest
from eval.PID.types import PIDGains, PIDConfig
from eval.PID.controller import compute_error, compute_pid_output, PIDController


# ---------------------------------------------------------------------------
# compute_error (Eq 5)
# ---------------------------------------------------------------------------

class TestComputeError:
    def test_perfect_score_no_sycophancy(self):
        """When rho_bar == rho_star and s_t == 0, error is zero."""
        assert compute_error(0.8, 0.8, 1.0, 0) == 0.0

    def test_below_target(self):
        """Positive error when rho_bar < rho_star."""
        e = compute_error(0.8, 0.5, 0.0, 0)
        assert e == pytest.approx(0.3)

    def test_above_target(self):
        """Negative error when rho_bar > rho_star."""
        e = compute_error(0.8, 1.0, 0.0, 0)
        assert e == pytest.approx(-0.2)

    def test_sycophancy_adds_penalty(self):
        """s_t=1 adds μ to the error signal."""
        e_no_syc = compute_error(0.8, 0.8, 1.0, 0)
        e_syc = compute_error(0.8, 0.8, 1.0, 1)
        assert e_syc - e_no_syc == pytest.approx(1.0)

    def test_mu_scaling(self):
        """Different μ values scale the sycophancy penalty."""
        e = compute_error(0.8, 0.8, 2.5, 1)
        assert e == pytest.approx(2.5)

    def test_zero_mu_ignores_sycophancy(self):
        """When μ=0, sycophancy signal has no effect."""
        assert compute_error(0.8, 0.5, 0.0, 1) == compute_error(0.8, 0.5, 0.0, 0)


# ---------------------------------------------------------------------------
# compute_pid_output (Eq 6)
# ---------------------------------------------------------------------------

class TestComputePIDOutput:
    def test_proportional_only(self):
        gains = PIDGains(Kp=1.0, Ki=0.0, Kd=0.0)
        u, p, i, d = compute_pid_output(gains, e_t=0.3, integral=0.0, e_prev=0.0)
        assert u == pytest.approx(0.3)
        assert p == pytest.approx(0.3)
        assert i == pytest.approx(0.0)
        assert d == pytest.approx(0.0)

    def test_integral_only(self):
        gains = PIDGains(Kp=0.0, Ki=0.5, Kd=0.0)
        u, p, i, d = compute_pid_output(gains, e_t=0.0, integral=2.0, e_prev=0.0)
        assert u == pytest.approx(1.0)
        assert i == pytest.approx(1.0)

    def test_derivative_only(self):
        gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0)
        u, p, i, d = compute_pid_output(gains, e_t=0.5, integral=0.0, e_prev=0.3)
        assert u == pytest.approx(0.2)
        assert d == pytest.approx(0.2)

    def test_all_terms(self):
        gains = PIDGains(Kp=1.0, Ki=0.5, Kd=0.2)
        u, p, i, d = compute_pid_output(gains, e_t=0.4, integral=1.0, e_prev=0.2)
        expected_p = 1.0 * 0.4
        expected_i = 0.5 * 1.0
        expected_d = 0.2 * (0.4 - 0.2)
        assert u == pytest.approx(expected_p + expected_i + expected_d)

    def test_zero_error(self):
        gains = PIDGains(Kp=1.0, Ki=1.0, Kd=1.0)
        u, _, _, _ = compute_pid_output(gains, e_t=0.0, integral=0.0, e_prev=0.0)
        assert u == 0.0

    def test_negative_error(self):
        gains = PIDGains(Kp=1.0, Ki=0.0, Kd=0.0)
        u, p, _, _ = compute_pid_output(gains, e_t=-0.5, integral=0.0, e_prev=0.0)
        assert u == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# PIDController (stateful)
# ---------------------------------------------------------------------------

class TestPIDController:
    def _make_config(self, **overrides):
        defaults = dict(
            gains=PIDGains(Kp=0.3, Ki=0.05, Kd=0.1),
            rho_star=0.8,
            gamma_beta=0.9,
            mu=1.0,
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )
        defaults.update(overrides)
        return PIDConfig(**defaults)

    def test_first_step_no_sycophancy(self):
        """First step cannot trigger sycophancy (no history)."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        result = ctrl.step(rho_bar=0.7, js_current=0.1, ov_current=0.5)
        assert result.s_t == 0
        assert result.e_t == pytest.approx(0.1)  # 0.8 - 0.7

    def test_step_advances_state(self):
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        ctrl.step(rho_bar=0.7)
        assert ctrl.t == 1
        ctrl.step(rho_bar=0.75)
        assert ctrl.t == 2

    def test_beta_stays_in_bounds(self):
        """Beta should always be clipped to [0, 1]."""
        cfg = self._make_config(gains=PIDGains(Kp=5.0, Ki=0.0, Kd=0.0))
        ctrl = PIDController(cfg, initial_beta=0.9)
        # Large positive error -> large u_t -> beta might exceed 1
        result = ctrl.step(rho_bar=0.0)
        assert 0.0 <= result.beta_new <= 1.0

    def test_integral_accumulates(self):
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        ctrl.step(rho_bar=0.7)  # e = 0.1
        ctrl.step(rho_bar=0.7)  # e = 0.1
        assert ctrl.state.integral == pytest.approx(0.2)

    def test_sycophancy_triggers_on_sharp_js_drop(self):
        """Sycophancy fires when JS drops sharply and overlap drops."""
        cfg = self._make_config(delta_s=0.05)
        ctrl = PIDController(cfg, initial_beta=0.5)
        # Round 0: high JS, high overlap
        ctrl.step(rho_bar=0.7, js_current=0.3, ov_current=0.8)
        # Round 1: JS drops by 0.2 (> delta_s) and overlap drops
        result = ctrl.step(rho_bar=0.7, js_current=0.1, ov_current=0.7)
        assert result.s_t == 1

    def test_converging_simulation(self):
        """Multi-step simulation with improving scores should reduce error."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        errors = []
        for rho in [0.5, 0.6, 0.7, 0.75, 0.78, 0.80]:
            result = ctrl.step(rho_bar=rho)
            errors.append(abs(result.e_t))
        # Errors should generally decrease as rho approaches rho_star
        assert errors[-1] < errors[0]


# ---------------------------------------------------------------------------
# First-step derivative kick
# ---------------------------------------------------------------------------

class TestFirstStepDerivativeKick:
    """The D-term uses (e_t - e_prev) where e_prev defaults to 0.0 on round 0.

    This means the very first D-term = Kd * (e_0 - 0.0) = Kd * e_0, which can
    be a large spike if the initial error is far from zero.  This is a known
    PID design limitation ("derivative kick").  These tests verify the behavior
    is predictable and bounded.
    """

    def _make_config(self, **overrides):
        defaults = dict(
            gains=PIDGains(Kp=0.0, Ki=0.0, Kd=1.0),  # D-term only
            rho_star=0.8,
            gamma_beta=0.0,   # No momentum so u_t is applied directly
            mu=0.0,
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )
        defaults.update(overrides)
        return PIDConfig(**defaults)

    def test_first_step_d_term_equals_error(self):
        """On round 0, D-term = Kd * (e_0 - 0) = Kd * e_0."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        result = ctrl.step(rho_bar=0.5)  # e_0 = 0.8 - 0.5 = 0.3
        assert result.d_term == pytest.approx(1.0 * 0.3)

    def test_second_step_d_term_uses_real_prev(self):
        """On round 1, D-term uses real e_prev from round 0."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        ctrl.step(rho_bar=0.5)   # e_0 = 0.3
        r1 = ctrl.step(rho_bar=0.6)  # e_1 = 0.2, D = Kd * (0.2 - 0.3) = -0.1
        assert r1.d_term == pytest.approx(-0.1)

    def test_first_step_kick_bounded_by_clamp(self):
        """Even with a large D-term kick, beta stays in [0, 1]."""
        cfg = self._make_config(gains=PIDGains(Kp=0.0, Ki=0.0, Kd=10.0))
        ctrl = PIDController(cfg, initial_beta=0.5)
        result = ctrl.step(rho_bar=0.0)  # e = 0.8, D = 10*0.8 = 8.0
        assert 0.0 <= result.beta_new <= 1.0

    def test_zero_initial_error_no_kick(self):
        """If initial error is zero, no derivative kick."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        result = ctrl.step(rho_bar=0.8)  # e = 0.0
        assert result.d_term == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integral windup behavior
# ---------------------------------------------------------------------------

class TestIntegralWindup:
    """The integral accumulates all past errors without bound.  If the error
    stays positive for many rounds, the integral grows large and the I-term
    can dominate.  These tests verify the behavior is consistent (not a bug
    fix — just documenting how the system behaves under sustained error).
    """

    def _make_config(self, **overrides):
        defaults = dict(
            gains=PIDGains(Kp=0.0, Ki=0.1, Kd=0.0),  # I-term only
            rho_star=0.8,
            gamma_beta=0.0,
            mu=0.0,
            delta_s=0.05,
            T_max=100,
            epsilon=0.01,
        )
        defaults.update(overrides)
        return PIDConfig(**defaults)

    def test_integral_grows_under_sustained_error(self):
        """Integral should equal sum of all past errors."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        for _ in range(10):
            ctrl.step(rho_bar=0.5)  # e = 0.3 each round
        assert ctrl.state.integral == pytest.approx(10 * 0.3)

    def test_i_term_proportional_to_integral(self):
        """I-term = Ki * integral."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        for _ in range(5):
            ctrl.step(rho_bar=0.5)
        result = ctrl.step(rho_bar=0.5)
        # integral = 6 * 0.3 = 1.8, I-term = 0.1 * 1.8 = 0.18
        assert result.i_term == pytest.approx(0.1 * 6 * 0.3)

    def test_integral_decreases_when_error_reverses(self):
        """If quality overshoots target, negative errors reduce the integral."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        # 5 rounds below target
        for _ in range(5):
            ctrl.step(rho_bar=0.5)  # e = +0.3
        integral_before = ctrl.state.integral  # 1.5
        # 5 rounds above target
        for _ in range(5):
            ctrl.step(rho_bar=1.0)  # e = -0.2
        integral_after = ctrl.state.integral
        assert integral_after < integral_before
        assert integral_after == pytest.approx(1.5 + 5 * (-0.2))

    def test_beta_clamps_despite_large_integral(self):
        """Even with a huge accumulated integral, beta stays in [0, 1]."""
        ctrl = PIDController(self._make_config(), initial_beta=0.5)
        for _ in range(100):
            result = ctrl.step(rho_bar=0.0)  # e = 0.8 each round
        assert 0.0 <= result.beta_new <= 1.0


# ---------------------------------------------------------------------------
# Sycophancy-inflated error exceeding 1.0
# ---------------------------------------------------------------------------

class TestSycophancyInflatedError:
    """When sycophancy fires (s_t=1), error = (rho_star - rho_bar) + mu.
    With default mu=1.0 and rho_bar < rho_star, error can exceed 1.0.
    These tests verify the controller handles this correctly.
    """

    def _make_config(self):
        return PIDConfig(
            gains=PIDGains(Kp=0.1, Ki=0.01, Kd=0.05),
            rho_star=0.8,
            gamma_beta=0.9,
            mu=1.0,
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )

    def test_error_exceeds_one(self):
        """With mu=1.0 and rho_bar=0.5, error = 0.3 + 1.0 = 1.3."""
        e = compute_error(rho_star=0.8, rho_bar=0.5, mu=1.0, s_t=1)
        assert e == pytest.approx(1.3)
        assert e > 1.0

    def test_max_possible_error(self):
        """Worst case: rho_bar=0, s_t=1 -> error = rho_star + mu."""
        e = compute_error(rho_star=0.8, rho_bar=0.0, mu=1.0, s_t=1)
        assert e == pytest.approx(1.8)

    def test_controller_handles_large_error(self):
        """Controller should still produce finite, clamped beta."""
        import math
        cfg = self._make_config()
        ctrl = PIDController(cfg, initial_beta=0.5)
        # Round 0: set up JS/ov history
        ctrl.step(rho_bar=0.5, js_current=0.4, ov_current=0.9)
        # Round 1: trigger sycophancy (JS drops 0.3 > delta_s, overlap drops)
        result = ctrl.step(rho_bar=0.5, js_current=0.1, ov_current=0.6)
        assert result.s_t == 1
        assert result.e_t > 1.0
        assert math.isfinite(result.u_t)
        assert 0.0 <= result.beta_new <= 1.0
