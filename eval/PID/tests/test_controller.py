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
