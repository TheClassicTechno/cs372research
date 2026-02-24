"""Integration tests — multi-step PID simulations with realistic scenarios."""

import math
import pytest
from eval.PID.types import PIDGains, PIDConfig
from eval.PID.controller import PIDController
from eval.PID.stability import check_stability, validate_gains
from eval.PID.termination import termination_bound, check_convergence


class TestConvergingSimulation:
    """Debate where scores steadily improve toward target."""

    def _make_controller(self):
        cfg = PIDConfig(
            gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
            rho_star=0.8,
            gamma_beta=0.9,
            mu=1.0,
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )
        # Validate with actual rho_star and mu so the sycophancy-corrected
        # bound is checked (e_max = max(0.8+1.0, 0.2) = 1.8).
        validate_gains(cfg.gains, cfg.T_max, cfg.gamma_beta,
                        rho_star=cfg.rho_star, mu=cfg.mu)
        return PIDController(cfg, initial_beta=0.5)

    def test_error_decreases(self):
        ctrl = self._make_controller()
        rho_values = [0.4, 0.5, 0.6, 0.65, 0.7, 0.74, 0.77, 0.79, 0.80]
        errors = []
        for rho in rho_values:
            result = ctrl.step(rho_bar=rho)
            errors.append(abs(result.e_t))
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.05  # Close to target

    def test_beta_settles(self):
        ctrl = self._make_controller()
        betas = []
        for rho in [0.5, 0.6, 0.7, 0.75, 0.78, 0.79, 0.80, 0.80, 0.80]:
            result = ctrl.step(rho_bar=rho)
            betas.append(result.beta_new)
        # Beta should stabilize as error approaches zero
        last_three = betas[-3:]
        spread = max(last_three) - min(last_three)
        assert spread < 0.2


class TestDivergingSimulation:
    """Debate where quality degrades — controller should push beta up."""

    def test_beta_increases_on_declining_quality(self):
        cfg = PIDConfig(
            gains=PIDGains(Kp=0.3, Ki=0.02, Kd=0.05),
            rho_star=0.8,
            gamma_beta=0.9,
            mu=0.0,  # No sycophancy for clarity
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )
        ctrl = PIDController(cfg, initial_beta=0.5)
        # Quality degrades
        rho_values = [0.7, 0.6, 0.5, 0.4, 0.3]
        results = [ctrl.step(rho_bar=rho) for rho in rho_values]
        # Error should be increasing (all positive since rho < rho_star)
        for r in results:
            assert r.e_t > 0.0
        # u_t should be positive (trying to correct)
        assert results[-1].u_t > 0.0


class TestSycophancyCorrectionSimulation:
    """Simulate sycophancy trigger mid-debate and verify the controller reacts."""

    def test_sycophancy_boosts_error(self):
        cfg = PIDConfig(
            gains=PIDGains(Kp=0.3, Ki=0.02, Kd=0.05),
            rho_star=0.8,
            gamma_beta=0.9,
            mu=1.0,
            delta_s=0.05,
            T_max=20,
            epsilon=0.01,
        )
        ctrl = PIDController(cfg, initial_beta=0.5)

        # Round 0: Normal
        r0 = ctrl.step(rho_bar=0.75, js_current=0.3, ov_current=0.8)
        assert r0.s_t == 0

        # Round 1: Sudden convergence (sycophancy)
        # JS drops by 0.2 (> delta_s=0.05), overlap drops
        r1 = ctrl.step(rho_bar=0.78, js_current=0.1, ov_current=0.6)
        assert r1.s_t == 1
        # Sycophancy adds mu=1.0 to error
        base_error = 0.8 - 0.78
        assert r1.e_t == pytest.approx(base_error + 1.0)

        # Round 2: Normal again (JS stabilizes)
        r2 = ctrl.step(rho_bar=0.78, js_current=0.1, ov_current=0.6)
        assert r2.s_t == 0  # No sharp drop from round 1 to round 2


class TestNumericalStability:
    """Edge cases for numerical precision."""

    def test_zero_scores(self):
        """All-zero inputs should not crash."""
        cfg = PIDConfig(
            gains=PIDGains(Kp=0.1, Ki=0.01, Kd=0.01),
            rho_star=0.0,
            gamma_beta=0.9,
        )
        ctrl = PIDController(cfg, initial_beta=0.5)
        result = ctrl.step(rho_bar=0.0)
        assert math.isfinite(result.u_t)
        assert math.isfinite(result.beta_new)

    def test_many_rounds_no_nan(self):
        """Run 100 rounds and verify no NaN/Inf values appear."""
        cfg = PIDConfig(
            gains=PIDGains(Kp=0.1, Ki=0.01, Kd=0.05),
            rho_star=0.8,
            gamma_beta=0.9,
        )
        ctrl = PIDController(cfg, initial_beta=0.5)
        for i in range(100):
            rho = 0.5 + 0.003 * i  # Slowly improving
            result = ctrl.step(rho_bar=rho)
            assert math.isfinite(result.e_t), f"NaN/Inf at round {i}"
            assert math.isfinite(result.u_t), f"NaN/Inf at round {i}"
            assert math.isfinite(result.beta_new), f"NaN/Inf at round {i}"
            assert 0.0 <= result.beta_new <= 1.0

    def test_extreme_gains_clipped(self):
        """Very large gains should still produce clipped beta in [0, 1]."""
        cfg = PIDConfig(
            gains=PIDGains(Kp=100.0, Ki=10.0, Kd=50.0),
            rho_star=0.8,
            gamma_beta=0.9,
        )
        ctrl = PIDController(cfg, initial_beta=0.5)
        result = ctrl.step(rho_bar=0.1)
        assert 0.0 <= result.beta_new <= 1.0


class TestTerminationIntegration:
    """Combine PID controller with termination bound logic."""

    def test_bound_vs_actual_rounds(self):
        """Actual convergence rounds should be within the theoretical bound."""
        kappa = 0.5
        d0 = 1.0
        epsilon = 0.01
        T_star = termination_bound(d0, epsilon, kappa)

        # Simulate contraction
        d_t = d0
        rounds = 0
        while d_t > epsilon and rounds < 1000:
            d_t *= kappa
            rounds += 1

        assert rounds <= math.ceil(T_star)

    def test_convergence_detection(self):
        """check_convergence should trigger when divergence hits threshold."""
        js_vals = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        epsilon = 0.01
        converged_at = None
        for i, js in enumerate(js_vals):
            if check_convergence(js, epsilon):
                converged_at = i
                break
        assert converged_at is not None
        assert js_vals[converged_at] <= epsilon
