"""L1 unit tests for PID controller behavior.

Tests cover:
    - Beta increases when rho_bar is low (high disagreement / poor quality)
    - Beta decreases when rho_bar is high (good quality)
    - Controller stabilizes over 20 rounds with constant rho_bar=0.8
    - No oscillatory instability (<=3 sign changes in beta sequence)
    - check_stability() and check_non_oscillation() from stability.py
"""

import pytest

from eval.PID.controller import PIDController
from eval.PID.types import PIDConfig, PIDGains
from eval.PID.stability import check_stability, check_non_oscillation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PIDConfig:
    """Build a PIDConfig with sensible defaults, allowing overrides."""
    defaults = dict(
        gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
        rho_star=0.8,
        gamma_beta=0.9,
        mu=1.0,
        delta_s=0.05,
        T_max=20,
        epsilon=0.01,
    )
    defaults.update(overrides)
    return PIDConfig(**defaults)


def _run_n_steps(controller: PIDController, rho_bar: float, n: int) -> list:
    """Run n PID steps with constant rho_bar and return the list of PIDStepResults."""
    results = []
    for _ in range(n):
        results.append(controller.step(rho_bar=rho_bar))
    return results


def _count_sign_changes(values: list[float]) -> int:
    """Count the number of sign changes in a sequence of deltas."""
    if len(values) < 2:
        return 0
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    changes = 0
    for i in range(1, len(deltas)):
        if deltas[i - 1] != 0.0 and deltas[i] != 0.0:
            if (deltas[i - 1] > 0) != (deltas[i] > 0):
                changes += 1
    return changes


# ---------------------------------------------------------------------------
# Beta increases when rho_bar is low (poor quality)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBetaIncreasesOnLowQuality:
    """When rho_bar < rho_star, the PID error is positive, so the controller
    should push beta upward (toward more exploration).

    With low rho_bar and low JS, the debate is in the STUCK quadrant,
    which forces beta up by delta_beta.
    """

    def test_beta_rises_after_one_step_low_rho(self):
        initial_beta = 0.5
        ctrl = PIDController(_make_config(), initial_beta=initial_beta)
        result = ctrl.step(rho_bar=0.3)
        assert result.beta_new > initial_beta

    def test_beta_rises_over_multiple_low_quality_rounds(self):
        ctrl = PIDController(_make_config(), initial_beta=0.3)
        results = _run_n_steps(ctrl, rho_bar=0.3, n=5)
        betas = [0.3] + [r.beta_new for r in results]
        # Overall trend should be upward
        assert betas[-1] > betas[0]

    def test_positive_error_when_below_target(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        result = ctrl.step(rho_bar=0.5)
        # e_t = rho_star - rho_bar = 0.8 - 0.5 = 0.3
        assert result.e_t == pytest.approx(0.3)
        assert result.e_t > 0.0


# ---------------------------------------------------------------------------
# Beta decreases when rho_bar is high (good quality)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBetaDecreasesOnHighQuality:
    """When rho_bar >= rho_star, the PID error is zero or negative. Combined
    with the gamma_beta momentum decay, beta should decrease or hold steady.

    With high rho_bar and low JS, the debate is in the CONVERGED quadrant,
    which applies gentle decay: beta *= gamma_beta.
    """

    def test_beta_decreases_after_one_step_high_rho(self):
        initial_beta = 0.5
        ctrl = PIDController(_make_config(), initial_beta=initial_beta)
        result = ctrl.step(rho_bar=0.95)
        assert result.beta_new < initial_beta

    def test_negative_error_when_above_target(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        result = ctrl.step(rho_bar=1.0)
        # e_t = 0.8 - 1.0 = -0.2
        assert result.e_t == pytest.approx(-0.2)
        assert result.e_t < 0.0

    def test_beta_trends_down_under_sustained_high_quality(self):
        ctrl = PIDController(_make_config(), initial_beta=0.8)
        results = _run_n_steps(ctrl, rho_bar=0.95, n=10)
        betas = [0.8] + [r.beta_new for r in results]
        assert betas[-1] < betas[0]


# ---------------------------------------------------------------------------
# Controller stabilizes (convergence test)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestControllerStabilization:
    """Running 20 rounds at constant rho_bar = rho_star = 0.8 should cause
    beta to converge. The error is zero each round (no sycophancy), so the
    P-term vanishes, the I-term stays near zero, and the D-term is zero.
    Beta should decay toward a steady-state value.
    """

    def test_beta_converges_at_target(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        results = _run_n_steps(ctrl, rho_bar=0.8, n=20)
        betas = [r.beta_new for r in results]
        # After 20 rounds at exactly rho_star, beta changes should be tiny
        last_few = betas[-5:]
        spread = max(last_few) - min(last_few)
        assert spread < 0.05, f"Beta did not converge: last 5 values = {last_few}"

    def test_beta_stays_bounded_during_convergence(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        results = _run_n_steps(ctrl, rho_bar=0.8, n=20)
        for r in results:
            assert 0.0 <= r.beta_new <= 1.0

    def test_error_stays_near_zero_at_target(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        results = _run_n_steps(ctrl, rho_bar=0.8, n=20)
        for r in results:
            assert abs(r.e_t) < 1e-9

    def test_convergence_from_high_initial_beta(self):
        ctrl = PIDController(_make_config(), initial_beta=0.95)
        results = _run_n_steps(ctrl, rho_bar=0.8, n=20)
        betas = [r.beta_new for r in results]
        last_few = betas[-5:]
        spread = max(last_few) - min(last_few)
        assert spread < 0.08, f"Beta did not converge from high start: last 5 = {last_few}"


# ---------------------------------------------------------------------------
# No oscillatory instability
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestNoOscillatoryInstability:
    """Beta should not exhibit wild oscillations. We check that the number
    of sign changes in consecutive beta deltas is at most 3 over a
    20-round simulation.
    """

    def test_low_sign_changes_constant_rho(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        results = _run_n_steps(ctrl, rho_bar=0.6, n=20)
        betas = [0.5] + [r.beta_new for r in results]
        changes = _count_sign_changes(betas)
        assert changes <= 3, f"Too many sign changes ({changes}) in beta sequence"

    def test_low_sign_changes_improving_rho(self):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        betas = [0.5]
        for rho in [0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8,
                     0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]:
            result = ctrl.step(rho_bar=rho)
            betas.append(result.beta_new)
        changes = _count_sign_changes(betas)
        assert changes <= 3, f"Too many sign changes ({changes}) in beta sequence"

    def test_beta_does_not_bounce_between_extremes(self):
        """Beta should not alternate between near-0 and near-1."""
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        results = _run_n_steps(ctrl, rho_bar=0.6, n=20)
        betas = [r.beta_new for r in results]
        for i in range(1, len(betas)):
            jump = abs(betas[i] - betas[i - 1])
            assert jump < 0.5, (
                f"Beta jumped by {jump:.3f} between rounds {i-1} and {i}: "
                f"{betas[i-1]:.3f} -> {betas[i]:.3f}"
            )


# ---------------------------------------------------------------------------
# check_stability() from stability.py
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCheckStabilityBehavior:
    """Verify check_stability returns correct verdicts for known gain configs."""

    def test_conservative_gains_are_stable(self):
        gains = PIDGains(Kp=0.05, Ki=0.005, Kd=0.01)
        assert check_stability(gains, T_max=20, gamma_beta=0.9) is True

    def test_aggressive_gains_are_unstable(self):
        gains = PIDGains(Kp=2.0, Ki=1.0, Kd=1.0)
        assert check_stability(gains, T_max=20, gamma_beta=0.9) is False

    def test_default_config_gains_are_stable(self):
        """The default gains used in _make_config should pass stability."""
        gains = PIDGains(Kp=0.15, Ki=0.01, Kd=0.03)
        assert check_stability(gains, T_max=20, gamma_beta=0.9) is True

    def test_zero_gamma_always_stable(self):
        gains = PIDGains(Kp=50.0, Ki=10.0, Kd=20.0)
        assert check_stability(gains, T_max=100, gamma_beta=0.0) is True

    def test_stability_sensitive_to_tmax(self):
        """Higher T_max makes the integral bound larger, potentially unstable."""
        gains = PIDGains(Kp=0.3, Ki=0.05, Kd=0.1)
        # Should be stable at low T_max
        assert check_stability(gains, T_max=5, gamma_beta=0.9) is True
        # May become unstable at high T_max due to integral accumulation
        assert check_stability(gains, T_max=100, gamma_beta=0.9) is False


# ---------------------------------------------------------------------------
# check_non_oscillation() from stability.py
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCheckNonOscillationBehavior:
    """Verify check_non_oscillation returns correct verdicts."""

    def test_conservative_gains_do_not_oscillate(self):
        gains = PIDGains(Kp=0.01, Ki=0.001, Kd=0.005)
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is True

    def test_aggressive_gains_oscillate(self):
        gains = PIDGains(Kp=0.5, Ki=0.1, Kd=0.2)
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is False

    def test_non_oscillation_stricter_than_stability(self):
        """A config can pass stability but fail non-oscillation."""
        gains = PIDGains(Kp=0.3, Ki=0.05, Kd=0.1)
        gamma = 0.9
        T_max = 10
        stable = check_stability(gains, T_max, gamma)
        non_osc = check_non_oscillation(gains, T_max, gamma)
        # If it passes non-oscillation, it must also pass stability
        if non_osc:
            assert stable

    def test_sycophancy_mu_can_break_non_oscillation(self):
        """Adding sycophancy penalty inflates e_max, which can break non-oscillation."""
        gains = PIDGains(Kp=0.01, Ki=0.001, Kd=0.005)
        # Without sycophancy
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is True
        # With large mu
        assert check_non_oscillation(
            gains, T_max=10, gamma_beta=0.9, rho_star=0.8, mu=5.0
        ) is False
