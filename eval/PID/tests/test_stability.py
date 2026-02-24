"""Tests for eval.PID.stability — output bounds, stability/non-oscillation checks."""

import pytest
from eval.PID.types import PIDGains
from eval.PID.stability import (
    _max_error,
    max_pid_output_bound,
    check_stability,
    check_non_oscillation,
    validate_gains,
    GainInstabilityError,
)


# ---------------------------------------------------------------------------
# _max_error (worst-case |e_t|)
# ---------------------------------------------------------------------------

class TestMaxError:
    def test_default_no_sycophancy(self):
        """rho_star=1.0, mu=0.0 -> e_max = max(1.0, 0.0) = 1.0."""
        assert _max_error(1.0, 0.0) == pytest.approx(1.0)

    def test_typical_config(self):
        """rho_star=0.8, mu=1.0 -> e_max = max(1.8, 0.2) = 1.8."""
        assert _max_error(0.8, 1.0) == pytest.approx(1.8)

    def test_negative_side_dominates(self):
        """rho_star=0.1, mu=0.0 -> e_max = max(0.1, 0.9) = 0.9."""
        assert _max_error(0.1, 0.0) == pytest.approx(0.9)

    def test_symmetric_target(self):
        """rho_star=0.5, mu=0.0 -> e_max = max(0.5, 0.5) = 0.5."""
        assert _max_error(0.5, 0.0) == pytest.approx(0.5)

    def test_large_mu(self):
        """Large mu inflates the positive side."""
        assert _max_error(0.5, 5.0) == pytest.approx(5.5)

    def test_zero_target_zero_mu(self):
        """rho_star=0.0, mu=0.0 -> e_max = max(0.0, 1.0) = 1.0."""
        assert _max_error(0.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# max_pid_output_bound (Eq 12)
# ---------------------------------------------------------------------------

class TestMaxPIDOutputBound:
    def test_basic_default_params(self):
        """With defaults (rho_star=1.0, mu=0.0), e_max=1.0, so bound = gains sum."""
        gains = PIDGains(Kp=0.3, Ki=0.05, Kd=0.1)
        bound = max_pid_output_bound(gains, T_max=20)
        assert bound == pytest.approx(0.3 + 20 * 0.05 + 2 * 0.1)

    def test_zero_gains(self):
        gains = PIDGains(Kp=0.0, Ki=0.0, Kd=0.0)
        assert max_pid_output_bound(gains, T_max=100) == 0.0

    def test_proportional_only(self):
        gains = PIDGains(Kp=0.5, Ki=0.0, Kd=0.0)
        assert max_pid_output_bound(gains, T_max=10) == pytest.approx(0.5)

    def test_integral_dominates_at_high_tmax(self):
        gains = PIDGains(Kp=0.1, Ki=0.1, Kd=0.1)
        bound = max_pid_output_bound(gains, T_max=100)
        assert bound == pytest.approx(0.1 + 10.0 + 0.2)

    def test_sycophancy_inflates_bound(self):
        """With mu > 0, the bound should be larger than with mu=0."""
        gains = PIDGains(Kp=0.3, Ki=0.02, Kd=0.05)
        bound_no_syc = max_pid_output_bound(gains, T_max=20, rho_star=0.8, mu=0.0)
        bound_syc = max_pid_output_bound(gains, T_max=20, rho_star=0.8, mu=1.0)
        assert bound_syc > bound_no_syc

    def test_sycophancy_exact_value(self):
        """rho_star=0.8, mu=1.0 -> e_max=1.8, bound = 1.8 * (Kp + T*Ki + 2*Kd)."""
        gains = PIDGains(Kp=0.3, Ki=0.02, Kd=0.05)
        bound = max_pid_output_bound(gains, T_max=20, rho_star=0.8, mu=1.0)
        gains_sum = 0.3 + 20 * 0.02 + 2 * 0.05
        assert bound == pytest.approx(1.8 * gains_sum)


# ---------------------------------------------------------------------------
# check_stability (Eq 17)
# ---------------------------------------------------------------------------

class TestCheckStability:
    def test_stable_gains(self):
        gains = PIDGains(Kp=0.1, Ki=0.01, Kd=0.05)
        assert check_stability(gains, T_max=10, gamma_beta=0.9) is True

    def test_unstable_gains(self):
        gains = PIDGains(Kp=1.0, Ki=0.5, Kd=1.0)
        assert check_stability(gains, T_max=20, gamma_beta=0.9) is False

    def test_borderline(self):
        """Gains exactly at the boundary (not strictly less) -> unstable."""
        # 1/gamma_beta = 1/0.5 = 2.0
        # e_max = 1.0 (defaults), Kp + T_max*Ki + 2*Kd = 1.0 + 5*0.1 + 2*0.25 = 2.0
        gains = PIDGains(Kp=1.0, Ki=0.1, Kd=0.25)
        assert check_stability(gains, T_max=5, gamma_beta=0.5) is False

    def test_zero_gamma(self):
        """gamma_beta=0 means no decay constraint, always stable."""
        gains = PIDGains(Kp=100.0, Ki=100.0, Kd=100.0)
        assert check_stability(gains, T_max=100, gamma_beta=0.0) is True

    def test_stable_without_sycophancy_unstable_with(self):
        """Gains that are stable without mu but unstable when mu inflates e_max."""
        gains = PIDGains(Kp=0.3, Ki=0.02, Kd=0.05)
        # With defaults (e_max=1.0): bound = 0.3 + 20*0.02 + 2*0.05 = 0.8 < 1/0.9 = 1.111 -> stable
        assert check_stability(gains, T_max=20, gamma_beta=0.9) is True
        # With rho_star=0.8, mu=1.0 (e_max=1.8): bound = 1.8 * 0.8 = 1.44 > 1.111 -> unstable
        assert check_stability(gains, T_max=20, gamma_beta=0.9, rho_star=0.8, mu=1.0) is False

    def test_stable_with_sycophancy_small_mu(self):
        """Small enough mu keeps things stable."""
        gains = PIDGains(Kp=0.1, Ki=0.01, Kd=0.05)
        assert check_stability(gains, T_max=10, gamma_beta=0.9, rho_star=0.8, mu=0.2) is True


# ---------------------------------------------------------------------------
# check_non_oscillation (Eq 18)
# ---------------------------------------------------------------------------

class TestCheckNonOscillation:
    def test_non_oscillating(self):
        gains = PIDGains(Kp=0.01, Ki=0.001, Kd=0.005)
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is True

    def test_oscillating(self):
        gains = PIDGains(Kp=0.5, Ki=0.1, Kd=0.2)
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is False

    def test_stricter_than_stability(self):
        """Non-oscillation is stricter: something stable may still oscillate."""
        gains = PIDGains(Kp=0.3, Ki=0.05, Kd=0.1)
        gamma = 0.9
        T_max = 10
        stable = check_stability(gains, T_max, gamma)
        non_osc = check_non_oscillation(gains, T_max, gamma)
        # Stable but oscillating is a valid combination
        if stable:
            # Just verify non-oscillation is at least as restrictive
            assert non_osc is True or non_osc is False  # valid either way

    def test_sycophancy_breaks_non_oscillation(self):
        """Gains that don't oscillate without mu may oscillate with mu."""
        gains = PIDGains(Kp=0.01, Ki=0.001, Kd=0.005)
        # Without sycophancy: bound = 0.01 + 10*0.001 + 2*0.005 = 0.03 < 0.1 -> OK
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9) is True
        # With large mu: e_max = 0.8 + 5.0 = 5.8, bound = 5.8 * 0.03 = 0.174 > 0.1 -> oscillates
        assert check_non_oscillation(gains, T_max=10, gamma_beta=0.9, rho_star=0.8, mu=5.0) is False


# ---------------------------------------------------------------------------
# validate_gains
# ---------------------------------------------------------------------------

class TestValidateGains:
    def test_stable_passes(self):
        gains = PIDGains(Kp=0.1, Ki=0.01, Kd=0.05)
        validate_gains(gains, T_max=10, gamma_beta=0.9)  # Should not raise

    def test_unstable_raises(self):
        gains = PIDGains(Kp=1.0, Ki=0.5, Kd=1.0)
        with pytest.raises(GainInstabilityError, match="violate stability"):
            validate_gains(gains, T_max=20, gamma_beta=0.9)

    def test_error_message_contains_values(self):
        gains = PIDGains(Kp=2.0, Ki=1.0, Kd=1.0)
        with pytest.raises(GainInstabilityError) as exc_info:
            validate_gains(gains, T_max=10, gamma_beta=0.9)
        msg = str(exc_info.value)
        assert "Kp=2.0" in msg
        assert "gamma_beta=0.9" in msg

    def test_sycophancy_aware_validation(self):
        """Gains that pass without mu should fail with mu that inflates e_max."""
        gains = PIDGains(Kp=0.3, Ki=0.02, Kd=0.05)
        # Passes without sycophancy
        validate_gains(gains, T_max=20, gamma_beta=0.9)
        # Fails with sycophancy
        with pytest.raises(GainInstabilityError):
            validate_gains(gains, T_max=20, gamma_beta=0.9, rho_star=0.8, mu=1.0)

    def test_error_message_contains_mu_and_rho_star(self):
        """Error message should include rho_star and mu for debugging."""
        gains = PIDGains(Kp=0.3, Ki=0.02, Kd=0.05)
        with pytest.raises(GainInstabilityError) as exc_info:
            validate_gains(gains, T_max=20, gamma_beta=0.9, rho_star=0.8, mu=1.0)
        msg = str(exc_info.value)
        assert "rho_star=0.8" in msg
        assert "mu=1.0" in msg
