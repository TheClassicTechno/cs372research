"""Tests for eval.PID.termination — bound formula, convergence, contraction sim."""

import math
import pytest
from eval.PID.termination import (
    termination_bound,
    check_convergence,
    expected_divergence,
    simulate_contraction,
)


class TestTerminationBound:
    def test_basic(self):
        """Standard case: d0=1.0, epsilon=0.01, kappa=0.5."""
        T = termination_bound(d0=1.0, epsilon=0.01, kappa=0.5)
        expected = math.log(1.0 / 0.01) / math.log(1.0 / 0.5)
        assert T == pytest.approx(expected)

    def test_already_converged(self):
        """d0 <= epsilon -> T* = 0."""
        assert termination_bound(d0=0.01, epsilon=0.01, kappa=0.5) == 0.0
        assert termination_bound(d0=0.005, epsilon=0.01, kappa=0.5) == 0.0

    def test_slow_contraction(self):
        """kappa close to 1 -> very many rounds needed."""
        T = termination_bound(d0=1.0, epsilon=0.01, kappa=0.99)
        assert T > 100

    def test_fast_contraction(self):
        """kappa close to 0 -> very few rounds needed."""
        T = termination_bound(d0=1.0, epsilon=0.01, kappa=0.1)
        assert T < 5

    def test_invalid_d0(self):
        with pytest.raises(ValueError, match="d0 must be > 0"):
            termination_bound(d0=0.0, epsilon=0.01, kappa=0.5)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            termination_bound(d0=1.0, epsilon=0.0, kappa=0.5)

    def test_invalid_kappa_zero(self):
        with pytest.raises(ValueError, match="kappa must be in"):
            termination_bound(d0=1.0, epsilon=0.01, kappa=0.0)

    def test_invalid_kappa_one(self):
        with pytest.raises(ValueError, match="kappa must be in"):
            termination_bound(d0=1.0, epsilon=0.01, kappa=1.0)

    def test_negative_d0(self):
        with pytest.raises(ValueError):
            termination_bound(d0=-1.0, epsilon=0.01, kappa=0.5)


class TestCheckConvergence:
    def test_converged(self):
        assert check_convergence(0.005, 0.01) is True

    def test_not_converged(self):
        assert check_convergence(0.05, 0.01) is False

    def test_exact_boundary(self):
        assert check_convergence(0.01, 0.01) is True

    def test_zero_divergence(self):
        assert check_convergence(0.0, 0.01) is True


class TestExpectedDivergence:
    def test_round_zero(self):
        assert expected_divergence(d0=1.0, kappa=0.5, t=0) == pytest.approx(1.0)

    def test_round_one(self):
        assert expected_divergence(d0=1.0, kappa=0.5, t=1) == pytest.approx(0.5)

    def test_round_ten(self):
        result = expected_divergence(d0=1.0, kappa=0.5, t=10)
        assert result == pytest.approx(0.5 ** 10)

    def test_kappa_zero(self):
        """kappa=0 -> all rounds after 0 have zero divergence."""
        assert expected_divergence(d0=1.0, kappa=0.0, t=1) == 0.0

    def test_kappa_one(self):
        """kappa=1 -> divergence never decreases."""
        assert expected_divergence(d0=1.0, kappa=1.0, t=100) == pytest.approx(1.0)

    def test_scaling_with_d0(self):
        d1 = expected_divergence(d0=2.0, kappa=0.5, t=3)
        d2 = expected_divergence(d0=1.0, kappa=0.5, t=3)
        assert d1 == pytest.approx(2.0 * d2)


class TestSimulateContraction:
    def test_converges(self):
        traj = simulate_contraction(d0=1.0, kappa=0.5, epsilon=0.01)
        assert traj[0] == 1.0
        assert traj[-1] <= 0.01

    def test_trajectory_decreasing(self):
        traj = simulate_contraction(d0=1.0, kappa=0.8, epsilon=0.01)
        for i in range(1, len(traj)):
            assert traj[i] < traj[i - 1]

    def test_kappa_zero_immediate(self):
        traj = simulate_contraction(d0=1.0, kappa=0.0, epsilon=0.01)
        assert len(traj) == 2  # [1.0, 0.0]
        assert traj[-1] == 0.0

    def test_max_steps_cap(self):
        """Very slow contraction should be capped at max_steps."""
        traj = simulate_contraction(d0=1.0, kappa=0.9999, epsilon=1e-10, max_steps=50)
        assert len(traj) <= 51  # d0 + up to 50 steps

    def test_already_converged(self):
        """d0 already below epsilon -> just [d0, d0*kappa]."""
        traj = simulate_contraction(d0=0.001, kappa=0.5, epsilon=0.01)
        assert len(traj) == 2
        assert traj[-1] <= 0.01
