"""Tests for eval.PID.beta_dynamics — clipped/unclipped update, steady-state."""

import pytest
from eval.PID.beta_dynamics import (
    update_beta_unclipped,
    update_beta_clipped,
    steady_state_beta,
)


class TestUpdateBetaUnclipped:
    def test_basic(self):
        result = update_beta_unclipped(beta=0.5, gamma_beta=0.9, u_t=0.1)
        assert result == pytest.approx(0.5 * 0.9 + 0.1)

    def test_can_exceed_one(self):
        result = update_beta_unclipped(beta=0.9, gamma_beta=0.9, u_t=0.5)
        assert result > 1.0

    def test_can_go_negative(self):
        result = update_beta_unclipped(beta=0.1, gamma_beta=0.9, u_t=-0.5)
        assert result < 0.0

    def test_zero_input(self):
        result = update_beta_unclipped(beta=0.5, gamma_beta=0.9, u_t=0.0)
        assert result == pytest.approx(0.45)

    def test_zero_gamma(self):
        result = update_beta_unclipped(beta=0.5, gamma_beta=0.0, u_t=0.3)
        assert result == pytest.approx(0.3)


class TestUpdateBetaClipped:
    def test_basic_within_bounds(self):
        result = update_beta_clipped(beta=0.5, gamma_beta=0.9, u_t=0.1)
        assert result == pytest.approx(0.55)

    def test_clips_at_one(self):
        result = update_beta_clipped(beta=0.9, gamma_beta=0.9, u_t=0.5)
        assert result == 1.0

    def test_clips_at_zero(self):
        result = update_beta_clipped(beta=0.1, gamma_beta=0.9, u_t=-0.5)
        assert result == 0.0

    def test_boundary_beta_one(self):
        result = update_beta_clipped(beta=1.0, gamma_beta=0.9, u_t=0.0)
        assert result == pytest.approx(0.9)

    def test_boundary_beta_zero(self):
        result = update_beta_clipped(beta=0.0, gamma_beta=0.9, u_t=0.0)
        assert result == 0.0

    def test_exact_boundary(self):
        """u_t chosen so result is exactly 1.0."""
        result = update_beta_clipped(beta=0.5, gamma_beta=0.9, u_t=0.55)
        assert result == 1.0

    def test_decay_without_input(self):
        """Beta decays toward zero when u_t=0."""
        beta = 0.8
        for _ in range(10):
            beta = update_beta_clipped(beta, gamma_beta=0.9, u_t=0.0)
        assert beta < 0.4  # Decayed significantly


class TestSteadyStateBeta:
    def test_basic(self):
        result = steady_state_beta(u=0.1, gamma_beta=0.9)
        assert result == pytest.approx(1.0)

    def test_zero_input(self):
        result = steady_state_beta(u=0.0, gamma_beta=0.9)
        assert result == pytest.approx(0.0)

    def test_negative_input(self):
        result = steady_state_beta(u=-0.05, gamma_beta=0.5)
        assert result == pytest.approx(-0.1)

    def test_gamma_one_raises(self):
        with pytest.raises(ValueError, match="gamma_beta must be < 1"):
            steady_state_beta(u=0.1, gamma_beta=1.0)

    def test_gamma_above_one_raises(self):
        with pytest.raises(ValueError, match="gamma_beta must be < 1"):
            steady_state_beta(u=0.1, gamma_beta=1.5)

    def test_small_gamma(self):
        """With low decay, steady state is close to u itself."""
        result = steady_state_beta(u=0.3, gamma_beta=0.0)
        assert result == pytest.approx(0.3)
