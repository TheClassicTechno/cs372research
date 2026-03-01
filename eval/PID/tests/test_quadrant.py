"""Tests for quadrant classification and quadrant-routed β updates.

RAudit paper references:
    - Table 1 (p.4): 4-quadrant control table
    - Algorithm 1 lines 13-29 (p.19): div/qual signals + actuator routing
    - Eq 7 (p.4): div(t) = 1[JS >= δ_JS], qual(t) = 1[ρ̄ >= ρ*]
"""

import pytest
from eval.PID.types import PIDGains, PIDConfig, Quadrant
from eval.PID.controller import classify_quadrant, PIDController


# ---------------------------------------------------------------------------
# classify_quadrant
# ---------------------------------------------------------------------------

class TestClassifyQuadrant:
    """Test quadrant classification from div/qual binary signals."""

    def test_stuck(self):
        """Low JS (¬div) + low ρ̄ (¬qual) → STUCK."""
        assert classify_quadrant(js=0.01, rho_bar=0.3, delta_js=0.05, rho_star=0.8) == Quadrant.STUCK

    def test_chaotic(self):
        """High JS (div) + low ρ̄ (¬qual) → CHAOTIC."""
        assert classify_quadrant(js=0.10, rho_bar=0.3, delta_js=0.05, rho_star=0.8) == Quadrant.CHAOTIC

    def test_converged(self):
        """Low JS (¬div) + high ρ̄ (qual) → CONVERGED."""
        assert classify_quadrant(js=0.01, rho_bar=0.9, delta_js=0.05, rho_star=0.8) == Quadrant.CONVERGED

    def test_healthy(self):
        """High JS (div) + high ρ̄ (qual) → HEALTHY."""
        assert classify_quadrant(js=0.10, rho_bar=0.9, delta_js=0.05, rho_star=0.8) == Quadrant.HEALTHY

    def test_boundary_js_equals_threshold(self):
        """JS exactly at δ_JS counts as div=True (>=, not >)."""
        assert classify_quadrant(js=0.05, rho_bar=0.3, delta_js=0.05, rho_star=0.8) == Quadrant.CHAOTIC

    def test_boundary_rho_equals_target(self):
        """ρ̄ exactly at ρ* counts as qual=True (>=, not >)."""
        assert classify_quadrant(js=0.01, rho_bar=0.8, delta_js=0.05, rho_star=0.8) == Quadrant.CONVERGED

    def test_both_boundaries(self):
        """Both JS=δ_JS and ρ̄=ρ* → HEALTHY."""
        assert classify_quadrant(js=0.05, rho_bar=0.8, delta_js=0.05, rho_star=0.8) == Quadrant.HEALTHY


# ---------------------------------------------------------------------------
# Quadrant-routed β updates via PIDController.step()
# ---------------------------------------------------------------------------

class TestQuadrantBetaUpdates:
    """Test that each quadrant produces the correct β update rule."""

    def test_stuck_increments_beta_no_decay(self):
        """Stuck → β ← min(β + Δβ, 1). Increment is FINAL (no decay after).

        RAudit Algorithm 1 L22 (p.19): β ← min(β + Δβ, 1)
        Table 1 (p.4): Stuck → "β↑ + EXPLORE"
        """
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), delta_beta=0.1, delta_js=0.05),
            initial_beta=0.5,
        )
        # js=0.01 < delta_js=0.05 (¬div), rho_bar=0.3 < rho_star=0.8 (¬qual) → STUCK
        result = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert result.beta_new == pytest.approx(0.6)  # 0.5 + 0.1
        assert result.quadrant == "stuck"

    def test_stuck_beta_capped_at_1(self):
        """Stuck β increment capped at 1.0."""
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), delta_beta=0.2, delta_js=0.05),
            initial_beta=0.95,
        )
        result = ctrl.step(rho_bar=0.3, js_current=0.01)  # STUCK
        assert result.beta_new == 1.0  # min(0.95 + 0.2, 1.0)

    def test_healthy_decays_beta(self):
        """Healthy → β ← clip(β · γ_β, 0, 1). Natural decay.

        RAudit Algorithm 1 L29 (p.19): β ← clip(β · γ_β, 0, 1)
        Table 1 (p.4): Healthy → "Natural decay"
        """
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), gamma_beta=0.9, delta_js=0.05),
            initial_beta=0.8,
        )
        # js=0.1 >= delta_js=0.05 (div), rho_bar=0.9 >= rho_star=0.8 (qual) → HEALTHY
        result = ctrl.step(rho_bar=0.9, js_current=0.1)
        assert result.beta_new == pytest.approx(0.72)  # 0.8 * 0.9
        assert result.quadrant == "healthy"

    def test_chaotic_pid_direct(self):
        """Chaotic → β ← clip(β + u_t, 0, 1). PID-directed, no γ_β.

        Algorithm 1 L24 (p.19): "Apply REFINE" (PID-directed)
        Table 1 (p.4): "Hold β + REFINE"
        """
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), delta_js=0.05),
            initial_beta=0.5,
        )
        # js=0.1 >= delta_js=0.05 (div), rho_bar=0.3 < rho_star=0.8 (¬qual) → CHAOTIC
        result = ctrl.step(rho_bar=0.3, js_current=0.1)
        # u_t is positive (quality below target), so beta should increase
        assert result.beta_new > 0.5
        assert result.quadrant == "chaotic"

    def test_converged_decays_beta(self):
        """Converged → β ← clip(β · γ_β, 0, 1). Same decay as Healthy.

        RAudit Algorithm 1 L29 (p.19): β ← clip(β · γ_β, 0, 1)
        Table 1 (p.4): "β↓ + CONSOLIDATE"
        """
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), gamma_beta=0.9, delta_js=0.05),
            initial_beta=0.8,
        )
        # js=0.01 < delta_js=0.05 (¬div), rho_bar=0.9 >= rho_star=0.8 (qual) → CONVERGED
        result = ctrl.step(rho_bar=0.9, js_current=0.01)
        assert result.beta_new == pytest.approx(0.72)  # 0.8 * 0.9, same as Healthy
        assert result.quadrant == "converged"

    def test_step_result_includes_div_qual_signals(self):
        """PIDStepResult exposes div_signal and qual_signal."""
        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), delta_js=0.05),
            initial_beta=0.5,
        )
        result = ctrl.step(rho_bar=0.9, js_current=0.1)  # HEALTHY
        assert result.div_signal is True
        assert result.qual_signal is True

        ctrl2 = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.01, 0.03), delta_js=0.05),
            initial_beta=0.5,
        )
        result2 = ctrl2.step(rho_bar=0.3, js_current=0.01)  # STUCK
        assert result2.div_signal is False
        assert result2.qual_signal is False

    def test_chaotic_no_gamma_momentum(self):
        """Chaotic β update uses γ_β=1.0 (no decay), only PID correction.

        This distinguishes Chaotic from Converged/Healthy where γ_β < 1
        applies momentum decay.
        """
        # Same inputs but different quadrants due to JS
        cfg = PIDConfig(gains=PIDGains(0.3, 0.0, 0.0), gamma_beta=0.5, delta_js=0.05)

        # CHAOTIC: β ← clip(β + u_t, 0, 1) — no gamma_beta decay
        ctrl_chaotic = PIDController(cfg, initial_beta=0.5)
        result_chaotic = ctrl_chaotic.step(rho_bar=0.3, js_current=0.1)  # CHAOTIC
        # e_t = 0.8 - 0.3 = 0.5, u_t = 0.3 * 0.5 = 0.15
        # beta_new = clip(0.5 * 1.0 + 0.15, 0, 1) = 0.65
        assert result_chaotic.beta_new == pytest.approx(0.65)
        assert result_chaotic.quadrant == "chaotic"

    def test_quadrant_string_values(self):
        """Quadrant enum .value matches expected strings."""
        assert Quadrant.STUCK.value == "stuck"
        assert Quadrant.CHAOTIC.value == "chaotic"
        assert Quadrant.CONVERGED.value == "converged"
        assert Quadrant.HEALTHY.value == "healthy"
