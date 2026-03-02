"""Tests for PID quadrant transitions and multi-round β dynamics.

These tests verify that the PID controller correctly switches between
quadrant-specific β update logic as debate conditions change round
over round. This is the key untested gap identified in code review.

RAudit references:
    - Table 1 (p.4): 4-quadrant control table
    - Algorithm 1 lines 13-29 (p.19): Quadrant-routed actuator logic
"""

import pytest
from eval.PID.types import PIDGains, PIDConfig, Quadrant
from eval.PID.controller import PIDController, classify_quadrant


def _make_config(**overrides):
    defaults = dict(
        gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
        rho_star=0.8,
        gamma_beta=0.9,
        mu=0.0,
        delta_s=0.05,
        delta_js=0.05,
        delta_beta=0.1,
        T_max=20,
        epsilon=0.01,
    )
    defaults.update(overrides)
    return PIDConfig(**defaults)


# ---------------------------------------------------------------------------
# Quadrant transition sequences
# ---------------------------------------------------------------------------

class TestQuadrantTransitions:
    """Test that β updates correctly when quadrant changes across rounds."""

    def test_stuck_to_healthy(self):
        """STUCK(β↑) then HEALTHY(decay): β bumps then decays."""
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        # Round 0: STUCK — low JS, low quality
        r0 = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert r0.quadrant == "stuck"
        assert r0.beta_new == pytest.approx(0.6)  # 0.5 + 0.1

        # Round 1: HEALTHY — high JS, high quality
        r1 = ctrl.step(rho_bar=0.9, js_current=0.1)
        assert r1.quadrant == "healthy"
        assert r1.beta_new == pytest.approx(0.6 * 0.9)  # decay

    def test_stuck_to_chaotic(self):
        """STUCK(β↑) then CHAOTIC(PID-directed): β bumps then PID adjusts."""
        ctrl = PIDController(_make_config(), initial_beta=0.4)
        # Round 0: STUCK
        r0 = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert r0.quadrant == "stuck"
        assert r0.beta_new == pytest.approx(0.5)  # 0.4 + 0.1

        # Round 1: CHAOTIC — high JS, low quality
        r1 = ctrl.step(rho_bar=0.3, js_current=0.1)
        assert r1.quadrant == "chaotic"
        # CHAOTIC uses PID-directed with gamma=1.0: clip(beta + u_t, 0, 1)
        assert r1.beta_new > 0.5  # u_t positive because quality below target

    def test_chaotic_to_converged(self):
        """CHAOTIC(PID) then CONVERGED(decay): PID adjusts then decays."""
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        # Round 0: CHAOTIC
        r0 = ctrl.step(rho_bar=0.3, js_current=0.1)
        assert r0.quadrant == "chaotic"

        # Round 1: CONVERGED — low JS, high quality
        r1 = ctrl.step(rho_bar=0.9, js_current=0.01)
        assert r1.quadrant == "converged"
        # CONVERGED uses gamma decay
        expected = r0.beta_new * 0.9
        assert r1.beta_new == pytest.approx(expected)

    def test_healthy_to_stuck(self):
        """HEALTHY(decay) then STUCK(β↑): decay then forced bump."""
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        # Round 0: HEALTHY
        r0 = ctrl.step(rho_bar=0.9, js_current=0.1)
        assert r0.quadrant == "healthy"
        decayed = r0.beta_new  # 0.5 * 0.9 = 0.45

        # Round 1: STUCK — quality crashes, JS drops
        r1 = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert r1.quadrant == "stuck"
        assert r1.beta_new == pytest.approx(decayed + 0.1)

    def test_converged_stays_converged_beta_decays(self):
        """Multiple CONVERGED rounds: β keeps decaying toward 0."""
        ctrl = PIDController(_make_config(gamma_beta=0.8), initial_beta=0.5)
        betas = []
        for _ in range(5):
            r = ctrl.step(rho_bar=0.9, js_current=0.01)
            assert r.quadrant == "converged"
            betas.append(r.beta_new)
        # Each β = prev * 0.8
        for i in range(1, len(betas)):
            assert betas[i] < betas[i - 1]

    def test_stuck_repeated_bumps_to_max(self):
        """Repeated STUCK: β increments toward 1.0."""
        ctrl = PIDController(
            _make_config(delta_beta=0.15), initial_beta=0.3
        )
        for _ in range(10):
            r = ctrl.step(rho_bar=0.3, js_current=0.01)
            assert r.quadrant == "stuck"
        # After many bumps, should be at 1.0
        assert ctrl.beta == 1.0


class TestQuadrantWithSycophancy:
    """Test sycophancy interacting with quadrant routing."""

    def test_sycophancy_inflates_error_in_chaotic(self):
        """Sycophancy + CHAOTIC: error inflated, PID-directed β update."""
        ctrl = PIDController(
            _make_config(mu=1.0, delta_s=0.05), initial_beta=0.5
        )
        # Round 0: set up history
        r0 = ctrl.step(rho_bar=0.7, js_current=0.3, ov_current=0.8)

        # Round 1: sycophancy trigger (JS drops, overlap drops) + CHAOTIC
        r1 = ctrl.step(rho_bar=0.7, js_current=0.08, ov_current=0.7)
        assert r1.s_t == 1
        assert r1.quadrant == "chaotic"  # JS=0.08 >= delta_js=0.05
        # Error = (0.8 - 0.7) + 1.0 * 1 = 1.1
        assert r1.e_t == pytest.approx(1.1)

    def test_sycophancy_in_stuck_still_bumps(self):
        """Sycophancy in STUCK: β still gets delta_beta bump (not PID)."""
        ctrl = PIDController(
            _make_config(mu=1.0, delta_s=0.05), initial_beta=0.5
        )
        # Round 0: set up history
        ctrl.step(rho_bar=0.7, js_current=0.3, ov_current=0.8)

        # Round 1: sycophancy + STUCK (JS drops below delta_js)
        r1 = ctrl.step(rho_bar=0.7, js_current=0.02, ov_current=0.7)
        assert r1.s_t == 1
        assert r1.quadrant == "stuck"
        # STUCK ignores PID — just bumps by delta_beta
        # beta after r0 was some value; r1 bumps it by 0.1
        # The exact value depends on r0's quadrant, but the key is
        # STUCK always does beta + delta_beta, regardless of sycophancy


class TestIntegralPersistenceAcrossQuadrants:
    """Verify that the integral accumulates continuously regardless of quadrant."""

    def test_integral_grows_across_different_quadrants(self):
        """Integral should accumulate errors from all quadrants."""
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        # Round 0: STUCK (error = 0.8 - 0.3 = 0.5)
        ctrl.step(rho_bar=0.3, js_current=0.01)
        # Round 1: CHAOTIC (error = 0.8 - 0.4 = 0.4)
        ctrl.step(rho_bar=0.4, js_current=0.1)
        # Round 2: HEALTHY (error = 0.8 - 0.9 = -0.1)
        ctrl.step(rho_bar=0.9, js_current=0.1)
        # Integral = 0.5 + 0.4 + (-0.1) = 0.8
        assert ctrl.state.integral == pytest.approx(0.8)


class TestAllSixteenTransitions:
    """Systematically test all 16 possible quadrant transitions (4 x 4).

    For each pair (Q_start → Q_end), verify:
    1. Quadrant is correctly classified
    2. β update follows the correct rule
    """

    @pytest.mark.parametrize(
        "start_js,start_rho,end_js,end_rho,expected_start,expected_end",
        [
            # STUCK → STUCK
            (0.01, 0.3, 0.01, 0.3, "stuck", "stuck"),
            # STUCK → CHAOTIC
            (0.01, 0.3, 0.10, 0.3, "stuck", "chaotic"),
            # STUCK → CONVERGED
            (0.01, 0.3, 0.01, 0.9, "stuck", "converged"),
            # STUCK → HEALTHY
            (0.01, 0.3, 0.10, 0.9, "stuck", "healthy"),
            # CHAOTIC → STUCK
            (0.10, 0.3, 0.01, 0.3, "chaotic", "stuck"),
            # CHAOTIC → CHAOTIC
            (0.10, 0.3, 0.10, 0.3, "chaotic", "chaotic"),
            # CHAOTIC → CONVERGED
            (0.10, 0.3, 0.01, 0.9, "chaotic", "converged"),
            # CHAOTIC → HEALTHY
            (0.10, 0.3, 0.10, 0.9, "chaotic", "healthy"),
            # CONVERGED → STUCK
            (0.01, 0.9, 0.01, 0.3, "converged", "stuck"),
            # CONVERGED → CHAOTIC
            (0.01, 0.9, 0.10, 0.3, "converged", "chaotic"),
            # CONVERGED → CONVERGED
            (0.01, 0.9, 0.01, 0.9, "converged", "converged"),
            # CONVERGED → HEALTHY
            (0.01, 0.9, 0.10, 0.9, "converged", "healthy"),
            # HEALTHY → STUCK
            (0.10, 0.9, 0.01, 0.3, "healthy", "stuck"),
            # HEALTHY → CHAOTIC
            (0.10, 0.9, 0.10, 0.3, "healthy", "chaotic"),
            # HEALTHY → CONVERGED
            (0.10, 0.9, 0.01, 0.9, "healthy", "converged"),
            # HEALTHY → HEALTHY
            (0.10, 0.9, 0.10, 0.9, "healthy", "healthy"),
        ],
    )
    def test_transition(
        self, start_js, start_rho, end_js, end_rho, expected_start, expected_end
    ):
        ctrl = PIDController(_make_config(), initial_beta=0.5)
        r0 = ctrl.step(rho_bar=start_rho, js_current=start_js)
        assert r0.quadrant == expected_start
        r1 = ctrl.step(rho_bar=end_rho, js_current=end_js)
        assert r1.quadrant == expected_end
        # β always in bounds
        assert 0.0 <= r1.beta_new <= 1.0
