"""Integration test: force quadrants, verify β + tone in logs.

Tests the full chain: PID quadrant → β update → tone bucket → prompt assembly.
Uses mock mode to avoid LLM calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from eval.PID.controller import PIDController, classify_quadrant
from eval.PID.types import PIDConfig, PIDGains, Quadrant
from multi_agent.prompts.registry import (
    beta_to_bucket,
    PromptRegistry,
    reset_registry_cache,
)


class TestQuadrantToneIntegration:
    """Integration: quadrant classification → β update → tone bucket → prompt."""

    def setup_method(self):
        reset_registry_cache()

    def test_multi_round_quadrant_transitions(self):
        """Simulate 3 rounds with varying quality, verify quadrant transitions."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(Kp=0.5, Ki=0.0, Kd=0.0),
                delta_js=0.05,
                delta_beta=0.15,
                gamma_beta=0.7,
            ),
            initial_beta=0.3,
        )

        quadrants_seen = set()
        tone_buckets = []

        # Round 1: Low quality, low diversity → STUCK → β↑
        r1 = ctrl.step(rho_bar=0.3, js_current=0.02)
        quadrants_seen.add(r1.quadrant)
        tone_buckets.append(beta_to_bucket(r1.beta_new))
        assert r1.quadrant == "stuck"
        assert r1.beta_new == pytest.approx(0.45)  # 0.3 + 0.15

        # Round 2: High quality, low diversity → CONVERGED → β decays
        r2 = ctrl.step(rho_bar=0.9, js_current=0.02)
        quadrants_seen.add(r2.quadrant)
        tone_buckets.append(beta_to_bucket(r2.beta_new))
        assert r2.quadrant == "converged"
        assert r2.beta_new == pytest.approx(0.45 * 0.7)  # decay

        # Round 3: High quality, high diversity → HEALTHY → β decays
        r3 = ctrl.step(rho_bar=0.85, js_current=0.1)
        quadrants_seen.add(r3.quadrant)
        tone_buckets.append(beta_to_bucket(r3.beta_new))
        assert r3.quadrant == "healthy"

        # Verify at least 2 different quadrants observed
        assert len(quadrants_seen) >= 2, f"Only saw: {quadrants_seen}"

    def test_tone_shifts_match_beta_direction(self):
        """When β increases, tone should move toward adversarial; when β decreases, toward collaborative."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(Kp=0.3, Ki=0.0, Kd=0.0),
                delta_js=0.05,
                delta_beta=0.3,
                gamma_beta=0.3,
            ),
            initial_beta=0.5,
        )

        # STUCK: β goes up → adversarial direction
        r1 = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert r1.quadrant == "stuck"
        bucket_up = beta_to_bucket(r1.beta_new)
        assert bucket_up in ("balanced", "adversarial")

        # CONVERGED: β decays → collaborative direction
        r2 = ctrl.step(rho_bar=0.95, js_current=0.01)
        assert r2.quadrant == "converged"
        bucket_down = beta_to_bucket(r2.beta_new)

        # β went up then down — second bucket should be ≤ first
        order = {"collaborative": 0, "balanced": 1, "adversarial": 2}
        assert order[bucket_down] <= order[bucket_up]

    def test_no_tone_in_propose_judge(self):
        """Propose and judge phases get no tone injection."""
        registry = PromptRegistry()

        propose_result = registry.build(
            role="macro", phase="propose", beta=0.9, user_prompt="test propose",
        )
        assert "ADVERSARIAL" not in propose_result.system_prompt
        assert propose_result.tone_file == ""

        judge_result = registry.build(
            role="judge", phase="judge", beta=None, user_prompt="test judge",
        )
        assert "ADVERSARIAL" not in judge_result.system_prompt
        assert judge_result.tone_file == ""

    def test_full_semantics_chain(self):
        """End-to-end: low quality → STUCK → β↑ → adversarial tone in prompt."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(Kp=0.3, Ki=0.0, Kd=0.0),
                delta_js=0.05,
                delta_beta=0.4,
            ),
            initial_beta=0.5,
        )

        # Force STUCK quadrant
        result = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert result.quadrant == "stuck"
        assert result.beta_new == pytest.approx(0.9)  # 0.5 + 0.4

        # Feed β into registry
        registry = PromptRegistry()
        build_result = registry.build(
            role="macro", phase="critique", beta=result.beta_new, user_prompt="test",
            block_order=["role_system", "phase_preamble", "tone"],
        )

        # Verify adversarial tone
        assert build_result.beta_bucket == "adversarial"
        assert "ADVERSARIAL" in build_result.system_prompt

    def test_chaotic_pid_directs_beta(self):
        """Chaotic regime: PID output directly controls β without decay."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(Kp=0.3, Ki=0.0, Kd=0.0),
                gamma_beta=0.5,  # Would halve β if decay applied
                delta_js=0.05,
            ),
            initial_beta=0.5,
        )
        # CHAOTIC: high JS, low quality
        result = ctrl.step(rho_bar=0.3, js_current=0.1)
        assert result.quadrant == "chaotic"
        # β = clip(0.5 * 1.0 + u_t, 0, 1) — note γ_β=1.0 for chaotic, not 0.5
        # e_t = 0.8 - 0.3 = 0.5, u_t = 0.3 * 0.5 = 0.15
        assert result.beta_new == pytest.approx(0.65)  # 0.5 + 0.15
