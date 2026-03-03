"""Tests for the modular prompt registry and β → tone bucket mapping.

RAudit paper references:
    - Section 3.5 (p.4): "Contentiousness (β): Modulates adversarial prompting."
    - Algorithm 1, line 5 (p.19): "Generate traces with contentiousness β^(t-1)"
    - Table 1 (p.4): Stuck → β↑ → adversarial; Converged → β↓ → collaborative
"""

import pytest
from multi_agent.prompts.registry import (
    beta_to_bucket,
    resolve_beta,
    get_registry,
    reset_registry_cache,
    PromptRegistry,
    PromptBuildResult,
)


# ---------------------------------------------------------------------------
# beta_to_bucket (RAudit-corrected semantics)
# ---------------------------------------------------------------------------

class TestBetaToBucket:
    """β → tone bucket mapping with CORRECTED RAudit semantics.

    RAudit: high β = contentious/explore, low β = collaborative/exploit.
    """

    def test_low_beta_collaborative(self):
        assert beta_to_bucket(0.0) == "collaborative"
        assert beta_to_bucket(0.32) == "collaborative"

    def test_mid_beta_balanced(self):
        assert beta_to_bucket(0.33) == "balanced"
        assert beta_to_bucket(0.66) == "balanced"

    def test_high_beta_adversarial(self):
        assert beta_to_bucket(0.67) == "adversarial"
        assert beta_to_bucket(1.0) == "adversarial"

    def test_boundary_0_33(self):
        """β = 0.33 → balanced (not collaborative)."""
        assert beta_to_bucket(0.33) == "balanced"

    def test_boundary_0_67(self):
        """β = 0.67 → adversarial (not balanced)."""
        assert beta_to_bucket(0.67) == "adversarial"

    def test_monotonicity(self):
        """Increasing β → monotonically increasing adversarial tone."""
        buckets = [beta_to_bucket(b) for b in [0.0, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0]]
        order = {"collaborative": 0, "balanced": 1, "adversarial": 2}
        numeric = [order[b] for b in buckets]
        assert numeric == sorted(numeric)


# ---------------------------------------------------------------------------
# resolve_beta
# ---------------------------------------------------------------------------

class TestResolveBeta:
    """Test unified beta resolution from PID _current_beta or agreeableness."""

    def test_propose_always_none(self):
        assert resolve_beta({"_current_beta": 0.7}, "propose") is None

    def test_judge_always_none(self):
        assert resolve_beta({"_current_beta": 0.7}, "judge") is None

    def test_critique_uses_current_beta(self):
        assert resolve_beta({"_current_beta": 0.42}, "critique") == 0.42

    def test_revise_uses_current_beta(self):
        assert resolve_beta({"_current_beta": 0.85}, "revise") == 0.85

    def test_critique_derives_from_agreeableness(self):
        """Without _current_beta, beta = 1.0 - agreeableness."""
        assert resolve_beta({"agreeableness": 0.3}, "critique") == pytest.approx(0.7)

    def test_revise_derives_from_agreeableness(self):
        assert resolve_beta({"agreeableness": 0.7}, "revise") == pytest.approx(0.3)

    def test_default_agreeableness(self):
        """Empty config → agreeableness=0.3 → beta=0.7."""
        assert resolve_beta({}, "critique") == pytest.approx(0.7)

    def test_current_beta_zero_is_not_none(self):
        """_current_beta=0.0 should be used, not fall through to agreeableness."""
        assert resolve_beta({"_current_beta": 0.0, "agreeableness": 0.3}, "critique") == 0.0


# ---------------------------------------------------------------------------
# PromptRegistry.build
# ---------------------------------------------------------------------------

class TestPromptRegistryBuild:
    """Test modular prompt assembly for different phases and β values."""

    def setup_method(self):
        reset_registry_cache()
        self.registry = PromptRegistry()

    def test_critique_adversarial(self):
        """High β critique → adversarial tone injected."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.9, user_prompt="test critique",
        )
        assert "ADVERSARIAL" in result.system_prompt
        assert "tone" in result.blocks_used
        assert result.beta_bucket == "adversarial"
        assert result.tone_file == "critique_adversarial.txt"
        assert result.user_prompt == "test critique"

    def test_critique_collaborative(self):
        """Low β critique → collaborative tone injected."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.1, user_prompt="test critique",
        )
        assert "COLLABORATIVE" in result.system_prompt
        assert result.beta_bucket == "collaborative"
        assert result.tone_file == "critique_collaborative.txt"

    def test_critique_balanced(self):
        """Mid β critique → balanced tone injected."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.5, user_prompt="test critique",
        )
        assert "BALANCED" in result.system_prompt
        assert result.beta_bucket == "balanced"

    def test_revise_adversarial(self):
        """High β revise → firm revision tone."""
        result = self.registry.build(
            role="value", phase="revise", beta=0.8, user_prompt="test revise",
        )
        assert "FIRM" in result.system_prompt
        assert result.tone_file == "revise_adversarial.txt"

    def test_revise_collaborative(self):
        """Low β revise → integrative revision tone."""
        result = self.registry.build(
            role="value", phase="revise", beta=0.1, user_prompt="test revise",
        )
        assert "INTEGRATIVE" in result.system_prompt
        assert result.tone_file == "revise_collaborative.txt"

    def test_propose_no_tone(self):
        """Propose phase gets no tone injection regardless of β."""
        result = self.registry.build(
            role="macro", phase="propose", beta=0.9, user_prompt="test propose",
        )
        assert "ADVERSARIAL" not in result.system_prompt
        assert "tone" not in result.blocks_used
        assert result.beta_bucket == ""
        assert result.tone_file == ""

    def test_judge_no_tone(self):
        """Judge phase gets no tone injection."""
        result = self.registry.build(
            role="judge", phase="judge", beta=None, user_prompt="test judge",
        )
        assert "Judge" in result.system_prompt
        assert "tone" not in result.blocks_used
        assert result.beta_bucket == ""

    def test_beta_none_no_tone(self):
        """When β is None, no tone injection even for critique/revise."""
        result = self.registry.build(
            role="macro", phase="critique", beta=None, user_prompt="test",
        )
        assert "tone" not in result.blocks_used
        assert result.beta_bucket == ""

    def test_no_numeric_beta_in_system_prompt(self):
        """β value should NOT appear as a number in the system prompt."""
        for beta_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = self.registry.build(
                role="macro", phase="critique", beta=beta_val, user_prompt="test",
            )
            # Check that the numeric beta doesn't appear
            assert str(beta_val) not in result.system_prompt

    def test_role_preamble_in_critique(self):
        """Critique system prompt includes role preamble."""
        result = self.registry.build(
            role="risk", phase="critique", beta=0.5, user_prompt="test",
        )
        assert "RISK" in result.system_prompt
        assert "critique" in result.system_prompt.lower() or "Provide" in result.system_prompt

    def test_custom_block_order_reverses_output(self):
        """Custom block_order changes the assembly order."""
        result_default = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
        )
        result_reversed = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
            block_order=["tone", "phase_preamble", "role_system", "causal_contract"],
        )
        # Both should have same blocks but in different order
        assert set(result_default.blocks_used) == set(result_reversed.blocks_used)
        assert result_default.blocks_used != result_reversed.blocks_used
        assert result_reversed.blocks_used[0] == "tone"

    def test_custom_block_order_subset(self):
        """Block order with subset of blocks — others excluded."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
            block_order=["role_system"],
        )
        assert result.blocks_used == ["role_system"]

    def test_prompt_file_overrides_role(self):
        """Override role file via prompt_file_overrides."""
        result = self.registry.build(
            role="macro", phase="propose",
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        assert "role_system" in result.blocks_used
        # Should load the slim variant


# ---------------------------------------------------------------------------
# get_registry (singleton)
# ---------------------------------------------------------------------------

class TestGetRegistry:
    """Test singleton registry access."""

    def setup_method(self):
        reset_registry_cache()
        # Also reset the module-level _registry singleton
        import multi_agent.prompts.registry as reg_mod
        reg_mod._registry = None

    def test_returns_registry_instance(self):
        reg = get_registry({"prompt_logging": {}})
        assert isinstance(reg, PromptRegistry)

    def test_same_instance_on_repeated_calls(self):
        reg1 = get_registry({"prompt_logging": {}})
        reg2 = get_registry({"prompt_logging": {}})
        assert reg1 is reg2


# ---------------------------------------------------------------------------
# Correctness chain: LOW ρ̄ → β↑ → adversarial (not agreeable!)
# ---------------------------------------------------------------------------

class TestCorrectSemanticsChain:
    """Verify the full chain: low quality → β↑ → adversarial tone.

    This is the critical fix. Previously: low quality → β↑ → agreeable (WRONG).
    Now: low quality → β↑ → adversarial (CORRECT per RAudit).
    """

    def test_low_quality_produces_adversarial(self):
        """When quality is below target, PID increases β, which maps to adversarial."""
        from eval.PID.controller import PIDController
        from eval.PID.types import PIDConfig, PIDGains

        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.5, 0.0, 0.0), delta_js=0.05, delta_beta=0.2),
            initial_beta=0.5,
        )
        # Low quality → STUCK quadrant → β bumped up
        result = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert result.beta_new > 0.5  # β increased
        bucket = beta_to_bucket(result.beta_new)
        # β = 0.7 → adversarial
        assert bucket == "adversarial", (
            f"Expected adversarial for β={result.beta_new}, got {bucket}"
        )

    def test_high_quality_produces_collaborative(self):
        """When quality exceeds target, β decays toward collaborative."""
        from eval.PID.controller import PIDController
        from eval.PID.types import PIDConfig, PIDGains

        ctrl = PIDController(
            PIDConfig(gains=PIDGains(0.15, 0.0, 0.0), gamma_beta=0.3, delta_js=0.05),
            initial_beta=0.5,
        )
        # High quality → CONVERGED → β decays (0.5 * 0.3 = 0.15)
        result = ctrl.step(rho_bar=0.95, js_current=0.01)
        assert result.beta_new < 0.5  # β decreased
        bucket = beta_to_bucket(result.beta_new)
        assert bucket == "collaborative", (
            f"Expected collaborative for β={result.beta_new}, got {bucket}"
        )
