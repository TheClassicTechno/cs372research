"""Tests for tone prompt content at different β settings.

Verifies:
1. Each β range (collaborative/balanced/adversarial) injects the correct
   tone keywords into critique and revise system prompts.
2. The full chain: PID β → beta_to_bucket → PromptRegistry.build → tone text.
3. Edge cases: β at exact boundaries, β=None, propose/judge phases.

RAudit references:
    - Section 3.5 (p.4): "Contentiousness (β): Modulates adversarial prompting."
    - Table 1 (p.4): Stuck → β↑ → adversarial; Converged → β↓ → collaborative
"""

import pytest
from multi_agent.prompts.registry import (
    beta_to_bucket,
    PromptRegistry,
    reset_registry_cache,
)
from eval.PID.controller import PIDController
from eval.PID.types import PIDGains, PIDConfig


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_registry_cache()
    yield
    reset_registry_cache()


# ---------------------------------------------------------------------------
# Tone content validation for critique phase
# ---------------------------------------------------------------------------

class TestCritiqueToneContent:
    """Verify critique tone files inject correct behavioral keywords."""

    def test_collaborative_critique_keywords(self):
        """β < 0.33 → collaborative: 'common ground', constructive framing."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="critique", beta=0.1, user_prompt="test")
        prompt = result.system_prompt.upper()
        assert "COLLABORATIVE" in prompt
        assert result.beta_bucket == "collaborative"

    def test_balanced_critique_keywords(self):
        """0.33 ≤ β < 0.67 → balanced: moderate tone."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="critique", beta=0.5, user_prompt="test")
        prompt = result.system_prompt.upper()
        assert "BALANCED" in prompt
        assert result.beta_bucket == "balanced"

    def test_adversarial_critique_keywords(self):
        """β ≥ 0.67 → adversarial: 'challenge', aggressive probing."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="critique", beta=0.9, user_prompt="test")
        prompt = result.system_prompt.upper()
        assert "ADVERSARIAL" in prompt
        assert result.beta_bucket == "adversarial"

    def test_critique_tone_changes_with_beta(self):
        """Different β values produce different tone content."""
        reg = PromptRegistry()
        r_low = reg.build(role="macro", phase="critique", beta=0.1, user_prompt="test")
        r_mid = reg.build(role="macro", phase="critique", beta=0.5, user_prompt="test")
        r_high = reg.build(role="macro", phase="critique", beta=0.9, user_prompt="test")

        # All three should have different tone files
        assert r_low.tone_file != r_mid.tone_file
        assert r_mid.tone_file != r_high.tone_file
        assert r_low.tone_file != r_high.tone_file

        # All three should have different system prompts (tone section differs)
        assert r_low.system_prompt != r_mid.system_prompt
        assert r_mid.system_prompt != r_high.system_prompt


# ---------------------------------------------------------------------------
# Tone content validation for revise phase
# ---------------------------------------------------------------------------

class TestReviseToneContent:
    """Verify revise tone files inject correct behavioral keywords."""

    def test_collaborative_revise_integrative(self):
        """Low β revise → integrative: willing to incorporate feedback."""
        reg = PromptRegistry()
        result = reg.build(role="value", phase="revise", beta=0.1, user_prompt="test")
        prompt = result.system_prompt.upper()
        assert "INTEGRATIVE" in prompt
        assert result.beta_bucket == "collaborative"
        assert result.tone_file == "revise_collaborative.txt"

    def test_adversarial_revise_firm(self):
        """High β revise → firm: stand by analysis, concede only genuine flaws."""
        reg = PromptRegistry()
        result = reg.build(role="value", phase="revise", beta=0.9, user_prompt="test")
        prompt = result.system_prompt.upper()
        assert "FIRM" in prompt
        assert result.beta_bucket == "adversarial"
        assert result.tone_file == "revise_adversarial.txt"

    def test_balanced_revise(self):
        """Mid β revise → balanced revision tone."""
        reg = PromptRegistry()
        result = reg.build(role="risk", phase="revise", beta=0.5, user_prompt="test")
        assert "BALANCED" in result.system_prompt.upper()
        assert result.beta_bucket == "balanced"
        assert result.tone_file == "revise_balanced.txt"


# ---------------------------------------------------------------------------
# β boundary behavior
# ---------------------------------------------------------------------------

class TestBetaBoundaries:
    """Test β values at exact bucket boundaries."""

    @pytest.mark.parametrize("beta,expected_bucket", [
        (0.0, "collaborative"),
        (0.329, "collaborative"),
        (0.33, "balanced"),
        (0.5, "balanced"),
        (0.669, "balanced"),
        (0.67, "adversarial"),
        (1.0, "adversarial"),
    ])
    def test_boundary_mapping(self, beta, expected_bucket):
        assert beta_to_bucket(beta) == expected_bucket

    @pytest.mark.parametrize("beta,expected_bucket", [
        (0.0, "collaborative"),
        (0.33, "balanced"),
        (0.67, "adversarial"),
    ])
    def test_boundary_in_registry(self, beta, expected_bucket):
        """Verify registry uses the same bucket as beta_to_bucket."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="critique", beta=beta, user_prompt="test")
        assert result.beta_bucket == expected_bucket


# ---------------------------------------------------------------------------
# Phases that should NOT get tone injection
# ---------------------------------------------------------------------------

class TestNoTonePhases:
    """Propose and judge phases must never receive tone injection."""

    @pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_propose_no_tone_any_beta(self, beta):
        """Propose ignores β completely."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="propose", beta=beta, user_prompt="test")
        assert result.tone_file == ""
        assert result.beta_bucket == ""
        assert "tone" not in result.blocks_used
        for keyword in ["ADVERSARIAL", "COLLABORATIVE", "BALANCED", "FIRM", "INTEGRATIVE"]:
            assert keyword not in result.system_prompt.upper() or keyword in result.system_prompt.split("##")[0].upper()

    def test_judge_no_tone(self):
        """Judge never gets tone."""
        reg = PromptRegistry()
        result = reg.build(role="judge", phase="judge", beta=0.9, user_prompt="test")
        assert result.tone_file == ""
        assert result.beta_bucket == ""

    def test_critique_without_beta_no_tone(self):
        """Critique with beta=None gets no tone."""
        reg = PromptRegistry()
        result = reg.build(role="macro", phase="critique", beta=None, user_prompt="test")
        assert result.tone_file == ""
        assert "tone" not in result.blocks_used


# ---------------------------------------------------------------------------
# Full chain: PID controller → β → bucket → tone
# ---------------------------------------------------------------------------

class TestPIDToBucketChain:
    """Verify the full PID → tone pipeline produces correct results."""

    def test_low_quality_produces_adversarial_critique(self):
        """Low ρ̄ → PID increases β → adversarial critique tone."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(0.5, 0.0, 0.0),
                delta_js=0.05,
                delta_beta=0.2,
            ),
            initial_beta=0.5,
        )
        # STUCK quadrant → β bumps
        result = ctrl.step(rho_bar=0.3, js_current=0.01)
        assert result.beta_new > 0.5

        bucket = beta_to_bucket(result.beta_new)
        reg = PromptRegistry()
        prompt_result = reg.build(
            role="macro", phase="critique", beta=result.beta_new, user_prompt="test"
        )
        assert prompt_result.beta_bucket == bucket
        # β = 0.7 → adversarial
        assert bucket == "adversarial"
        assert "ADVERSARIAL" in prompt_result.system_prompt.upper()

    def test_high_quality_produces_collaborative_critique(self):
        """High ρ̄ → PID decays β → collaborative critique tone."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(0.15, 0.0, 0.0),
                gamma_beta=0.3,
                delta_js=0.05,
            ),
            initial_beta=0.5,
        )
        # CONVERGED → β decays (0.5 * 0.3 = 0.15)
        result = ctrl.step(rho_bar=0.95, js_current=0.01)
        assert result.beta_new < 0.33

        bucket = beta_to_bucket(result.beta_new)
        reg = PromptRegistry()
        prompt_result = reg.build(
            role="macro", phase="critique", beta=result.beta_new, user_prompt="test"
        )
        assert bucket == "collaborative"
        assert "COLLABORATIVE" in prompt_result.system_prompt.upper()

    def test_mid_quality_produces_balanced(self):
        """Medium ρ̄ → moderate β → balanced tone."""
        ctrl = PIDController(
            PIDConfig(
                gains=PIDGains(0.15, 0.0, 0.0),
                gamma_beta=0.9,
                delta_js=0.05,
            ),
            initial_beta=0.5,
        )
        # HEALTHY → β decays slightly (0.5 * 0.9 = 0.45)
        result = ctrl.step(rho_bar=0.85, js_current=0.1)
        # β = 0.45 → balanced
        assert 0.33 <= result.beta_new < 0.67

        bucket = beta_to_bucket(result.beta_new)
        assert bucket == "balanced"

    def test_revise_tone_follows_same_beta(self):
        """Both critique and revise phases get tone from the same β."""
        reg = PromptRegistry()
        beta = 0.8
        critique = reg.build(role="macro", phase="critique", beta=beta, user_prompt="test")
        revise = reg.build(role="macro", phase="revise", beta=beta, user_prompt="test")
        assert critique.beta_bucket == "adversarial"
        assert revise.beta_bucket == "adversarial"
        # But they use different tone files
        assert critique.tone_file == "critique_adversarial.txt"
        assert revise.tone_file == "revise_adversarial.txt"


# ---------------------------------------------------------------------------
# Role-specific prompts with tone injection
# ---------------------------------------------------------------------------

class TestRolesWithTone:
    """Verify tone injection works for all agent roles."""

    @pytest.mark.parametrize("role", ["macro", "value", "risk", "technical"])
    def test_all_roles_get_tone_in_critique(self, role):
        """All roles get tone injected in critique phase."""
        reg = PromptRegistry()
        result = reg.build(role=role, phase="critique", beta=0.9, user_prompt="test")
        assert "tone" in result.blocks_used
        assert result.beta_bucket == "adversarial"

    @pytest.mark.parametrize("role", ["macro", "value", "risk", "technical"])
    def test_all_roles_get_tone_in_revise(self, role):
        """All roles get tone injected in revise phase."""
        reg = PromptRegistry()
        result = reg.build(role=role, phase="revise", beta=0.1, user_prompt="test")
        assert "tone" in result.blocks_used
        assert result.beta_bucket == "collaborative"

    @pytest.mark.parametrize("role", ["macro", "value", "risk", "technical"])
    def test_role_preamble_preserved_with_tone(self, role):
        """Role preamble should still be present when tone is injected."""
        reg = PromptRegistry()
        result = reg.build(role=role, phase="critique", beta=0.5, user_prompt="test")
        assert role.upper() in result.system_prompt
