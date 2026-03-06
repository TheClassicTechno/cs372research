"""L2 composition invariant tests for the prompt profile system.

Validates:
  - Every file declared in an agent profile exists on disk.
  - Every declared system file appears in the rendered system prompt (no silent drops).
  - Tone is correctly absent when beta=None and present when beta is provided.
  - Invalid file path in profile raises PromptCompositionError.
  - All agent profiles pass validation across all phases.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from multi_agent.prompts.profile_loader import (
    load_agent_profile,
    validate_agent_profile,
    reset_profile_cache,
    _PROMPT_DIR,
)
from multi_agent.prompts.registry import (
    PromptRegistry,
    PromptCompositionError,
    reset_registry_cache,
    _load_prompt_file,
)

# ---------------------------------------------------------------------------
# All agent profile names (discovered from config/agent_profiles/*.yaml)
# ---------------------------------------------------------------------------

_PROFILE_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "agent_profiles"

_ALL_PROFILE_NAMES = sorted(
    p.stem for p in _PROFILE_DIR.glob("*.yaml") if p.is_file()
)


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_registry_cache()
    reset_profile_cache()
    yield
    reset_registry_cache()
    reset_profile_cache()


# ---------------------------------------------------------------------------
# 1. Every file declared in agent profile exists on disk
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestProfileFilesExist:
    """Every prompt file path declared in an agent profile must exist on disk."""

    @pytest.mark.parametrize("profile_name", _ALL_PROFILE_NAMES)
    def test_all_declared_files_exist(self, profile_name):
        profile = load_agent_profile(profile_name)
        errors = validate_agent_profile(profile_name, profile)
        file_errors = [e for e in errors if "not found" in e]
        assert not file_errors, (
            f"Profile '{profile_name}' references missing files:\n"
            + "\n".join(file_errors)
        )


# ---------------------------------------------------------------------------
# 2. Every declared system file appears in rendered system prompt
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNoSilentDrops:
    """Every system_prompts entry must contribute content to the rendered prompt."""

    @pytest.mark.parametrize("profile_name", _ALL_PROFILE_NAMES)
    def test_all_system_blocks_render(self, profile_name):
        profile = load_agent_profile(profile_name)
        sys_prompts = profile.get("system_prompts", {})
        registry = PromptRegistry()

        for phase, file_list in sys_prompts.items():
            if not isinstance(file_list, list):
                continue

            # For tone, use beta=0.5 so tone resolves; otherwise None
            has_tone = "tone" in file_list
            beta = 0.5 if has_tone else None

            result = registry.build_from_profile(
                role=profile_name.split("_")[0],  # e.g. "macro" from "macro_standard"
                phase=phase,
                profile=profile,
                beta=beta,
            )

            # Every non-tone entry should appear in blocks_used
            for entry in file_list:
                if entry == "tone" and beta is None:
                    continue
                if entry == "tone":
                    # Tone was provided with beta, should be in blocks_used
                    assert entry in result.blocks_used, (
                        f"Profile '{profile_name}' phase '{phase}': "
                        f"tone block not in blocks_used despite beta={beta}"
                    )
                else:
                    # Load the file content and verify it appears in system_prompt
                    content = _load_prompt_file(entry)
                    if content:
                        assert entry in result.blocks_used, (
                            f"Profile '{profile_name}' phase '{phase}': "
                            f"block '{entry}' silently dropped from system prompt"
                        )


# ---------------------------------------------------------------------------
# 3. Tone correctly absent when beta=None, present when beta provided
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTonePresenceByBeta:
    """Tone block behavior depends on beta value."""

    def _get_profile_with_tone(self):
        """Find a profile that declares 'tone' in at least one phase."""
        for name in _ALL_PROFILE_NAMES:
            profile = load_agent_profile(name)
            for phase, blocks in profile.get("system_prompts", {}).items():
                if isinstance(blocks, list) and "tone" in blocks:
                    return name, profile, phase
        pytest.skip("No profile with tone block found")

    def test_tone_absent_when_beta_none(self):
        name, profile, phase = self._get_profile_with_tone()
        registry = PromptRegistry()
        role = name.split("_")[0]

        result = registry.build_from_profile(
            role=role, phase=phase, profile=profile, beta=None,
        )
        assert "tone" not in result.blocks_used, (
            f"Tone should not appear in blocks_used when beta=None "
            f"(profile={name}, phase={phase})"
        )
        assert result.beta_bucket == ""
        assert result.tone_file == ""

    @pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
    def test_tone_present_when_beta_provided(self, beta):
        name, profile, phase = self._get_profile_with_tone()
        registry = PromptRegistry()
        role = name.split("_")[0]

        result = registry.build_from_profile(
            role=role, phase=phase, profile=profile, beta=beta,
        )
        assert "tone" in result.blocks_used, (
            f"Tone should appear in blocks_used when beta={beta} "
            f"(profile={name}, phase={phase})"
        )
        assert result.beta_bucket != ""
        assert result.tone_file != ""


# ---------------------------------------------------------------------------
# 4. Invalid file path in profile raises PromptCompositionError
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestInvalidFilePathRaisesError:
    """A profile declaring a non-existent file must raise PromptCompositionError."""

    def test_invalid_system_file_raises(self):
        fake_profile = {
            "system_prompts": {
                "propose": ["nonexistent/path/that_does_not_exist.txt"],
            },
            "user_prompts": {
                "propose": {
                    "template": "phases/proposal_allocation.txt",
                    "sections": ["context"],
                },
            },
        }
        registry = PromptRegistry()

        # The file won't be found, producing empty content.
        # _validate_composition should raise PromptCompositionError
        # because the declared block is missing from blocks_loaded.
        with pytest.raises(PromptCompositionError):
            registry.build_from_profile(
                role="macro",
                phase="propose",
                profile=fake_profile,
                beta=None,
            )

    def test_invalid_tone_file_with_beta_raises(self):
        """If tone is declared but the tone file doesn't resolve, it should fail."""
        # We mock _load_prompt_file to simulate a missing tone file
        fake_profile = {
            "system_prompts": {
                "critique": [
                    "system_contract/system_causal_contract.txt",
                    "roles/macro.txt",
                    "tone",
                ],
            },
            "user_prompts": {
                "critique": {
                    "template": "phases/critique_allocation.txt",
                    "sections": ["context"],
                },
            },
        }
        registry = PromptRegistry()

        # Patch _TONE_FILES to return a non-existent file for the bucket
        with patch(
            "multi_agent.prompts.registry._TONE_FILES",
            {
                ("critique", "balanced"): "nonexistent_tone.txt",
                ("critique", "collaborative"): "nonexistent_tone.txt",
                ("critique", "adversarial"): "nonexistent_tone.txt",
                ("revise", "balanced"): "nonexistent_tone.txt",
                ("revise", "collaborative"): "nonexistent_tone.txt",
                ("revise", "adversarial"): "nonexistent_tone.txt",
            },
        ):
            # beta=0.5 → balanced bucket → tries to load nonexistent_tone.txt
            with pytest.raises(PromptCompositionError):
                registry.build_from_profile(
                    role="macro",
                    phase="critique",
                    profile=fake_profile,
                    beta=0.5,
                )


# ---------------------------------------------------------------------------
# 5. All agent profiles x all phases pass validation
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAllProfilesValidate:
    """Every agent profile YAML must pass structural validation."""

    @pytest.mark.parametrize("profile_name", _ALL_PROFILE_NAMES)
    def test_profile_passes_validation(self, profile_name):
        profile = load_agent_profile(profile_name)
        errors = validate_agent_profile(profile_name, profile)
        assert not errors, (
            f"Profile '{profile_name}' failed validation:\n"
            + "\n".join(errors)
        )

    def test_all_profiles_load_without_error(self):
        """Smoke test: every profile YAML loads without exceptions."""
        for name in _ALL_PROFILE_NAMES:
            profile = load_agent_profile(name)
            assert isinstance(profile, dict), (
                f"Profile '{name}' did not load as dict"
            )
            assert "system_prompts" in profile, (
                f"Profile '{name}' missing system_prompts key"
            )
            assert "user_prompts" in profile, (
                f"Profile '{name}' missing user_prompts key"
            )
