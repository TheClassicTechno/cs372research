"""L3 integration tests: prompt manifest and config loading.

Loads each debate config from config/debate/*.yaml and verifies that:
- Configs with an ``agents:`` field have valid agent profile references
- Profile file paths exist on disk
- Prompt manifest can be built from config without errors

All tests use mock mode -- no API calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multi_agent.prompts.registry import build_prompt_manifest, reset_registry_cache


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBATE_CONFIG_DIR = REPO_ROOT / "config" / "debate"
AGENT_PROFILE_DIR = REPO_ROOT / "config" / "agent_profiles"
PROMPT_DIR = REPO_ROOT / "multi_agent" / "prompts"


def _discover_debate_configs() -> list[Path]:
    """Return all .yaml files in config/debate/."""
    if not DEBATE_CONFIG_DIR.is_dir():
        return []
    return sorted(DEBATE_CONFIG_DIR.glob("*.yaml"))


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


DEBATE_CONFIGS = _discover_debate_configs()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDebateConfigLoading:
    """Each debate YAML in config/debate/ must be loadable and valid."""

    def setup_method(self):
        reset_registry_cache()

    @pytest.mark.parametrize(
        "config_path",
        DEBATE_CONFIGS,
        ids=[p.stem for p in DEBATE_CONFIGS],
    )
    def test_config_yaml_loads_without_error(self, config_path: Path):
        """Config file is valid YAML."""
        data = _load_yaml(config_path)
        assert isinstance(data, dict), f"{config_path.name} did not parse as dict"

    @pytest.mark.parametrize(
        "config_path",
        DEBATE_CONFIGS,
        ids=[p.stem for p in DEBATE_CONFIGS],
    )
    def test_agent_profiles_reference_valid_files(self, config_path: Path):
        """If config has agents: field, each profile name should resolve to a file."""
        data = _load_yaml(config_path)
        agent_section = data.get("agent", {})
        agents = agent_section.get("agents", {})
        if not agents:
            pytest.skip(f"{config_path.name} has no agents: field")

        for role, profile_name in agents.items():
            # Agent profiles might be in profiles/ dir or as direct filenames
            profile_path = AGENT_PROFILE_DIR / f"{profile_name}.yaml"
            alt_path = PROMPT_DIR / "roles" / f"{profile_name}.txt"
            assert profile_path.exists() or alt_path.exists(), (
                f"Profile '{profile_name}' for role '{role}' in {config_path.name} "
                f"not found at {profile_path} or {alt_path}"
            )


@pytest.mark.integration
class TestProfileFilesExist:
    """All profile YAML files in the profiles directory should be loadable."""

    def test_profile_directory_exists(self):
        assert AGENT_PROFILE_DIR.is_dir(), (
            f"Profile directory does not exist: {AGENT_PROFILE_DIR}"
        )

    def test_all_profile_yamls_load(self):
        """Every .yaml in profiles/ must parse without error."""
        profiles = sorted(AGENT_PROFILE_DIR.glob("*.yaml"))
        assert len(profiles) > 0, "No profile YAML files found"

        for profile_path in profiles:
            data = _load_yaml(profile_path)
            assert isinstance(data, dict), (
                f"Profile {profile_path.name} did not parse as dict"
            )


@pytest.mark.integration
class TestPromptManifestBuild:
    """Prompt manifest can be built from a valid config dict."""

    def setup_method(self):
        reset_registry_cache()

    def test_manifest_from_default_config(self):
        """Build prompt manifest from default DebateConfig."""
        from multi_agent.config import DebateConfig

        config = DebateConfig(mock=True, console_display=False)
        config_dict = config.to_dict()
        manifest = build_prompt_manifest(config_dict)
        assert isinstance(manifest, dict), "Manifest should be a dict"

    def test_manifest_has_expected_structure(self):
        """Manifest should contain per-role file info."""
        from multi_agent.config import AgentRole, DebateConfig

        config = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            console_display=False,
        )
        config_dict = config.to_dict()
        manifest = build_prompt_manifest(config_dict)
        assert isinstance(manifest, dict)
        # Manifest should have at least some content
        assert len(manifest) > 0, "Manifest is empty"

    @pytest.mark.parametrize(
        "profile_name",
        ["default", "diverse_agents", "minimal", "no_scaffold"],
    )
    def test_manifest_from_each_profile(self, profile_name: str):
        """Build manifest with each known profile name."""
        from multi_agent.config import DebateConfig

        profile_path = AGENT_PROFILE_DIR / f"{profile_name}.yaml"
        if not profile_path.exists():
            pytest.skip(f"Profile {profile_name}.yaml not found")

        config = DebateConfig(
            mock=True,
            prompt_profile=profile_name,
            console_display=False,
        )
        config_dict = config.to_dict()
        manifest = build_prompt_manifest(config_dict)
        assert isinstance(manifest, dict)
