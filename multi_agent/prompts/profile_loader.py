"""Agent profile loader — loads per-agent profile YAMLs from config/agent_profiles/.

Each profile YAML is the single source of truth for all prompts an agent receives.
Explicit file paths, no dispatch keys.

Usage:
    profile = load_agent_profile("macro_diverse")
    profiles = get_agent_profiles({"macro": "macro_diverse", "value": "value_diverse"})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROFILE_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "agent_profiles"
_PROMPT_DIR = Path(__file__).resolve().parent

# Cache loaded profiles to avoid repeated disk reads
_profile_cache: dict[str, dict] = {}


class ProfileLoadError(Exception):
    """Raised when a profile YAML cannot be loaded or is malformed."""


class ProfileValidationError(Exception):
    """Raised when a profile fails structural validation."""


def reset_profile_cache() -> None:
    """Clear the profile cache. Call between test runs."""
    _profile_cache.clear()


def load_agent_profile(name: str) -> dict:
    """Load an agent profile YAML by name.

    Args:
        name: Profile name (e.g. "macro_diverse"). Looks up
              ``config/agent_profiles/{name}.yaml``.

    Returns:
        Parsed YAML dict with ``system_prompts`` and ``user_prompts`` keys.

    Raises:
        ProfileLoadError: If the file doesn't exist or is malformed.
    """
    if name in _profile_cache:
        return _profile_cache[name]

    path = _PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileLoadError(f"Agent profile not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ProfileLoadError(f"Agent profile is not a YAML mapping: {path}")

    # Basic structural validation
    if "system_prompts" not in raw:
        raise ProfileLoadError(f"Agent profile missing 'system_prompts': {path}")
    if "user_prompts" not in raw:
        raise ProfileLoadError(f"Agent profile missing 'user_prompts': {path}")

    _profile_cache[name] = raw
    return raw


def get_agent_profiles(
    agent_map: dict[str, str],
    judge_profile_name: str = "judge_standard",
) -> dict[str, dict]:
    """Load profiles for all agents in a debate.

    Args:
        agent_map: Mapping of role name to profile name,
                   e.g. ``{"macro": "macro_diverse", "value": "value_diverse"}``.
        judge_profile_name: Profile name for the judge agent.

    Returns:
        Dict mapping role name to loaded profile dict.
        Includes a "judge" key for the judge profile.
    """
    profiles: dict[str, dict] = {}
    for role, profile_name in agent_map.items():
        profiles[role] = load_agent_profile(profile_name)
    profiles["judge"] = load_agent_profile(judge_profile_name)
    return profiles


def validate_agent_profile(name: str, profile: dict) -> list[str]:
    """Validate an agent profile for structural correctness.

    Checks:
        - All declared prompt files exist on disk
        - Required phases are defined (propose/critique/revise for agents, judge for judge)
        - User prompt templates have 'template' and 'sections' keys

    Args:
        name: Profile name (for error messages).
        profile: Parsed profile dict.

    Returns:
        List of warning/error messages (empty = valid).
    """
    errors: list[str] = []
    sys_prompts = profile.get("system_prompts", {})
    usr_prompts = profile.get("user_prompts", {})

    # Check each phase's system prompt files exist
    for phase, blocks in sys_prompts.items():
        if not isinstance(blocks, list):
            errors.append(f"[{name}] system_prompts.{phase} must be a list")
            continue
        for entry in blocks:
            if entry == "tone":
                continue  # resolved at runtime
            file_path = _PROMPT_DIR / entry
            if not file_path.exists():
                errors.append(
                    f"[{name}] system_prompts.{phase}: file not found: {entry}"
                )

    # Check each phase's user prompt config
    for phase, config in usr_prompts.items():
        if not isinstance(config, dict):
            errors.append(f"[{name}] user_prompts.{phase} must be a dict")
            continue
        if "template" not in config:
            errors.append(f"[{name}] user_prompts.{phase} missing 'template'")
        else:
            tmpl_path = _PROMPT_DIR / config["template"]
            if not tmpl_path.exists():
                errors.append(
                    f"[{name}] user_prompts.{phase}: template not found: {config['template']}"
                )
        if "sections" not in config:
            errors.append(f"[{name}] user_prompts.{phase} missing 'sections'")

    return errors


def validate_all_profiles(
    agent_map: dict[str, str],
    judge_profile_name: str = "judge_standard",
) -> list[str]:
    """Validate all profiles in a debate configuration.

    Args:
        agent_map: Mapping of role name to profile name.
        judge_profile_name: Profile name for the judge.

    Returns:
        List of all validation errors across all profiles.
    """
    all_errors: list[str] = []
    for role, profile_name in agent_map.items():
        try:
            profile = load_agent_profile(profile_name)
            all_errors.extend(validate_agent_profile(profile_name, profile))
        except ProfileLoadError as e:
            all_errors.append(f"[{role}] {e}")

    try:
        judge = load_agent_profile(judge_profile_name)
        all_errors.extend(validate_agent_profile(judge_profile_name, judge))
    except ProfileLoadError as e:
        all_errors.append(f"[judge] {e}")

    return all_errors
