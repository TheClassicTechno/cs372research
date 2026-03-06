"""Root conftest: shared fixtures and marker registration for all test layers.

L1/L2 (fast):        Pure unit + prompt tests, no I/O, no API
L3/L4 (integration): Graph state, mock LLM, artifacts
L5 (live):           Real API calls, auto-skipped by default
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Shared test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_observation() -> dict:
    """Minimal observation dict for debate nodes."""
    return {
        "universe": ["AAPL", "MSFT", "NVDA"],
        "timestamp": "2025-01-15T00:00:00Z",
        "portfolio_state": {"cash": 100_000.0},
        "text_context": "## Q4 2024 Market Summary\nStrong tech earnings.",
    }


@pytest.fixture
def sample_obs_dict(sample_observation) -> dict:
    """Observation as a flat dict (matches LangGraph state format)."""
    return sample_observation


@pytest.fixture
def mock_config() -> dict:
    """Minimal config dict for debate tests (old profile system)."""
    return {
        "roles": ["macro", "value", "risk"],
        "max_rounds": 1,
        "mock": True,
        "model_name": "mock",
        "temperature": 0.0,
        "prompt_profile": "default",
        "prompt_file_overrides": {},
        "role_overrides": {},
        "agent_profiles": {},
        "judge_profile": {},
        "parallel_agents": False,
        "verbose": False,
        "console_display": False,
        "log_tokens": False,
        "log_rendered_prompts": False,
        "log_prompt_manifest": False,
        "prompt_logging": {},
        "logging_mode": "off",
        "pid_enabled": False,
        "allocation_mode": True,
        "skip_pipeline": True,
        "no_rate_limit": False,
        "llm_stagger_ms": 0,
        "max_concurrent_llm": 0,
        "sector_config": None,
    }


@pytest.fixture
def mock_config_with_profiles() -> dict:
    """Config dict using the new agent profile system."""
    from multi_agent.prompts.profile_loader import get_agent_profiles

    agent_map = {
        "macro": "macro_standard",
        "value": "value_standard",
        "risk": "risk_standard",
    }
    profiles = get_agent_profiles(agent_map, "judge_standard")
    judge_profile = profiles.pop("judge")

    return {
        "roles": ["macro", "value", "risk"],
        "max_rounds": 1,
        "mock": True,
        "model_name": "mock",
        "temperature": 0.0,
        "prompt_profile": "default",
        "prompt_file_overrides": {},
        "role_overrides": {},
        "agent_profiles": profiles,
        "judge_profile": judge_profile,
        "parallel_agents": False,
        "verbose": False,
        "console_display": False,
        "log_tokens": False,
        "log_rendered_prompts": False,
        "log_prompt_manifest": False,
        "prompt_logging": {},
        "logging_mode": "off",
        "pid_enabled": False,
        "allocation_mode": True,
        "skip_pipeline": True,
        "no_rate_limit": False,
        "llm_stagger_ms": 0,
        "max_concurrent_llm": 0,
        "sector_config": None,
    }


@pytest.fixture
def mock_config_dict(mock_config) -> dict:
    """Alias for mock_config."""
    return mock_config


@pytest.fixture
def pid_config() -> dict:
    """PID configuration dict for testing."""
    return {
        "gains": {"Kp": 0.15, "Ki": 0.01, "Kd": 0.03},
        "rho_star": 0.8,
        "epsilon": 0.001,
    }


@pytest.fixture
def mock_crit_response() -> dict:
    """Mock CRIT scoring response."""
    return {
        "pillar_scores": {
            "logical_validity": 0.75,
            "evidential_support": 0.70,
            "alternative_consideration": 0.65,
            "causal_alignment": 0.80,
        },
        "diagnostics": {
            "contradictions_detected": False,
            "unsupported_claims_detected": False,
            "ignored_critiques_detected": False,
            "premature_certainty_detected": False,
            "causal_overreach_detected": False,
            "conclusion_drift_detected": False,
        },
        "explanations": {
            "logical_validity": "Sound reasoning chain.",
            "evidential_support": "Most claims backed by evidence.",
            "alternative_consideration": "Some alternatives considered.",
            "causal_alignment": "Good causal claim classification.",
        },
        "rho_bar": 0.725,
    }
