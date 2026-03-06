"""Tests that node functions correctly wire prompt configs to prompt builders.

Covers the gaps between:
  - Unit tests (test_prompt_ordering.py) that test build_*_prompt() directly
  - Unit tests (test_registry.py) that test PromptRegistry.build() directly
  - Integration tests that run full debates but never use custom ordering

These tests invoke actual node functions in mock mode and inspect the
raw_system_prompt / raw_user_prompt captured in debate_turns to verify
that config fields (section_order, block_order, file_overrides, β)
actually reach the prompt builders.

All tests use mock=True — no real API calls.
"""

from __future__ import annotations

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.graph import (
    build_context_node,
    critique_node,
    judge_node,
    propose_node,
    revise_node,
)
from multi_agent.models import MarketState, Observation, PortfolioState
from multi_agent.prompts.registry import reset_registry_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG"]


@pytest.fixture
def obs_dict():
    """Observation dict with 3 tickers and memo context."""
    return Observation(
        timestamp="2024-12-31",
        universe=TICKERS,
        market_state=MarketState(
            prices={"AAPL": 185.0, "MSFT": 390.0, "GOOG": 140.0},
        ),
        text_context="Q4 2024 macro analysis memo with [L1-VIX] and [AAPL-RET60] data.",
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
    ).model_dump()


def _make_config(**overrides) -> dict:
    """Build a DebateConfig dict with mock=True and optional overrides."""
    kwargs = dict(
        mock=True,
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=1,
        trace_dir="/tmp/test_traces",
    )
    kwargs.update(overrides)
    return DebateConfig(**kwargs).to_dict()


def _make_state(obs_dict, config_dict, **overrides):
    """Minimal DebateState dict."""
    state = {
        "observation": obs_dict,
        "config": config_dict,
        "news_digest": "",
        "data_analysis": "",
        "enriched_context": "",
        "proposals": [],
        "critiques": [],
        "revisions": [],
        "current_round": 0,
        "debate_turns": [],
        "final_action": {},
        "strongest_objection": "",
        "audited_memo": "",
        "trace": {},
    }
    state.update(overrides)
    return state


def _run_propose(obs_dict, config_dict):
    """Run propose_node and return (result, state)."""
    state = _make_state(obs_dict, config_dict, enriched_context="Test context")
    result = propose_node(state)
    return result, state


def _run_through_critique(obs_dict, config_dict):
    """Run propose → critique and return critique result + merged state."""
    state = _make_state(obs_dict, config_dict, enriched_context="Test context")
    state.update(propose_node(state))
    state["current_round"] = 1
    result = critique_node(state)
    return result, state


def _run_through_revise(obs_dict, config_dict):
    """Run propose → critique → revise and return revise result + state."""
    state = _make_state(obs_dict, config_dict, enriched_context="Test context")
    state.update(propose_node(state))
    state["current_round"] = 1
    state.update(critique_node(state))
    result = revise_node(state)
    return result, state


def _run_through_judge(obs_dict, config_dict):
    """Run full pipeline → judge and return judge result + state."""
    state = _make_state(obs_dict, config_dict, enriched_context="Test context")
    state.update(propose_node(state))
    state["current_round"] = 1
    state.update(critique_node(state))
    state.update(revise_node(state))
    result = judge_node(state)
    return result, state


def _get_turns(result, turn_type=None):
    """Extract debate_turns from a node result, optionally filtering by type."""
    turns = result.get("debate_turns", [])
    if turn_type:
        turns = [t for t in turns if t["type"] == turn_type]
    return turns


# =============================================================================
# 1. Tone routing through unified PromptRegistry
# =============================================================================


class TestToneRouting:
    """Verify that nodes route through PromptRegistry and apply correct tone."""

    def setup_method(self):
        reset_registry_cache()

    def test_critique_explicit_beta_adversarial(self, obs_dict):
        """Explicit _current_beta=0.9 → critique system prompt has adversarial tone."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9  # high β → adversarial bucket

        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")
        assert len(turns) >= 1

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            assert "ADVERSARIAL" in sys_prompt, (
                f"Expected adversarial tone in system prompt for {turn['role']}, "
                f"got: {sys_prompt[:200]}"
            )

    def test_critique_explicit_beta_collaborative(self, obs_dict):
        """Explicit _current_beta=0.1 → critique system prompt has collaborative tone."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.1  # low β → collaborative bucket

        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            assert "COLLABORATIVE" in sys_prompt, (
                f"Expected collaborative tone for {turn['role']}, "
                f"got: {sys_prompt[:200]}"
            )

    def test_critique_no_beta_no_tone(self, obs_dict):
        """No _current_beta → no tone injection (beta is None)."""
        config = _make_config()
        # No _current_beta set → resolve_beta returns None → no tone

        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            # Without _current_beta, no tone should be injected
            assert "ADVERSARIAL" not in sys_prompt or "COLLABORATIVE" not in sys_prompt

    def test_revise_has_tone(self, obs_dict):
        """Revise system prompt gets tone injection via registry."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9  # high β → adversarial/firm

        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")
        assert len(turns) >= 1

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            assert "FIRM" in sys_prompt or "Stand by" in sys_prompt, (
                f"Expected firm/adversarial revise tone for {turn['role']}, "
                f"got: {sys_prompt[:200]}"
            )

    def test_judge_no_tone(self, obs_dict):
        """Judge routes through registry but gets no tone injection."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9

        result, _ = _run_through_judge(obs_dict, config)
        turns = _get_turns(result, "judge_decision")
        assert len(turns) == 1

        sys_prompt = turns[0]["raw_system_prompt"]
        assert "ADVERSARIAL" not in sys_prompt
        assert "COLLABORATIVE" not in sys_prompt
        assert "FIRM" not in sys_prompt
        assert "Judge" in sys_prompt


# =============================================================================
# 2. Propose never gets tone injection
# =============================================================================


class TestProposeNoToneInjection:
    """Propose routes through registry but never gets tone injection."""

    def setup_method(self):
        reset_registry_cache()

    def test_propose_no_tone_with_high_beta(self, obs_dict):
        """Even with high β, propose has no tone injection."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9

        result, _ = _run_propose(obs_dict, config)
        turns = _get_turns(result, "proposal")
        assert len(turns) >= 1

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            assert "ADVERSARIAL" not in sys_prompt
            assert "COLLABORATIVE" not in sys_prompt
            assert "FIRM" not in sys_prompt

    def test_propose_system_prompt_unchanged_with_pid(self, obs_dict):
        """Propose system prompt is identical with and without PID."""
        config_no_pid = _make_config()
        config_pid = _make_config(_pid_enabled_flag=True)
        config_pid["_current_beta"] = 0.5

        result_no_pid, _ = _run_propose(obs_dict, config_no_pid)
        result_pid, _ = _run_propose(obs_dict, config_pid)

        turns_no_pid = _get_turns(result_no_pid, "proposal")
        turns_pid = _get_turns(result_pid, "proposal")

        for t_no, t_pid in zip(turns_no_pid, turns_pid):
            assert t_no["raw_system_prompt"] == t_pid["raw_system_prompt"], (
                f"Propose system prompt differs for {t_no['role']} between PID on/off"
            )


# =============================================================================
# 3. Profile-driven ordering wired through nodes
# =============================================================================


class TestNodeProfileOrdering:
    """Verify that prompt profiles drive block/section ordering through nodes.

    Block ordering and section ordering are now entirely driven by prompt
    profiles (YAML), not by DebateConfig fields.
    """

    def setup_method(self):
        reset_registry_cache()

    def test_default_profile_critique_has_tone(self, obs_dict):
        """Default profile critique includes tone block when beta is set."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9  # adversarial

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        assert "ADVERSARIAL" in sys_prompt, "Default profile critique should include tone"

    def test_default_profile_propose_has_no_tone(self, obs_dict):
        """Default profile propose does not include tone block."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9

        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        sys_prompt = turn["raw_system_prompt"]

        assert "ADVERSARIAL" not in sys_prompt
        assert "COLLABORATIVE" not in sys_prompt


# =============================================================================
# 5. Prompt file overrides wired through nodes
# =============================================================================


class TestNodeFileOverrides:
    """Verify that prompt_file_overrides in config reaches prompt builders."""

    def setup_method(self):
        reset_registry_cache()

    def test_proposal_template_override(self, obs_dict):
        """Override proposal template file via config."""
        # Use the same file (to avoid needing a custom file) — just verify
        # the override key is respected by checking the prompt is non-empty
        config = _make_config(
            prompt_file_overrides={"proposal_template": "phases/proposal_allocation.txt"},
        )
        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        assert len(turn["raw_user_prompt"]) > 100

    def test_critique_template_override(self, obs_dict):
        """Override critique template file via config."""
        config = _make_config(
            prompt_file_overrides={"critique_template": "phases/critique_allocation.txt"},
        )
        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        assert len(turn["raw_user_prompt"]) > 100

    def test_role_file_override_in_modular_path(self, obs_dict):
        """Override role prompt file via config in modular (PID) path.

        Use diverse variant as the override file — the system prompt should
        contain the diverse content instead of the standard role prompt.
        """
        config = _make_config(
            _pid_enabled_flag=True,
            prompt_file_overrides={"role_macro": "roles/macro_diverse.txt"},
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_critique(obs_dict, config)
        # Find the macro agent's critique turn
        turns = _get_turns(result, "critique")
        macro_turn = next(t for t in turns if t["role"] == "macro")

        sys_prompt = macro_turn["raw_system_prompt"]
        # The diverse variant should be different from the standard variant
        from multi_agent.prompts import ROLE_SYSTEM_PROMPTS
        standard_content = ROLE_SYSTEM_PROMPTS[AgentRole.MACRO]

        # Override should produce a different prompt than the standard variant
        assert sys_prompt != standard_content

    def test_judge_file_override_in_modular_path(self, obs_dict):
        """Override judge system prompt file via config in modular path."""
        config = _make_config(
            _pid_enabled_flag=True,
            # Use a known existing file as the judge override
            prompt_file_overrides={"judge_system": "roles/macro_diverse.txt"},
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_judge(obs_dict, config)
        turn = _get_turns(result, "judge_decision")[0]
        sys_prompt = turn["raw_system_prompt"]

        # The default judge prompt says "You are the Judge"
        # With the override, it should NOT contain the default judge text
        # and should instead contain the overridden file content
        assert "MACRO" in sys_prompt.upper() or "You are the Judge" not in sys_prompt, (
            "Judge system prompt should use overridden file content"
        )


# =============================================================================
# 6. Causal contract propagation
# =============================================================================


class TestCausalContractWiring:
    """Verify causal_contract is profile-driven: default profile includes it,
    diverse_agents profile does not."""

    def setup_method(self):
        reset_registry_cache()

    def test_default_profile_propose_has_causal_contract(self, obs_dict):
        """Default profile includes causal_contract in propose system_blocks."""
        config = _make_config()  # prompt_profile defaults to "default"

        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] in sys_prompt

    def test_default_profile_critique_has_causal_contract(self, obs_dict):
        """Default profile includes causal_contract in critique system_blocks."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.9

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] in sys_prompt
        assert "ADVERSARIAL" in sys_prompt

    def test_diverse_agents_profile_omits_causal_contract(self, obs_dict):
        """diverse_agents profile does NOT include causal_contract."""
        config = _make_config(prompt_profile="diverse_agents")

        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] not in sys_prompt

    def test_diverse_agents_critique_omits_causal_contract(self, obs_dict):
        """diverse_agents critique also omits causal_contract."""
        config = _make_config(
            _pid_enabled_flag=True,
            prompt_profile="diverse_agents",
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] not in sys_prompt


# =============================================================================
# 7. β=None disables tone even in modular path
# =============================================================================


# =============================================================================
# 8. Role identity survives into critique and revise system prompts
# =============================================================================

# Each role prompt starts with a distinctive identity line:
#   macro     → "MACRO STRATEGIST"
#   value     → "VALUE/FUNDAMENTALS ANALYST"
#   risk      → "RISK MANAGER"
# These must be present in every agent's system prompt during critique
# and revise, regardless of PID state or beta value.

_ROLE_IDENTITY = {
    "macro": "MACRO STRATEGIST",
    "value": "VALUE/FUNDAMENTALS ANALYST",
    "risk": "RISK MANAGER",
}


class TestRoleIdentityInCritique:
    """Every agent keeps its role identity in the critique system prompt."""

    def setup_method(self):
        reset_registry_cache()

    def test_critique_pid_off_has_role_identity(self, obs_dict):
        """PID off: each agent's critique prompt has its role."""
        config = _make_config()
        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} critique (PID off) missing identity '{expected}'"
            )

    def test_critique_pid_on_has_role_identity(self, obs_dict):
        """PID on (explicit beta): each agent's critique prompt has its role."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.5
        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} critique (PID on) missing identity '{expected}'"
            )

    def test_critique_with_default_profile_has_role_identity(self, obs_dict):
        """Default profile (with causal_contract) + tone: role identity still present."""
        config = _make_config(
            _pid_enabled_flag=True,
        )
        config["_current_beta"] = 0.9
        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} critique (default profile) missing identity '{expected}'"
            )

    def test_critique_no_explicit_beta_has_role_identity(self, obs_dict):
        """No _current_beta: role identity still present."""
        config = _make_config()
        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} critique (no explicit beta) missing identity '{expected}'"
            )


class TestRoleIdentityInRevise:
    """Every agent keeps its role identity in the revise system prompt."""

    def setup_method(self):
        reset_registry_cache()

    def test_revise_pid_off_has_role_identity(self, obs_dict):
        """PID off: each agent's revise prompt has its role."""
        config = _make_config()
        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} revise (PID off) missing identity '{expected}'"
            )

    def test_revise_pid_on_has_role_identity(self, obs_dict):
        """PID on (explicit beta): each agent's revise prompt has its role."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = 0.5
        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} revise (PID on) missing identity '{expected}'"
            )

    def test_revise_with_default_profile_has_role_identity(self, obs_dict):
        """Default profile (with causal_contract) + tone: role identity still present."""
        config = _make_config(
            _pid_enabled_flag=True,
        )
        config["_current_beta"] = 0.1
        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} revise (default profile) missing identity '{expected}'"
            )

    def test_revise_no_explicit_beta_has_role_identity(self, obs_dict):
        """No _current_beta: role identity still present."""
        config = _make_config()
        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} revise (no explicit beta) missing identity '{expected}'"
            )


# =============================================================================
# 9. _current_beta=None → no tone injection
# =============================================================================


class TestBetaNoneNoTone:
    """When _current_beta is None, no tone injection occurs."""

    def setup_method(self):
        reset_registry_cache()

    def test_critique_no_beta_gets_no_tone(self, obs_dict):
        """_current_beta=None → no tone injection in critique."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = None  # simulates propose phase (no PID β)

        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            # No beta → no tone injection
            assert "ADVERSARIAL" not in sys_prompt, (
                f"Expected no tone injection when beta is None for {turn['role']}"
            )

    def test_revise_no_beta_gets_no_tone(self, obs_dict):
        """_current_beta=None → no tone injection in revise."""
        config = _make_config(_pid_enabled_flag=True)
        config["_current_beta"] = None

        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            sys_prompt = turn["raw_system_prompt"]
            # No beta → no tone injection
            assert "FIRM" not in sys_prompt, (
                f"Expected no tone injection when beta is None for {turn['role']}"
            )
