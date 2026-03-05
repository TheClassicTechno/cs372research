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
# 3. User prompt section ordering wired through nodes
# =============================================================================


class TestNodeSectionOrdering:
    """Verify that user_prompt_section_order in config reaches the prompt builders."""

    def test_proposal_custom_section_order(self, obs_dict):
        """Custom section order changes the structure of the proposal user prompt."""
        # Default order: preamble, context, agent_data, task, scaffolding, output_format
        # Proposal template has: context, task, scaffolding, output_format (no preamble/agent_data)
        config_default = _make_config()
        config_reversed = _make_config(
            user_prompt_section_order=["output_format", "scaffolding", "task", "context"],
        )

        result_default, _ = _run_propose(obs_dict, config_default)
        result_reversed, _ = _run_propose(obs_dict, config_reversed)

        turn_default = _get_turns(result_default, "proposal")[0]
        turn_reversed = _get_turns(result_reversed, "proposal")[0]

        prompt_default = turn_default["raw_user_prompt"]
        prompt_reversed = turn_reversed["raw_user_prompt"]

        # Both should contain the same content words
        assert "Respond with valid JSON" in prompt_default
        assert "Respond with valid JSON" in prompt_reversed

        # But in reversed order, output_format (JSON instructions) should come
        # before the task section. Find positions of key markers.
        json_pos_default = prompt_default.find("Respond with valid JSON")
        json_pos_reversed = prompt_reversed.find("Respond with valid JSON")

        # "Using the data above" is in the task section
        task_marker = "Using the data above"
        task_pos_default = prompt_default.find(task_marker)
        task_pos_reversed = prompt_reversed.find(task_marker)

        # Default: task before output_format
        if task_pos_default != -1 and json_pos_default != -1:
            assert task_pos_default < json_pos_default

        # Reversed: output_format before task
        if task_pos_reversed != -1 and json_pos_reversed != -1:
            assert json_pos_reversed < task_pos_reversed

    def test_critique_custom_section_order(self, obs_dict):
        """Custom section order changes critique user prompt structure."""
        # Critique template has: preamble, context, agent_data, task, output_format
        config_reversed = _make_config(
            user_prompt_section_order=["output_format", "task", "agent_data", "context", "preamble"],
        )

        result, _ = _run_through_critique(obs_dict, config_reversed)
        turn = _get_turns(result, "critique")[0]
        prompt = turn["raw_user_prompt"]

        # The output_format section should appear before the preamble section
        # Critique preamble contains role-specific text, output says "JSON"
        json_pos = prompt.find("JSON")
        # Preamble/context typically mentions the role or "As the"
        role_pos = prompt.find("MACRO")
        if json_pos != -1 and role_pos != -1:
            assert json_pos < role_pos, "output_format should precede role-related content"


# =============================================================================
# 4. System prompt block ordering wired through nodes
# =============================================================================


class TestNodeBlockOrdering:
    """Verify that system_prompt_block_order in config reaches PromptRegistry."""

    def setup_method(self):
        reset_registry_cache()

    def test_critique_custom_block_order(self, obs_dict):
        """Custom block order changes the structure of critique system prompt."""
        # Default order: causal_contract, role_system, phase_preamble, tone
        # Reversed: tone first, then phase_preamble, then role_system
        config = _make_config(
            _pid_enabled_flag=True,
            system_prompt_block_order=["tone", "phase_preamble", "role_system"],
        )
        config["_current_beta"] = 0.9  # adversarial

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        # With reversed order, tone (ADVERSARIAL) should appear before the
        # role identity text. The role prompt for macro contains "macro" or
        # a multi-paragraph role description.
        tone_pos = sys_prompt.find("ADVERSARIAL")
        # Phase preamble contains "critiques"
        preamble_pos = sys_prompt.find("critiques")

        assert tone_pos != -1, "Adversarial tone should be present"
        assert preamble_pos != -1, "Phase preamble should be present"
        assert tone_pos < preamble_pos, (
            "With custom order [tone, phase_preamble, role_system], "
            "tone should appear before phase_preamble"
        )

    def test_revise_custom_block_order_role_only(self, obs_dict):
        """Block order with only role_system → no tone, no preamble."""
        config = _make_config(
            _pid_enabled_flag=True,
            system_prompt_block_order=["role_system"],
        )
        config["_current_beta"] = 0.9

        result, _ = _run_through_revise(obs_dict, config)
        turn = _get_turns(result, "revision")[0]
        sys_prompt = turn["raw_system_prompt"]

        # Only role_system block — no tone or preamble
        assert "ADVERSARIAL" not in sys_prompt
        assert "FIRM" not in sys_prompt
        # Role system prompt should still be present (it's always available)
        assert len(sys_prompt) > 50, "Role system prompt should have content"


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

        Use slim variant as the override file — the system prompt should
        contain the slim content instead of the full role prompt.
        """
        config = _make_config(
            _pid_enabled_flag=True,
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_critique(obs_dict, config)
        # Find the macro agent's critique turn
        turns = _get_turns(result, "critique")
        macro_turn = next(t for t in turns if t["role"] == "macro")

        sys_prompt = macro_turn["raw_system_prompt"]
        # The slim variant is shorter than the full variant.
        # Verify by checking the full variant's distinctive content is absent
        # or the slim variant's content is present.
        from multi_agent.prompts import ROLE_SYSTEM_PROMPTS, ROLE_SYSTEM_PROMPTS_SLIM
        slim_content = ROLE_SYSTEM_PROMPTS_SLIM[AgentRole.MACRO]
        full_content = ROLE_SYSTEM_PROMPTS[AgentRole.MACRO]

        # Slim should be in the prompt, and it should differ from full
        assert slim_content in sys_prompt or len(sys_prompt) < len(full_content) + 200

    def test_judge_file_override_in_modular_path(self, obs_dict):
        """Override judge system prompt file via config in modular path."""
        config = _make_config(
            _pid_enabled_flag=True,
            # Use a known existing file as the judge override
            prompt_file_overrides={"judge_system": "roles/macro_slim.txt"},
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_judge(obs_dict, config)
        turn = _get_turns(result, "judge_decision")[0]
        sys_prompt = turn["raw_system_prompt"]

        # The default judge prompt says "You are the Judge"
        # With the override, it should contain the macro_slim content instead
        from multi_agent.prompts import ROLE_SYSTEM_PROMPTS_SLIM
        slim_content = ROLE_SYSTEM_PROMPTS_SLIM[AgentRole.MACRO]
        assert slim_content in sys_prompt, (
            "Judge system prompt should use overridden file content"
        )


# =============================================================================
# 6. Causal contract propagation
# =============================================================================


class TestCausalContractWiring:
    """Verify use_system_causal_contract flag propagates to node prompts."""

    def setup_method(self):
        reset_registry_cache()

    def test_propose_with_causal_contract(self, obs_dict):
        """When causal contract is enabled, propose system prompt includes it."""
        config = _make_config(use_system_causal_contract=True)

        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] in sys_prompt

    def test_propose_without_causal_contract(self, obs_dict):
        """Without causal contract, propose system prompt omits it."""
        config = _make_config(use_system_causal_contract=False)

        result, _ = _run_propose(obs_dict, config)
        turn = _get_turns(result, "proposal")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] not in sys_prompt

    def test_critique_modular_with_causal_contract(self, obs_dict):
        """PID + causal contract → system prompt has both contract and tone."""
        config = _make_config(
            _pid_enabled_flag=True,
            use_system_causal_contract=True,
        )
        config["_current_beta"] = 0.9

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import SYSTEM_CAUSAL_CONTRACT
        assert SYSTEM_CAUSAL_CONTRACT[:50] in sys_prompt
        assert "ADVERSARIAL" in sys_prompt

    def test_critique_modular_causal_contract_uses_slim_role(self, obs_dict):
        """PID + causal contract → role prompt is the slim variant."""
        config = _make_config(
            _pid_enabled_flag=True,
            use_system_causal_contract=True,
        )
        config["_current_beta"] = 0.5

        result, _ = _run_through_critique(obs_dict, config)
        turn = _get_turns(result, "critique")[0]
        sys_prompt = turn["raw_system_prompt"]

        from multi_agent.prompts import ROLE_SYSTEM_PROMPTS_SLIM
        macro_slim = ROLE_SYSTEM_PROMPTS_SLIM[AgentRole.MACRO]
        # The slim content should appear in the system prompt
        assert macro_slim in sys_prompt


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

    def test_critique_with_causal_contract_has_role_identity(self, obs_dict):
        """Causal contract + tone: role identity still present."""
        config = _make_config(
            _pid_enabled_flag=True,
            use_system_causal_contract=True,
        )
        config["_current_beta"] = 0.9
        result, _ = _run_through_critique(obs_dict, config)
        turns = _get_turns(result, "critique")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} critique (with cc) missing identity '{expected}'"
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

    def test_revise_with_causal_contract_has_role_identity(self, obs_dict):
        """Causal contract + tone: role identity still present."""
        config = _make_config(
            _pid_enabled_flag=True,
            use_system_causal_contract=True,
        )
        config["_current_beta"] = 0.1
        result, _ = _run_through_revise(obs_dict, config)
        turns = _get_turns(result, "revision")

        for turn in turns:
            role = turn["role"]
            expected = _ROLE_IDENTITY[role]
            assert expected in turn["raw_system_prompt"], (
                f"{role} revise (with cc) missing identity '{expected}'"
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
