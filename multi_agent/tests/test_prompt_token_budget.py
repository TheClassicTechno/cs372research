"""L2 token budget tests for debate prompts.

Validates that system and user prompts stay within token-budget guardrails
so they fit inside LLM context windows with room for completion.

Uses tiktoken if available; falls back to len(text) // 4 approximation.
"""

from __future__ import annotations

import pytest

from multi_agent.prompts import (
    build_proposal_user_prompt,
    build_critique_prompt,
    build_revision_prompt,
    build_judge_prompt,
    ROLE_SYSTEM_PROMPTS,
    resolve_prompt_profile,
)
from multi_agent.prompts.registry import (
    PromptRegistry,
    reset_registry_cache,
)

# ---------------------------------------------------------------------------
# Token counting — optional tiktoken dependency
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _enc = tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except (ImportError, ModuleNotFoundError):
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4

# ---------------------------------------------------------------------------
# Shared mock context
# ---------------------------------------------------------------------------

_CONTEXT = "## Portfolio\n- Cash: $100,000\n- Universe: AAPL, MSFT\n\nMemo text."

_MY_PROPOSAL = (
    '{"allocation": {"AAPL": 0.6, "MSFT": 0.4}, '
    '"justification": "Growth", "confidence": 0.8, '
    '"risks_or_falsifiers": "Downturn", '
    '"claims": [{"claim_text": "AAPL up", "pearl_level": "L1", '
    '"variables": ["AAPL"], "assumptions": [], "confidence": 0.7}]}'
)

_ALL_PROPOSALS = [
    {"role": "macro", "proposal": _MY_PROPOSAL},
    {"role": "value", "proposal": _MY_PROPOSAL},
    {"role": "risk", "proposal": _MY_PROPOSAL},
    {"role": "technical", "proposal": _MY_PROPOSAL},
]

_CRITIQUES_RECEIVED = [
    {
        "from_role": "value",
        "objection": "Too concentrated in AAPL.",
        "falsifier": "AAPL earnings beat by >10%",
    },
]

_REVISIONS = [
    {"role": "macro", "action": _MY_PROPOSAL, "confidence": 0.8},
    {"role": "value", "action": _MY_PROPOSAL, "confidence": 0.7},
]

_ALL_CRITIQUES_TEXT = "- [VALUE]: Too concentrated in AAPL."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_user(phase: str, role: str = "macro") -> str:
    if phase == "propose":
        return build_proposal_user_prompt(_CONTEXT)
    elif phase == "critique":
        return build_critique_prompt(
            role=role, context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS, my_proposal=_MY_PROPOSAL,
        )
    elif phase == "revise":
        return build_revision_prompt(
            role=role, context=_CONTEXT,
            my_proposal=_MY_PROPOSAL, critiques_received=_CRITIQUES_RECEIVED,
        )
    elif phase == "judge":
        return build_judge_prompt(
            context=_CONTEXT, revisions=_REVISIONS,
            all_critiques_text=_ALL_CRITIQUES_TEXT,
        )
    raise ValueError(f"Unknown phase: {phase}")


def _render_system(role: str, phase: str, beta: float | None = None) -> str:
    config = {"prompt_profile": "default"}
    profile = resolve_prompt_profile(config, role, phase)
    registry = PromptRegistry()
    result = registry.build(
        role=role, phase=phase, beta=beta,
        user_prompt="", block_order=profile["system_blocks"],
    )
    return result.system_prompt


# ---------------------------------------------------------------------------
# System prompt token budgets
# ---------------------------------------------------------------------------

_SYSTEM_LIMITS = {
    "propose": 1500,
    "critique": 2000,
    "revise": 2000,
    "judge": 1200,
}


@pytest.fixture(autouse=True)
def _reset():
    reset_registry_cache()
    yield
    reset_registry_cache()


@pytest.mark.fast
class TestSystemPromptTokenBudgets:
    """System prompts must stay within token limits."""

    @pytest.mark.parametrize("phase,limit", list(_SYSTEM_LIMITS.items()))
    def test_system_prompt_within_budget(self, phase, limit):
        role = "macro" if phase != "judge" else "judge"
        beta = 0.5 if phase in ("critique", "revise") else None
        sys_prompt = _render_system(role, phase, beta=beta)
        tokens = _count_tokens(sys_prompt)
        assert tokens <= limit, (
            f"{phase} system prompt exceeds budget: {tokens} tokens > {limit} limit"
        )


# ---------------------------------------------------------------------------
# User prompt token budgets
# ---------------------------------------------------------------------------

_USER_LIMITS = {
    "propose": 8000,
    "critique": 6000,
    "revise": 7000,
    "judge": 8000,
}


@pytest.mark.fast
class TestUserPromptTokenBudgets:
    """User prompts must stay within token limits."""

    @pytest.mark.parametrize("phase,limit", list(_USER_LIMITS.items()))
    def test_user_prompt_within_budget(self, phase, limit):
        user_prompt = _render_user(phase)
        tokens = _count_tokens(user_prompt)
        assert tokens <= limit, (
            f"{phase} user prompt exceeds budget: {tokens} tokens > {limit} limit"
        )


# ---------------------------------------------------------------------------
# Per-role system prompt budget
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPerRoleSystemPromptBudget:
    """Each role's raw system prompt text must be <= 1500 tokens."""

    @pytest.mark.parametrize(
        "role", ["macro", "value", "risk", "technical", "sentiment", "devils_advocate"]
    )
    def test_role_system_prompt_within_budget(self, role):
        raw_text = ROLE_SYSTEM_PROMPTS[role]
        tokens = _count_tokens(raw_text)
        assert tokens <= 1500, (
            f"Role {role} system prompt exceeds 1500-token budget: {tokens} tokens"
        )


# ---------------------------------------------------------------------------
# Tone injection budget
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestToneInjectionBudget:
    """Tone text injected by PID must be < 1000 tokens."""

    @pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("phase", ["critique", "revise"])
    def test_tone_injection_within_budget(self, phase, beta):
        # Build with tone vs. without tone, measure the delta
        config = {"prompt_profile": "default"}
        profile = resolve_prompt_profile(config, "macro", phase)
        registry = PromptRegistry()

        result_with_tone = registry.build(
            role="macro", phase=phase, beta=beta,
            user_prompt="", block_order=profile["system_blocks"],
        )
        # Build without tone by excluding 'tone' from block order
        blocks_no_tone = [b for b in profile["system_blocks"] if b != "tone"]
        result_without_tone = registry.build(
            role="macro", phase=phase, beta=None,
            user_prompt="", block_order=blocks_no_tone,
        )

        tone_text = result_with_tone.system_prompt.replace(
            result_without_tone.system_prompt, ""
        )
        tone_tokens = _count_tokens(tone_text)
        assert tone_tokens < 1000, (
            f"Tone injection for {phase} beta={beta} exceeds 1000-token budget: "
            f"{tone_tokens} tokens"
        )
