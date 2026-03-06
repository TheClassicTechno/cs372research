"""L2 context propagation tests for debate prompts.

Validates that runtime data flows correctly into rendered prompts:
  - Sector constraints propagate into critique user prompt.
  - Enriched context variables propagate into proposal user prompt.
  - Other agents' proposals appear in critique user prompt.
  - Critique text appears in revision user prompt.
  - Empty optional context fields do not break rendering.
"""

from __future__ import annotations

import pytest

from multi_agent.prompts import (
    build_proposal_user_prompt,
    build_critique_prompt,
    build_revision_prompt,
    build_judge_prompt,
)

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

_CONTEXT = "## Portfolio\n- Cash: $100,000\n- Universe: AAPL, MSFT\n\nMemo text."

_MY_PROPOSAL = (
    '{"allocation": {"AAPL": 0.6, "MSFT": 0.4}, '
    '"justification": "Growth", "confidence": 0.8, '
    '"risks_or_falsifiers": "Downturn", '
    '"claims": []}'
)

_ALL_PROPOSALS = [
    {"role": "macro", "proposal": _MY_PROPOSAL},
    {"role": "value", "proposal": "Value agent proposal with DEEP_VALUE_SIGNAL."},
    {"role": "risk", "proposal": "Risk agent proposal with TAIL_RISK_ALERT."},
]

_CRITIQUES_RECEIVED = [
    {
        "from_role": "value",
        "objection": "AAPL is overweight given AAPL_VALUATION_CONCERN.",
        "falsifier": "AAPL earnings beat by >10%",
    },
    {
        "from_role": "risk",
        "objection": "Portfolio vol exceeds threshold: HIGH_VOL_MARKER.",
    },
]

_REVISIONS = [
    {"role": "macro", "action": _MY_PROPOSAL, "confidence": 0.8},
    {"role": "value", "action": _MY_PROPOSAL, "confidence": 0.7},
]


# ---------------------------------------------------------------------------
# 1. Sector constraints propagate into critique user prompt
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestSectorConstraintsPropagation:
    """sector_constraints text must appear in rendered critique/revise/judge prompts."""

    _SECTOR_TEXT = "SECTOR CONSTRAINT: Technology sector max 40%."

    def test_sector_constraints_in_critique(self):
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
            sector_constraints=self._SECTOR_TEXT,
        )
        assert self._SECTOR_TEXT in rendered, (
            "sector_constraints not found in critique user prompt"
        )

    def test_sector_constraints_in_proposal(self):
        rendered = build_proposal_user_prompt(
            _CONTEXT,
            sector_constraints=self._SECTOR_TEXT,
        )
        assert self._SECTOR_TEXT in rendered, (
            "sector_constraints not found in proposal user prompt"
        )

    def test_sector_constraints_in_revision(self):
        rendered = build_revision_prompt(
            role="macro",
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=_CRITIQUES_RECEIVED,
            sector_constraints=self._SECTOR_TEXT,
        )
        assert self._SECTOR_TEXT in rendered, (
            "sector_constraints not found in revision user prompt"
        )

    def test_sector_constraints_in_judge(self):
        rendered = build_judge_prompt(
            context=_CONTEXT,
            revisions=_REVISIONS,
            all_critiques_text="none",
            sector_constraints=self._SECTOR_TEXT,
        )
        assert self._SECTOR_TEXT in rendered, (
            "sector_constraints not found in judge user prompt"
        )


# ---------------------------------------------------------------------------
# 2. Enriched context variables propagate into proposal
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestEnrichedContextPropagation:
    """The full context string (enriched with portfolio/memo data) must
    appear verbatim in the proposal user prompt."""

    def test_context_in_proposal(self):
        enriched = (
            "## Portfolio\n- Cash: $250,000\n- Positions: AAPL 100 shares\n"
            "- Universe: AAPL, MSFT, GOOG\n\n"
            "Enriched memo with [L1-VIX] and [AAPL-RET60] data."
        )
        rendered = build_proposal_user_prompt(enriched)
        # Key fragments from the enriched context must appear
        assert "$250,000" in rendered
        assert "AAPL, MSFT, GOOG" in rendered
        assert "[L1-VIX]" in rendered
        assert "[AAPL-RET60]" in rendered

    def test_context_in_critique(self):
        enriched = "Enriched context with UNIQUE_CONTEXT_MARKER_12345."
        rendered = build_critique_prompt(
            role="macro",
            context=enriched,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
        )
        assert "UNIQUE_CONTEXT_MARKER_12345" in rendered


# ---------------------------------------------------------------------------
# 3. Agent data (other proposals) into critique user prompt
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAgentDataPropagation:
    """Other agents' proposals must appear in the critique user prompt."""

    def test_other_proposals_in_critique(self):
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
        )
        # The value and risk proposals (which are "others" for macro) should appear
        assert "DEEP_VALUE_SIGNAL" in rendered, (
            "Value agent's proposal text not found in macro's critique prompt"
        )
        assert "TAIL_RISK_ALERT" in rendered, (
            "Risk agent's proposal text not found in macro's critique prompt"
        )

    def test_own_proposal_in_critique(self):
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
        )
        # My own proposal should appear in the agent_data section
        assert "AAPL" in rendered
        assert "MSFT" in rendered

    def test_role_labels_in_critique(self):
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
        )
        # Other agents' role labels should appear (uppercased)
        assert "VALUE" in rendered
        assert "RISK" in rendered


# ---------------------------------------------------------------------------
# 4. Critique text into revision user prompt
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestCritiqueTextInRevision:
    """Critiques received must appear in the revision user prompt."""

    def test_critique_objections_in_revision(self):
        rendered = build_revision_prompt(
            role="macro",
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=_CRITIQUES_RECEIVED,
        )
        assert "AAPL_VALUATION_CONCERN" in rendered, (
            "Value agent's critique text not found in revision prompt"
        )
        assert "HIGH_VOL_MARKER" in rendered, (
            "Risk agent's critique text not found in revision prompt"
        )

    def test_critique_from_role_in_revision(self):
        rendered = build_revision_prompt(
            role="macro",
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=_CRITIQUES_RECEIVED,
        )
        assert "VALUE" in rendered
        assert "RISK" in rendered

    def test_falsifier_in_revision(self):
        rendered = build_revision_prompt(
            role="macro",
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=_CRITIQUES_RECEIVED,
        )
        # The first critique has a falsifier
        assert "AAPL earnings beat" in rendered


# ---------------------------------------------------------------------------
# 5. Empty optional context does not break rendering
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestEmptyOptionalContext:
    """Rendering must succeed with empty/minimal optional fields."""

    def test_empty_sector_constraints(self):
        rendered = build_proposal_user_prompt(_CONTEXT, sector_constraints="")
        assert len(rendered) > 0

    def test_empty_sector_constraints_critique(self):
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
            sector_constraints="",
        )
        assert len(rendered) > 0

    def test_empty_critiques_received(self):
        rendered = build_revision_prompt(
            role="macro",
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=[],
        )
        assert len(rendered) > 0
        # Should include the fallback text for no critiques
        assert "No critiques" in rendered or len(rendered) > 100

    def test_empty_disagreements(self):
        rendered = build_judge_prompt(
            context=_CONTEXT,
            revisions=_REVISIONS,
            all_critiques_text="",
            strongest_disagreements="",
        )
        assert len(rendered) > 0

    def test_minimal_context(self):
        rendered = build_proposal_user_prompt("Minimal.")
        assert len(rendered) > 0
        assert "Minimal." in rendered

    def test_empty_all_proposals_for_critique(self):
        """Critique with only own proposal (no others) should not crash."""
        rendered = build_critique_prompt(
            role="macro",
            context=_CONTEXT,
            all_proposals=[{"role": "macro", "proposal": _MY_PROPOSAL}],
            my_proposal=_MY_PROPOSAL,
        )
        assert len(rendered) > 0
