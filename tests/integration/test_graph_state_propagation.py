"""L3 integration tests: graph state propagation through mock debate.

Runs full debate pipelines in mock mode (no API calls) and inspects
the raw LangGraph state dict to verify that proposals, critiques,
revisions, debate_turns, and final allocations propagate correctly.

All tests use DebateConfig(mock=True) + MultiAgentRunner.run_returning_state().
"""

from __future__ import annotations

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.models import MarketState, Observation, PortfolioState
from multi_agent.runner import MultiAgentRunner


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA"]

ROLES_3 = [AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK]


def _make_observation() -> Observation:
    """Minimal observation for mock debate runs."""
    return Observation(
        universe=TICKERS,
        timestamp="2025-01-15T00:00:00Z",
        market_state=MarketState(
            prices={"AAPL": 185.0, "MSFT": 390.0, "NVDA": 140.0},
        ),
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
        text_context="Market summary text.",
    )


def _run_mock_debate(max_rounds: int = 1, roles: list | None = None, **overrides) -> dict:
    """Run a mock debate and return the raw state dict."""
    config = DebateConfig(
        mock=True,
        roles=roles or list(ROLES_3),
        max_rounds=max_rounds,
        parallel_agents=False,
        console_display=False,
        trace_dir="/tmp/test_traces_state_prop",
        **overrides,
    )
    runner = MultiAgentRunner(config)
    return runner.run_returning_state(_make_observation())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProposalPersistence:
    """Proposals must persist in state after the propose phase."""

    def test_proposals_not_empty(self):
        state = _run_mock_debate()
        proposals = state.get("proposals", [])
        assert len(proposals) > 0, "proposals should not be empty after debate"

    def test_proposals_have_role_field(self):
        state = _run_mock_debate()
        for p in state["proposals"]:
            assert "role" in p, "Each proposal must have a 'role' field"

    def test_proposals_have_action_dict(self):
        state = _run_mock_debate()
        for p in state["proposals"]:
            assert "action_dict" in p, "Each proposal must have 'action_dict'"
            alloc = p["action_dict"].get("allocation", {})
            assert len(alloc) > 0, "Proposal allocation must not be empty"


@pytest.mark.integration
class TestCritiquesReferenceProposals:
    """Critiques must have content referencing proposals."""

    def test_critiques_not_empty(self):
        state = _run_mock_debate()
        critiques = state.get("critiques", [])
        assert len(critiques) > 0, "critiques should not be empty"

    def test_critiques_have_content(self):
        state = _run_mock_debate()
        for c in state["critiques"]:
            # Each critique should have either 'critiques' list or 'self_critique'
            has_content = (
                len(c.get("critiques", [])) > 0
                or len(c.get("self_critique", "")) > 0
            )
            assert has_content, f"Critique from {c.get('role')} has no content"


@pytest.mark.integration
class TestRevisionsReferenceCritiques:
    """Revisions must exist and reference prior critiques."""

    def test_revisions_not_empty(self):
        state = _run_mock_debate()
        revisions = state.get("revisions", [])
        assert len(revisions) > 0, "revisions should not be empty"

    def test_revisions_have_action_dict(self):
        state = _run_mock_debate()
        for r in state["revisions"]:
            assert "action_dict" in r, "Each revision must have 'action_dict'"


@pytest.mark.integration
class TestAllocationsChangeAfterCritique:
    """Revisions should differ from proposals (mock reduces confidence)."""

    def test_revision_confidence_differs_from_proposal(self):
        state = _run_mock_debate()
        for role in ROLES_3:
            proposal = next(
                (p for p in state["proposals"] if p["role"] == role), None
            )
            revision = next(
                (r for r in state["revisions"] if r["role"] == role), None
            )
            if proposal and revision:
                p_conf = proposal["action_dict"].get("confidence", 0.5)
                r_conf = revision["action_dict"].get("confidence", 0.5)
                # Mock revisions reduce confidence by 0.1
                assert p_conf != r_conf or revision["action_dict"].get("revision_notes"), (
                    f"Revision for {role} should differ from proposal"
                )


@pytest.mark.integration
class TestDebateTurnsAccumulate:
    """debate_turns list must grow with each phase."""

    def test_turns_accumulate_single_round(self):
        state = _run_mock_debate(max_rounds=1)
        turns = state.get("debate_turns", [])
        # At minimum: proposals + critiques + revisions + judge = 4 phases
        assert len(turns) >= 4, (
            f"Expected at least 4 debate turns (1 round), got {len(turns)}"
        )

    def test_turns_increase_with_rounds(self):
        state_1 = _run_mock_debate(max_rounds=1)
        state_2 = _run_mock_debate(max_rounds=2)
        turns_1 = len(state_1.get("debate_turns", []))
        turns_2 = len(state_2.get("debate_turns", []))
        assert turns_2 > turns_1, (
            f"2 rounds ({turns_2} turns) should have more turns than 1 round ({turns_1})"
        )


@pytest.mark.integration
class TestFinalAllocationSumsToOne:
    """Final allocation weights must sum to approximately 1.0."""

    def test_final_allocation_sums_to_one(self):
        state = _run_mock_debate()
        final_action = state.get("final_action", {})
        allocation = final_action.get("allocation", {})
        assert len(allocation) > 0, "Final allocation must not be empty"
        total = sum(allocation.values())
        assert abs(total - 1.0) < 0.01, (
            f"Final allocation sum {total:.4f} should be ~1.0"
        )

    def test_final_allocation_values_in_range(self):
        state = _run_mock_debate()
        final_action = state.get("final_action", {})
        allocation = final_action.get("allocation", {})
        for ticker, weight in allocation.items():
            assert 0.0 <= weight <= 1.0, (
                f"Weight for {ticker} = {weight} is outside [0, 1]"
            )


@pytest.mark.integration
class TestRoundCounterIncrements:
    """current_round should reflect the number of completed rounds."""

    def test_round_counter_after_single_round(self):
        state = _run_mock_debate(max_rounds=1)
        # After 1 round, current_round should be at least 1
        assert state.get("current_round", 0) >= 1

    def test_round_counter_after_multi_round(self):
        state = _run_mock_debate(max_rounds=3)
        assert state.get("current_round", 0) >= 1


@pytest.mark.integration
class TestConvergence:
    """3 agents, 3 rounds, mock mode -- disagreement should not diverge."""

    def test_disagreement_stays_bounded(self):
        state = _run_mock_debate(max_rounds=3, roles=list(ROLES_3))
        revisions = state.get("revisions", [])

        # Collect all final revision allocations
        allocs = []
        for r in revisions:
            a = r.get("action_dict", {}).get("allocation", {})
            if a:
                allocs.append(a)

        if len(allocs) < 2:
            pytest.skip("Not enough revisions to measure disagreement")

        # Compute max pairwise difference per ticker
        tickers = set()
        for a in allocs:
            tickers.update(a.keys())

        max_diff = 0.0
        for t in tickers:
            weights = [a.get(t, 0.0) for a in allocs]
            diff = max(weights) - min(weights)
            max_diff = max(max_diff, diff)

        # In mock mode, proposals are equal-weight and revisions only
        # slightly adjust confidence -- allocations should be close
        assert max_diff < 0.5, (
            f"Max pairwise allocation diff {max_diff:.4f} exceeds 0.5 "
            f"-- agents are diverging"
        )
