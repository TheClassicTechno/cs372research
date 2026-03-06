"""L3 integration tests: sequential vs parallel agent equivalence.

Runs the same mock debate in both sequential (parallel_agents=False) and
parallel (parallel_agents=True) modes, then verifies structural
equivalence -- same asset set, similar weights, same turn counts,
both summing to ~1.0.

NOT exact equality: parallel fan-out merges via operator.add in
thread-completion order, so per-ticker weights can differ slightly.
"""

from __future__ import annotations

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.models import MarketState, Observation, PortfolioState
from multi_agent.runner import MultiAgentRunner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA"]
ROLES = [AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK]


def _make_observation() -> Observation:
    return Observation(
        universe=TICKERS,
        timestamp="2025-01-15T00:00:00Z",
        market_state=MarketState(
            prices={"AAPL": 185.0, "MSFT": 390.0, "NVDA": 140.0},
        ),
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
        text_context="Market summary text for parallel test.",
    )


def _run_debate(parallel: bool) -> dict:
    config = DebateConfig(
        mock=True,
        roles=list(ROLES),
        max_rounds=1,
        parallel_agents=parallel,
        console_display=False,
        trace_dir="/tmp/test_traces_parallel_eq",
    )
    runner = MultiAgentRunner(config)
    return runner.run_returning_state(_make_observation())


@pytest.fixture(scope="module")
def sequential_state():
    return _run_debate(parallel=False)


@pytest.fixture(scope="module")
def parallel_state():
    return _run_debate(parallel=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestParallelEquivalence:
    """Structural equivalence between sequential and parallel debate runs."""

    def test_same_asset_set_in_final_allocation(self, sequential_state, parallel_state):
        """Both modes produce allocations for the same tickers."""
        seq_alloc = sequential_state.get("final_action", {}).get("allocation", {})
        par_alloc = parallel_state.get("final_action", {}).get("allocation", {})
        assert set(seq_alloc.keys()) == set(par_alloc.keys()), (
            f"Asset sets differ: seq={set(seq_alloc.keys())} vs par={set(par_alloc.keys())}"
        )

    def test_per_ticker_weights_within_tolerance(self, sequential_state, parallel_state):
        """Per-ticker weights differ by less than 0.15."""
        seq_alloc = sequential_state.get("final_action", {}).get("allocation", {})
        par_alloc = parallel_state.get("final_action", {}).get("allocation", {})

        all_tickers = set(seq_alloc.keys()) | set(par_alloc.keys())
        for ticker in all_tickers:
            seq_w = seq_alloc.get(ticker, 0.0)
            par_w = par_alloc.get(ticker, 0.0)
            diff = abs(seq_w - par_w)
            assert diff < 0.15, (
                f"Weight diff for {ticker}: seq={seq_w:.4f} par={par_w:.4f} "
                f"diff={diff:.4f} exceeds 0.15"
            )

    def test_both_sum_to_one(self, sequential_state, parallel_state):
        """Both modes produce allocations summing to approximately 1.0."""
        for label, state in [("sequential", sequential_state), ("parallel", parallel_state)]:
            alloc = state.get("final_action", {}).get("allocation", {})
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"{label} allocation sums to {total:.4f}, expected ~1.0"
            )

    def test_same_turn_counts(self, sequential_state, parallel_state):
        """Both modes produce the same number of debate turns."""
        seq_turns = len(sequential_state.get("debate_turns", []))
        par_turns = len(parallel_state.get("debate_turns", []))
        assert seq_turns == par_turns, (
            f"Turn count mismatch: seq={seq_turns} par={par_turns}"
        )

    def test_same_proposal_roles(self, sequential_state, parallel_state):
        """Both modes produce proposals from the same set of roles."""
        seq_roles = sorted(p["role"] for p in sequential_state.get("proposals", []))
        par_roles = sorted(p["role"] for p in parallel_state.get("proposals", []))
        assert seq_roles == par_roles, (
            f"Proposal roles differ: seq={seq_roles} par={par_roles}"
        )

    def test_same_revision_roles(self, sequential_state, parallel_state):
        """Both modes produce revisions from the same set of roles."""
        seq_roles = sorted(r["role"] for r in sequential_state.get("revisions", []))
        par_roles = sorted(r["role"] for r in parallel_state.get("revisions", []))
        assert seq_roles == par_roles, (
            f"Revision roles differ: seq={seq_roles} par={par_roles}"
        )
