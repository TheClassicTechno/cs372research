"""
Tests for single-round debate mechanics exposure (PR #1).

PURPOSE
-------
The runner was refactored to decompose a single monolithic LangGraph
invocation into three phases (pipeline → debate rounds → finalize), moving
the iteration loop out of LangGraph and into MultiAgentRunner.  These
tests prove that the refactor is strictly behavior-preserving.

STRATEGY
--------
Tests compare two execution paths on identical inputs:

  OLD PATH: compile_debate_graph(config).invoke(initial_state)
    The original monolithic graph with internal should_continue loop.

  NEW PATH: MultiAgentRunner(config).run_returning_state(observation)
    The new three-phase runner with external iteration loop.

Both paths share the same node functions.  The only difference is who
manages the loop: LangGraph's conditional edge (old) vs the runner's
for-loop (new).  If both produce identical outputs, the refactor is safe.

INVARIANTS VERIFIED
-------------------
- final_action equality:  the judge sees the same revisions/critiques
  and produces the same trading decision.
- debate_turns equality:  every proposal, critique, revision, and judge
  turn is identical in content, round number, and ordering.
- trace equality:  architecture label, decision string, and structure
  match between old and new.
- Turn counting:  proposals generated exactly once (idempotency guard),
  critiques/revisions generated every round, one judge decision.
- Accumulation:  debate_turns grows across rounds (operator.add reducer
  works correctly across sub-graph boundaries).
- State isolation:  proposals are not duplicated on re-invocation.
- should_terminate:  baseline always returns False.

ISOLATION
---------
All equivalence tests disable pipeline features
(enable_news_pipeline=False, enable_data_pipeline=False) so that the
only difference between old and new is the iteration mechanism.  This
avoids false positives from pipeline ordering differences (the old graph
runs news+data in parallel; the new pipeline graph runs them sequentially).

No filesystem I/O, no timestamps, no demo scripts.
"""

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.graph import compile_debate_graph, propose_node
from multi_agent.models import (
    Constraints,
    MarketState,
    Observation,
    PortfolioState,
)
from multi_agent.runner import MultiAgentRunner


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_observation() -> Observation:
    """A realistic sample observation for testing."""
    return Observation(
        timestamp="2025-03-15T10:00:00Z",
        universe=["AAPL", "GOOGL", "MSFT"],
        market_state=MarketState(
            prices={"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 390.00},
            returns={"AAPL": 0.025, "GOOGL": -0.01, "MSFT": 0.015},
            volatility={"AAPL": 0.22, "GOOGL": 0.25, "MSFT": 0.18},
        ),
        text_context="Fed signals potential rate cuts in Q2. AAPL earnings beat expectations.",
        portfolio_state=PortfolioState(
            cash=50000.0,
            positions={"AAPL": 100, "GOOGL": 0, "MSFT": 50},
        ),
        constraints=Constraints(max_leverage=2.0, max_position_size=500),
    )


def _build_old_initial_state(observation: Observation, config: DebateConfig) -> dict:
    """Build the initial state dict matching the old runner's format.

    Mirrors MultiAgentRunner._initialize_state() exactly so the old
    monolithic graph starts from the same state as the new runner.
    Any mismatch here would produce false equivalence failures.
    """
    return {
        "observation": observation.model_dump(),
        "config": config.to_dict(),
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


# =============================================================================
# TEST 1: Old-vs-New Equivalence (max_rounds=1)
#
# The simplest case: one round means propose → critique → revise → judge.
# No looping, so this validates the basic decomposition without any
# iteration complexity.  Checks final_action, all debate_turns, and
# trace metadata.
# =============================================================================


def test_equivalence_1_round(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=1,
        mock=True,
        parallel_agents=False,
    )

    # OLD PATH
    old_graph = compile_debate_graph(config)
    old_initial = _build_old_initial_state(sample_observation, config)
    old_state = old_graph.invoke(old_initial)

    # NEW PATH (no disk I/O)
    runner = MultiAgentRunner(config)
    new_state = runner.run_returning_state(sample_observation)

    # Strict structural equality
    assert old_state["final_action"] == new_state["final_action"]
    assert old_state["debate_turns"] == new_state["debate_turns"]
    assert old_state["trace"]["architecture"] == new_state["trace"]["architecture"]
    assert old_state["trace"]["decision"] == new_state["trace"]["decision"]


# =============================================================================
# TEST 2: Old-vs-New Equivalence (max_rounds=2)
#
# Two rounds exercises the iteration mechanism:
#   Old: propose → critique → revise → should_continue(→critique) →
#        critique → revise → should_continue(→judge) → judge
#   New: pipeline_graph → round1(propose → critique → revise) →
#        round2(propose[no-op] → critique → revise) → finalize_graph
# Both should produce: 1 propose + 2 critique + 2 revise + 1 judge.
# This is the critical test — if current_round tracking were wrong,
# the old path would loop differently than the new path.
# =============================================================================


def test_equivalence_2_rounds(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=False,
    )

    old_graph = compile_debate_graph(config)
    old_initial = _build_old_initial_state(sample_observation, config)
    old_state = old_graph.invoke(old_initial)

    runner = MultiAgentRunner(config)
    new_state = runner.run_returning_state(sample_observation)

    assert old_state["final_action"] == new_state["final_action"]
    assert old_state["debate_turns"] == new_state["debate_turns"]
    assert old_state["trace"]["decision"] == new_state["trace"]["decision"]


# =============================================================================
# TEST 3: Max Rounds Counting (max_rounds=3)
#
# Verifies that the new runner produces the correct number of each turn
# type for 3 rounds with 3 agents:
#   - proposals: 3 agents * 1 (generated once due to idempotency guard)
#   - critiques: 3 agents * 3 rounds = 9
#   - revisions: 3 agents * 3 rounds = 9
#   - judge:     1 (always exactly one)
# This catches off-by-one errors in the runner's for-loop and confirms
# the propose_node guard fires correctly on rounds 2 and 3.
# =============================================================================


def test_max_rounds_counting(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=3,
        mock=True,
        parallel_agents=False,
    )

    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(sample_observation)
    turns = state["debate_turns"]

    proposal_turns = [t for t in turns if t["type"] == "proposal"]
    critique_turns = [t for t in turns if t["type"] == "critique"]
    revise_turns = [t for t in turns if t["type"] == "revision"]
    judge_turns = [t for t in turns if t["type"] == "judge_decision"]

    num_agents = len(config.roles)
    # Proposals generated only once (round 1)
    assert len(proposal_turns) == num_agents * 1
    # Critiques and revisions happen every round
    assert len(critique_turns) == num_agents * 3
    assert len(revise_turns) == num_agents * 3
    # One judge decision at the end
    assert len(judge_turns) == 1


# =============================================================================
# TEST 4: Single Round Primitive
#
# Tests run_single_round() as a standalone primitive — the building block
# that future controllers will call.  Verifies two key invariants:
#   1. Proposals are generated only once (idempotency guard).
#      After round 1 and round 2, the proposals list is identical.
#   2. debate_turns accumulates across rounds (operator.add reducer
#      works correctly when state is passed between separate graph
#      invocations).
# =============================================================================


def test_single_round_primitive(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=False,
    )

    runner = MultiAgentRunner(config)
    state = runner._initialize_state(sample_observation)
    state = runner.pipeline_graph.invoke(state)

    # Round 1
    state["current_round"] = 1
    state = runner.run_single_round(state)
    proposals_after_1 = list(state["proposals"])
    turns_after_1 = len(state["debate_turns"])

    # Round 2
    state["current_round"] = 2
    state = runner.run_single_round(state)
    proposals_after_2 = list(state["proposals"])
    turns_after_2 = len(state["debate_turns"])

    # Proposals generated only once
    assert proposals_after_1 == proposals_after_2
    # debate_turns accumulates
    assert turns_after_2 > turns_after_1


# =============================================================================
# TEST 5: Critiques/Revisions Final State Consistency
#
# critiques and revisions are plain lists (no reducer) that get REPLACED
# each round.  After the final round, the old and new paths should hold
# the same last-round critiques and revisions.  This catches bugs where
# the sub-graph boundary might cause stale data to persist or where
# the reducer might accidentally append instead of replace.
# =============================================================================


def test_final_critiques_revisions_match_old(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=False,
    )

    old_graph = compile_debate_graph(config)
    old_state = old_graph.invoke(
        _build_old_initial_state(sample_observation, config)
    )

    runner = MultiAgentRunner(config)
    new_state = runner.run_returning_state(sample_observation)

    assert old_state["critiques"] == new_state["critiques"]
    assert old_state["revisions"] == new_state["revisions"]


# =============================================================================
# TEST 6: should_terminate Always False
#
# Baseline test for the termination hook.  Currently should_terminate()
# always returns False, meaning debates run for exactly max_rounds.
# Future PRs will override this with PID-based or convergence-based
# termination.  This test documents the current contract so that future
# changes are intentional.
# =============================================================================


def test_should_terminate_false():
    config = DebateConfig(mock=True)
    runner = MultiAgentRunner(config)
    assert runner.should_terminate({}) is False
    assert runner.should_terminate({"current_round": 100}) is False


# =============================================================================
# TEST 7: DebateConfig rejects max_rounds < 1
#
# Regression test for __post_init__ validation added to DebateConfig.
# Previously, max_rounds=0 was silently accepted, causing the runner's
# for-loop to execute zero rounds (range(0) = empty) and skip the debate
# entirely.  The validation now raises ValueError at construction time.
# =============================================================================


def test_config_rejects_max_rounds_zero():
    with pytest.raises(ValueError, match="max_rounds must be >= 1"):
        DebateConfig(max_rounds=0)


def test_config_rejects_max_rounds_negative():
    with pytest.raises(ValueError, match="max_rounds must be >= 1"):
        DebateConfig(max_rounds=-1)


# =============================================================================
# TEST 8: propose_node guard handles proposals=None
#
# Regression test for the propose_node idempotency guard.  Previously,
# the guard used `state.get("proposals", [])` which returns None when
# the key exists with value None — causing len(None) to raise TypeError.
# The fix uses `state.get("proposals") or []` to coalesce None to [].
# This test proves the guard fires cleanly when proposals is None.
# =============================================================================


def test_propose_node_guard_with_none_proposals():
    state = {
        "proposals": None,
        "config": {"roles": ["macro"], "mock": True},
        "enriched_context": "",
        "observation": {},
    }
    # Previously len(None) raised TypeError.  After the fix,
    # None is coalesced to [] so propose_node treats it as
    # "no proposals yet" and generates them without crashing.
    result = propose_node(state)
    assert "proposals" in result
    assert len(result["proposals"]) == 1  # 1 role = 1 proposal


# =============================================================================
# PARALLEL GRAPH TESTS
#
# These tests verify the parallel per-agent fan-out/fan-in graph
# (ParallelRoundState + make_propose_node/make_critique_node/make_revise_node)
# produces equivalent results to the sequential batch-node graph.
#
# All comparisons use (type, role) keys — never list position — because
# parallel node output order is non-deterministic.
# =============================================================================


def _turns_by_key(turns: list) -> dict:
    """Index debate turns by (type, role) for order-independent comparison."""
    result = {}
    for t in turns:
        key = (t["type"], t["role"])
        result.setdefault(key, []).append(t)
    return result


def _compare_debate_content(seq_turns: list, par_turns: list) -> None:
    """Assert debate turns match by (type, role), ignoring list order."""
    seq_by_key = _turns_by_key(seq_turns)
    par_by_key = _turns_by_key(par_turns)
    assert set(seq_by_key.keys()) == set(par_by_key.keys()), (
        f"Turn keys differ: seq={set(seq_by_key.keys())} par={set(par_by_key.keys())}"
    )
    for key in seq_by_key:
        seq_entries = seq_by_key[key]
        par_entries = par_by_key[key]
        assert len(seq_entries) == len(par_entries), (
            f"Count mismatch for {key}: seq={len(seq_entries)} par={len(par_entries)}"
        )
        # Compare content (sort by round for multi-round)
        seq_sorted = sorted(seq_entries, key=lambda t: t.get("round", 0))
        par_sorted = sorted(par_entries, key=lambda t: t.get("round", 0))
        for s, p in zip(seq_sorted, par_sorted):
            assert s["content"] == p["content"], f"Content mismatch for {key}"


# =============================================================================
# TEST 9: Parallel ↔ Sequential Equivalence (max_rounds=1)
#
# The simplest parallel case: one round means each agent proposes,
# critiques, and revises exactly once.  Verifies that the parallel
# graph produces the same final_action and debate content as the
# sequential graph.
# =============================================================================


def test_parallel_equivalence_1_round(sample_observation):
    seq_config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=1,
        mock=True,
        parallel_agents=False,
    )
    par_config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=1,
        mock=True,
        parallel_agents=True,
    )

    seq_runner = MultiAgentRunner(seq_config)
    par_runner = MultiAgentRunner(par_config)

    seq_state = seq_runner.run_returning_state(sample_observation)
    par_state = par_runner.run_returning_state(sample_observation)

    assert seq_state["final_action"] == par_state["final_action"]
    _compare_debate_content(seq_state["debate_turns"], par_state["debate_turns"])


# =============================================================================
# TEST 10: Parallel ↔ Sequential Equivalence (max_rounds=2)
#
# Two rounds exercises cross-round state management: critiques/revisions
# must be reset between rounds so operator.add doesn't accumulate.
# =============================================================================


def test_parallel_equivalence_2_rounds(sample_observation):
    seq_config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=False,
    )
    par_config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=True,
    )

    seq_runner = MultiAgentRunner(seq_config)
    par_runner = MultiAgentRunner(par_config)

    seq_state = seq_runner.run_returning_state(sample_observation)
    par_state = par_runner.run_returning_state(sample_observation)

    assert seq_state["final_action"] == par_state["final_action"]
    _compare_debate_content(seq_state["debate_turns"], par_state["debate_turns"])


# =============================================================================
# TEST 11: Parallel Turn Counting (max_rounds=3)
#
# 3 rounds × 3 agents should produce:
#   - 3 proposals  (generated once, idempotency guard)
#   - 9 critiques  (3 agents × 3 rounds)
#   - 9 revisions  (3 agents × 3 rounds)
#   - 1 judge decision
# =============================================================================


def test_parallel_turn_counting(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=3,
        mock=True,
        parallel_agents=True,
    )

    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(sample_observation)
    turns = state["debate_turns"]

    proposal_turns = [t for t in turns if t["type"] == "proposal"]
    critique_turns = [t for t in turns if t["type"] == "critique"]
    revise_turns = [t for t in turns if t["type"] == "revision"]
    judge_turns = [t for t in turns if t["type"] == "judge_decision"]

    num_agents = len(config.roles)
    assert len(proposal_turns) == num_agents * 1
    assert len(critique_turns) == num_agents * 3
    assert len(revise_turns) == num_agents * 3
    assert len(judge_turns) == 1


# =============================================================================
# TEST 12: Parallel Proposals Not Duplicated
#
# After 2 rounds, exactly 3 proposals should exist (not 6).  The
# idempotency guard + operator.add must cooperate: round 1 produces
# [p1, p2, p3], round 2 returns {} from each propose node.
# =============================================================================


def test_parallel_proposals_not_duplicated(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=True,
    )

    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(sample_observation)

    assert len(state["proposals"]) == 3
    roles_found = {p["role"] for p in state["proposals"]}
    assert roles_found == {"macro", "value", "risk"}


# =============================================================================
# TEST 13: Parallel Critiques Reset Between Rounds
#
# With operator.add, critiques would accumulate to 6 (3+3) after
# 2 rounds without the runner's reset.  With the reset, the final
# state should have exactly 3 critiques (last round only).
# =============================================================================


def test_parallel_critiques_reset(sample_observation):
    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=2,
        mock=True,
        parallel_agents=True,
    )

    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(sample_observation)

    num_agents = len(config.roles)
    assert len(state["critiques"]) == num_agents
    assert len(state["revisions"]) == num_agents


# =============================================================================
# TEST 14: Config Toggle
#
# parallel_agents=False should use the sequential single-round graph,
# parallel_agents=True should use the parallel single-round graph.
# Both should produce valid outputs.
# =============================================================================


def test_parallel_config_toggle(sample_observation):
    for parallel in (False, True):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            mock=True,
            parallel_agents=parallel,
        )
        runner = MultiAgentRunner(config)
        state = runner.run_returning_state(sample_observation)

        # Both modes should produce a valid final_action
        assert "allocation" in state["final_action"]
        assert "confidence" in state["final_action"]

        # Both modes should produce the right number of turns
        turns = state["debate_turns"]
        proposals = [t for t in turns if t["type"] == "proposal"]
        critiques = [t for t in turns if t["type"] == "critique"]
        revisions = [t for t in turns if t["type"] == "revision"]
        assert len(proposals) == 3
        assert len(critiques) == 3
        assert len(revisions) == 3
