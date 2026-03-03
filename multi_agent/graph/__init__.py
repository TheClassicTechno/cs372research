"""
LangGraph-based multi-agent debate orchestrator.

This package defines ALL node functions and graph builders for the debate
system.  It serves two execution paths:

  1. MONOLITHIC GRAPH (original, preserved for backward compatibility):
       build_debate_graph / compile_debate_graph
       Runs the full pipeline → propose → critique/revise loop → judge → trace
       as a single LangGraph invocation.  The internal should_continue edge
       decides when to stop looping.

  2. DECOMPOSED SUB-GRAPHS (new, used by MultiAgentRunner):
       build_pipeline_graph   — build_context → END
       build_single_round_graph — propose → critique → revise → END
       build_finalize_graph   — judge → build_trace → END
       The runner calls each sub-graph separately, owning the iteration
       loop itself so external controllers (PID, agreeableness tuners)
       can intervene between rounds.

Sub-modules:
  state       — DebateState, ParallelRoundState TypedDicts
  llm         — _call_llm, _parse_json, _log_prompt, semaphore
  allocation  — normalize_allocation, _enforce_max_weight
  mocks       — _mock_proposal, _mock_critique, _mock_revision, _mock_judge
  display     — verbose display helpers
  nodes       — all node functions, factories, judge, trace builders
  builders    — all graph build_*/compile_* functions

All symbols are re-exported here so that existing imports like
``from multi_agent.graph import X`` continue to work unchanged.
"""

# --- state ---
from .state import DebateState, ParallelRoundState

# --- llm ---
from .llm import (
    _call_llm,
    _compact_user_prompt,
    _log_prompt,
    _LLM_SEMAPHORE,
    _parse_json,
    logger,
    prompt_logger,
)

# --- allocation ---
from .allocation import normalize_allocation, _enforce_max_weight

# --- mocks ---
from .mocks import _mock_proposal, _mock_critique, _mock_revision, _mock_judge

# --- display ---
from .display import (
    _ROLE_LABELS,
    _print_allocation,
    _print_critique_summary,
    _verbose_proposal,
    _verbose_critique,
    _verbose_revision,
    _verbose_judge,
)

# --- nodes ---
from .nodes import (
    build_context_node,
    propose_node,
    critique_node,
    revise_node,
    _sync_noop,
    make_propose_node,
    make_critique_node,
    make_revise_node,
    should_continue,
    judge_node,
    _get_vote_direction,
    _get_median_size,
    aggregate_proposals_node,
    build_mv_trace_node,
    build_trace_node,
)

# --- builders ---
from .builders import (
    build_debate_graph,
    compile_debate_graph,
    build_parallel_single_round_graph,
    compile_parallel_single_round_graph,
    build_propose_graph,
    compile_propose_graph,
    build_critique_graph,
    compile_critique_graph,
    build_revise_graph,
    compile_revise_graph,
    build_parallel_propose_graph,
    compile_parallel_propose_graph,
    build_parallel_critique_graph,
    compile_parallel_critique_graph,
    build_parallel_revise_graph,
    compile_parallel_revise_graph,
    build_pipeline_graph,
    compile_pipeline_graph,
    build_single_round_graph,
    compile_single_round_graph,
    build_finalize_graph,
    compile_finalize_graph,
    build_majority_vote_graph,
    compile_majority_vote_graph,
)

__all__ = [
    # state
    "DebateState",
    "ParallelRoundState",
    # llm
    "_call_llm",
    "_compact_user_prompt",
    "_log_prompt",
    "_LLM_SEMAPHORE",
    "_parse_json",
    "logger",
    "prompt_logger",
    # allocation
    "normalize_allocation",
    "_enforce_max_weight",
    # mocks
    "_mock_proposal",
    "_mock_critique",
    "_mock_revision",
    "_mock_judge",
    # display
    "_ROLE_LABELS",
    "_print_allocation",
    "_print_critique_summary",
    "_verbose_proposal",
    "_verbose_critique",
    "_verbose_revision",
    "_verbose_judge",
    # nodes
    "build_context_node",
    "propose_node",
    "critique_node",
    "revise_node",
    "_sync_noop",
    "make_propose_node",
    "make_critique_node",
    "make_revise_node",
    "should_continue",
    "judge_node",
    "_get_vote_direction",
    "_get_median_size",
    "aggregate_proposals_node",
    "build_mv_trace_node",
    "build_trace_node",
    # builders
    "build_debate_graph",
    "compile_debate_graph",
    "build_parallel_single_round_graph",
    "compile_parallel_single_round_graph",
    "build_propose_graph",
    "compile_propose_graph",
    "build_critique_graph",
    "compile_critique_graph",
    "build_revise_graph",
    "compile_revise_graph",
    "build_parallel_propose_graph",
    "compile_parallel_propose_graph",
    "build_parallel_critique_graph",
    "compile_parallel_critique_graph",
    "build_parallel_revise_graph",
    "compile_parallel_revise_graph",
    "build_pipeline_graph",
    "compile_pipeline_graph",
    "build_single_round_graph",
    "compile_single_round_graph",
    "build_finalize_graph",
    "compile_finalize_graph",
    "build_majority_vote_graph",
    "compile_majority_vote_graph",
]
