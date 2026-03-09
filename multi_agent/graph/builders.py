"""Graph construction functions for the LangGraph debate orchestrator.

All build_* and compile_* functions that assemble StateGraph objects
from node functions.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..config import DebateConfig
from .nodes import (
    _sync_noop,
    aggregate_proposals_node,
    build_context_node,
    build_mv_trace_node,
    build_trace_node,
    critique_node,
    judge_node,
    make_critique_node,
    make_propose_node,
    make_revise_node,
    propose_node,
    revise_node,
    should_continue,
)
from .state import DebateState, ParallelRoundState


# =============================================================================
# PARALLEL SINGLE-ROUND GRAPH CONSTRUCTION
# =============================================================================


def build_parallel_single_round_graph(config: DebateConfig) -> StateGraph:
    """Parallel single round: per-agent fan-out → sync → fan-out → sync → fan-out → END.

    Target topology for 3 agents (macro, value, risk):

        [START] ┬→ [propose_macro]  ┐
                ├→ [propose_value]  ├→ [sync_propose] ┬→ [critique_macro]  ┐
                └→ [propose_risk]   ┘                  ├→ [critique_value]  ├→ [sync_critique] ┬→ [revise_macro]  ┐
                                                       └→ [critique_risk]   ┘                  ├→ [revise_value]  ├→ [END]
                                                                                               └→ [revise_risk]   ┘

    Uses ParallelRoundState with operator.add reducers on proposals,
    critiques, and revisions so that parallel nodes can each contribute
    single-element lists that get merged at the sync barriers.

    The runner resets critiques/revisions to [] between rounds because
    operator.add would otherwise accumulate across rounds.
    """
    roles = list(config.roles)
    graph = StateGraph(ParallelRoundState)

    # Sync barriers (no-op pass-throughs for fan-in)
    graph.add_node("sync_propose", _sync_noop)
    graph.add_node("sync_critique", _sync_noop)

    # Per-agent nodes + edges
    for role in roles:
        graph.add_node(f"propose_{role}", make_propose_node(role))
        graph.add_edge(START, f"propose_{role}")
        graph.add_edge(f"propose_{role}", "sync_propose")

        graph.add_node(f"critique_{role}", make_critique_node(role))
        graph.add_edge("sync_propose", f"critique_{role}")
        graph.add_edge(f"critique_{role}", "sync_critique")

        graph.add_node(f"revise_{role}", make_revise_node(role))
        graph.add_edge("sync_critique", f"revise_{role}")
        graph.add_edge(f"revise_{role}", END)

    return graph


def compile_parallel_single_round_graph(config: DebateConfig):
    """Build and compile the parallel single-round graph, ready for invocation."""
    return build_parallel_single_round_graph(config).compile()


# =============================================================================
# PER-PHASE SUB-GRAPHS (used by PID controller for per-phase intervention)
# =============================================================================
#
# The PID controller needs to set beta (tone)
# BETWEEN propose, critique, and revise phases within a single round.
# The existing single-round graphs run all three atomically, so we
# provide per-phase graphs that the runner can invoke individually.
#
# Sequential (for parallel_agents=False):
#   build_propose_graph   — START → propose → END
#   build_critique_graph  — START → critique → END
#   build_revise_graph    — START → revise → END
#
# Parallel (for parallel_agents=True):
#   build_parallel_propose_graph   — START → propose_* → END
#   build_parallel_critique_graph  — START → critique_* → END
#   build_parallel_revise_graph    — START → revise_* → END
# =============================================================================


def build_propose_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: propose only.

    START → propose → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("propose", propose_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", END)
    return graph


def build_critique_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: critique only.

    START → critique → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("critique", critique_node)
    graph.add_edge(START, "critique")
    graph.add_edge("critique", END)
    return graph


def build_revise_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: revise only.

    START → revise → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "revise")
    graph.add_edge("revise", END)
    return graph


def build_parallel_propose_graph(config: DebateConfig) -> StateGraph:
    """Parallel propose: per-agent fan-out → END.

    START ┬→ propose_macro ┐
          ├→ propose_value ├→ END
          └→ propose_risk  ┘

    Uses ParallelRoundState for operator.add on proposals.
    """
    roles = list(config.roles)
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"propose_{role}", make_propose_node(role))
        graph.add_edge(START, f"propose_{role}")
        graph.add_edge(f"propose_{role}", END)
    return graph


def build_parallel_critique_graph(config: DebateConfig) -> StateGraph:
    """Parallel critique: per-agent fan-out → END.

    Uses ParallelRoundState for operator.add on critiques.
    """
    roles = list(config.roles)
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"critique_{role}", make_critique_node(role))
        graph.add_edge(START, f"critique_{role}")
        graph.add_edge(f"critique_{role}", END)
    return graph


def build_parallel_revise_graph(config: DebateConfig) -> StateGraph:
    """Parallel revise: per-agent fan-out → END.

    Uses ParallelRoundState for operator.add on revisions.
    """
    roles = list(config.roles)
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"revise_{role}", make_revise_node(role))
        graph.add_edge(START, f"revise_{role}")
        graph.add_edge(f"revise_{role}", END)
    return graph


def compile_propose_graph(config: DebateConfig):
    """Build and compile the sequential propose graph."""
    return build_propose_graph(config).compile()


def compile_critique_graph(config: DebateConfig):
    """Build and compile the sequential critique graph."""
    return build_critique_graph(config).compile()


def compile_revise_graph(config: DebateConfig):
    """Build and compile the sequential revise graph."""
    return build_revise_graph(config).compile()


def compile_parallel_propose_graph(config: DebateConfig):
    """Build and compile the parallel propose graph."""
    return build_parallel_propose_graph(config).compile()


def compile_parallel_critique_graph(config: DebateConfig):
    """Build and compile the parallel critique graph."""
    return build_parallel_critique_graph(config).compile()


def compile_parallel_revise_graph(config: DebateConfig):
    """Build and compile the parallel revise graph."""
    return build_parallel_revise_graph(config).compile()


# =============================================================================
# MONOLITHIC GRAPH CONSTRUCTION
# =============================================================================


def build_debate_graph(config: DebateConfig) -> StateGraph:
    """
    Build the LangGraph debate graph based on configuration.

    The graph structure adapts to config:
      - Pipeline nodes are included only if enabled
      - The critique->revise loop runs for config.max_rounds iterations
      - Adversarial agent is injected into the roles list
    """
    graph = StateGraph(DebateState)

    # --- Add all nodes ---
    graph.add_node("build_context", build_context_node)
    graph.add_node("propose", propose_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_node("judge", judge_node)
    graph.add_node("build_trace", build_trace_node)

    # --- Edges: START -> build_context (no pipeline) ---
    graph.add_edge(START, "build_context")

    # --- Edges: debate flow ---
    graph.add_edge("build_context", "propose")
    graph.add_edge("propose", "critique")
    graph.add_edge("critique", "revise")

    # Conditional: more rounds or go to judge
    graph.add_conditional_edges(
        "revise",
        should_continue,
        {"critique": "critique", "judge": "judge"},
    )

    graph.add_edge("judge", "build_trace")
    graph.add_edge("build_trace", END)

    return graph


def compile_debate_graph(config: DebateConfig):
    """Build and compile the debate graph, ready for invocation."""
    graph = build_debate_graph(config)
    return graph.compile()


# =============================================================================
# SINGLE-ROUND SUB-GRAPH CONSTRUCTION
# =============================================================================
#
# These three builders decompose the monolithic debate graph into phases
# that the runner can invoke independently.  This is the key architectural
# change: by moving the iteration loop out of LangGraph and into the
# runner, we create a seam where external controllers can observe and
# modify state between debate rounds.
#
# All three use StateGraph(DebateState) so that LangGraph's reducers
# (Annotated[list, operator.add] for debate_turns) are properly applied.
# The runner never does manual state.update(node(state)) — every state
# transition goes through graph.invoke().
#
# Equivalence with the monolithic graph is verified by tests in
# test_round_exposure.py which run both paths on identical inputs and
# assert identical outputs (final_action, debate_turns, trace, etc.).
# =============================================================================


def build_pipeline_graph(config: DebateConfig) -> StateGraph:
    """Pipeline: build_context → END.

    In allocation mode, there are no preprocessing pipeline stages.
    The build_context node constructs the enriched context directly
    from the memo observation.
    """
    graph = StateGraph(DebateState)
    graph.add_node("build_context", build_context_node)
    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", END)
    return graph


def build_single_round_graph(config: DebateConfig) -> StateGraph:
    """One debate round: propose → critique → revise → END.

    The runner calls this once per round.  propose_node is idempotent:
    on round 1 it generates proposals; on rounds 2+ it detects existing
    proposals and returns {} (no-op), so critique and revise operate on
    the latest revisions.

    State flow per round:
      - debate_turns: appended via operator.add reducer (accumulates)
      - proposals: set once in round 1, preserved thereafter (plain list)
      - critiques/revisions: replaced each round (plain list, no reducer)
      - current_round: set by runner before invocation, updated by
        revise_node to current_round+1 (kept for monolithic compat)
    """
    graph = StateGraph(DebateState)
    graph.add_node("propose", propose_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "critique")
    graph.add_edge("critique", "revise")
    graph.add_edge("revise", END)
    return graph


def build_finalize_graph(config: DebateConfig) -> StateGraph:
    """Judge synthesis + trace: judge → build_trace → END.

    Called once after all debate rounds complete.  The judge sees the
    final revisions and critiques from the last round and produces the
    final trading decision.  build_trace constructs the auditable trace
    including all accumulated debate_turns.
    """
    graph = StateGraph(DebateState)
    graph.add_node("judge", judge_node)
    graph.add_node("build_trace", build_trace_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", "build_trace")
    graph.add_edge("build_trace", END)
    return graph


def compile_pipeline_graph(config: DebateConfig):
    """Build and compile the pipeline graph, ready for invocation."""
    return build_pipeline_graph(config).compile()


def compile_single_round_graph(config: DebateConfig):
    """Build and compile the single-round graph, ready for invocation."""
    return build_single_round_graph(config).compile()


def compile_finalize_graph(config: DebateConfig):
    """Build and compile the finalize graph, ready for invocation."""
    return build_finalize_graph(config).compile()


# =============================================================================
# MAJORITY VOTE GRAPH CONSTRUCTION
# =============================================================================


def build_majority_vote_graph(config: DebateConfig) -> StateGraph:
    """
    Build a LangGraph majority-vote graph.

    Graph structure (no critique/revise/judge):
      [START] -> [pipeline nodes] -> [build_context] -> [propose]
              -> [aggregate] -> [build_mv_trace] -> [END]
    """
    graph = StateGraph(DebateState)

    graph.add_node("build_context", build_context_node)
    graph.add_node("propose", propose_node)
    graph.add_node("aggregate", aggregate_proposals_node)
    graph.add_node("build_mv_trace", build_mv_trace_node)

    # --- Edges: START -> build_context (no pipeline) ---
    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "propose")
    graph.add_edge("propose", "aggregate")
    graph.add_edge("aggregate", "build_mv_trace")
    graph.add_edge("build_mv_trace", END)

    return graph


def compile_majority_vote_graph(config: DebateConfig):
    """Build and compile the majority-vote graph, ready for invocation."""
    graph = build_majority_vote_graph(config)
    return graph.compile()


def build_round_robin_graph(config: DebateConfig) -> StateGraph:
    """Round robin graph:
    
    START -> propose_agent1 -> round_robin_agent2 -> round_robin_agent3 ... -> END
    """
    from .nodes import make_round_robin_propose_node, make_round_robin_node
    roles = list(config.roles)
    if not roles:
        raise ValueError("Roles list is empty")
        
    graph = StateGraph(DebateState)
    
    first_role = roles[0]
    graph.add_node(f"propose_{first_role}", make_round_robin_propose_node(first_role))
    graph.add_edge(START, f"propose_{first_role}")
    
    prev_node = f"propose_{first_role}"
    
    # Loop max_rounds times
    max_rounds = config.max_rounds
    total_steps = max_rounds * len(roles)
    
    # Starting from the second role in the first round (index 1)
    current_idx = 1
    node_count = 1
    
    while node_count < total_steps:
        # Get the role for this step
        role = roles[current_idx % len(roles)]
        
        # We need unique node names for each step if they can loop.
        # But LangGraph nodes must be unique. We can name them by step.
        node_name = f"step_{node_count}_rr_{role}"
        
        graph.add_node(node_name, make_round_robin_node(role))
        graph.add_edge(prev_node, node_name)
        
        prev_node = node_name
        current_idx += 1
        node_count += 1

    # End after the last node
    graph.add_edge(prev_node, END)
    
    return graph

def compile_round_robin_graph(config: DebateConfig):
    return build_round_robin_graph(config).compile()
