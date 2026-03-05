"""State TypedDicts for the LangGraph debate orchestrator."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class DebateState(TypedDict):
    """State that flows through the LangGraph debate graph."""

    # --- Inputs (set once at invocation) ---
    observation: dict
    config: dict

    # --- Pipeline outputs (set by pipeline nodes) ---
    news_digest: str
    data_analysis: str
    enriched_context: str

    # --- Debate state (replaced each round) ---
    proposals: list  # [{role, action_dict, raw_response}]
    critiques: list  # [{role, critiques: [...], self_critique}]
    revisions: list  # [{role, action_dict, revision_notes}]
    current_round: int

    # --- Accumulated across all rounds (append-only) ---
    debate_turns: Annotated[list, operator.add]

    # --- Final outputs ---
    final_action: dict
    strongest_objection: str
    audited_memo: str
    trace: dict


class ParallelRoundState(TypedDict):
    """State for parallel single-round graph.

    Identical to DebateState but with Annotated[list, operator.add] on
    proposals, critiques, and revisions.  This is required because
    LangGraph raises InvalidUpdateError when parallel nodes write to a
    non-annotated field.  Each per-agent node returns a single-element
    list, and operator.add merges them at the sync barrier.

    debate_turns already uses operator.add in DebateState — unchanged here.
    """

    # --- Inputs (set once at invocation) ---
    observation: dict
    config: dict

    # --- Pipeline outputs (set by pipeline nodes) ---
    news_digest: str
    data_analysis: str
    enriched_context: str

    # --- Debate state (parallel: accumulated via operator.add) ---
    proposals: Annotated[list, operator.add]
    critiques: Annotated[list, operator.add]
    revisions: Annotated[list, operator.add]
    current_round: int

    # --- Accumulated across all rounds (append-only) ---
    debate_turns: Annotated[list, operator.add]

    # --- Final outputs ---
    final_action: dict
    strongest_objection: str
    audited_memo: str
    trace: dict
