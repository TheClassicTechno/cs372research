"""logging_v2 — Append-only causal execution trace system.

A parallel logging system that captures every LLM call, decision boundary,
and state snapshot in an immutable JSONL event stream.  Runs alongside
the existing DebateLogger without modifying its behavior.

Enables:
- Deterministic replay of ablation runs (including interventions)
- Pre-intervention state extraction (the CRIT scores that triggered
  an intervention, which the v1 logger overwrites)
- Full causal traceability of pipeline decisions
- Paper-grade audit trails

Usage:
    from logging_v2 import EventLogger, EventReplayLLM
    from logging_v2.loader import load_event_log, get_pre_intervention_crit
"""

from .event_logger import EventLogger
from .events import (
    SCHEMA_VERSION,
    VALID_EVENT_TYPES,
    DecisionBoundaryEvent,
    ErrorEvent,
    LLMCallEvent,
    RunMetadataEvent,
    StateSnapshotEvent,
    compute_allocation_diff,
    compute_crit_diff,
    make_event,
    normalize_crit_scores,
)
from .replay import EventReplayLLM
from .replay_strict import StrictReplayLLM

__all__ = [
    "EventLogger",
    "EventReplayLLM",
    "StrictReplayLLM",
    "SCHEMA_VERSION",
    "VALID_EVENT_TYPES",
    "make_event",
    "RunMetadataEvent",
    "LLMCallEvent",
    "DecisionBoundaryEvent",
    "StateSnapshotEvent",
    "ErrorEvent",
    "compute_allocation_diff",
    "compute_crit_diff",
    "normalize_crit_scores",
]
