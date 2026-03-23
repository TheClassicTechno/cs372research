"""Event schema definitions for the causal execution trace system.

All events share a common envelope (EventEnvelope) and specialize by
event_type.  Each event is a frozen dataclass with a ``to_dict()`` method
that produces a JSON-serializable dict.

Schema version: "v2" — v2 adds workflow-agnostic context, artifact storage,
and the make_event() factory for producing compact WAL events.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


SCHEMA_VERSION = "v2"

# ── Valid Event Types (v2) ───────────────────────────────────────────
VALID_EVENT_TYPES = frozenset({
    # Core v2 types
    "llm_call_start",
    "llm_call_end",
    "artifact_created",
    "evaluation_computed",
    "control_flow",
    "state_snapshot",
    "error",
    # Legacy types (backward compat)
    "run_metadata",
    "llm_call",
    "decision_boundary",
})

# Monotonic logical clock shared across the process.  Guarded externally
# by EventLogger._lock — never incremented without holding the lock.
_logical_clock: int = 0
_clock_lock = threading.Lock()


def _next_logical_clock() -> int:
    """Atomically increment and return the next logical clock value."""
    global _logical_clock
    with _clock_lock:
        _logical_clock += 1
        return _logical_clock


def _hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _new_event_id() -> str:
    """Generate a globally unique event ID."""
    return str(uuid.uuid4())


# ── Common Envelope ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EventEnvelope:
    """Fields present on every event."""

    event_id: str
    event_type: str
    schema_version: str
    debate_id: str
    run_id: str
    experiment: str
    round_num: int
    logical_clock: int
    wall_time_ns: int
    thread_id: str
    parent_event_id: Optional[str]
    causal_chain_id: str


def _make_envelope(
    event_type: str,
    debate_id: str,
    run_id: str,
    experiment: str,
    round_num: int,
    logical_clock: int,
    parent_event_id: Optional[str] = None,
    causal_chain_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the envelope dict for an event."""
    return {
        "event_id": _new_event_id(),
        "event_type": event_type,
        "schema_version": SCHEMA_VERSION,
        "debate_id": debate_id,
        "run_id": run_id,
        "experiment": experiment,
        "round_num": round_num,
        "logical_clock": logical_clock,
        "wall_time_ns": time.monotonic_ns(),
        "thread_id": str(threading.current_thread().ident),
        "parent_event_id": parent_event_id,
        "causal_chain_id": causal_chain_id or str(uuid.uuid4()),
    }


# ── v2 Event Factory ─────────────────────────────────────────────────


def make_event(
    event_type: str,
    run_id: str,
    logical_clock: int,
    payload: dict[str, Any],
    *,
    parent_event_id: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create an event dict in the v2 schema.

    This is the canonical way to produce events for the WAL going forward.
    Events are small, atomic, and workflow-agnostic.

    Args:
        event_type: One of VALID_EVENT_TYPES.
        run_id: The run identifier.
        logical_clock: Monotonically increasing clock value.
        payload: Event-specific data (must be small — no full prompts).
        parent_event_id: Optional link to a parent event.
        context: Optional workflow context dict with keys:
            workflow, stage, agent_id.

    Returns:
        A JSON-serializable event dict.
    """
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Invalid event_type {event_type!r}. "
            f"Valid types: {sorted(VALID_EVENT_TYPES)}"
        )

    ctx = context or {}
    return {
        "event_id": _new_event_id(),
        "parent_event_id": parent_event_id,
        "run_id": run_id,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "logical_clock": logical_clock,
        "context": {
            "workflow": ctx.get("workflow"),
            "stage": ctx.get("stage"),
            "agent_id": ctx.get("agent_id"),
        },
        "payload": payload,
    }


# ── Event Type: run_metadata ─────────────────────────────────────────


@dataclass
class RunMetadataEvent:
    """First event in every event log file.  Captures full run config."""

    config_snapshot: dict[str, Any]
    config_hash: str
    start_time: str  # ISO 8601 UTC
    git_commit_hash: Optional[str] = None
    ticker_universe: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    max_rounds: int = 0

    def to_dict(
        self,
        debate_id: str,
        run_id: str,
        experiment: str,
        logical_clock: int,
    ) -> dict[str, Any]:
        env = _make_envelope(
            "run_metadata", debate_id, run_id, experiment,
            round_num=0, logical_clock=logical_clock,
        )
        env.update({
            "config_snapshot": self.config_snapshot,
            "config_hash": self.config_hash,
            "start_time": self.start_time,
            "git_commit_hash": self.git_commit_hash,
            "ticker_universe": self.ticker_universe,
            "roles": self.roles,
            "max_rounds": self.max_rounds,
        })
        return env


# ── Event Type: llm_call ─────────────────────────────────────────────


@dataclass
class LLMCallEvent:
    """Captures every _call_llm invocation — agents, CRIT, judge."""

    phase: str
    role: str
    call_index: int
    call_context: str  # "initial", "intervention_retry", "crit_retry", "parse_retry"

    system_prompt: str
    user_prompt: str
    response: str
    raw_response: str
    parsed_output: Optional[dict[str, Any]]
    parse_success: bool

    system_prompt_hash: str
    user_prompt_hash: str
    response_hash: str

    model_name: str
    provider: str
    temperature: float
    latency_ms: float

    def to_dict(
        self,
        debate_id: str,
        run_id: str,
        experiment: str,
        round_num: int,
        logical_clock: int,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> dict[str, Any]:
        env = _make_envelope(
            "llm_call", debate_id, run_id, experiment,
            round_num=round_num, logical_clock=logical_clock,
            parent_event_id=parent_event_id,
            causal_chain_id=causal_chain_id,
        )
        env.update({
            "phase": self.phase,
            "role": self.role,
            "call_index": self.call_index,
            "call_context": self.call_context,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response": self.response,
            "raw_response": self.raw_response,
            "parsed_output": self.parsed_output,
            "parse_success": self.parse_success,
            "system_prompt_hash": self.system_prompt_hash,
            "user_prompt_hash": self.user_prompt_hash,
            "response_hash": self.response_hash,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "latency_ms": self.latency_ms,
        })
        return env

    @staticmethod
    def from_call(
        *,
        phase: str,
        role: str,
        call_index: int,
        call_context: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        raw_response: Optional[str] = None,
        parsed_output: Optional[dict[str, Any]] = None,
        parse_success: bool = True,
        model_name: str,
        provider: str,
        temperature: float,
        latency_ms: float,
    ) -> "LLMCallEvent":
        """Factory that auto-computes hashes."""
        return LLMCallEvent(
            phase=phase,
            role=role,
            call_index=call_index,
            call_context=call_context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            raw_response=raw_response if raw_response is not None else response,
            parsed_output=parsed_output,
            parse_success=parse_success,
            system_prompt_hash=_hash_text(system_prompt),
            user_prompt_hash=_hash_text(user_prompt),
            response_hash=_hash_text(response),
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            latency_ms=latency_ms,
        )


# ── Event Type: decision_boundary ────────────────────────────────────


@dataclass
class DecisionBoundaryEvent:
    """Captures every pipeline decision point (intervention, PID, convergence)."""

    boundary_type: str  # "intervention_eval", "intervention_fire",
                        # "intervention_skip", "crit_threshold",
                        # "pid_step", "convergence_check"
    stage: str          # "post_revision", "post_crit"
    inputs: dict[str, Any]
    decision: str       # "fire", "skip", "converged", "continue"
    outputs: dict[str, Any]

    def to_dict(
        self,
        debate_id: str,
        run_id: str,
        experiment: str,
        round_num: int,
        logical_clock: int,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> dict[str, Any]:
        env = _make_envelope(
            "decision_boundary", debate_id, run_id, experiment,
            round_num=round_num, logical_clock=logical_clock,
            parent_event_id=parent_event_id,
            causal_chain_id=causal_chain_id,
        )
        env.update({
            "boundary_type": self.boundary_type,
            "stage": self.stage,
            "inputs": self.inputs,
            "decision": self.decision,
            "outputs": self.outputs,
        })
        return env


# ── Event Type: state_snapshot ───────────────────────────────────────


@dataclass
class StateSnapshotEvent:
    """Captures pipeline state at phase boundaries."""

    snapshot_type: str      # "post_propose", "post_critique", "post_revise",
                            # "post_crit", "post_intervention_retry",
                            # "post_crit_retry", "final_portfolio"

    allocations: dict[str, dict[str, float]]  # {role: {ticker: weight}}

    allocation_diff: Optional[dict[str, dict[str, float]]] = None
    crit_scores: Optional[dict[str, dict[str, Any]]] = None
    crit_diff: Optional[dict[str, dict[str, Any]]] = None
    rho_bar: Optional[float] = None
    js_divergence: Optional[float] = None
    evidence_overlap: Optional[float] = None

    def to_dict(
        self,
        debate_id: str,
        run_id: str,
        experiment: str,
        round_num: int,
        logical_clock: int,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> dict[str, Any]:
        env = _make_envelope(
            "state_snapshot", debate_id, run_id, experiment,
            round_num=round_num, logical_clock=logical_clock,
            parent_event_id=parent_event_id,
            causal_chain_id=causal_chain_id,
        )
        env.update({
            "snapshot_type": self.snapshot_type,
            "allocations": self.allocations,
            "allocation_diff": self.allocation_diff,
            "crit_scores": self.crit_scores,
            "crit_diff": self.crit_diff,
            "rho_bar": self.rho_bar,
            "js_divergence": self.js_divergence,
            "evidence_overlap": self.evidence_overlap,
        })
        return env


# ── Event Type: error ────────────────────────────────────────────────


@dataclass
class ErrorEvent:
    """Captures errors and failures."""

    error_type: str    # "parse_failure", "llm_error", "validation_error"
    message: str
    phase: Optional[str] = None
    role: Optional[str] = None
    retryable: bool = False
    details: Optional[dict[str, Any]] = None

    def to_dict(
        self,
        debate_id: str,
        run_id: str,
        experiment: str,
        round_num: int,
        logical_clock: int,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> dict[str, Any]:
        env = _make_envelope(
            "error", debate_id, run_id, experiment,
            round_num=round_num, logical_clock=logical_clock,
            parent_event_id=parent_event_id,
            causal_chain_id=causal_chain_id,
        )
        env.update({
            "error_type": self.error_type,
            "message": self.message,
            "phase": self.phase,
            "role": self.role,
            "retryable": self.retryable,
            "details": self.details,
        })
        return env


# ── Diff Utilities ───────────────────────────────────────────────────


def compute_allocation_diff(
    prev: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute per-role, per-ticker allocation deltas.

    Returns: {role: {ticker: delta}} where delta = current - prev.
    Only includes non-zero deltas.
    """
    diff: dict[str, dict[str, float]] = {}
    all_roles = set(prev.keys()) | set(current.keys())
    for role in sorted(all_roles):
        prev_alloc = prev.get(role, {})
        curr_alloc = current.get(role, {})
        all_tickers = set(prev_alloc.keys()) | set(curr_alloc.keys())
        role_diff: dict[str, float] = {}
        for ticker in sorted(all_tickers):
            delta = curr_alloc.get(ticker, 0.0) - prev_alloc.get(ticker, 0.0)
            if abs(delta) > 1e-6:
                role_diff[ticker] = round(delta, 6)
        if role_diff:
            diff[role] = role_diff
    return diff


def compute_crit_diff(
    prev: Optional[dict[str, dict[str, Any]]],
    current: Optional[dict[str, dict[str, Any]]],
) -> Optional[dict[str, dict[str, Any]]]:
    """Compute per-role CRIT score deltas.

    Uses the normalized schema:
    {role: {"rho_i": float, "pillars": {pillar: float}}}

    Returns: {role: {"rho_i_delta": float, "pillars": {pillar: delta}}}
    """
    if prev is None or current is None:
        return None

    diff: dict[str, dict[str, Any]] = {}
    all_roles = set(prev.keys()) | set(current.keys())
    for role in sorted(all_roles):
        prev_data = prev.get(role, {})
        curr_data = current.get(role, {})

        role_diff: dict[str, Any] = {}

        prev_rho = prev_data.get("rho_i", 0.0)
        curr_rho = curr_data.get("rho_i", 0.0)
        if prev_rho is not None and curr_rho is not None:
            delta = curr_rho - prev_rho
            if abs(delta) > 1e-6:
                role_diff["rho_i_delta"] = round(delta, 6)

        prev_pillars = prev_data.get("pillars", {})
        curr_pillars = curr_data.get("pillars", {})
        pillar_diff: dict[str, float] = {}
        for pillar in ("logical_validity", "evidential_support",
                       "alternative_consideration", "causal_alignment"):
            p = prev_pillars.get(pillar, 0.0)
            c = curr_pillars.get(pillar, 0.0)
            if p is not None and c is not None:
                d = c - p
                if abs(d) > 1e-6:
                    pillar_diff[pillar] = round(d, 6)
        if pillar_diff:
            role_diff["pillars"] = pillar_diff

        if role_diff:
            diff[role] = role_diff

    return diff if diff else None


def normalize_crit_scores(
    agent_scores: dict,
) -> dict[str, dict[str, Any]]:
    """Normalize CRIT scores to the canonical schema.

    Accepts either:
    - RoundCritResult.agent_scores (Pydantic models)
    - Raw dicts from round_state.json

    Returns:
    {
        role: {
            "rho_i": float,
            "pillars": {
                "logical_validity": float,
                "evidential_support": float,
                "alternative_consideration": float,
                "causal_alignment": float,
            }
        }
    }
    """
    result: dict[str, dict[str, Any]] = {}
    for role, cr in agent_scores.items():
        if hasattr(cr, "pillar_scores"):
            # Pydantic CritResult object
            ps = cr.pillar_scores
            result[role] = {
                "rho_i": cr.rho_bar,
                "pillars": {
                    "logical_validity": ps.logical_validity,
                    "evidential_support": ps.evidential_support,
                    "alternative_consideration": ps.alternative_consideration,
                    "causal_alignment": ps.causal_alignment,
                },
            }
        elif isinstance(cr, dict):
            # Raw dict (from round_state.json or similar)
            pillars = cr.get("pillars", cr.get("pillar_scores", {}))
            # Handle abbreviated keys
            result[role] = {
                "rho_i": cr.get("rho_i", cr.get("rho_bar", 0.0)),
                "pillars": {
                    "logical_validity": pillars.get("logical_validity", pillars.get("LV", 0.0)),
                    "evidential_support": pillars.get("evidential_support", pillars.get("ES", 0.0)),
                    "alternative_consideration": pillars.get("alternative_consideration", pillars.get("AC", 0.0)),
                    "causal_alignment": pillars.get("causal_alignment", pillars.get("CA", 0.0)),
                },
            }
    return result
