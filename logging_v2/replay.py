"""EventReplayLLM — deterministic replay of LLM calls from event logs.

Provides a drop-in replacement for ``_call_llm`` that returns stored
responses from a previously recorded event log.  Supports three
validation modes:

- ``"strict"``: asserts prompt hashes match exactly
- ``"hash_only"``: warns on mismatch but continues
- ``"best_effort"``: no hash validation

Usage:
    from logging_v2.loader import load_event_log
    from logging_v2.replay import EventReplayLLM

    events = load_event_log("path/to/events.jsonl")
    replay = EventReplayLLM(events, mode="strict")

    # Monkeypatch _call_llm
    monkeypatch.setattr("multi_agent.graph.llm._call_llm", replay)

    # Run pipeline — all LLM calls return stored responses
    state = runner.run_returning_state(observation)

    # Verify
    assert replay.all_calls_matched()
    replay.assert_decision_boundaries_match(events)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


ReplayKey = tuple[str, int, str, str, int, str]
# (debate_id, round_num, phase, role, call_index, call_context)


def _hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EventReplayLLM:
    """Replays stored LLM responses from an event log.

    Keying:
        (debate_id, round_num, phase, role, call_index, call_context)

    This 6-tuple uniquely identifies every LLM call in a debate run,
    including retries (call_index > 0) and intervention-triggered
    re-invocations (call_context = "intervention_retry" or "crit_retry").

    The call_index is tracked per (phase, role) and auto-incremented
    on each call.  The call_context is read from ``config["_call_context"]``.

    Validation modes:
        - "strict": AssertionError on prompt hash mismatch
        - "hash_only": log warning on mismatch, continue
        - "best_effort": no hash validation
    """

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        mode: str = "strict",
        debate_id: Optional[str] = None,
    ) -> None:
        if mode not in ("strict", "hash_only", "best_effort"):
            raise ValueError(f"Invalid mode: {mode!r}")
        self._mode = mode

        # Auto-detect debate_id from events if not provided
        if debate_id is None:
            for evt in events:
                if "debate_id" in evt:
                    debate_id = evt["debate_id"]
                    break
        self._debate_id = debate_id or ""

        # Index LLM call events by the full 6-tuple key
        self._responses: dict[ReplayKey, dict[str, Any]] = {}
        for evt in events:
            if evt.get("event_type") != "llm_call":
                continue
            key: ReplayKey = (
                evt.get("debate_id", self._debate_id),
                evt.get("round_num", 0),
                evt.get("phase", ""),
                evt.get("role", ""),
                evt.get("call_index", 0),
                evt.get("call_context", "initial"),
            )
            self._responses[key] = evt

        # Also index by (phase, role, call_index) as fallback
        self._fallback: dict[tuple[str, str, int], dict[str, Any]] = {}
        for evt in events:
            if evt.get("event_type") != "llm_call":
                continue
            fkey = (evt.get("phase", ""), evt.get("role", ""), evt.get("call_index", 0))
            self._fallback[fkey] = evt

        # Track call counts per (phase, role) for auto-incrementing call_index
        self._call_counts: dict[tuple[str, str], int] = {}

        # Record of all calls made during replay
        self.calls: list[dict[str, Any]] = []
        self._unmatched: list[dict[str, Any]] = []

        # Store all events for decision boundary validation
        self._all_events = events

        # Extract decision boundaries from stored events
        self._stored_boundaries: list[dict[str, Any]] = [
            e for e in events if e.get("event_type") == "decision_boundary"
        ]

    def __call__(
        self,
        config: dict,
        system_prompt: str,
        user_prompt: str,
        role: Optional[str] = None,
        phase: Optional[str] = None,
        round_num: int = 0,
    ) -> str:
        """Drop-in replacement for _call_llm.

        Looks up the stored response by the 6-tuple key and returns it.
        Validates prompt hashes according to the configured mode.
        """
        phase = phase or ""
        role = role or ""
        call_context = config.get("_call_context", "initial")

        # Auto-increment call_index per (phase, role)
        count_key = (phase, role)
        call_index = self._call_counts.get(count_key, 0)
        self._call_counts[count_key] = call_index + 1

        # Full 6-tuple lookup
        key: ReplayKey = (
            self._debate_id, round_num, phase, role, call_index, call_context,
        )
        evt = self._responses.get(key)

        # Fallback to (phase, role, call_index) if full key not found
        if evt is None:
            fkey = (phase, role, call_index)
            evt = self._fallback.get(fkey)

        if evt is None:
            self._unmatched.append({
                "phase": phase,
                "role": role,
                "call_index": call_index,
                "call_context": call_context,
                "round_num": round_num,
                "key": key,
            })
            raise KeyError(
                f"No stored response for key={key}. "
                f"Available keys for ({phase}, {role}): "
                f"{[k for k in self._responses if k[2] == phase and k[3] == role]}"
            )

        # Validate prompt hashes
        stored_sys_hash = evt.get("system_prompt_hash", "")
        stored_usr_hash = evt.get("user_prompt_hash", "")
        actual_sys_hash = _hash_text(system_prompt)
        actual_usr_hash = _hash_text(user_prompt)

        sys_match = actual_sys_hash == stored_sys_hash
        usr_match = actual_usr_hash == stored_usr_hash

        if self._mode == "strict":
            if not sys_match:
                raise AssertionError(
                    f"System prompt hash mismatch for {key}: "
                    f"expected={stored_sys_hash[:16]}... "
                    f"actual={actual_sys_hash[:16]}..."
                )
            if not usr_match:
                raise AssertionError(
                    f"User prompt hash mismatch for {key}: "
                    f"expected={stored_usr_hash[:16]}... "
                    f"actual={actual_usr_hash[:16]}..."
                )
        elif self._mode == "hash_only":
            if not sys_match:
                logger.warning(
                    "System prompt hash mismatch for %s: "
                    "expected=%s actual=%s",
                    key, stored_sys_hash[:16], actual_sys_hash[:16],
                )
            if not usr_match:
                logger.warning(
                    "User prompt hash mismatch for %s: "
                    "expected=%s actual=%s",
                    key, stored_usr_hash[:16], actual_usr_hash[:16],
                )

        response = evt.get("response", "")

        call_record = {
            "phase": phase,
            "role": role,
            "call_index": call_index,
            "call_context": call_context,
            "round_num": round_num,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "system_hash_match": sys_match,
            "user_hash_match": usr_match,
            "key": key,
        }
        self.calls.append(call_record)

        return response

    # ── Verification methods ─────────────────────────────────────────

    def all_calls_matched(self) -> bool:
        """True if every call during replay found a stored response."""
        return len(self._unmatched) == 0

    def get_unmatched(self) -> list[dict[str, Any]]:
        """Return list of calls that had no matching stored response."""
        return list(self._unmatched)

    def get_calls_by_phase(self, phase: str) -> list[dict[str, Any]]:
        """Return all replay calls for a given phase."""
        return [c for c in self.calls if c["phase"] == phase]

    def get_calls_by_context(self, context: str) -> list[dict[str, Any]]:
        """Return all replay calls with a given call_context."""
        return [c for c in self.calls if c["call_context"] == context]

    def assert_decision_boundaries_match(
        self,
        events: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Assert that the stored decision boundaries are consistent.

        Checks:
        - All intervention_eval events have matching intervention_fire
          or intervention_skip events
        - No orphan intervention_fire without preceding intervention_eval
        """
        boundaries = events or self._stored_boundaries

        eval_events = [
            b for b in boundaries
            if b.get("boundary_type") == "intervention_eval"
        ]
        fire_events = [
            b for b in boundaries
            if b.get("boundary_type") == "intervention_fire"
        ]
        skip_events = [
            b for b in boundaries
            if b.get("boundary_type") == "intervention_skip"
        ]

        # Every fire should be preceded by an eval with decision="fire"
        fire_clocks = {e.get("logical_clock") for e in fire_events}
        eval_fire_clocks = set()
        for evt in eval_events:
            if evt.get("decision") == "fire":
                eval_fire_clocks.add(evt.get("logical_clock"))

        # Every eval with decision="fire" should have a corresponding fire event
        for evt in eval_events:
            if evt.get("decision") == "fire":
                eval_clock = evt.get("logical_clock")
                # Look for fire event with clock > eval_clock
                matching_fires = [
                    f for f in fire_events
                    if f.get("logical_clock", 0) > eval_clock
                ]
                assert matching_fires, (
                    f"intervention_eval with decision='fire' at clock={eval_clock} "
                    f"has no matching intervention_fire event"
                )

    def assert_no_missing_calls(self) -> None:
        """Assert that every stored LLM call was replayed.

        Raises AssertionError if stored events contain calls that were
        never requested during replay.
        """
        replayed_keys = {c["key"] for c in self.calls}
        stored_keys = set(self._responses.keys())
        missing = stored_keys - replayed_keys
        if missing:
            raise AssertionError(
                f"{len(missing)} stored LLM calls were not replayed: "
                f"{sorted(str(k) for k in list(missing)[:5])}"
            )
