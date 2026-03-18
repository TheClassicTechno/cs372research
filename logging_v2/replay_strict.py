"""StrictReplayLLM — thread-safe deterministic replay of LLM calls.

No fallbacks, no partial keys, no silent skipping.  Every runtime call
must match a stored event by its unique 4-tuple key.  Any deviation is
a hard failure.

Replay key: ``(round_num, phase, role, call_index)``

Thread-safe: all mutable state is guarded by a ``threading.Lock``,
supporting ``parallel_agents=True`` (concurrent LLM calls from
LangGraph parallel nodes).

``debate_id`` is excluded because the runner generates a fresh one each run.
``call_context`` is enforced as a separate assertion after key matching.

Usage:
    from logging_v2.loader import load_event_log
    from logging_v2.replay_strict import StrictReplayLLM

    events = load_event_log("path/to/events.jsonl")
    replay = StrictReplayLLM(events, strict=True)

    monkeypatch.setattr("multi_agent.graph.nodes._call_llm", replay)
    monkeypatch.setattr("multi_agent.runner._call_llm", replay)

    state = runner.run_returning_state(observation)

    replay.assert_all_consumed()
    replay.assert_no_diffs()
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Optional


# ── Replay key ────────────────────────────────────────────────────────

ReplayKey = tuple[int, str, str, int]
# (round_num, phase, role, call_index)


# ── Custom exceptions ─────────────────────────────────────────────────

class ReplayExhaustedError(RuntimeError):
    """Runtime key has no matching stored event."""


class ReplayKeyMismatchError(RuntimeError):
    """call_context mismatch for a matched key."""


class PromptDriftError(RuntimeError):
    """Prompt hash mismatch between runtime and stored event."""


class ExecutionParamMismatchError(RuntimeError):
    """Execution parameter (model, temperature, provider) mismatch."""

    def __init__(self, param: str, expected: Any, actual: Any) -> None:
        self.param = param
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{param} mismatch: expected={expected!r}, actual={actual!r}"
        )


class ReplayIncompleteError(RuntimeError):
    """Events remain unconsumed after replay finished."""


class ReplayDiffError(RuntimeError):
    """Debug mode collected mismatches (raised by assert_no_diffs)."""


# ── Helpers ───────────────────────────────────────────────────────────

def _hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Strict Replay Engine ─────────────────────────────────────────────

class StrictReplayLLM:
    """Thread-safe strict replay of LLM calls from an event log.

    Uses key-based lookup: each runtime call builds a
    ``(round_num, phase, role, call_index)`` key and looks up the
    matching stored event.  No sequence cursor, no ordering assumptions
    within a phase — fully compatible with ``parallel_agents=True``.

    All mutable state is guarded by ``self._lock``.

    Args:
        events: Full event list (all types).  Only ``llm_call`` events
            are used; others are ignored.
        strict: If True (default), mismatches raise immediately.
            If False ("debug mode"), mismatches are collected and can
            be inspected via ``diffs`` / ``assert_no_diffs()``.
    """

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        strict: bool = True,
    ) -> None:
        # Index llm_call events by unique 4-tuple key
        self._events_by_key: dict[ReplayKey, dict[str, Any]] = {}
        for evt in events:
            if evt.get("event_type") != "llm_call":
                continue
            key: ReplayKey = (
                evt.get("round_num", 0),
                evt.get("phase", ""),
                evt.get("role", ""),
                evt.get("call_index", 0),
            )
            if key in self._events_by_key:
                raise ValueError(f"Duplicate replay key: {key}")
            self._events_by_key[key] = evt

        self._lock = threading.Lock()
        self._strict: bool = strict
        self._call_index_tracker: dict[tuple[int, str, str], int] = {}
        self._consumed: set[ReplayKey] = set()
        self._diffs: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    @property
    def event_count(self) -> int:
        """Total number of stored LLM call events."""
        return len(self._events_by_key)

    @property
    def diffs(self) -> list[dict[str, Any]]:
        """Collected mismatches (debug mode only)."""
        with self._lock:
            return list(self._diffs)

    def __call__(
        self,
        config: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        role: Optional[str] = None,
        phase: Optional[str] = None,
        round_num: int = 0,
    ) -> str:
        """Drop-in replacement for ``_call_llm``.

        Looks up the stored event by key, validates call_context +
        prompt hashes + execution parameters, and returns the stored
        response.  Thread-safe.
        """
        phase = phase or ""
        role = role or ""
        call_context = config.get("_call_context", "initial")

        # 1. Deterministic call_index (under lock for thread safety)
        with self._lock:
            ci_key = (round_num, phase, role)
            call_index = self._call_index_tracker.get(ci_key, 0)
            self._call_index_tracker[ci_key] = call_index + 1

        # 2. Key-based lookup
        current_key: ReplayKey = (round_num, phase, role, call_index)
        expected = self._events_by_key.get(current_key)

        if expected is None:
            msg = (
                f"No stored event for key={current_key}, "
                f"call_context={call_context!r}. "
                f"Available keys: {sorted(self._events_by_key.keys())}"
            )
            if self._strict:
                raise ReplayExhaustedError(msg)
            with self._lock:
                self._diffs.append({
                    "type": "missing_key",
                    "key": current_key,
                    "message": msg,
                })
            return ""

        # 3. Mark consumed (under lock)
        with self._lock:
            self._consumed.add(current_key)

        # 4. call_context match (strict enforcement)
        expected_context = expected.get("call_context", "initial")
        if call_context != expected_context:
            msg = (
                f"call_context mismatch for key={current_key}: "
                f"expected={expected_context!r}, actual={call_context!r}"
            )
            if self._strict:
                raise ReplayKeyMismatchError(msg)
            with self._lock:
                self._diffs.append({
                    "type": "call_context_mismatch",
                    "key": current_key,
                    "expected_context": expected_context,
                    "actual_context": call_context,
                    "message": msg,
                })

        # 5. Prompt hash match
        actual_sys_hash = _hash_text(system_prompt)
        actual_usr_hash = _hash_text(user_prompt)
        stored_sys_hash = expected.get("system_prompt_hash", "")
        stored_usr_hash = expected.get("user_prompt_hash", "")

        sys_hash_match = actual_sys_hash == stored_sys_hash
        usr_hash_match = actual_usr_hash == stored_usr_hash

        if self._strict:
            if not sys_hash_match:
                raise PromptDriftError(
                    f"System prompt hash mismatch for key={current_key}: "
                    f"expected={stored_sys_hash[:16]}..., "
                    f"actual={actual_sys_hash[:16]}..."
                )
            if not usr_hash_match:
                raise PromptDriftError(
                    f"User prompt hash mismatch for key={current_key}: "
                    f"expected={stored_usr_hash[:16]}..., "
                    f"actual={actual_usr_hash[:16]}..."
                )
        else:
            if not sys_hash_match:
                with self._lock:
                    self._diffs.append({
                        "type": "system_prompt_drift",
                        "key": current_key,
                        "expected_hash": stored_sys_hash[:16],
                        "actual_hash": actual_sys_hash[:16],
                    })
            if not usr_hash_match:
                with self._lock:
                    self._diffs.append({
                        "type": "user_prompt_drift",
                        "key": current_key,
                        "expected_hash": stored_usr_hash[:16],
                        "actual_hash": actual_usr_hash[:16],
                    })

        # 6. Execution parameter match (model, temperature, provider)
        model_name = config.get("model_name", "")
        temperature = config.get("temperature", 0.0)
        provider = config.get("llm_provider", "")

        stored_model = expected.get("model_name", "")
        stored_temp = expected.get("temperature", 0.0)
        stored_provider = expected.get("provider", "")

        if self._strict:
            if model_name != stored_model:
                raise ExecutionParamMismatchError(
                    "model_name", stored_model, model_name,
                )
            if abs(temperature - stored_temp) > 1e-6:
                raise ExecutionParamMismatchError(
                    "temperature", stored_temp, temperature,
                )
            if provider != stored_provider:
                raise ExecutionParamMismatchError(
                    "provider", stored_provider, provider,
                )
        else:
            with self._lock:
                if model_name != stored_model:
                    self._diffs.append({
                        "type": "model_mismatch",
                        "key": current_key,
                        "expected": stored_model,
                        "actual": model_name,
                    })
                if abs(temperature - stored_temp) > 1e-6:
                    self._diffs.append({
                        "type": "temperature_mismatch",
                        "key": current_key,
                        "expected": stored_temp,
                        "actual": temperature,
                    })
                if provider != stored_provider:
                    self._diffs.append({
                        "type": "provider_mismatch",
                        "key": current_key,
                        "expected": stored_provider,
                        "actual": provider,
                    })

        # 7. Record call and return stored response
        response = expected.get("response", "")

        with self._lock:
            self.calls.append({
                "phase": phase,
                "role": role,
                "call_index": call_index,
                "call_context": call_context,
                "round_num": round_num,
                "key": current_key,
                "system_hash_match": sys_hash_match,
                "user_hash_match": usr_hash_match,
                "model_name": model_name,
                "temperature": temperature,
                "provider": provider,
            })

        return response

    # ── Verification methods ─────────────────────────────────────────

    def assert_all_consumed(self) -> None:
        """Assert every stored event was consumed during replay.

        Raises ReplayIncompleteError if any keys were never matched.
        """
        with self._lock:
            unconsumed = set(self._events_by_key.keys()) - self._consumed
        if unconsumed:
            sample = sorted(unconsumed)[:5]
            raise ReplayIncompleteError(
                f"{len(unconsumed)} event(s) unconsumed: {sample}"
            )

    def assert_no_diffs(self) -> None:
        """Assert no mismatches were collected (debug mode).

        Raises ReplayDiffError if any diffs exist.
        """
        with self._lock:
            diffs = list(self._diffs)
        if diffs:
            summary = "; ".join(
                d.get("message", str(d)) for d in diffs[:5]
            )
            raise ReplayDiffError(
                f"{len(diffs)} mismatch(es) collected: {summary}"
            )
