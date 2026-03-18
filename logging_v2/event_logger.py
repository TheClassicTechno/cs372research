"""Append-only, thread-safe event logger for the causal execution trace system.

Writes one JSONL file per debate run:
    logging_v2/runs/<experiment>/<run_id>/events.jsonl

Each event is a single JSON line.  The file is opened in append mode —
no seek, no truncate, no overwrite.  Thread safety is guaranteed by
a threading.Lock that guards both the logical clock increment and
the file write as a single atomic operation.

Usage:
    logger = EventLogger(
        experiment="vskarich_ablation_10",
        run_id="run_2026-03-11_23-36-59",
        debate_id="86ef5e54-...",
    )
    logger.log_run_metadata(config_snapshot={...}, start_time="...")
    logger.log_llm_call(phase="propose", role="macro", ...)
    logger.close()
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .events import (
    DecisionBoundaryEvent,
    ErrorEvent,
    LLMCallEvent,
    RunMetadataEvent,
    StateSnapshotEvent,
    compute_allocation_diff,
    compute_crit_diff,
    normalize_crit_scores,
    _hash_text,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]  # cs372research/
_DEFAULT_BASE_DIR = _REPO_ROOT / "logging_v2" / "runs"


def _get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


class EventLogger:
    """Append-only causal event logger.

    Thread-safe.  Each ``log_*`` method:
    1. Acquires ``self._lock``
    2. Increments ``self._logical_clock``
    3. Serializes the event to JSON
    4. Writes one line to the JSONL file
    5. Releases the lock

    The logical clock provides total ordering across all events in this
    debate run, even when LLM calls run in parallel threads.
    """

    def __init__(
        self,
        experiment: str,
        run_id: str,
        debate_id: str,
        *,
        base_dir: Optional[Path] = None,
        store_full_text: bool = True,
    ) -> None:
        self._experiment = experiment
        self._run_id = run_id
        self._debate_id = debate_id
        self._store_full_text = store_full_text

        self._logical_clock: int = 0
        self._lock = threading.Lock()

        # Auto-track call_index per (round_num, phase, role) — thread-safe
        # under self._lock.  Replaces the broken config["_call_index"] path
        # that was always 0.
        self._call_index_tracker: dict[tuple[int, str, str], int] = {}

        # Track latest state for diff computation
        self._last_allocations: Optional[dict[str, dict[str, float]]] = None
        self._last_crit_scores: Optional[dict[str, dict[str, Any]]] = None

        # Track causal chain IDs for retry linkage
        self._causal_chains: dict[str, str] = {}

        out_dir = (base_dir or _DEFAULT_BASE_DIR) / experiment / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "events.jsonl"
        # Line-buffered append mode — each write is atomic for lines < PIPE_BUF
        self._file = open(self._path, "a", buffering=1, encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        """Path to the events.jsonl file."""
        return self._path

    def _next_clock(self) -> int:
        """Increment and return the logical clock. MUST be called under lock."""
        self._logical_clock += 1
        return self._logical_clock

    def _write_event(self, event_dict: dict[str, Any]) -> str:
        """Serialize and write one event line. MUST be called under lock.

        Returns the event_id.
        """
        line = json.dumps(event_dict, separators=(",", ":"), default=str)
        self._file.write(line + "\n")
        return event_dict["event_id"]

    def _get_causal_chain(self, phase: str, role: str) -> str:
        """Get or create a causal chain ID for a (phase, role) pair.

        Retry calls share the same causal_chain_id as the initial call.
        """
        key = f"{phase}:{role}"
        if key not in self._causal_chains:
            import uuid
            self._causal_chains[key] = str(uuid.uuid4())
        return self._causal_chains[key]

    def reset_causal_chain(self, phase: str, role: str) -> None:
        """Reset the causal chain for a (phase, role) pair.

        Call this at the start of each new phase to ensure initial and
        retry calls within the same logical step share a chain, but
        different phases get distinct chains.
        """
        key = f"{phase}:{role}"
        import uuid
        self._causal_chains[key] = str(uuid.uuid4())

    # ── Public logging methods ───────────────────────────────────────

    def log_run_metadata(
        self,
        config_snapshot: dict[str, Any],
        start_time: str,
        *,
        ticker_universe: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
        max_rounds: int = 0,
    ) -> str:
        """Log the run_metadata event (MUST be the first event)."""
        config_json = json.dumps(config_snapshot, sort_keys=True, default=str)
        config_hash = _hash_text(config_json)

        event = RunMetadataEvent(
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            start_time=start_time,
            git_commit_hash=_get_git_commit(),
            ticker_universe=ticker_universe or [],
            roles=roles or [],
            max_rounds=max_rounds,
        )

        with self._lock:
            clock = self._next_clock()
            event_dict = event.to_dict(
                debate_id=self._debate_id,
                run_id=self._run_id,
                experiment=self._experiment,
                logical_clock=clock,
            )
            return self._write_event(event_dict)

    def log_llm_call(
        self,
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
        round_num: int,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log an LLM call event. Returns the event_id."""
        # Conditionally strip full text
        sys_text = system_prompt if self._store_full_text else ""
        usr_text = user_prompt if self._store_full_text else ""
        resp_text = response if self._store_full_text else ""
        raw_text = (raw_response or response) if self._store_full_text else ""

        event = LLMCallEvent.from_call(
            phase=phase,
            role=role,
            call_index=call_index,
            call_context=call_context,
            system_prompt=sys_text,
            user_prompt=usr_text,
            response=resp_text,
            raw_response=raw_text,
            parsed_output=parsed_output,
            parse_success=parse_success,
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            latency_ms=latency_ms,
        )
        # Always compute hashes from original text, not stored text
        event = LLMCallEvent(
            phase=event.phase,
            role=event.role,
            call_index=event.call_index,
            call_context=event.call_context,
            system_prompt=sys_text,
            user_prompt=usr_text,
            response=resp_text,
            raw_response=raw_text,
            parsed_output=event.parsed_output,
            parse_success=event.parse_success,
            system_prompt_hash=_hash_text(system_prompt),
            user_prompt_hash=_hash_text(user_prompt),
            response_hash=_hash_text(response),
            model_name=event.model_name,
            provider=event.provider,
            temperature=event.temperature,
            latency_ms=event.latency_ms,
        )

        chain_id = self._get_causal_chain(phase, role)

        with self._lock:
            # Auto-compute call_index per (round_num, phase, role).
            # The call_index parameter is ignored — this tracker is the
            # single source of truth, ensuring retries get index 1, 2, …
            ci_key = (round_num, phase, role)
            auto_call_index = self._call_index_tracker.get(ci_key, 0)
            self._call_index_tracker[ci_key] = auto_call_index + 1

            clock = self._next_clock()
            event_dict = event.to_dict(
                debate_id=self._debate_id,
                run_id=self._run_id,
                experiment=self._experiment,
                round_num=round_num,
                logical_clock=clock,
                parent_event_id=parent_event_id,
                causal_chain_id=chain_id,
            )
            # Override call_index with the auto-tracked value
            event_dict["call_index"] = auto_call_index
            return self._write_event(event_dict)

    def log_decision_boundary(
        self,
        *,
        boundary_type: str,
        stage: str,
        inputs: dict[str, Any],
        decision: str,
        outputs: dict[str, Any],
        round_num: int,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> str:
        """Log a decision boundary event. Returns the event_id."""
        event = DecisionBoundaryEvent(
            boundary_type=boundary_type,
            stage=stage,
            inputs=inputs,
            decision=decision,
            outputs=outputs,
        )

        with self._lock:
            clock = self._next_clock()
            event_dict = event.to_dict(
                debate_id=self._debate_id,
                run_id=self._run_id,
                experiment=self._experiment,
                round_num=round_num,
                logical_clock=clock,
                parent_event_id=parent_event_id,
                causal_chain_id=causal_chain_id,
            )
            return self._write_event(event_dict)

    def log_state_snapshot(
        self,
        *,
        snapshot_type: str,
        allocations: dict[str, dict[str, float]],
        round_num: int,
        crit_scores: Optional[dict[str, dict[str, Any]]] = None,
        rho_bar: Optional[float] = None,
        js_divergence: Optional[float] = None,
        evidence_overlap: Optional[float] = None,
        parent_event_id: Optional[str] = None,
        causal_chain_id: Optional[str] = None,
    ) -> str:
        """Log a state snapshot event. Returns the event_id.

        Automatically computes allocation_diff and crit_diff from the
        previous snapshot.
        """
        alloc_diff = None
        if self._last_allocations is not None:
            alloc_diff = compute_allocation_diff(self._last_allocations, allocations)

        crit_diff = None
        if crit_scores is not None:
            crit_diff = compute_crit_diff(self._last_crit_scores, crit_scores)

        event = StateSnapshotEvent(
            snapshot_type=snapshot_type,
            allocations=allocations,
            allocation_diff=alloc_diff if alloc_diff else None,
            crit_scores=crit_scores,
            crit_diff=crit_diff,
            rho_bar=rho_bar,
            js_divergence=js_divergence,
            evidence_overlap=evidence_overlap,
        )

        # Update tracking state
        self._last_allocations = allocations
        if crit_scores is not None:
            self._last_crit_scores = crit_scores

        with self._lock:
            clock = self._next_clock()
            event_dict = event.to_dict(
                debate_id=self._debate_id,
                run_id=self._run_id,
                experiment=self._experiment,
                round_num=round_num,
                logical_clock=clock,
                parent_event_id=parent_event_id,
                causal_chain_id=causal_chain_id,
            )
            return self._write_event(event_dict)

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        round_num: int,
        phase: Optional[str] = None,
        role: Optional[str] = None,
        retryable: bool = False,
        details: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log an error event. Returns the event_id."""
        event = ErrorEvent(
            error_type=error_type,
            message=message,
            phase=phase,
            role=role,
            retryable=retryable,
            details=details,
        )

        with self._lock:
            clock = self._next_clock()
            event_dict = event.to_dict(
                debate_id=self._debate_id,
                run_id=self._run_id,
                experiment=self._experiment,
                round_num=round_num,
                logical_clock=clock,
                parent_event_id=parent_event_id,
            )
            return self._write_event(event_dict)

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the event log file."""
        if not self._closed:
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass
