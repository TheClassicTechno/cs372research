"""Append-only, thread-safe event logger with WAL-based segment files.

Writes segmented JSONL files per debate run:
    logging_v2/runs/<experiment>/<run_id>/events/segment_000000.jsonl
    logging_v2/runs/<experiment>/<run_id>/events/segment_000001.jsonl
    ...

Legacy single-file format (events.jsonl) is still supported by the loader.

Each event is a single JSON line.  The file is opened in append mode —
no seek, no truncate, no overwrite.  Thread safety is guaranteed by
a threading.Lock that guards both the logical clock increment and
the file write as a single atomic operation.

Segment rotation: when the active segment reaches ``segment_max_events``
events, it is closed and a new segment is opened.  Only CLOSED segments
(not the active one) are eligible for S3 upload.

Restart safety: ``event_index.txt`` persists the logical clock counter
so that a restarted logger picks up where it left off.

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
import uuid
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
    make_event,
    normalize_crit_scores,
    VALID_EVENT_TYPES,
    _hash_text,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]  # cs372research/
_DEFAULT_BASE_DIR = _REPO_ROOT / "logging_v2" / "runs"

DEFAULT_SEGMENT_MAX_EVENTS = 1000


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
    """Append-only causal event logger with WAL-based segment files.

    Thread-safe.  Each ``log_*`` method:
    1. Acquires ``self._lock``
    2. Increments ``self._logical_clock``
    3. Serializes the event to JSON
    4. Writes one line to the active segment JSONL file
    5. Persists event_index to disk
    6. Rotates segment if threshold reached
    7. Releases the lock

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
        segment_max_events: int = DEFAULT_SEGMENT_MAX_EVENTS,
    ) -> None:
        self._experiment = experiment
        self._run_id = run_id
        self._debate_id = debate_id
        self._store_full_text = store_full_text
        self._segment_max_events = segment_max_events

        self._lock = threading.Lock()

        # Linear causal chain: each event parents to the previous one
        self._last_event_id: Optional[str] = None

        # Auto-track call_index per (round_num, phase, role) — thread-safe
        # under self._lock.  Replaces the broken config["_call_index"] path
        # that was always 0.
        self._call_index_tracker: dict[tuple[int, str, str], int] = {}

        # Track latest state for diff computation
        self._last_allocations: Optional[dict[str, dict[str, float]]] = None
        self._last_crit_scores: Optional[dict[str, dict[str, Any]]] = None

        # Track causal chain IDs for retry linkage
        self._causal_chains: dict[str, str] = {}

        # --- Directory setup ---
        self._run_dir = (base_dir or _DEFAULT_BASE_DIR) / experiment / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_dir = self._run_dir / "events"
        self._events_dir.mkdir(parents=True, exist_ok=True)

        # --- Artifact storage ---
        self._artifacts_dir = self._run_dir / "artifacts" / "llm_calls"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_counter = self._find_next_artifact_index()

        # --- Restore event_index from disk (restart safety) ---
        self._event_index_path = self._run_dir / "event_index.txt"
        self._logical_clock = self._load_event_index()

        # --- Segment state ---
        self._segment_index = self._find_next_segment_index()
        self._segment_event_count = self._count_events_in_active_segment()
        self._active_segment_path = self._segment_path(self._segment_index)
        self._file = open(self._active_segment_path, "a", buffering=1, encoding="utf-8")
        self._closed = False

        # Legacy compatibility: expose path to run_dir
        self._path = self._run_dir / "events.jsonl"

    @property
    def path(self) -> Path:
        """Path to the run directory's legacy events.jsonl location."""
        return self._path

    @property
    def run_dir(self) -> Path:
        """Path to the run directory."""
        return self._run_dir

    @property
    def events_dir(self) -> Path:
        """Path to the events/ segment directory."""
        return self._events_dir

    # ── Segment helpers ─────────────────────────────────────────────

    def _segment_path(self, index: int) -> Path:
        """Return the path for segment N."""
        return self._events_dir / f"segment_{index:06d}.jsonl"

    def _find_next_segment_index(self) -> int:
        """Find the highest existing segment index, or 0 if none exist."""
        existing = sorted(self._events_dir.glob("segment_*.jsonl"))
        if not existing:
            return 0
        # Parse the index from the last segment filename
        last = existing[-1].stem  # "segment_000003"
        return int(last.split("_")[1])

    def _count_events_in_active_segment(self) -> int:
        """Count lines in the active segment (for restart recovery)."""
        path = self._segment_path(self._segment_index)
        if not path.exists():
            return 0
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _rotate_segment(self) -> None:
        """Close the active segment and open a new one. MUST be called under lock."""
        self._file.flush()
        self._file.close()
        self._segment_index += 1
        self._segment_event_count = 0
        self._active_segment_path = self._segment_path(self._segment_index)
        self._file = open(self._active_segment_path, "a", buffering=1, encoding="utf-8")

    def get_closed_segments(self) -> list[Path]:
        """Return paths of all CLOSED segments (all except the active one).

        Only closed segments are safe for S3 upload.
        """
        result = []
        for i in range(self._segment_index):
            p = self._segment_path(i)
            if p.exists():
                result.append(p)
        return result

    def get_all_segments(self) -> list[Path]:
        """Return paths of all segment files (closed + active), sorted."""
        return sorted(self._events_dir.glob("segment_*.jsonl"))

    # ── Artifact helpers ───────────────────────────────────────────

    def _find_next_artifact_index(self) -> int:
        """Find the next sequential artifact index (restart recovery)."""
        existing = sorted(self._artifacts_dir.glob("*.json"))
        if not existing:
            return 0
        last = existing[-1].stem  # e.g. "000123"
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)

    def _write_artifact(self, data: dict[str, Any]) -> str:
        """Write an artifact JSON file to artifacts/llm_calls/.

        Args:
            data: The artifact content dict. Must include at minimum:
                call_id, timestamp, model, system_prompt, user_prompt,
                response, metadata.

        Returns:
            Relative path from run_dir (e.g. "artifacts/llm_calls/000123.json").
        """
        index = self._artifact_counter
        self._artifact_counter += 1
        filename = f"{index:06d}.json"
        artifact_path = self._artifacts_dir / filename
        artifact_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        return f"artifacts/llm_calls/{filename}"

    @property
    def artifacts_dir(self) -> Path:
        """Path to the artifacts/llm_calls/ directory."""
        return self._artifacts_dir

    # ── Event index persistence ─────────────────────────────────────

    def _load_event_index(self) -> int:
        """Load the persisted event index, or 0 if not found."""
        if self._event_index_path.exists():
            try:
                text = self._event_index_path.read_text(encoding="utf-8").strip()
                return int(text)
            except (ValueError, OSError):
                pass
        return 0

    def _persist_event_index(self) -> None:
        """Write the current logical clock to event_index.txt. MUST be called under lock."""
        try:
            self._event_index_path.write_text(
                str(self._logical_clock), encoding="utf-8"
            )
        except OSError:
            pass  # Best-effort; don't crash the logger

    # ── Core write path ─────────────────────────────────────────────

    def _next_clock(self) -> int:
        """Increment and return the logical clock. MUST be called under lock."""
        self._logical_clock += 1
        return self._logical_clock

    def _to_v2(self, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Wrap a legacy flat event dict into v2 {context, payload} envelope.

        Extracts known envelope fields; everything else becomes payload.
        Also enforces parent_event_id chaining and adds timestamp.
        MUST be called under lock.
        """
        # Fields that belong in the v2 envelope (not payload)
        _ENVELOPE = {
            "event_id", "event_type", "run_id", "logical_clock",
            "parent_event_id",
        }

        # Already a v2 event (has context + payload)?  Just chain parent.
        if "context" in event_dict and "payload" in event_dict:
            if event_dict.get("parent_event_id") is None:
                event_dict["parent_event_id"] = self._last_event_id
            event_dict["timestamp"] = event_dict.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            )
            return event_dict

        # --- Wrap legacy v1 flat event into v2 shape ---

        # Build context from known fields (don't remove originals from payload)
        context = {
            "workflow": event_dict.get("experiment", self._experiment),
            "stage": event_dict.get("stage") or event_dict.get("phase"),
            "agent_id": event_dict.get("role"),
        }
        # Remove only the duplicate 'experiment' from payload
        event_dict.pop("experiment", None)

        # Extract envelope fields
        envelope: dict[str, Any] = {}
        for k in list(_ENVELOPE):
            if k in event_dict:
                envelope[k] = event_dict.pop(k)

        # Remove legacy envelope cruft from payload (keep debate_id)
        for k in ("wall_time_ns", "thread_id", "causal_chain_id"):
            event_dict.pop(k, None)

        # Set parent_event_id via linear chain
        parent = envelope.get("parent_event_id")
        if parent is None:
            parent = self._last_event_id

        return {
            "event_id": envelope["event_id"],
            "parent_event_id": parent,
            "run_id": envelope.get("run_id", self._run_id),
            "event_type": envelope["event_type"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logical_clock": envelope["logical_clock"],
            "context": context,
            "payload": event_dict,  # everything left over
        }

    def _write_event(self, event_dict: dict[str, Any]) -> str:
        """Wrap in v2, serialize, write one event line. MUST be called under lock.

        Returns the event_id.
        """
        v2 = self._to_v2(event_dict)
        line = json.dumps(v2, separators=(",", ":"), default=str)
        self._file.write(line + "\n")
        self._segment_event_count += 1
        self._persist_event_index()

        # Update linear causal chain
        self._last_event_id = v2["event_id"]

        # Rotate segment if threshold reached
        if self._segment_event_count >= self._segment_max_events:
            self._rotate_segment()

        return v2["event_id"]

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
        token_usage: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an LLM call event. Returns the event_id.

        Writes full text to artifact only. WAL gets a compact event
        with hashes and artifact_path.
        """
        with self._lock:
            # Auto-compute call_index per (round_num, phase, role).
            ci_key = (round_num, phase, role)
            auto_call_index = self._call_index_tracker.get(ci_key, 0)
            self._call_index_tracker[ci_key] = auto_call_index + 1

            # Write artifact with full LLM interaction data
            artifact_data = {
                "call_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": response,
                "metadata": {
                    "latency_ms": int(latency_ms),
                    "token_usage": token_usage,
                },
            }
            artifact_path = self._write_artifact(artifact_data)

            # Compact WAL event — no full text
            clock = self._next_clock()
            event_dict: dict[str, Any] = {
                "event_id": str(uuid.uuid4()),
                "event_type": "llm_call",
                "run_id": self._run_id,
                "logical_clock": clock,
                "parent_event_id": parent_event_id,
                "experiment": self._experiment,
                "phase": phase,
                "role": role,
                "round_num": round_num,
                "call_index": auto_call_index,
                "call_context": call_context,
                "model_name": model_name,
                "provider": provider,
                "temperature": temperature,
                "latency_ms": latency_ms,
                "parse_success": parse_success,
                "system_prompt_hash": _hash_text(system_prompt),
                "user_prompt_hash": _hash_text(user_prompt),
                "response_hash": _hash_text(response),
                "artifact_path": artifact_path,
            }
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

    # ── v2 Generic Event Emitter ────────────────────────────────────

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        context: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Emit a v2-schema event to the WAL.

        This is the primary interface for new event types. Events are
        small and atomic — large data belongs in artifacts.

        Args:
            event_type: One of VALID_EVENT_TYPES.
            payload: Event-specific data (keep small).
            context: Workflow context {workflow, stage, agent_id}.
            parent_event_id: Optional parent event linkage.

        Returns:
            The event_id of the emitted event.
        """
        with self._lock:
            clock = self._next_clock()
            event = make_event(
                event_type=event_type,
                run_id=self._run_id,
                logical_clock=clock,
                payload=payload,
                parent_event_id=parent_event_id,
                context=context,
            )
            return self._write_event(event)

    def emit_llm_call_start(
        self,
        *,
        model: str,
        context: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Emit an llm_call_start event before an LLM call.

        Returns the event_id (use as parent_event_id for the
        corresponding llm_call_end).
        """
        return self.emit(
            "llm_call_start",
            {"model": model},
            context=context,
            parent_event_id=parent_event_id,
        )

    def emit_llm_call_end(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response: str,
        model: str,
        latency_ms: float = 0.0,
        token_usage: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Write artifact + emit llm_call_end to WAL.

        1. Writes the full LLM interaction to an artifact file.
        2. Emits an artifact_created event.
        3. Emits an llm_call_end event referencing the artifact.

        Returns the event_id of the llm_call_end event.
        """
        call_id = str(uuid.uuid4())
        response_hash = _hash_text(response)

        # 1. Write artifact
        artifact_data = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "metadata": {
                "latency_ms": int(latency_ms),
                "token_usage": token_usage,
            },
        }

        with self._lock:
            artifact_path = self._write_artifact(artifact_data)

        # 2. Emit artifact_created
        self.emit(
            "artifact_created",
            {"artifact_path": artifact_path, "call_id": call_id},
            context=context,
        )

        # 3. Emit llm_call_end (compact — no full text)
        return self.emit(
            "llm_call_end",
            {
                "call_id": call_id,
                "response_hash": response_hash,
                "artifact_path": artifact_path,
                "model": model,
                "latency_ms": latency_ms,
            },
            context=context,
            parent_event_id=parent_event_id,
        )

    def emit_evaluation(
        self,
        *,
        metric_name: str,
        scores: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Emit an evaluation_computed event."""
        return self.emit(
            "evaluation_computed",
            {"metric_name": metric_name, "scores": scores},
            context=context,
            parent_event_id=parent_event_id,
        )

    def emit_control_flow(
        self,
        *,
        decision: str,
        reason: str,
        details: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Emit a control_flow event for pipeline decisions."""
        payload: dict[str, Any] = {"decision": decision, "reason": reason}
        if details:
            payload["details"] = details
        return self.emit(
            "control_flow",
            payload,
            context=context,
            parent_event_id=parent_event_id,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the event log file.

        After close(), the active segment becomes a closed segment
        (eligible for S3 upload).
        """
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
