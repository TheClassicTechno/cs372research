"""Event log loading and query utilities.

Provides:
- ``load_event_log(path)`` — parse events from segments or legacy events.jsonl
- ``filter_events(...)`` — filter by any combination of fields
- ``get_pre_intervention_crit(events)`` — extract the CRIT snapshot
  taken BEFORE an intervention fired (the key query that was impossible
  with the v1 logger)
- ``get_llm_calls(events)`` — extract all LLM call events
- ``get_decision_boundaries(events)`` — extract all decision boundary events
- ``get_state_snapshots(events)`` — extract all state snapshot events
- ``get_intervention_timeline(events)`` — extract intervention event sequence

Supports two layouts:
  - Segment-based: ``run_dir/events/segment_*.jsonl``
  - Legacy single-file: ``run_dir/events.jsonl``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load events from a single JSONL file."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Malformed JSON on line {line_num} of {path.name}: {e.msg}",
                    e.doc, e.pos,
                ) from e
    return events


def load_event_log(path: Path | str) -> list[dict[str, Any]]:
    """Load events into a list of dicts, ordered by logical_clock.

    ``path`` can be:
    - A run directory containing ``events/segment_*.jsonl`` (new format)
    - A run directory containing ``events.jsonl`` (legacy format)
    - A direct path to an ``events.jsonl`` file (legacy format)

    The loader tries segment-based loading first, then falls back to
    legacy single-file loading.

    Returns:
        List of event dicts sorted by logical_clock.

    Raises:
        FileNotFoundError: If no event files are found.
        json.JSONDecodeError: If any line is malformed JSON.
    """
    path = Path(path)

    # If path points directly to a .jsonl file, load it
    if path.is_file() and path.suffix == ".jsonl":
        events = _load_jsonl(path)
        events.sort(key=lambda e: e.get("logical_clock", 0))
        return events

    # path is a directory — try segment-based first
    if path.is_dir():
        events_dir = path / "events"
        segments = sorted(events_dir.glob("segment_*.jsonl")) if events_dir.is_dir() else []

        if segments:
            # Segment-based loading: concatenate all segments in order
            all_events: list[dict[str, Any]] = []
            for seg_path in segments:
                all_events.extend(_load_jsonl(seg_path))
            all_events.sort(key=lambda e: e.get("logical_clock", 0))
            return all_events

        # Fall back to legacy events.jsonl in the directory
        legacy_path = path / "events.jsonl"
        if legacy_path.exists():
            events = _load_jsonl(legacy_path)
            events.sort(key=lambda e: e.get("logical_clock", 0))
            return events

        raise FileNotFoundError(
            f"No event files found in {path} "
            f"(checked events/segment_*.jsonl and events.jsonl)"
        )

    # path doesn't exist or isn't a file/directory we understand
    raise FileNotFoundError(f"Event log path not found: {path}")


def filter_events(
    events: list[dict[str, Any]],
    *,
    event_type: Optional[str] = None,
    phase: Optional[str] = None,
    role: Optional[str] = None,
    call_context: Optional[str] = None,
    boundary_type: Optional[str] = None,
    snapshot_type: Optional[str] = None,
    round_num: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Filter events by any combination of fields.

    All filters are AND-combined.  None means "don't filter on this field".
    """
    result = events
    if event_type is not None:
        result = [e for e in result if e.get("event_type") == event_type]
    if phase is not None:
        result = [e for e in result if e.get("phase") == phase]
    if role is not None:
        result = [e for e in result if e.get("role") == role]
    if call_context is not None:
        result = [e for e in result if e.get("call_context") == call_context]
    if boundary_type is not None:
        result = [e for e in result if e.get("boundary_type") == boundary_type]
    if snapshot_type is not None:
        result = [e for e in result if e.get("snapshot_type") == snapshot_type]
    if round_num is not None:
        result = [e for e in result if e.get("round_num") == round_num]
    return result


def get_llm_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all LLM call events."""
    return filter_events(events, event_type="llm_call")


def get_decision_boundaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all decision boundary events."""
    return filter_events(events, event_type="decision_boundary")


def get_state_snapshots(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all state snapshot events."""
    return filter_events(events, event_type="state_snapshot")


def get_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all error events."""
    return filter_events(events, event_type="error")


def get_pre_intervention_crit(
    events: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Get the CRIT state snapshot taken BEFORE the intervention fired.

    This is the key query that was impossible with the v1 logger.
    The v1 logger overwrites CRIT/<agent>/response.txt on re-scoring,
    so the pre-intervention CRIT scores (e.g., CA=0.68) are lost.

    Returns the first 'post_crit' state_snapshot that is immediately
    followed (in logical_clock order) by an 'intervention_fire'
    decision_boundary event.

    Returns:
        The pre-intervention CRIT state snapshot dict, or None if no
        intervention was fired in this debate.
    """
    for i, evt in enumerate(events):
        if (evt.get("event_type") == "state_snapshot"
                and evt.get("snapshot_type") == "post_crit"):
            # Look ahead for the next decision boundary
            for j in range(i + 1, len(events)):
                next_evt = events[j]
                if next_evt.get("event_type") == "decision_boundary":
                    if next_evt.get("boundary_type") == "intervention_fire":
                        return evt  # Pre-intervention CRIT snapshot
                    if next_evt.get("boundary_type") in (
                        "intervention_eval", "intervention_skip",
                    ):
                        break  # Intervention was evaluated but didn't fire
                    break  # Different boundary type
    return None


def get_intervention_timeline(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract the full intervention event sequence.

    Returns all events related to interventions, in logical_clock order:
    - decision_boundary events with boundary_type starting with "intervention_"
    - state_snapshot events with snapshot_type containing "intervention" or "crit_retry"
    - llm_call events with call_context "intervention_retry" or "crit_retry"
    """
    result = []
    for evt in events:
        etype = evt.get("event_type")
        if etype == "decision_boundary":
            bt = evt.get("boundary_type", "")
            if bt.startswith("intervention_"):
                result.append(evt)
        elif etype == "state_snapshot":
            st = evt.get("snapshot_type", "")
            if "intervention" in st or "crit_retry" in st:
                result.append(evt)
        elif etype == "llm_call":
            cc = evt.get("call_context", "")
            if cc in ("intervention_retry", "crit_retry"):
                result.append(evt)
    return result


def get_run_metadata(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Extract the run_metadata event (should be first)."""
    for evt in events:
        if evt.get("event_type") == "run_metadata":
            return evt
    return None


def validate_event_ordering(events: list[dict[str, Any]]) -> list[str]:
    """Validate that events have strictly increasing logical_clock values.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    prev_clock = -1
    for i, evt in enumerate(events):
        clock = evt.get("logical_clock", -1)
        if clock <= prev_clock:
            errors.append(
                f"Event {i}: logical_clock={clock} is not greater than "
                f"previous={prev_clock} (event_type={evt.get('event_type')})"
            )
        prev_clock = clock
    return errors


def count_events_by_type(events: list[dict[str, Any]]) -> dict[str, int]:
    """Count events by event_type."""
    counts: dict[str, int] = {}
    for evt in events:
        etype = evt.get("event_type", "unknown")
        counts[etype] = counts.get(etype, 0) + 1
    return counts
