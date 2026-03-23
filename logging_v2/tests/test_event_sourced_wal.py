"""Tests for the event-sourced WAL and artifact storage system.

Verifies:
1. WAL events are append-only and ordered
2. Each event has event_id and logical_clock
3. Artifact files are written correctly
4. WAL correctly references artifact paths
5. No large payloads in WAL
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from logging_v2.event_logger import EventLogger
from logging_v2.events import VALID_EVENT_TYPES, make_event, SCHEMA_VERSION
from logging_v2.loader import load_event_log


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_runs"


@pytest.fixture
def logger(run_dir: Path) -> EventLogger:
    el = EventLogger(
        experiment="test_exp",
        run_id="run_001",
        debate_id="debate_abc",
        base_dir=run_dir,
    )
    yield el
    el.close()


# ── 1. WAL events are append-only and ordered ────────────────────────


class TestWALOrdering:
    def test_logical_clock_strictly_increases(self, logger: EventLogger):
        logger.log_run_metadata(
            config_snapshot={"model": "test"},
            start_time="2026-01-01T00:00:00Z",
        )
        logger.log_llm_call(
            phase="propose", role="agent_a", call_index=0,
            call_context="initial", system_prompt="sys", user_prompt="usr",
            response="resp", model_name="gpt-4", provider="openai",
            temperature=0.7, latency_ms=100.0, round_num=1,
        )
        logger.log_error(
            error_type="test_error", message="boom", round_num=1,
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        clocks = [e["logical_clock"] for e in events]
        assert clocks == sorted(clocks)
        assert len(set(clocks)) == len(clocks), "Clocks must be unique"
        for i in range(1, len(clocks)):
            assert clocks[i] > clocks[i - 1]

    def test_events_are_append_only(self, logger: EventLogger, run_dir: Path):
        """Events written to segment files are append-only JSONL."""
        for i in range(5):
            logger.emit(
                "control_flow",
                {"decision": "continue", "reason": f"step {i}"},
                context={"workflow": "test"},
            )
        logger.close()

        segments = logger.get_all_segments()
        assert len(segments) >= 1
        for seg in segments:
            lines = seg.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                evt = json.loads(line)
                assert "event_id" in evt
                assert "logical_clock" in evt


# ── 2. Each event has event_id and logical_clock ─────────────────────


class TestEventSchema:
    def test_v2_event_has_required_fields(self):
        event = make_event(
            event_type="control_flow",
            run_id="run_001",
            logical_clock=1,
            payload={"decision": "continue", "reason": "test"},
            context={"workflow": "test_wf", "stage": "init"},
        )
        assert "event_id" in event
        assert "run_id" in event
        assert "event_type" in event
        assert "timestamp" in event
        assert "logical_clock" in event
        assert "context" in event
        assert "payload" in event
        assert event["context"]["workflow"] == "test_wf"
        assert event["context"]["stage"] == "init"
        assert event["context"]["agent_id"] is None

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValueError, match="Invalid event_type"):
            make_event(
                event_type="not_a_valid_type",
                run_id="run_001",
                logical_clock=1,
                payload={},
            )

    def test_all_required_event_types_exist(self):
        expected = {
            "llm_call_start", "llm_call_end", "artifact_created",
            "evaluation_computed", "control_flow", "state_snapshot", "error",
        }
        assert expected.issubset(VALID_EVENT_TYPES)

    def test_schema_version_is_v2(self):
        assert SCHEMA_VERSION == "v2"

    def test_emit_produces_event_with_id_and_clock(self, logger: EventLogger):
        event_id = logger.emit(
            "control_flow",
            {"decision": "stop", "reason": "done"},
        )
        assert event_id  # non-empty string
        logger.close()

        events = load_event_log(logger.run_dir)
        assert len(events) == 1
        evt = events[0]
        assert evt["event_id"] == event_id
        assert evt["logical_clock"] == 1


# ── 3. Artifact files are written correctly ──────────────────────────


class TestArtifactStorage:
    def test_artifact_written_on_llm_call(self, logger: EventLogger):
        logger.log_llm_call(
            phase="propose", role="agent_a", call_index=0,
            call_context="initial",
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            response="4",
            model_name="gpt-4", provider="openai",
            temperature=0.0, latency_ms=50.0, round_num=1,
        )
        logger.close()

        artifacts = list(logger.artifacts_dir.glob("*.json"))
        assert len(artifacts) == 1

        data = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert data["model"] == "gpt-4"
        assert data["system_prompt"] == "You are a helpful assistant."
        assert data["user_prompt"] == "What is 2+2?"
        assert data["response"] == "4"
        assert "call_id" in data
        assert "timestamp" in data
        assert "metadata" in data
        assert data["metadata"]["latency_ms"] == 50

    def test_artifact_sequential_naming(self, logger: EventLogger):
        for i in range(3):
            logger.log_llm_call(
                phase="propose", role=f"agent_{i}", call_index=0,
                call_context="initial",
                system_prompt="sys", user_prompt="usr",
                response=f"resp_{i}",
                model_name="gpt-4", provider="openai",
                temperature=0.0, latency_ms=10.0, round_num=1,
            )
        logger.close()

        artifacts = sorted(logger.artifacts_dir.glob("*.json"))
        assert len(artifacts) == 3
        assert artifacts[0].name == "000000.json"
        assert artifacts[1].name == "000001.json"
        assert artifacts[2].name == "000002.json"

    def test_emit_llm_call_end_writes_artifact(self, logger: EventLogger):
        event_id = logger.emit_llm_call_end(
            system_prompt="You are an analyst.",
            user_prompt="Analyze AAPL.",
            response="AAPL looks bullish.",
            model="claude-3",
            latency_ms=200.0,
            context={"workflow": "analysis", "stage": "propose", "agent_id": "macro"},
        )
        logger.close()

        # Artifact exists
        artifacts = list(logger.artifacts_dir.glob("*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert data["response"] == "AAPL looks bullish."
        assert data["model"] == "claude-3"

        # WAL has artifact_created + llm_call_end events
        events = load_event_log(logger.run_dir)
        types = [e["event_type"] for e in events]
        assert "artifact_created" in types
        assert "llm_call_end" in types


# ── 4. WAL correctly references artifact paths ──────────────────────


class TestWALArtifactLinkage:
    def test_llm_call_event_has_artifact_path(self, logger: EventLogger):
        logger.log_llm_call(
            phase="propose", role="agent_a", call_index=0,
            call_context="initial",
            system_prompt="sys", user_prompt="usr", response="resp",
            model_name="gpt-4", provider="openai",
            temperature=0.0, latency_ms=10.0, round_num=1,
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        llm_events = [e for e in events if e.get("event_type") == "llm_call"]
        assert len(llm_events) == 1
        evt = llm_events[0]
        payload = evt.get("payload", evt)
        assert "artifact_path" in payload
        assert payload["artifact_path"].startswith("artifacts/llm_calls/")
        assert payload["artifact_path"].endswith(".json")

    def test_artifact_path_resolves_to_real_file(self, logger: EventLogger):
        logger.log_llm_call(
            phase="propose", role="agent_a", call_index=0,
            call_context="initial",
            system_prompt="sys", user_prompt="usr", response="resp",
            model_name="gpt-4", provider="openai",
            temperature=0.0, latency_ms=10.0, round_num=1,
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        llm_evt = [e for e in events if e["event_type"] == "llm_call"][0]
        payload = llm_evt.get("payload", llm_evt)
        artifact_full_path = logger.run_dir / payload["artifact_path"]
        assert artifact_full_path.exists()

        data = json.loads(artifact_full_path.read_text(encoding="utf-8"))
        assert data["response"] == "resp"

    def test_v2_llm_call_end_references_artifact(self, logger: EventLogger):
        logger.emit_llm_call_end(
            system_prompt="sys", user_prompt="usr", response="hello",
            model="test-model", latency_ms=5.0,
            context={"workflow": "test"},
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        end_events = [
            e for e in events if e.get("event_type") == "llm_call_end"
        ]
        assert len(end_events) == 1
        evt = end_events[0]
        assert "artifact_path" in evt
        assert "response_hash" in evt

        # Verify artifact exists at referenced path
        artifact_path = logger.run_dir / evt["artifact_path"]
        assert artifact_path.exists()


# ── 5. No large payloads in WAL ──────────────────────────────────────


class TestNoLargePayloads:
    def test_v2_wal_events_are_compact(self, logger: EventLogger):
        """v2-style events should not contain full prompts/responses."""
        large_prompt = "x" * 10_000
        large_response = "y" * 10_000

        logger.emit_llm_call_end(
            system_prompt=large_prompt,
            user_prompt=large_prompt,
            response=large_response,
            model="test-model",
            latency_ms=100.0,
            context={"workflow": "test"},
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        for evt in events:
            serialized = json.dumps(evt)
            assert len(serialized) < 2000, (
                f"WAL event too large ({len(serialized)} bytes): "
                f"event_type={evt.get('event_type')}"
            )

    def test_emit_evaluation_is_compact(self, logger: EventLogger):
        logger.emit_evaluation(
            metric_name="crit",
            scores={"agent_a": {"rho_i": 0.85}},
            context={"workflow": "debate", "stage": "post_crit"},
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        assert len(events) == 1
        serialized = json.dumps(events[0])
        assert len(serialized) < 2000


# ── Directory structure ──────────────────────────────────────────────


class TestDirectoryStructure:
    def test_run_dir_structure(self, logger: EventLogger):
        logger.log_llm_call(
            phase="propose", role="agent_a", call_index=0,
            call_context="initial",
            system_prompt="sys", user_prompt="usr", response="resp",
            model_name="gpt-4", provider="openai",
            temperature=0.0, latency_ms=10.0, round_num=1,
        )
        logger.close()

        assert (logger.run_dir / "events").is_dir()
        assert (logger.run_dir / "artifacts").is_dir()
        assert (logger.run_dir / "artifacts" / "llm_calls").is_dir()
        assert len(list((logger.run_dir / "events").glob("segment_*.jsonl"))) >= 1
        assert len(list((logger.run_dir / "artifacts" / "llm_calls").glob("*.json"))) == 1


# ── Context handling ─────────────────────────────────────────────────


class TestContextHandling:
    def test_context_is_workflow_agnostic(self, logger: EventLogger):
        logger.emit(
            "control_flow",
            {"decision": "continue", "reason": "test"},
            context={
                "workflow": "codegen",
                "stage": "planning",
                "agent_id": "planner_01",
            },
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        evt = events[0]
        # Loader flattens context fields to top level
        assert evt["workflow"] == "codegen"
        assert evt["stage"] == "planning"
        assert evt["agent_id"] == "planner_01"

    def test_context_defaults_to_none(self, logger: EventLogger):
        logger.emit("error", {"message": "oops"})
        logger.close()

        events = load_event_log(logger.run_dir)
        evt = events[0]
        # None context values are omitted by the flattener
        assert evt.get("workflow") is None
        assert evt.get("stage") is None
        assert evt.get("agent_id") is None


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompat:
    def test_legacy_log_methods_still_work(self, logger: EventLogger):
        logger.log_run_metadata(
            config_snapshot={"model": "gpt-4"},
            start_time="2026-01-01T00:00:00Z",
            roles=["macro", "quant"],
        )
        logger.log_llm_call(
            phase="propose", role="macro", call_index=0,
            call_context="initial",
            system_prompt="sys", user_prompt="usr", response="resp",
            model_name="gpt-4", provider="openai",
            temperature=0.7, latency_ms=100.0, round_num=1,
        )
        logger.log_state_snapshot(
            snapshot_type="post_propose",
            allocations={"macro": {"AAPL": 0.5, "GOOG": 0.5}},
            round_num=1,
        )
        logger.log_decision_boundary(
            boundary_type="convergence_check",
            stage="post_revision",
            inputs={"js_div": 0.01},
            decision="converged",
            outputs={},
            round_num=1,
        )
        logger.log_error(
            error_type="parse_failure",
            message="bad json",
            round_num=1,
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        types = [e["event_type"] for e in events]
        assert "run_metadata" in types
        assert "llm_call" in types
        assert "state_snapshot" in types
        assert "decision_boundary" in types
        assert "error" in types

    def test_legacy_llm_call_compact_wal_with_artifact(
        self, logger: EventLogger,
    ):
        """log_llm_call writes compact WAL (no full text) + artifact."""
        logger.log_llm_call(
            phase="propose", role="macro", call_index=0,
            call_context="initial",
            system_prompt="sys prompt", user_prompt="usr prompt",
            response="the response",
            model_name="gpt-4", provider="openai",
            temperature=0.7, latency_ms=100.0, round_num=1,
        )
        logger.close()

        events = load_event_log(logger.run_dir)
        llm = [e for e in events if e["event_type"] == "llm_call"][0]
        # WAL should NOT have full text (flattened view)
        assert "system_prompt" not in llm or llm.get("system_prompt") == ""
        assert "user_prompt" not in llm or llm.get("user_prompt") == ""
        # WAL should have hashes and artifact link
        assert "response_hash" in llm
        assert "artifact_path" in llm
        # Artifact should have full text
        artifact_path = logger.run_dir / llm["artifact_path"]
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert data["response"] == "the response"
