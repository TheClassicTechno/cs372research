"""Perfect Replay Contract — deterministic causal replay of the LLM interaction trace.

Guarantees:
- Exact 1:1 key-based event matching (no fallbacks)
- Thread-safe for parallel_agents=True
- call_context strictly enforced
- Prompt hash + execution parameter verification
- Exhaustive consumption (no unused events)
- Failure on missing, duplicate, or context-corrupted events

Uses a corrected events.jsonl fixture (with auto-tracked call_index)
to replay through the full pipeline and verify equivalence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from logging_v2.loader import (
    get_decision_boundaries,
    get_intervention_timeline,
    get_llm_calls,
    get_pre_intervention_crit,
    get_run_metadata,
    load_event_log,
    validate_event_ordering,
)
from logging_v2.replay_strict import (
    ExecutionParamMismatchError,
    PromptDriftError,
    ReplayDiffError,
    ReplayExhaustedError,
    ReplayIncompleteError,
    ReplayKeyMismatchError,
    StrictReplayLLM,
)

# ── Constants ─────────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "perfect_replay_fixture"


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_events():
    """Load the corrected events.jsonl fixture."""
    return load_event_log(FIXTURE_DIR / "events.jsonl")


@pytest.fixture(scope="module")
def fixture_observation():
    """Reconstruct the Observation from fixture data."""
    from multi_agent.models import MarketState, Observation, PortfolioState

    obs_data = json.loads((FIXTURE_DIR / "observation.json").read_text())
    memo_text = (FIXTURE_DIR / "memo.txt").read_text()

    return Observation(
        universe=obs_data["universe"],
        timestamp=obs_data["timestamp"],
        market_state=MarketState(prices=obs_data["prices"]),
        portfolio_state=PortfolioState(cash=obs_data["cash"], positions={}),
        text_context=memo_text,
    )


@pytest.fixture(scope="module")
def fixture_config(fixture_events):
    """Build DebateConfig from the fixture's run_metadata snapshot."""
    from multi_agent.config import DebateConfig

    meta = get_run_metadata(fixture_events)
    cfg = meta["config_snapshot"]

    return DebateConfig(
        roles=cfg["roles"],
        max_rounds=cfg.get("max_rounds", 1),
        propose_only=cfg.get("propose_only", False),
        judge_type=cfg.get("judge_type", "llm"),
        model_name=cfg.get("model_name", "gpt-4.1-nano"),
        llm_provider=cfg.get("llm_provider", "openai"),
        temperature=cfg.get("temperature", 0.3),
        parallel_agents=cfg.get("parallel_agents", True),
        mock=False,
        verbose=False,
        console_display=False,
        logging_mode="off",
        event_logging=True,
        event_logging_store_full_text=True,
        agent_profiles=cfg.get("agent_profiles", {}),
        agent_profile_names=cfg.get("agent_profile_names", {}),
        judge_profile=cfg.get("judge_profile", {}),
        intervention_config=cfg.get("intervention_config"),
        crit_model_name=cfg.get("model_name", "gpt-4.1-nano"),
        crit_system_template=cfg.get("crit_system_template", ""),
        crit_user_template=cfg.get("crit_user_template", ""),
        sector_config=cfg.get("sector_config"),
        allocation_constraints=cfg.get("allocation_constraints"),
    )


@pytest.fixture(scope="module")
def perfect_replay_run(fixture_events, fixture_config, fixture_observation, tmp_path_factory):
    """Run the full pipeline with StrictReplayLLM and return results.

    This is the central fixture: it replays every LLM call from the
    corrected events.jsonl through the actual pipeline, verifying
    key + call_context + prompt hash + execution parameter matching
    at every step.
    """
    from unittest.mock import patch

    tmp_path = tmp_path_factory.mktemp("perfect_replay")

    replay_llm = StrictReplayLLM(fixture_events, strict=True)

    # TODO: Replace monkeypatching with explicit LLM injection via runner/pipeline.
    with (
        patch("multi_agent.graph.nodes._call_llm", replay_llm),
        patch("multi_agent.runner._call_llm", replay_llm),
        patch("logging_v2.event_logger._DEFAULT_BASE_DIR", tmp_path),
    ):
        from multi_agent.runner import MultiAgentRunner
        runner = MultiAgentRunner(fixture_config)
        state = runner.run_returning_state(fixture_observation)

    # Load replay-generated events
    replay_event_files = list(tmp_path.rglob("events.jsonl"))
    replay_events = load_event_log(replay_event_files[0]) if replay_event_files else []

    return {
        "state": state,
        "replay_llm": replay_llm,
        "replay_events": replay_events,
        "original_events": fixture_events,
    }


# ── Test Classes ──────────────────────────────────────────────────────


class TestExactReplayEquivalence:
    """Verify the pipeline replays to completion with exact event matching."""

    def test_replay_completes_without_error(self, perfect_replay_run):
        """Pipeline runs to completion under strict replay."""
        assert perfect_replay_run["state"] is not None

    def test_all_events_consumed(self, perfect_replay_run):
        """Every stored LLM event was consumed during replay."""
        perfect_replay_run["replay_llm"].assert_all_consumed()

    def test_exact_call_count(self, perfect_replay_run):
        """Replay made exactly as many calls as the fixture has events."""
        replay_llm = perfect_replay_run["replay_llm"]
        assert len(replay_llm.calls) == replay_llm.event_count

    def test_replay_keys_match_original(self, perfect_replay_run):
        """The set of (phase, role, call_index, call_context) keys is identical."""
        original_llm_calls = get_llm_calls(perfect_replay_run["original_events"])
        original_keys = {
            (e["phase"], e["role"], e["call_index"], e.get("call_context", "initial"))
            for e in original_llm_calls
        }
        replay_keys = {
            (c["phase"], c["role"], c["call_index"], c["call_context"])
            for c in perfect_replay_run["replay_llm"].calls
        }
        assert replay_keys == original_keys


class TestInterventionBoundaryMatch:
    """Verify intervention fires correctly during replay."""

    def test_intervention_fires_during_replay(self, perfect_replay_run):
        """An intervention_fire event must exist in replay events."""
        fires = [
            e for e in perfect_replay_run["replay_events"]
            if e.get("event_type") == "decision_boundary"
            and e.get("boundary_type") == "intervention_fire"
        ]
        assert len(fires) >= 1, "No intervention_fire in replay events"

    def test_intervention_targets_correct_agents(self, perfect_replay_run):
        """Intervention fire targets should include agents with CA < threshold."""
        fires = [
            e for e in perfect_replay_run["replay_events"]
            if e.get("event_type") == "decision_boundary"
            and e.get("boundary_type") == "intervention_fire"
        ]
        assert fires, "No intervention_fire events found"
        for fire in fires:
            # Targets are in inputs.target_roles or outputs.roles_receiving_nudge
            target_roles = (
                fire.get("inputs", {}).get("target_roles", [])
                or fire.get("outputs", {}).get("roles_receiving_nudge", [])
            )
            assert len(target_roles) >= 1, (
                f"intervention_fire at lc={fire.get('logical_clock')} has no targets"
            )

    def test_pre_intervention_crit_preserved(self, perfect_replay_run):
        """Replay-generated post_crit snapshot precedes intervention_fire.

        Note: get_pre_intervention_crit() requires llm_call events to be
        absent between post_crit and intervention_fire.  Replay events
        lack llm_call events (monkeypatched), so the query works directly.
        If it returns None, verify the structure manually.
        """
        replay_events = perfect_replay_run["replay_events"]
        pre_crit = get_pre_intervention_crit(replay_events)
        if pre_crit is not None:
            # Ideal case: function found it
            crit_scores = pre_crit.get("crit_scores", {})
            has_low_ca = any(
                data.get("pillars", {}).get("causal_alignment", 1.0) < 0.75
                for data in crit_scores.values()
            )
            assert has_low_ca, (
                f"Pre-intervention CRIT has no agent with CA < 0.75: "
                f"{json.dumps(crit_scores, indent=2)}"
            )
        else:
            # Verify structurally: post_crit snapshot exists AND
            # intervention_fire exists after it
            post_crit = [
                e for e in replay_events
                if e.get("event_type") == "state_snapshot"
                and e.get("snapshot_type") == "post_crit"
            ]
            fires = [
                e for e in replay_events
                if e.get("boundary_type") == "intervention_fire"
            ]
            assert post_crit, "No post_crit snapshot in replay events"
            assert fires, "No intervention_fire in replay events"
            assert post_crit[0]["logical_clock"] < fires[0]["logical_clock"], (
                "post_crit must precede intervention_fire"
            )

    def test_intervention_trace_exact_match(self, perfect_replay_run):
        """Intervention boundary + snapshot events match between original and replay.

        Note: llm_call events (intervention_retry, crit_retry) only appear
        in the original (real _call_llm logs them) but not in replay
        (monkeypatched).  We compare only boundary + snapshot events.
        """
        orig = get_intervention_timeline(perfect_replay_run["original_events"])
        replay = get_intervention_timeline(perfect_replay_run["replay_events"])

        def _non_llm(events):
            return [e for e in events if e.get("event_type") != "llm_call"]

        def _trace_key(e):
            return (
                e.get("boundary_type", e.get("snapshot_type")),
                e.get("decision"),
                e.get("stage"),
            )

        orig_set = {_trace_key(e) for e in _non_llm(orig)}
        replay_set = {_trace_key(e) for e in _non_llm(replay)}
        assert replay_set == orig_set


class TestPromptHashMatch:
    """Verify all prompt hashes match during replay."""

    def test_all_prompt_hashes_match(self, perfect_replay_run):
        """Every call in replay_llm.calls has matching prompt hashes."""
        for call in perfect_replay_run["replay_llm"].calls:
            assert call["system_hash_match"], (
                f"System prompt hash mismatch for key={call['key']}"
            )
            assert call["user_hash_match"], (
                f"User prompt hash mismatch for key={call['key']}"
            )


class TestExecutionParameterMatch:
    """Verify model_name, temperature, provider match stored values."""

    def test_model_params_match(self, perfect_replay_run):
        """Every call uses the same model/temp/provider as the stored event."""
        original_llm_calls = get_llm_calls(perfect_replay_run["original_events"])
        original_by_key = {}
        for e in original_llm_calls:
            key = (e.get("round_num", 0), e.get("phase", ""),
                   e.get("role", ""), e.get("call_index", 0))
            original_by_key[key] = e

        for call in perfect_replay_run["replay_llm"].calls:
            original = original_by_key[call["key"]]
            assert call["model_name"] == original.get("model_name", ""), (
                f"model_name mismatch for key={call['key']}"
            )
            assert abs(call["temperature"] - original.get("temperature", 0.0)) < 1e-6, (
                f"temperature mismatch for key={call['key']}"
            )
            assert call["provider"] == original.get("provider", ""), (
                f"provider mismatch for key={call['key']}"
            )


class TestSequenceIntegrity:
    """Verify logical_clock ordering in replay-generated events."""

    def test_logical_clock_strictly_monotonic(self, perfect_replay_run):
        """validate_event_ordering() returns no errors."""
        errors = validate_event_ordering(perfect_replay_run["replay_events"])
        assert not errors, f"Ordering errors: {errors}"

    def test_sequence_indices_contiguous(self, perfect_replay_run):
        """Logical clock values form [1, 2, 3, ...]."""
        clocks = [
            e.get("logical_clock", 0)
            for e in perfect_replay_run["replay_events"]
        ]
        expected = list(range(1, len(clocks) + 1))
        assert clocks == expected, (
            f"Non-contiguous logical_clock: got {clocks[:10]}..., "
            f"expected {expected[:10]}..."
        )


class TestBoundarySequenceEquality:
    """Verify decision boundary sets are identical."""

    def test_boundary_set_exact_match(self, perfect_replay_run):
        """Decision boundary (type, decision, stage) sets match."""
        orig_boundaries = get_decision_boundaries(
            perfect_replay_run["original_events"]
        )
        replay_boundaries = get_decision_boundaries(
            perfect_replay_run["replay_events"]
        )
        orig_set = {
            (b["boundary_type"], b["decision"], b.get("stage"))
            for b in orig_boundaries
        }
        replay_set = {
            (b["boundary_type"], b["decision"], b.get("stage"))
            for b in replay_boundaries
        }
        assert replay_set == orig_set


class TestCallContextEnforcement:
    """Verify call_context mismatch causes failure."""

    def test_call_context_mismatch_fails(self, fixture_events):
        """Corrupting call_context in events causes ReplayKeyMismatchError."""
        corrupted = copy.deepcopy(fixture_events)
        # Find an intervention_retry call and change its context to "initial"
        found = False
        for e in corrupted:
            if (
                e.get("event_type") == "llm_call"
                and e.get("call_context") == "intervention_retry"
            ):
                e["call_context"] = "initial"
                found = True
                break
        assert found, "No intervention_retry call found in fixture to corrupt"

        replay = StrictReplayLLM(corrupted, strict=True)

        # Simulate the call sequence to hit the mismatch.
        # The caller sets call_context="intervention_retry" but the
        # corrupted event expects "initial".
        original_llm_calls = get_llm_calls(fixture_events)
        with pytest.raises(ReplayKeyMismatchError, match="call_context mismatch"):
            for evt in original_llm_calls:
                config = {
                    "_call_context": evt.get("call_context", "initial"),
                    "model_name": evt.get("model_name", ""),
                    "temperature": evt.get("temperature", 0.0),
                    "llm_provider": evt.get("provider", ""),
                }
                replay(
                    config,
                    evt.get("system_prompt", ""),
                    evt.get("user_prompt", ""),
                    role=evt.get("role", ""),
                    phase=evt.get("phase", ""),
                    round_num=evt.get("round_num", 0),
                )


class TestFailureOnCorruptedInput:
    """Verify strict failures on missing or duplicate events."""

    def test_fail_on_missing_event(self, fixture_events):
        """Removing one llm_call causes ReplayExhaustedError."""
        modified = list(fixture_events)
        llm_indices = [
            i for i, e in enumerate(modified)
            if e.get("event_type") == "llm_call"
        ]
        assert len(llm_indices) >= 2
        del modified[llm_indices[1]]

        replay = StrictReplayLLM(modified, strict=True)
        original_llm_calls = get_llm_calls(fixture_events)

        with pytest.raises((ReplayExhaustedError, ReplayIncompleteError)):
            for evt in original_llm_calls:
                config = {
                    "_call_context": evt.get("call_context", "initial"),
                    "model_name": evt.get("model_name", ""),
                    "temperature": evt.get("temperature", 0.0),
                    "llm_provider": evt.get("provider", ""),
                }
                replay(
                    config,
                    evt.get("system_prompt", ""),
                    evt.get("user_prompt", ""),
                    role=evt.get("role", ""),
                    phase=evt.get("phase", ""),
                    round_num=evt.get("round_num", 0),
                )
            # If all calls succeeded, the removed event is unconsumed
            replay.assert_all_consumed()

    def test_fail_on_duplicate_keys(self, fixture_events):
        """Duplicate replay keys in constructor raises ValueError."""
        modified = copy.deepcopy(fixture_events)
        first_llm = next(e for e in modified if e.get("event_type") == "llm_call")
        dup = copy.deepcopy(first_llm)
        dup["logical_clock"] = 9999
        modified.append(dup)

        with pytest.raises(ValueError, match="Duplicate replay key"):
            StrictReplayLLM(modified, strict=True)

    def test_fail_on_extra_runtime_call(self, fixture_events):
        """Extra runtime call beyond stored events raises ReplayExhaustedError."""
        replay = StrictReplayLLM(fixture_events, strict=True)

        # Consume all events
        original_llm_calls = get_llm_calls(fixture_events)
        for evt in original_llm_calls:
            config = {
                "_call_context": evt.get("call_context", "initial"),
                "model_name": evt.get("model_name", ""),
                "temperature": evt.get("temperature", 0.0),
                "llm_provider": evt.get("provider", ""),
            }
            replay(
                config,
                evt.get("system_prompt", ""),
                evt.get("user_prompt", ""),
                role=evt.get("role", ""),
                phase=evt.get("phase", ""),
                round_num=evt.get("round_num", 0),
            )

        # One more call with a key that doesn't exist
        with pytest.raises(ReplayExhaustedError):
            replay(
                {"_call_context": "initial", "model_name": "", "temperature": 0.0, "llm_provider": ""},
                "sys", "usr", role="macro", phase="propose", round_num=99,
            )


class TestDebugMode:
    """Verify debug mode (strict=False) collects diffs instead of raising."""

    def test_debug_mode_collects_diffs(self, fixture_events):
        """strict=False collects mismatches instead of raising."""
        # Corrupt a call_context to induce a mismatch
        corrupted = copy.deepcopy(fixture_events)
        for e in corrupted:
            if e.get("event_type") == "llm_call" and e.get("call_context") == "intervention_retry":
                e["call_context"] = "WRONG_CONTEXT"
                break

        replay = StrictReplayLLM(corrupted, strict=False)
        original_llm_calls = get_llm_calls(fixture_events)

        for evt in original_llm_calls:
            config = {
                "_call_context": evt.get("call_context", "initial"),
                "model_name": evt.get("model_name", ""),
                "temperature": evt.get("temperature", 0.0),
                "llm_provider": evt.get("provider", ""),
            }
            replay(
                config,
                evt.get("system_prompt", ""),
                evt.get("user_prompt", ""),
                role=evt.get("role", ""),
                phase=evt.get("phase", ""),
                round_num=evt.get("round_num", 0),
            )

        assert len(replay.diffs) > 0, "Expected diffs from corrupted call_context"

    def test_debug_mode_fails_at_end(self, fixture_events):
        """assert_no_diffs() raises ReplayDiffError when diffs exist."""
        corrupted = copy.deepcopy(fixture_events)
        for e in corrupted:
            if e.get("event_type") == "llm_call" and e.get("call_context") == "intervention_retry":
                e["call_context"] = "WRONG_CONTEXT"
                break

        replay = StrictReplayLLM(corrupted, strict=False)
        original_llm_calls = get_llm_calls(fixture_events)

        for evt in original_llm_calls:
            config = {
                "_call_context": evt.get("call_context", "initial"),
                "model_name": evt.get("model_name", ""),
                "temperature": evt.get("temperature", 0.0),
                "llm_provider": evt.get("provider", ""),
            }
            replay(
                config,
                evt.get("system_prompt", ""),
                evt.get("user_prompt", ""),
                role=evt.get("role", ""),
                phase=evt.get("phase", ""),
                round_num=evt.get("round_num", 0),
            )

        with pytest.raises(ReplayDiffError):
            replay.assert_no_diffs()
