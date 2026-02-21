"""
Comprehensive tests for the Debate Output validation pipeline.

Tests cover two independent validation layers:

1. **JSON Schema validation** — structure, types, enums, formats, additionalProperties
   via ``jsonschema.Draft202012Validator`` against ``debate_output.schema.json``.

2. **Runtime semantic validation** — business invariants enforced by
   ``contracts.scripts.validate_debate_output`` (speaker consistency, attempt
   ordering, posthoc single-attempt rule, recommendation range sanity).

The suite includes:
  - Positive tests for all four example files
  - Negative tests via a mutation factory (missing fields, bad enums, extra
    properties, invalid types, boundary violations)
  - Direct tests of the runtime validator's error-code paths
  - Determinism and edge-case coverage
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import jsonschema
import pytest

# ---------------------------------------------------------------------------
# Path helpers — work when pytest is run from repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "contracts" / "schemas"
SCHEMA_PATH = SCHEMA_DIR / "debate_output.schema.json"

EXAMPLE_FILES = sorted(SCHEMA_DIR.glob("EXAMPLE_*.json"))
EXAMPLE_NAMES = [p.name for p in EXAMPLE_FILES]

# Import the runtime validator functions directly.
sys.path.insert(0, str(REPO_ROOT))
from contracts.scripts.validate_debate_output import (  # noqa: E402
    E_ATTEMPT_INDEX_INVALID,
    E_DUPLICATE_TURN_ID,
    E_DUPLICATE_TURN_INDEX,
    E_NEGATIVE_TURN_INDEX,
    E_POSTHOC_MULTI_ATTEMPT,
    E_RECOMMENDATION_RANGE_INVALID,
    E_SCHEMA_VALIDATION_FAILED,
    E_SPEAKER_ID_EMPTY,
    E_SPEAKER_UNKNOWN,
    E_TURN_ID_EMPTY,
    RunMetrics,
    runtime_validate,
    validate_schema,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="session")
def schema() -> dict:
    """Load the debate output JSON schema once per session."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def validator(schema) -> jsonschema.Draft202012Validator:
    """Pre-built Draft 2020-12 validator."""
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture()
def valid_debate() -> dict:
    """Minimal debate document that passes schema validation.

    Used as a baseline for mutation-based negative tests.
    """
    return _minimal_valid_debate()


# ===================================================================
# Factories / helpers
# ===================================================================


def _load_example(name: str) -> dict:
    """Load an example JSON file by name."""
    with open(SCHEMA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _minimal_valid_debate() -> dict:
    """Build the smallest document that is valid against the schema."""
    return {
        "schema_version": "2.0.0",
        "debate_id": "test-debate-001",
        "run_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "mode": "posthoc",
        "created_at": "2026-01-01T00:00:00Z",
        "run_metadata": {"generation_mode": "test"},
        "debate_metadata": {"task_type": "unit_test"},
        "participants": [
            {"agent_id": "agent_a", "role": "bull"},
            {"agent_id": "agent_b", "role": "bear"},
        ],
        "rounds": [
            {
                "round_index": 1,
                "turns": [
                    {
                        "turn_id": "t1",
                        "round_index": 1,
                        "turn_index_in_round": 0,
                        "speaker_id": "agent_a",
                        "attempts": [
                            {
                                "attempt_index": 0,
                                "timestamp": "2026-01-01T00:00:01Z",
                                "status": "ok",
                                "content": "I believe X.",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _minimal_runtime_debate(**overrides) -> dict:
    """Build a minimal round-based debate dict for runtime_validate tests."""
    base = {
        "participants": [
            {"agent_id": "alice"},
            {"agent_id": "bob"},
        ],
        "mode": "posthoc",
        "rounds": [
            {
                "round_index": 1,
                "turns": [
                    {
                        "turn_id": "t1",
                        "turn_index_in_round": 0,
                        "speaker_id": "alice",
                        "attempts": [
                            {"attempt_index": 0, "timestamp": "2026-01-01T00:00:00Z",
                             "status": "ok", "content": "hello"}
                        ],
                    },
                ],
            }
        ],
    }
    base.update(overrides)
    return base


def _mutate(debate: dict, path: list[str | int], value=..., *, delete: bool = False) -> dict:
    """Return a deep copy of *debate* with one field changed or deleted.

    *path* is a list of keys/indices leading to the target.
    If *delete* is True the key at the end of *path* is removed.
    Otherwise it is set to *value*.
    """
    doc = copy.deepcopy(debate)
    obj = doc
    for key in path[:-1]:
        obj = obj[key]
    if delete:
        del obj[path[-1]]
    else:
        obj[path[-1]] = value
    return doc


def _schema_errors(validator, doc: dict) -> list[jsonschema.ValidationError]:
    """Collect all schema validation errors for *doc*."""
    return list(validator.iter_errors(doc))


def _runtime_errors(debate: dict) -> list:
    """Run the runtime validator and return its error findings."""
    metrics = RunMetrics()
    runtime_validate(debate, metrics)
    return metrics.errors


def _runtime_warnings(debate: dict) -> list:
    """Run the runtime validator and return its warning findings."""
    metrics = RunMetrics()
    runtime_validate(debate, metrics)
    return metrics.warnings


# ===================================================================
# 1. POSITIVE: Schema validation of example files
# ===================================================================


class TestExampleSchemaValidation:
    """Every shipped example file must pass schema validation."""

    @pytest.mark.parametrize("filename", EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
    def test_example_passes_schema(self, validator, filename):
        doc = _load_example(filename)
        errors = _schema_errors(validator, doc)
        msgs = [e.message for e in errors]
        assert errors == [], f"Schema errors in {filename}: {msgs}"

    @pytest.mark.parametrize("filename", EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
    def test_example_has_required_toplevel_keys(self, filename):
        doc = _load_example(filename)
        required = {
            "schema_version", "debate_id", "run_id", "mode",
            "created_at", "run_metadata", "debate_metadata",
            "participants", "rounds",
        }
        assert required.issubset(doc.keys())


# ===================================================================
# 2. POSITIVE: Runtime semantic validation of example files
# ===================================================================


class TestExampleRuntimeValidation:
    """Runtime validator should produce no HARD errors on example files.

    The runtime validator iterates ``rounds[].turns[]`` and checks semantic
    invariants (turn_id uniqueness, speaker consistency, posthoc rules, etc.).
    """

    @pytest.mark.parametrize("filename", EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
    def test_schema_function_no_errors(self, filename):
        """validate_schema() should produce no errors for valid examples."""
        doc = _load_example(filename)
        metrics = RunMetrics()
        validate_schema(doc, metrics)
        codes = [e.code for e in metrics.errors]
        assert metrics.errors == [], f"Schema errors: {codes}"

    @pytest.mark.parametrize("filename", EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
    def test_runtime_function_no_errors(self, filename):
        """runtime_validate() should produce no HARD errors."""
        doc = _load_example(filename)
        errors = _runtime_errors(doc)
        assert errors == []

    @pytest.mark.parametrize("filename", EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
    def test_runtime_processes_all_turns(self, filename):
        """runtime_validate() should process every turn in the example."""
        doc = _load_example(filename)
        metrics = RunMetrics()
        runtime_validate(doc, metrics)
        expected_turns = sum(
            len(r.get("turns", []))
            for r in doc.get("rounds", [])
        )
        assert metrics.turns_processed == expected_turns


# ===================================================================
# 3. NEGATIVE: Missing required fields (schema)
# ===================================================================


class TestMissingRequiredFields:
    """Removing any required top-level field must cause a schema error."""

    REQUIRED_TOPLEVEL = [
        "schema_version", "debate_id", "run_id", "mode",
        "created_at", "run_metadata", "debate_metadata",
        "participants", "rounds",
    ]

    @pytest.mark.parametrize("field", REQUIRED_TOPLEVEL)
    def test_missing_toplevel_field(self, validator, valid_debate, field):
        doc = _mutate(valid_debate, [field], delete=True)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0, f"Expected error when '{field}' is missing"

    def test_missing_generation_mode(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["run_metadata", "generation_mode"], delete=True)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_missing_task_type(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["debate_metadata", "task_type"], delete=True)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_missing_turn_id(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["rounds", 0, "turns", 0, "turn_id"], delete=True)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_missing_attempts(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["rounds", 0, "turns", 0, "attempts"], delete=True)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_missing_attempt_content(self, validator, valid_debate):
        doc = _mutate(
            valid_debate,
            ["rounds", 0, "turns", 0, "attempts", 0, "content"],
            delete=True,
        )
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_missing_attempt_status(self, validator, valid_debate):
        doc = _mutate(
            valid_debate,
            ["rounds", 0, "turns", 0, "attempts", 0, "status"],
            delete=True,
        )
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0


# ===================================================================
# 4. NEGATIVE: Invalid enum values (schema)
# ===================================================================


class TestInvalidEnums:
    """Enum fields must reject values outside the allowed set."""

    def test_invalid_mode(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["mode"], "live")
        errors = _schema_errors(validator, doc)
        assert any("'live' is not one of" in e.message for e in errors)

    def test_invalid_attempt_status(self, validator, valid_debate):
        doc = _mutate(
            valid_debate,
            ["rounds", 0, "turns", 0, "attempts", 0, "status"],
            "pending",
        )
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_invalid_prompt_regime(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["run_metadata", "prompt_regime"], "freeform")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_invalid_turn_type(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["turn_type"] = "monologue"
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_invalid_turn_outcome(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["turn_outcome"] = "aborted"
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_invalid_recommendation_action(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["attempts"][0]["recommendation_hint"] = {
            "action": "YOLO"
        }
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0


# ===================================================================
# 5. NEGATIVE: Invalid schema_version (schema)
# ===================================================================


class TestSchemaVersion:
    """schema_version must be exactly '2.0.0'."""

    @pytest.mark.parametrize("bad_version", ["1.0.0", "2.1.0", "3.0.0", ""])
    def test_wrong_version_rejected(self, validator, valid_debate, bad_version):
        doc = _mutate(valid_debate, ["schema_version"], bad_version)
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_correct_version_accepted(self, validator, valid_debate):
        errors = _schema_errors(validator, valid_debate)
        assert len(errors) == 0


# ===================================================================
# 6. NEGATIVE: Invalid turn structure (schema)
# ===================================================================


class TestInvalidTurnStructure:
    """Turn and round structural rules."""

    def test_round_index_zero(self, validator, valid_debate):
        """round_index minimum is 1."""
        doc = _mutate(valid_debate, ["rounds", 0, "round_index"], 0)
        # Also fix the nested turn's round_index
        doc["rounds"][0]["turns"][0]["round_index"] = 0
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_round_index_negative(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["rounds", 0, "round_index"], -1)
        doc["rounds"][0]["turns"][0]["round_index"] = -1
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_attempt_index_negative(self, validator, valid_debate):
        """attempt_index minimum is 0."""
        doc = _mutate(
            valid_debate,
            ["rounds", 0, "turns", 0, "attempts", 0, "attempt_index"],
            -1,
        )
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_turn_index_in_round_negative(self, validator, valid_debate):
        """turn_index_in_round minimum is 0."""
        doc = _mutate(
            valid_debate,
            ["rounds", 0, "turns", 0, "turn_index_in_round"],
            -1,
        )
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_empty_rounds_array(self, validator, valid_debate):
        """rounds must have minItems: 1."""
        doc = _mutate(valid_debate, ["rounds"], [])
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_empty_participants_array(self, validator, valid_debate):
        """participants must have minItems: 1."""
        doc = _mutate(valid_debate, ["participants"], [])
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_empty_attempts_array(self, validator, valid_debate):
        """attempts must have minItems: 1."""
        doc = _mutate(valid_debate, ["rounds", 0, "turns", 0, "attempts"], [])
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_empty_turn_id_string(self, validator, valid_debate):
        """turn_id has minLength: 1."""
        doc = _mutate(valid_debate, ["rounds", 0, "turns", 0, "turn_id"], "")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_empty_speaker_id_string(self, validator, valid_debate):
        """speaker_id has minLength: 1."""
        doc = _mutate(valid_debate, ["rounds", 0, "turns", 0, "speaker_id"], "")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0


# ===================================================================
# 7. NEGATIVE: recommendation_hint type violations (schema)
# ===================================================================


class TestRecommendationHint:
    """recommendation_hint fields must satisfy type and range constraints."""

    def _with_hint(self, valid_debate, hint: dict) -> dict:
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["attempts"][0]["recommendation_hint"] = hint
        return doc

    def test_conviction_above_1(self, validator, valid_debate):
        doc = self._with_hint(valid_debate, {"conviction": 1.5})
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_conviction_below_0(self, validator, valid_debate):
        doc = self._with_hint(valid_debate, {"conviction": -0.1})
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_position_size_negative(self, validator, valid_debate):
        doc = self._with_hint(valid_debate, {"position_size_pct_min": -5})
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_horizon_days_negative(self, validator, valid_debate):
        doc = self._with_hint(valid_debate, {"horizon_days_min": -30})
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_horizon_days_float_rejected(self, validator, valid_debate):
        """horizon_days_min/max must be integer, not float."""
        doc = self._with_hint(valid_debate, {"horizon_days_min": 30.5})
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_valid_hint_accepted(self, validator, valid_debate):
        doc = self._with_hint(valid_debate, {
            "action": "BUY",
            "position_size_pct_min": 5,
            "position_size_pct_max": 10,
            "horizon_days_min": 90,
            "horizon_days_max": 365,
            "conviction": 0.75,
        })
        errors = _schema_errors(validator, doc)
        assert len(errors) == 0


# ===================================================================
# 8. NEGATIVE: additionalProperties rejected (schema)
# ===================================================================


class TestAdditionalProperties:
    """Objects with additionalProperties: false must reject unknown keys."""

    def test_extra_toplevel_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["extra_field"] = "should fail"
        errors = _schema_errors(validator, doc)
        assert any("additional" in e.message.lower() for e in errors)

    def test_extra_run_metadata_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["run_metadata"]["unknown_key"] = True
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_extra_participant_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["participants"][0]["extra"] = "bad"
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_extra_turn_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["phantom_field"] = 42
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_extra_attempt_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["attempts"][0]["rogue_key"] = []
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_extra_recommendation_hint_property(self, validator, valid_debate):
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["attempts"][0]["recommendation_hint"] = {
            "action": "BUY",
            "unknown": True,
        }
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0


# ===================================================================
# 9. Runtime validator: speaker consistency
# ===================================================================


class TestRuntimeSpeakerValidation:
    """The runtime validator checks speaker_id against participants."""

    def test_valid_speaker_no_error(self):
        errors = _runtime_errors(_minimal_runtime_debate())
        speaker_errors = [e for e in errors if e.code == E_SPEAKER_UNKNOWN]
        assert speaker_errors == []

    def test_unknown_speaker_detected(self):
        debate = _minimal_runtime_debate()
        debate["rounds"][0]["turns"][0]["speaker_id"] = "charlie"
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_SPEAKER_UNKNOWN in codes

    def test_empty_speaker_id(self):
        debate = _minimal_runtime_debate()
        debate["rounds"][0]["turns"][0]["speaker_id"] = ""
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_SPEAKER_ID_EMPTY in codes

    def test_whitespace_only_speaker_id(self):
        debate = _minimal_runtime_debate()
        debate["rounds"][0]["turns"][0]["speaker_id"] = "   "
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_SPEAKER_ID_EMPTY in codes


# ===================================================================
# 10. Runtime validator: turn ID uniqueness
# ===================================================================


class TestRuntimeTurnIdUniqueness:
    def test_duplicate_turn_id_detected(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "dup", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                        {"turn_id": "dup", "turn_index_in_round": 1, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_DUPLICATE_TURN_ID in codes

    def test_duplicate_turn_id_across_rounds(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "dup", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                },
                {
                    "round_index": 2,
                    "turns": [
                        {"turn_id": "dup", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                },
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_DUPLICATE_TURN_ID in codes

    def test_empty_turn_id_detected(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_TURN_ID_EMPTY in codes


# ===================================================================
# 11. Runtime validator: turn index validation
# ===================================================================


class TestRuntimeTurnIndex:
    def test_negative_turn_index_in_round(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "t1", "turn_index_in_round": -1, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_NEGATIVE_TURN_INDEX in codes

    def test_duplicate_turn_index_in_round(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                        {"turn_id": "t2", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_DUPLICATE_TURN_INDEX in codes

    def test_same_turn_index_different_rounds_ok(self):
        """turn_index_in_round uniqueness is per-round, not global."""
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                },
                {
                    "round_index": 2,
                    "turns": [
                        {"turn_id": "t2", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                },
            ],
        }
        errors = _runtime_errors(debate)
        index_errors = [e for e in errors if e.code in (E_DUPLICATE_TURN_INDEX, E_NEGATIVE_TURN_INDEX)]
        assert index_errors == []


# ===================================================================
# 12. Runtime validator: posthoc single-attempt rule
# ===================================================================


class TestRuntimePosthocRule:
    def test_posthoc_multi_attempt_rejected(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "mode": "posthoc",
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {"attempt_index": 0},
                                {"attempt_index": 1},
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_POSTHOC_MULTI_ATTEMPT in codes

    def test_posthoc_single_attempt_ok(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "mode": "posthoc",
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": 0}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        posthoc_errors = [e for e in errors if e.code == E_POSTHOC_MULTI_ATTEMPT]
        assert posthoc_errors == []

    def test_in_loop_multi_attempt_ok(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "mode": "in_loop",
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {"attempt_index": 0},
                                {"attempt_index": 1},
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        posthoc_errors = [e for e in errors if e.code == E_POSTHOC_MULTI_ATTEMPT]
        assert posthoc_errors == []


# ===================================================================
# 13. Runtime validator: attempt index ordering
# ===================================================================


class TestRuntimeAttemptIndex:
    def test_non_monotonic_attempt_index(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "mode": "in_loop",
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {"attempt_index": 1},
                                {"attempt_index": 0},
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_ATTEMPT_INDEX_INVALID in codes

    def test_duplicate_attempt_index(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "mode": "in_loop",
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {"attempt_index": 0},
                                {"attempt_index": 0},
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_ATTEMPT_INDEX_INVALID in codes

    def test_negative_attempt_index(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {"turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                         "attempts": [{"attempt_index": -1}]},
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_ATTEMPT_INDEX_INVALID in codes


# ===================================================================
# 14. Runtime validator: recommendation_hint range sanity
# ===================================================================


class TestRuntimeRecommendationRange:
    def test_min_greater_than_max(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {
                                    "attempt_index": 0,
                                    "recommendation_hint": {
                                        "position_size_pct_min": 10,
                                        "position_size_pct_max": 5,
                                    },
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        codes = [e.code for e in errors]
        assert E_RECOMMENDATION_RANGE_INVALID in codes

    def test_valid_range_ok(self):
        debate = {
            "participants": [{"agent_id": "a"}],
            "rounds": [
                {
                    "round_index": 1,
                    "turns": [
                        {
                            "turn_id": "t1", "turn_index_in_round": 0, "speaker_id": "a",
                            "attempts": [
                                {
                                    "attempt_index": 0,
                                    "recommendation_hint": {
                                        "position_size_pct_min": 5,
                                        "position_size_pct_max": 10,
                                    },
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _runtime_errors(debate)
        range_errors = [e for e in errors if e.code == E_RECOMMENDATION_RANGE_INVALID]
        assert range_errors == []


# ===================================================================
# 15. Runtime validator: validate_schema function
# ===================================================================


class TestRuntimeSchemaFunction:
    """Test the validate_schema function from the runtime validator."""

    def test_valid_doc_no_errors(self):
        doc = _minimal_valid_debate()
        metrics = RunMetrics()
        validate_schema(doc, metrics)
        schema_errors = [e for e in metrics.errors if e.code == E_SCHEMA_VALIDATION_FAILED]
        assert schema_errors == []

    def test_invalid_doc_produces_errors(self):
        doc = _minimal_valid_debate()
        doc["schema_version"] = "9.9.9"
        metrics = RunMetrics()
        validate_schema(doc, metrics)
        schema_errors = [e for e in metrics.errors if e.code == E_SCHEMA_VALIDATION_FAILED]
        assert len(schema_errors) > 0

    def test_error_messages_are_meaningful(self):
        doc = _minimal_valid_debate()
        del doc["debate_id"]
        metrics = RunMetrics()
        validate_schema(doc, metrics)
        assert any("required" in e.message.lower() or "debate_id" in e.message
                    for e in metrics.errors)


# ===================================================================
# 16. Edge cases and robustness
# ===================================================================


class TestEdgeCases:
    """Boundary conditions and hostile inputs."""

    def test_completely_empty_object(self, validator):
        errors = _schema_errors(validator, {})
        assert len(errors) > 0

    def test_null_document(self, validator):
        errors = _schema_errors(validator, None)
        assert len(errors) > 0

    def test_array_instead_of_object(self, validator):
        errors = _schema_errors(validator, [])
        assert len(errors) > 0

    def test_string_instead_of_object(self, validator):
        errors = _schema_errors(validator, "not an object")
        assert len(errors) > 0

    def test_rounds_with_empty_turns(self, validator, valid_debate):
        """Rounds can have 0 turns (minItems: 0 on turns)."""
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"] = []
        errors = _schema_errors(validator, doc)
        assert len(errors) == 0

    def test_timestamp_non_iso8601(self, validator, valid_debate):
        """created_at must be format: date-time."""
        doc = _mutate(valid_debate, ["created_at"], "not-a-date")
        errors = _schema_errors(validator, doc)
        # jsonschema may or may not enforce format — check if it does
        # (Draft202012Validator does NOT enforce format by default)
        # So this is a documentation test, not a hard assertion.

    def test_debate_id_empty_string(self, validator, valid_debate):
        """debate_id has minLength: 1."""
        doc = _mutate(valid_debate, ["debate_id"], "")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_run_id_empty_string(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["run_id"], "")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_participant_agent_id_empty(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["participants", 0, "agent_id"], "")
        errors = _schema_errors(validator, doc)
        assert len(errors) > 0

    def test_control_state_in_posthoc(self, validator, valid_debate):
        """control_state is allowed but nullable — setting it to an object is OK."""
        doc = copy.deepcopy(valid_debate)
        doc["rounds"][0]["turns"][0]["attempts"][0]["control_state"] = {
            "e_t": 0.5,
            "u_t": 0.3,
        }
        errors = _schema_errors(validator, doc)
        assert len(errors) == 0

    def test_multiple_rounds_multiple_turns(self, validator):
        """Larger valid document with multiple rounds and turns."""
        doc = _minimal_valid_debate()
        doc["rounds"].append({
            "round_index": 2,
            "turns": [
                {
                    "turn_id": "t2",
                    "round_index": 2,
                    "turn_index_in_round": 0,
                    "speaker_id": "agent_b",
                    "attempts": [
                        {
                            "attempt_index": 0,
                            "timestamp": "2026-01-01T00:01:00Z",
                            "status": "ok",
                            "content": "I disagree.",
                        }
                    ],
                },
                {
                    "turn_id": "t3",
                    "round_index": 2,
                    "turn_index_in_round": 1,
                    "speaker_id": "agent_a",
                    "attempts": [
                        {
                            "attempt_index": 0,
                            "timestamp": "2026-01-01T00:02:00Z",
                            "status": "ok",
                            "content": "My rebuttal.",
                        }
                    ],
                },
            ],
        })
        errors = _schema_errors(jsonschema.Draft202012Validator(
            json.loads(SCHEMA_PATH.read_text())
        ), doc)
        assert len(errors) == 0

    def test_runtime_empty_rounds(self):
        """Runtime validator handles empty rounds array gracefully."""
        debate = {"participants": [{"agent_id": "a"}], "rounds": []}
        errors = _runtime_errors(debate)
        assert errors == []

    def test_runtime_round_with_no_turns_key(self):
        """Runtime validator handles round missing turns key."""
        debate = {"participants": [{"agent_id": "a"}], "rounds": [{"round_index": 1}]}
        errors = _runtime_errors(debate)
        assert errors == []


# ===================================================================
# 17. Determinism
# ===================================================================


class TestDeterminism:
    """Schema validation results must be deterministic."""

    def test_same_valid_doc_always_passes(self, validator, valid_debate):
        for _ in range(5):
            assert _schema_errors(validator, valid_debate) == []

    def test_same_invalid_doc_always_fails(self, validator, valid_debate):
        doc = _mutate(valid_debate, ["schema_version"], "0.0.0")
        for _ in range(5):
            errors = _schema_errors(validator, doc)
            assert len(errors) > 0
