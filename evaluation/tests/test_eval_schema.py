"""
Schema validation tests for evaluation/schemas/eval.schema.json.

Coverage:
  - Positive: all 3 provided example files pass Draft 2020-12 validation
  - Negative / adversarial: inline JSON objects that must be rejected
    * Missing required top-level fields
    * Invalid enum values (evaluation_mode, overall_verdict, rca verdict, t3 rung)
    * Wrong schema_version
    * Extra unexpected properties at multiple levels
    * Missing experiment_config.label
    * Invalid control.policy.type
    * Invalid signal structure (missing required fields)
    * Invalid actuator structure (missing fields, invalid enum)
    * Wrong data types for various fields
    * Numeric range violations (crit gamma, pid retry_count)
    * turnEval missing turn_id / extra properties
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
SCHEMA_PATH = SCHEMA_DIR / "eval.schema.json"
EXAMPLES_DIR = SCHEMA_DIR / "examples"

EXAMPLE_FILES = [
    "eval_schema_post_hoc_example_1.json",
    "eval_schema_in_loop_crit_rca_no_pid_example_1.json",
    "eval_schema_in_loop_pid_rca_example_1.json",
]


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator(schema):
    return Draft202012Validator(schema)


def _load_example(filename: str) -> dict:
    with open(EXAMPLES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def _minimal_valid() -> dict:
    """Smallest document that satisfies all required fields."""
    return {
        "schema_version": "1.2.0",
        "debate_id": "test_debate",
        "run_id": "test_run",
        "evaluation_mode": "posthoc",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "eval_metadata": {"evaluator_version": "v1"},
        "run_summary": {"overall_verdict": "pass"},
    }


def _assert_fails(validator, doc, min_errors=1):
    errors = list(validator.iter_errors(doc))
    assert len(errors) >= min_errors, (
        f"Expected >= {min_errors} error(s), got {len(errors)}"
    )


def _assert_passes(validator, doc):
    errors = list(validator.iter_errors(doc))
    assert errors == [], (
        f"Expected 0 errors, got {len(errors)}: "
        + "; ".join(e.message[:100] for e in errors[:5])
    )


# ===================================================================
# POSITIVE — provided examples pass schema validation
# ===================================================================


class TestExampleSchemaValidation:
    """Every example file in evaluation/schemas/examples/ must pass."""

    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_example_passes_schema(self, validator, filename):
        doc = _load_example(filename)
        _assert_passes(validator, doc)

    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_example_has_required_toplevel_keys(self, filename):
        doc = _load_example(filename)
        required = [
            "schema_version", "debate_id", "run_id",
            "evaluation_mode", "evaluated_at",
            "eval_metadata", "run_summary",
        ]
        for key in required:
            assert key in doc, f"{filename} missing required key '{key}'"

    def test_minimal_valid_document_passes(self, validator):
        _assert_passes(validator, _minimal_valid())


# ===================================================================
# NEGATIVE — missing required top-level fields
# ===================================================================


class TestMissingRequiredFields:

    @pytest.mark.parametrize("field", [
        "schema_version", "debate_id", "run_id",
        "evaluation_mode", "evaluated_at",
        "eval_metadata", "run_summary",
    ])
    def test_missing_toplevel_required_field(self, validator, field):
        doc = _minimal_valid()
        del doc[field]
        _assert_fails(validator, doc)

    def test_missing_evaluator_version(self, validator):
        doc = _minimal_valid()
        doc["eval_metadata"] = {}
        _assert_fails(validator, doc)

    def test_missing_overall_verdict(self, validator):
        doc = _minimal_valid()
        doc["run_summary"] = {}
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — invalid enum values
# ===================================================================


class TestInvalidEnums:

    def test_invalid_evaluation_mode(self, validator):
        doc = _minimal_valid()
        doc["evaluation_mode"] = "realtime"
        _assert_fails(validator, doc)

    def test_invalid_overall_verdict(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["overall_verdict"] = "maybe"
        _assert_fails(validator, doc)

    def test_invalid_rca_verdict(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["rca_summary"] = {"verdict": "unknown"}
        _assert_fails(validator, doc)

    def test_invalid_t3_required_rung(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["t3_summary"] = {"required_rung": "L99"}
        _assert_fails(validator, doc)

    def test_invalid_t3_detected_rung(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["t3_summary"] = {"detected_rung": "L4"}
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — wrong schema_version
# ===================================================================


class TestSchemaVersion:

    @pytest.mark.parametrize("version", ["1.0.0", "2.0.0", ""])
    def test_wrong_version_rejected(self, validator, version):
        doc = _minimal_valid()
        doc["schema_version"] = version
        _assert_fails(validator, doc)

    def test_correct_version_accepted(self, validator):
        doc = _minimal_valid()
        _assert_passes(validator, doc)


# ===================================================================
# NEGATIVE — extra unexpected properties
# ===================================================================


class TestAdditionalProperties:

    def test_extra_toplevel_property(self, validator):
        doc = _minimal_valid()
        doc["unexpected_field"] = "surprise"
        _assert_fails(validator, doc)

    def test_extra_eval_metadata_property(self, validator):
        doc = _minimal_valid()
        doc["eval_metadata"]["unknown_key"] = True
        _assert_fails(validator, doc)

    def test_extra_run_summary_property(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["extra"] = 42
        _assert_fails(validator, doc)

    def test_extra_crit_summary_property(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["crit_summary"] = {"gamma_mean": 0.5, "extra": True}
        _assert_fails(validator, doc)

    def test_extra_turn_eval_property(self, validator):
        doc = _minimal_valid()
        doc["turn_evaluations"] = [{"turn_id": "t1", "unknown": True}]
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — experiment_config issues
# ===================================================================


class TestExperimentConfig:

    def test_missing_label(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {"category": "test"}
        _assert_fails(validator, doc)

    def test_null_experiment_config_accepted(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = None
        _assert_passes(validator, doc)

    def test_valid_experiment_config(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "baseline",
            "category": "core_ablation",
            "interventions": {"crit_in_loop": False, "rca_in_loop": False},
            "control": None,
            "extra_dimensions": None,
            "notes": None,
        }
        _assert_passes(validator, doc)

    def test_extra_interventions_property(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "interventions": {"crit_in_loop": False, "unknown_flag": True},
        }
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — control block issues
# ===================================================================


class TestControlBlock:

    def test_invalid_policy_type(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "policy": {"type": "quantum_optimizer"},
            },
        }
        _assert_fails(validator, doc)

    def test_control_missing_enabled(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {"policy": {"type": "pid"}},
        }
        _assert_fails(validator, doc)

    def test_valid_pid_control_block(self, validator):
        doc = _minimal_valid()
        doc["evaluation_mode"] = "in_loop"
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "policy": {"type": "pid", "parameters": {"Kp": 0.8, "Ki": 0.2, "Kd": 0.1}},
                "signals": [{"signal_id": "s1", "source": "rca", "metric": "consistency"}],
                "actuators": [{"actuator_id": "a1", "type": "retry"}],
                "termination": {
                    "max_rounds": 6,
                    "max_retries_per_turn": 3,
                    "stop_when": [{"signal_id": "s1", "op": ">=", "value": 0.95}],
                },
                "logging": {"log_level": "per_step", "store_step_series": True},
            },
        }
        _assert_passes(validator, doc)

    def test_extra_control_property(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {"enabled": False, "unknown_key": 42},
        }
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — invalid signal structure
# ===================================================================


class TestSignalStructure:

    def test_signal_missing_signal_id(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [{"source": "rca", "metric": "x"}],
            },
        }
        _assert_fails(validator, doc)

    def test_signal_missing_source(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [{"signal_id": "s1", "metric": "x"}],
            },
        }
        _assert_fails(validator, doc)

    def test_signal_missing_metric(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [{"signal_id": "s1", "source": "rca"}],
            },
        }
        _assert_fails(validator, doc)

    def test_signal_invalid_direction(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [
                    {"signal_id": "s1", "source": "rca", "metric": "x", "direction": "sideways"}
                ],
            },
        }
        _assert_fails(validator, doc)

    def test_signal_invalid_aggregation(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [
                    {"signal_id": "s1", "source": "rca", "metric": "x", "aggregation": "all_at_once"}
                ],
            },
        }
        _assert_fails(validator, doc)

    def test_signal_extra_property(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "signals": [
                    {"signal_id": "s1", "source": "rca", "metric": "x", "extra_field": True}
                ],
            },
        }
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — invalid actuator structure
# ===================================================================


class TestActuatorStructure:

    def test_actuator_missing_actuator_id(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "actuators": [{"type": "retry"}],
            },
        }
        _assert_fails(validator, doc)

    def test_actuator_missing_type(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "actuators": [{"actuator_id": "a1"}],
            },
        }
        _assert_fails(validator, doc)

    def test_actuator_invalid_type(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "actuators": [{"actuator_id": "a1", "type": "quantum_engine"}],
            },
        }
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE — wrong data types
# ===================================================================


class TestWrongDataTypes:

    def test_debate_id_integer(self, validator):
        doc = _minimal_valid()
        doc["debate_id"] = 12345
        _assert_fails(validator, doc)

    def test_evaluated_at_integer(self, validator):
        doc = _minimal_valid()
        doc["evaluated_at"] = 12345
        _assert_fails(validator, doc)

    def test_run_summary_string(self, validator):
        doc = _minimal_valid()
        doc["run_summary"] = "pass"
        _assert_fails(validator, doc)

    def test_eval_metadata_array(self, validator):
        doc = _minimal_valid()
        doc["eval_metadata"] = [1, 2, 3]
        _assert_fails(validator, doc)

    def test_crit_gamma_above_1(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["crit_summary"] = {"gamma_mean": 1.5}
        _assert_fails(validator, doc)

    def test_crit_gamma_below_0(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["crit_summary"] = {"gamma_mean": -0.1}
        _assert_fails(validator, doc)

    def test_pid_retry_count_negative(self, validator):
        doc = _minimal_valid()
        doc["run_summary"]["pid_summary"] = {"retry_count_total": -1}
        _assert_fails(validator, doc)

    def test_debate_id_empty_string(self, validator):
        doc = _minimal_valid()
        doc["debate_id"] = ""
        _assert_fails(validator, doc)

    def test_run_id_empty_string(self, validator):
        doc = _minimal_valid()
        doc["run_id"] = ""
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE / POSITIVE — turn_evaluations
# ===================================================================


class TestTurnEvaluations:

    def test_turn_eval_missing_turn_id(self, validator):
        doc = _minimal_valid()
        doc["turn_evaluations"] = [{"crit": None}]
        _assert_fails(validator, doc)

    def test_turn_eval_valid(self, validator):
        doc = _minimal_valid()
        doc["turn_evaluations"] = [
            {
                "turn_id": "t1",
                "crit": {"gamma_mean": 0.7, "theta_mean": 0.6, "threshold_pass": True},
            },
        ]
        _assert_passes(validator, doc)

    def test_turn_eval_null_accepted(self, validator):
        doc = _minimal_valid()
        doc["turn_evaluations"] = None
        _assert_passes(validator, doc)

    def test_turn_eval_extra_property_rejected(self, validator):
        doc = _minimal_valid()
        doc["turn_evaluations"] = [{"turn_id": "t1", "unknown_field": True}]
        _assert_fails(validator, doc)


# ===================================================================
# NEGATIVE / POSITIVE — control_trace
# ===================================================================


class TestControlTrace:

    def test_control_trace_null_accepted(self, validator):
        doc = _minimal_valid()
        doc["control_trace"] = None
        _assert_passes(validator, doc)

    def test_control_trace_valid_step(self, validator):
        doc = _minimal_valid()
        doc["control_trace"] = [
            {
                "step_index": 0,
                "timestamp": "2026-01-01T00:00:00Z",
                "observations": [{"signal_id": "s1", "value": 0.88}],
                "actions": [{"actuator_id": "a1", "payload": None}],
                "notes": None,
            },
        ]
        _assert_passes(validator, doc)

    def test_control_trace_missing_step_index(self, validator):
        doc = _minimal_valid()
        doc["control_trace"] = [{"timestamp": "2026-01-01T00:00:00Z"}]
        _assert_fails(validator, doc)

    def test_control_trace_negative_step_index(self, validator):
        doc = _minimal_valid()
        doc["control_trace"] = [{"step_index": -1}]
        _assert_fails(validator, doc)


# ===================================================================
# POSITIVE — stop_when / termination
# ===================================================================


class TestStopCondition:

    def test_invalid_stop_when_op(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "termination": {
                    "stop_when": [{"signal_id": "s1", "op": "!==", "value": 0.5}]
                },
            },
        }
        _assert_fails(validator, doc)

    def test_stop_when_missing_value(self, validator):
        doc = _minimal_valid()
        doc["experiment_config"] = {
            "label": "test",
            "control": {
                "enabled": True,
                "termination": {
                    "stop_when": [{"signal_id": "s1", "op": ">="}]
                },
            },
        }
        _assert_fails(validator, doc)
