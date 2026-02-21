"""
Semantic validation tests for evaluation/utils/validate_eval_schema.py.

Coverage — rules enforced by validate_semantics():
  Rule 1: control.enabled=True requires evaluation_mode="in_loop"
  Rule 4: PID policy must define Kp, Ki, Kd
  Rule 5: termination.stop_when must reference declared signal_ids
  Plus: valid control blocks pass without error
"""

from pathlib import Path

import pytest

from evaluation.utils.validate_eval_schema import validate_semantics

DUMMY_PATH = Path("test_file.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_doc(**overrides) -> dict:
    """Minimal document for semantic checks (schema shape not enforced here)."""
    doc = {
        "schema_version": "1.2.0",
        "debate_id": "test",
        "run_id": "run_001",
        "evaluation_mode": "posthoc",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "eval_metadata": {"evaluator_version": "v1"},
        "run_summary": {"overall_verdict": "pass"},
    }
    doc.update(overrides)
    return doc


# ===================================================================
# Rule 1: control.enabled=True requires evaluation_mode="in_loop"
# ===================================================================


class TestControlEnabledRequiresInLoop:

    def test_control_enabled_with_posthoc_raises(self):
        doc = _base_doc(
            evaluation_mode="posthoc",
            experiment_config={
                "label": "test",
                "control": {"enabled": True},
            },
        )
        with pytest.raises(ValueError, match="control.enabled=True requires evaluation_mode='in_loop'"):
            validate_semantics(doc, DUMMY_PATH)

    def test_control_enabled_with_in_loop_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {"type": "threshold"},
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)  # must not raise

    def test_control_disabled_with_posthoc_passes(self):
        doc = _base_doc(
            evaluation_mode="posthoc",
            experiment_config={
                "label": "test",
                "control": {"enabled": False},
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_control_null_passes(self):
        doc = _base_doc(
            experiment_config={
                "label": "test",
                "control": None,
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_no_experiment_config_passes(self):
        doc = _base_doc()
        validate_semantics(doc, DUMMY_PATH)

    def test_experiment_config_without_control_key_passes(self):
        doc = _base_doc(
            experiment_config={"label": "test"},
        )
        validate_semantics(doc, DUMMY_PATH)


# ===================================================================
# Rule 4: PID policy must define Kp, Ki, Kd
# ===================================================================


class TestPidPolicyParameters:

    def test_pid_missing_kp_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {
                        "type": "pid",
                        "parameters": {"Ki": 0.2, "Kd": 0.1},
                    },
                },
            },
        )
        with pytest.raises(ValueError, match="PID policy must define Kp, Ki, Kd"):
            validate_semantics(doc, DUMMY_PATH)

    def test_pid_missing_ki_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {
                        "type": "pid",
                        "parameters": {"Kp": 0.8, "Kd": 0.1},
                    },
                },
            },
        )
        with pytest.raises(ValueError, match="PID policy must define Kp, Ki, Kd"):
            validate_semantics(doc, DUMMY_PATH)

    def test_pid_missing_kd_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {
                        "type": "pid",
                        "parameters": {"Kp": 0.8, "Ki": 0.2},
                    },
                },
            },
        )
        with pytest.raises(ValueError, match="PID policy must define Kp, Ki, Kd"):
            validate_semantics(doc, DUMMY_PATH)

    def test_pid_empty_parameters_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {"type": "pid", "parameters": {}},
                },
            },
        )
        with pytest.raises(ValueError, match="PID policy must define Kp, Ki, Kd"):
            validate_semantics(doc, DUMMY_PATH)

    def test_pid_null_parameters_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {"type": "pid", "parameters": None},
                },
            },
        )
        with pytest.raises(ValueError, match="PID policy must define Kp, Ki, Kd"):
            validate_semantics(doc, DUMMY_PATH)

    def test_pid_all_params_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {
                        "type": "pid",
                        "parameters": {"Kp": 0.8, "Ki": 0.2, "Kd": 0.1},
                    },
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_non_pid_policy_skips_param_check(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "policy": {"type": "threshold"},
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_no_policy_key_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {"enabled": True},
            },
        )
        validate_semantics(doc, DUMMY_PATH)


# ===================================================================
# Rule 5: termination.stop_when references declared signal_ids
# ===================================================================


class TestTerminationSignalIdReference:

    def test_unknown_signal_id_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "signals": [{"signal_id": "s_rca"}],
                    "termination": {
                        "stop_when": [
                            {"signal_id": "s_nonexistent", "op": ">=", "value": 0.9}
                        ]
                    },
                },
            },
        )
        with pytest.raises(ValueError, match="unknown signal_id 's_nonexistent'"):
            validate_semantics(doc, DUMMY_PATH)

    def test_known_signal_id_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "signals": [{"signal_id": "s_rca"}],
                    "termination": {
                        "stop_when": [
                            {"signal_id": "s_rca", "op": ">=", "value": 0.95}
                        ]
                    },
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_empty_stop_when_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "signals": [],
                    "termination": {"stop_when": []},
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_null_stop_when_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "signals": [],
                    "termination": {"stop_when": None},
                },
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_no_termination_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {"enabled": True},
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_multiple_signals_one_bad_reference_raises(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "test",
                "control": {
                    "enabled": True,
                    "signals": [
                        {"signal_id": "s_rca"},
                        {"signal_id": "s_crit"},
                    ],
                    "termination": {
                        "stop_when": [
                            {"signal_id": "s_rca", "op": ">=", "value": 0.9},
                            {"signal_id": "s_missing", "op": ">=", "value": 0.8},
                        ]
                    },
                },
            },
        )
        with pytest.raises(ValueError, match="unknown signal_id 's_missing'"):
            validate_semantics(doc, DUMMY_PATH)


# ===================================================================
# POSITIVE — full valid control blocks pass end-to-end
# ===================================================================


class TestValidControlPasses:

    def test_full_pid_control_block_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "full_pid_test",
                "category": "integration",
                "interventions": {"crit_in_loop": False, "rca_in_loop": True},
                "control": {
                    "enabled": True,
                    "policy": {
                        "type": "pid",
                        "parameters": {"Kp": 0.8, "Ki": 0.2, "Kd": 0.1},
                    },
                    "signals": [
                        {"signal_id": "s_rca", "source": "rca", "metric": "trace_consistency_rate"}
                    ],
                    "actuators": [
                        {"actuator_id": "retry1", "type": "retry", "limits": {"max": 3}}
                    ],
                    "termination": {
                        "max_rounds": 6,
                        "max_retries_per_turn": 3,
                        "stop_when": [
                            {"signal_id": "s_rca", "op": ">=", "value": 0.95}
                        ],
                    },
                    "logging": {"log_level": "per_step", "store_step_series": True},
                },
                "extra_dimensions": None,
                "notes": "Full integration test.",
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_posthoc_no_control_passes(self):
        doc = _base_doc(
            evaluation_mode="posthoc",
            experiment_config={
                "label": "baseline",
                "control": None,
            },
        )
        validate_semantics(doc, DUMMY_PATH)

    def test_in_loop_disabled_control_passes(self):
        doc = _base_doc(
            evaluation_mode="in_loop",
            experiment_config={
                "label": "disabled",
                "control": {"enabled": False},
            },
        )
        validate_semantics(doc, DUMMY_PATH)
