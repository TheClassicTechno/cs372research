"""Tests for scripts/ablation/runner.py — single ablation executor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.ablation.config import BASELINE
from scripts.ablation.runner import (
    _save_trace,
    build_debate_config,
    build_observation,
    run_single_ablation,
    stability_precheck,
)


# ===========================================================================
# stability_precheck
# ===========================================================================


class TestStabilityPrecheck:
    def test_baseline_is_stable(self):
        stable, non_osc = stability_precheck(BASELINE)
        assert stable is True

    def test_aggressive_gains_may_be_unstable(self):
        params = dict(BASELINE)
        params["Kp"] = 2.0
        params["Ki"] = 1.0
        params["Kd"] = 1.0
        stable, _ = stability_precheck(params)
        assert stable is False

    def test_returns_tuple_of_bools(self):
        result = stability_precheck(BASELINE)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], bool)

    def test_zero_gains_stable(self):
        params = dict(BASELINE)
        params["Kp"] = 0.0
        params["Ki"] = 0.0
        params["Kd"] = 0.0
        stable, non_osc = stability_precheck(params)
        assert stable is True
        assert non_osc is True


# ===========================================================================
# build_debate_config
# ===========================================================================


class TestBuildDebateConfig:
    def test_returns_debate_config(self):
        config = build_debate_config(BASELINE, mock=True)
        assert hasattr(config, "roles")
        assert hasattr(config, "pid_config")
        assert config.mock is True

    def test_pid_config_set(self):
        config = build_debate_config(BASELINE, mock=True)
        assert config.pid_config is not None
        assert config.pid_config.gains.Kp == BASELINE["Kp"]
        assert config.pid_config.gains.Ki == BASELINE["Ki"]
        assert config.pid_config.gains.Kd == BASELINE["Kd"]

    def test_roles_mapped(self):
        config = build_debate_config(BASELINE, mock=True)
        assert len(config.roles) == len(BASELINE["roles"])

    def test_custom_params_applied(self):
        params = dict(BASELINE)
        params["Kp"] = 0.25
        params["rho_star"] = 0.9
        params["model_name"] = "gpt-4o"
        config = build_debate_config(params, mock=True)
        assert config.pid_config.gains.Kp == 0.25
        assert config.pid_config.rho_star == 0.9
        assert config.model_name == "gpt-4o"

    def test_pid_log_metrics_enabled(self):
        config = build_debate_config(BASELINE, mock=True)
        assert config.pid_log_metrics is True

    def test_parallel_agents_disabled(self):
        config = build_debate_config(BASELINE, mock=True)
        assert config.parallel_agents is False



# ===========================================================================
# build_observation
# ===========================================================================


class TestBuildObservation:
    def test_neutral_scenario(self):
        obs = build_observation(BASELINE, "neutral")
        assert obs is not None
        assert obs.universe == BASELINE["tickers"]

    def test_bullish_scenario(self):
        obs = build_observation(BASELINE, "bullish")
        assert obs is not None

    def test_unknown_scenario_falls_back_to_neutral(self):
        obs = build_observation(BASELINE, "nonexistent_scenario")
        assert obs is not None

    def test_custom_tickers(self):
        params = dict(BASELINE)
        params["tickers"] = ["XOM", "CAT"]
        obs = build_observation(params, "neutral")
        assert obs.universe == ["XOM", "CAT"]


# ===========================================================================
# run_single_ablation
# ===========================================================================


class TestRunSingleAblation:
    def _make_run_config(self, **overrides):
        config = dict(BASELINE)
        config.update({
            "run_id": "test_run",
            "group": "test",
            "param": "test",
            "value": "test",
            "scenario": "neutral",
            "replicate": 0,
        })
        config.update(overrides)
        return config

    def test_unstable_run_skipped(self):
        config = self._make_run_config(Kp=2.0, Ki=1.0, Kd=1.0)
        result = run_single_ablation(config, mock=True)
        assert "skipped" in result["status"]
        assert result["stability_check"] is False

    def test_completed_run_has_metrics(self):
        config = self._make_run_config()
        result = run_single_ablation(config, mock=True)
        assert result["status"] == "completed"
        assert "final_rho_bar" in result
        assert "final_beta" in result
        assert "elapsed_seconds" in result

    def test_run_id_preserved(self):
        config = self._make_run_config(run_id="my_special_run")
        result = run_single_ablation(config, mock=True)
        assert result["run_id"] == "my_special_run"

    def test_config_params_in_result(self):
        config = self._make_run_config()
        result = run_single_ablation(config, mock=True)
        assert result["Kp"] == BASELINE["Kp"]
        assert result["Ki"] == BASELINE["Ki"]
        assert result["scenario"] == "neutral"

    def test_trace_saved_when_dir_provided(self, tmp_path):
        config = self._make_run_config()
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        result = run_single_ablation(config, mock=True, trace_dir=trace_dir)
        if result["status"] == "completed":
            traces = list(trace_dir.glob("*.json"))
            assert len(traces) >= 1

    def test_stability_labels_in_result(self):
        config = self._make_run_config()
        result = run_single_ablation(config, mock=True)
        if result["status"] == "completed":
            assert "control_stable" in result
            assert "behavioral_stable" in result
            assert "stability_check" in result


# ===========================================================================
# _save_trace
# ===========================================================================


class TestSaveTrace:
    def test_creates_json_file(self, tmp_path):
        trace = MagicMock()
        trace.model_dump_json.return_value = '{"test": true}'
        _save_trace(tmp_path, "test_run", trace)
        assert (tmp_path / "test_run.json").exists()

    def test_file_content(self, tmp_path):
        trace = MagicMock()
        trace.model_dump_json.return_value = '{"hello": "world"}'
        _save_trace(tmp_path, "test_run", trace)
        content = (tmp_path / "test_run.json").read_text()
        assert '"hello"' in content

    def test_creates_dir_if_missing(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        trace = MagicMock()
        trace.model_dump_json.return_value = "{}"
        _save_trace(subdir, "test_run", trace)
        assert (subdir / "test_run.json").exists()

    def test_handles_serialization_error(self, tmp_path):
        trace = MagicMock()
        trace.model_dump_json.side_effect = TypeError("not serializable")
        # Should not raise, just log warning
        _save_trace(tmp_path, "test_run", trace)
