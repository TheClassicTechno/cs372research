"""Tests for scripts/ablation/quarterly_runner.py — 4-quarter episode runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.ablation.config import BASELINE, INVEST_QUARTERS
from scripts.ablation.quarterly_runner import (
    _aggregate_episodes,
    _synthetic_quarterly_observations,
    run_quarterly_ablation,
)


# ===========================================================================
# _synthetic_quarterly_observations
# ===========================================================================


class TestSyntheticQuarterlyObservations:
    def test_returns_four_quarters(self):
        obs = _synthetic_quarterly_observations(["AAPL", "NVDA"], "neutral")
        assert len(obs) == 4
        for q in INVEST_QUARTERS:
            assert q in obs

    def test_all_quarters_same_observation(self):
        obs = _synthetic_quarterly_observations(["AAPL"], "neutral")
        # Synthetic fallback uses same observation for all quarters
        first = obs[INVEST_QUARTERS[0]]
        for q in INVEST_QUARTERS[1:]:
            assert obs[q] is first

    def test_unknown_scenario_falls_back_to_neutral(self):
        obs = _synthetic_quarterly_observations(["AAPL"], "nonexistent")
        assert len(obs) == 4

    def test_bullish_scenario(self):
        obs = _synthetic_quarterly_observations(["AAPL"], "bullish")
        first = obs[INVEST_QUARTERS[0]]
        assert first is not None


# ===========================================================================
# _aggregate_episodes
# ===========================================================================


class TestAggregateEpisodes:
    def _make_episode(self, **overrides):
        base = {
            "final_rho_bar": 0.75,
            "mean_rho_bar": 0.65,
            "final_beta": 0.45,
            "beta_range": 0.1,
            "steady_state_error": 0.05,
            "beta_overshoot": 0.08,
            "settling_round": 3,
            "rho_variance": 0.02,
            "mean_JS": 0.05,
            "rho_sign_change_count": 1,
            "rho_contraction_rate": 0.85,
            "empirical_kappa": 0.85,
            "paranoia_rate": 0.1,
            "realignment_rate": 0.3,
            "net_effect": 0.2,
            "healthy_persistence": 0.8,
            "regression_rate": 0.1,
            "chaotic_escape_rate": 0.5,
            "transition_entropy": 0.3,
            "escalation_count": 0,
            "stuck_rounds": 1,
            "sycophancy_count": 0,
            "rounds_used": 5,
            "beta_oscillation_flag": False,
            "rho_oscillation_flag": False,
            "rho_limit_cycle_flag": False,
            "JS_monotonicity_flag": False,
            "converged_single": True,
            "convergence_window_met": True,
            "control_stable": True,
            "behavioral_stable": True,
            "quadrant_stuck_pct": 0.1,
            "quadrant_chaotic_pct": 0.1,
            "quadrant_converged_pct": 0.4,
            "quadrant_healthy_pct": 0.4,
            "dominant_quadrant": "healthy",
            "stochastic_regime": False,
        }
        base.update(overrides)
        return base

    def test_numeric_mean(self):
        episodes = [
            self._make_episode(final_rho_bar=0.7),
            self._make_episode(final_rho_bar=0.8),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["final_rho_bar_mean"] == pytest.approx(0.75, abs=0.001)

    def test_numeric_std(self):
        episodes = [
            self._make_episode(final_rho_bar=0.7),
            self._make_episode(final_rho_bar=0.8),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["final_rho_bar_std"] > 0

    def test_single_episode_std_zero(self):
        episodes = [self._make_episode()]
        agg = _aggregate_episodes(episodes)
        assert agg["final_rho_bar_std"] == 0.0

    def test_boolean_rate(self):
        episodes = [
            self._make_episode(behavioral_stable=True),
            self._make_episode(behavioral_stable=True),
            self._make_episode(behavioral_stable=False),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["behavioral_stable_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_boolean_rate_all_true(self):
        episodes = [self._make_episode(control_stable=True)] * 4
        agg = _aggregate_episodes(episodes)
        assert agg["control_stable_rate"] == 1.0

    def test_quadrant_pct_mean(self):
        episodes = [
            self._make_episode(quadrant_healthy_pct=0.6),
            self._make_episode(quadrant_healthy_pct=0.4),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["quadrant_healthy_pct_mean"] == pytest.approx(0.5, abs=0.01)

    def test_dominant_quadrant_mode(self):
        episodes = [
            self._make_episode(dominant_quadrant="healthy"),
            self._make_episode(dominant_quadrant="healthy"),
            self._make_episode(dominant_quadrant="converged"),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["dominant_quadrant"] == "healthy"

    def test_per_quarter_rho_breakdown(self):
        episodes = [
            self._make_episode(final_rho_bar=0.7),
            self._make_episode(final_rho_bar=0.8),
            self._make_episode(final_rho_bar=0.75),
            self._make_episode(final_rho_bar=0.82),
        ]
        agg = _aggregate_episodes(episodes)
        assert len(agg["per_quarter_rho"]) == 4
        assert agg["per_quarter_rho"][0] == pytest.approx(0.7, abs=0.001)

    def test_per_quarter_converged_breakdown(self):
        episodes = [
            self._make_episode(convergence_window_met=True),
            self._make_episode(convergence_window_met=False),
            self._make_episode(convergence_window_met=True),
            self._make_episode(convergence_window_met=True),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["per_quarter_converged"] == [True, False, True, True]

    def test_stochastic_regime_from_first_episode(self):
        episodes = [
            self._make_episode(stochastic_regime=True),
            self._make_episode(stochastic_regime=True),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["stochastic_regime"] is True

    def test_none_numeric_values_skipped(self):
        episodes = [
            self._make_episode(empirical_kappa=None),
            self._make_episode(empirical_kappa=0.9),
        ]
        agg = _aggregate_episodes(episodes)
        assert agg["empirical_kappa_mean"] == pytest.approx(0.9, abs=0.001)


# ===========================================================================
# run_quarterly_ablation
# ===========================================================================


class TestRunQuarterlyAblation:
    def _make_run_config(self, **overrides):
        config = dict(BASELINE)
        config.update({
            "run_id": "test_quarterly",
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
        result = run_quarterly_ablation(config, mock=True)
        assert "skipped" in result["status"]

    def test_completed_run_has_episodes(self):
        config = self._make_run_config()
        result = run_quarterly_ablation(config, mock=True)
        if result.get("status") == "completed":
            assert result["episodes_completed"] > 0
            assert result["num_episodes"] == 4

    def test_result_has_data_source(self):
        config = self._make_run_config()
        result = run_quarterly_ablation(config, mock=True)
        if result.get("status") == "completed":
            assert result["data_source"] in ("real", "synthetic")

    def test_result_has_aggregated_metrics(self):
        config = self._make_run_config()
        result = run_quarterly_ablation(config, mock=True)
        if result.get("status") == "completed":
            # Should have _mean suffixed aggregated metrics
            assert "final_rho_bar_mean" in result

    def test_result_has_config_columns(self):
        config = self._make_run_config()
        result = run_quarterly_ablation(config, mock=True)
        assert result["run_id"] == "test_quarterly"
        assert result["group"] == "test"
        assert result["Kp"] == BASELINE["Kp"]

    def test_trace_saved_per_quarter(self, tmp_path):
        config = self._make_run_config()
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        result = run_quarterly_ablation(config, mock=True, trace_dir=trace_dir)
        if result.get("status") == "completed":
            traces = list(trace_dir.glob("*.json"))
            assert len(traces) >= 1

    def test_elapsed_seconds_tracked(self):
        config = self._make_run_config()
        result = run_quarterly_ablation(config, mock=True)
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] >= 0
