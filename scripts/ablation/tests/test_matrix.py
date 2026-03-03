"""Tests for scripts/ablation/matrix.py — run matrix generation."""

from __future__ import annotations

import pytest

from scripts.ablation.config import BASELINE, NUM_RANDOM_GAIN_SAMPLES, SWEEP_GROUPS
from scripts.ablation.matrix import (
    _generate_random_gain_samples,
    _make_run_label,
    _run_id,
    count_runs,
    generate_run_matrix,
)


# ===========================================================================
# _run_id
# ===========================================================================


class TestRunId:
    def test_basic_neutral(self):
        rid = _run_id("gains", "Kp", "0.15", "neutral", 0)
        assert rid == "gains_Kp-0.15"

    def test_non_neutral_scenario_appended(self):
        rid = _run_id("gains", "Kp", "0.15", "bullish", 0)
        assert "bullish" in rid

    def test_replicate_appended(self):
        rid = _run_id("gains", "Kp", "0.15", "neutral", 2)
        assert "_r2" in rid

    def test_replicate_zero_not_appended(self):
        rid = _run_id("gains", "Kp", "0.15", "neutral", 0)
        assert "_r0" not in rid

    def test_slashes_sanitized(self):
        rid = _run_id("gains", "Kp", "0.15/extra", "neutral", 0)
        assert "/" not in rid

    def test_spaces_sanitized(self):
        rid = _run_id("gains", "Kp", "some value", "neutral", 0)
        assert " " not in rid


# ===========================================================================
# _make_run_label
# ===========================================================================


class TestMakeRunLabel:
    def test_single_param_override(self):
        param, value = _make_run_label("gains", {"Kp": 0.3}, 0)
        assert param == "Kp"
        assert "0.3" in value

    def test_labeled_group_interactions(self):
        param, value = _make_run_label("interactions", {"Kp": 0.3, "Ki": 0.01}, 0)
        assert param == "config"
        assert value == "aggressive_p"

    def test_labeled_group_phases(self):
        param, value = _make_run_label("phases", {}, 2)
        assert param == "config"
        assert value == "revise_only"

    def test_models_group(self):
        param, value = _make_run_label("models", {"model_name": "gpt-4o"}, 0)
        assert param == "model"
        assert value == "gpt-4o"

    def test_random_gain_samples(self):
        param, value = _make_run_label("random_gain_samples", {"Kp": 0.1}, 3)
        assert param == "gains"
        assert value == "random_3"

    def test_multi_param_override(self):
        overrides = {"Kp": 0.3, "Ki": 0.05}
        param, value = _make_run_label("some_group", overrides, 0)
        assert param == "multi"
        assert "Kp" in value
        assert "Ki" in value

    def test_baseline_match_returns_fallback(self):
        # Override with the same value as baseline → no diff
        param, value = _make_run_label("gains", {"Kp": BASELINE["Kp"]}, 5)
        assert param == "param"
        assert "baseline" in value


# ===========================================================================
# _generate_random_gain_samples
# ===========================================================================


class TestRandomGainSamples:
    def test_correct_count(self):
        samples = _generate_random_gain_samples(42)
        assert len(samples) == NUM_RANDOM_GAIN_SAMPLES

    def test_keys_present(self):
        samples = _generate_random_gain_samples(42)
        for s in samples:
            assert "Kp" in s
            assert "Ki" in s
            assert "Kd" in s

    def test_values_in_range(self):
        samples = _generate_random_gain_samples(42)
        for s in samples:
            assert 0.0 <= s["Kp"] <= 0.4
            assert 0.0 <= s["Ki"] <= 0.1
            assert 0.0 <= s["Kd"] <= 0.2

    def test_seed_reproducibility(self):
        a = _generate_random_gain_samples(123)
        b = _generate_random_gain_samples(123)
        assert a == b

    def test_different_seeds_differ(self):
        a = _generate_random_gain_samples(1)
        b = _generate_random_gain_samples(2)
        assert a != b

    def test_none_seed_produces_samples(self):
        samples = _generate_random_gain_samples(None)
        assert len(samples) == NUM_RANDOM_GAIN_SAMPLES


# ===========================================================================
# generate_run_matrix
# ===========================================================================


class TestGenerateRunMatrix:
    def test_baseline_always_included(self):
        matrix = generate_run_matrix(groups=["gains"], scenarios=["neutral"])
        baseline_runs = [r for r in matrix if r["group"] == "baseline"]
        assert len(baseline_runs) >= 1

    def test_single_group_single_scenario(self):
        matrix = generate_run_matrix(
            groups=["quality"], scenarios=["neutral"], replicates=1,
        )
        # baseline + 4 quality configs
        quality_runs = [r for r in matrix if r["group"] == "quality"]
        assert len(quality_runs) == 4

    def test_replicates_multiplies(self):
        matrix = generate_run_matrix(
            groups=["quality"], scenarios=["neutral"], replicates=3,
        )
        quality_runs = [r for r in matrix if r["group"] == "quality"]
        assert len(quality_runs) == 12  # 4 configs * 3 replicates

    def test_multi_scenario_multiplies(self):
        matrix = generate_run_matrix(
            groups=["quality"], scenarios=["bullish", "neutral"], replicates=1,
        )
        quality_runs = [r for r in matrix if r["group"] == "quality"]
        assert len(quality_runs) == 8  # 4 configs * 2 scenarios

    def test_run_config_has_required_metadata(self):
        matrix = generate_run_matrix(groups=["gains"], scenarios=["neutral"])
        for r in matrix:
            assert "run_id" in r
            assert "group" in r
            assert "param" in r
            assert "value" in r
            assert "scenario" in r
            assert "replicate" in r

    def test_run_config_inherits_baseline(self):
        matrix = generate_run_matrix(groups=["quality"], scenarios=["neutral"])
        # Quality overrides rho_star; other params should come from baseline
        for r in matrix:
            assert "Kp" in r
            assert "Ki" in r
            assert "roles" in r

    def test_run_ids_unique(self):
        matrix = generate_run_matrix(
            groups=["gains", "quality"], scenarios=["neutral"], replicates=2,
        )
        ids = [r["run_id"] for r in matrix]
        assert len(ids) == len(set(ids)), "Duplicate run_ids found"

    def test_random_gain_samples_populated(self):
        matrix = generate_run_matrix(
            groups=["random_gain_samples"], scenarios=["neutral"],
            replicates=1, seed=42,
        )
        rgs_runs = [r for r in matrix if r["group"] == "random_gain_samples"]
        assert len(rgs_runs) == NUM_RANDOM_GAIN_SAMPLES

    def test_none_groups_includes_all(self):
        matrix = generate_run_matrix(
            groups=None, scenarios=["neutral"], replicates=1, seed=42,
        )
        groups_seen = {r["group"] for r in matrix}
        # Should include baseline + at least several sweep groups
        assert "baseline" in groups_seen
        assert "gains" in groups_seen
        assert "quality" in groups_seen

    def test_scenario_recorded_in_config(self):
        matrix = generate_run_matrix(
            groups=["quality"], scenarios=["bullish"], replicates=1,
        )
        for r in matrix:
            if r["group"] == "quality":
                assert r["scenario"] == "bullish"


# ===========================================================================
# count_runs
# ===========================================================================


class TestCountRuns:
    def test_matches_matrix_length(self):
        groups = ["gains"]
        scenarios = ["neutral"]
        replicates = 2
        count = count_runs(groups, scenarios, replicates)
        matrix = generate_run_matrix(groups, scenarios, replicates, seed=42)
        assert count == len(matrix)

    def test_none_groups(self):
        count = count_runs(None, ["neutral"], 1)
        assert count > 100  # Should be a large number

    def test_multi_scenario(self):
        single = count_runs(["gains"], ["neutral"], 1)
        triple = count_runs(["gains"], ["bullish", "neutral", "riskoff"], 1)
        assert triple == single * 3

    def test_replicates_multiply(self):
        one = count_runs(["gains"], ["neutral"], 1)
        three = count_runs(["gains"], ["neutral"], 3)
        assert three == one * 3
