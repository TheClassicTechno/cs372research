"""Tests for scripts/ablation/io.py — output writing and validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.ablation.io import build_suite_summary, validate_run, write_results


# ===========================================================================
# validate_run
# ===========================================================================


class TestValidateRun:
    def test_valid_completed_run_no_warnings(self, sample_completed_results):
        warnings = validate_run(sample_completed_results[0])
        assert warnings == []

    def test_non_completed_skips_validation(self):
        result = {"run_id": "test", "status": "skipped_unstable"}
        warnings = validate_run(result)
        assert warnings == []

    def test_quadrant_sum_mismatch(self):
        result = {
            "run_id": "bad_quad",
            "status": "completed",
            "quadrant_stuck_pct": 0.5,
            "quadrant_chaotic_pct": 0.5,
            "quadrant_converged_pct": 0.5,
            "quadrant_healthy_pct": 0.5,
        }
        warnings = validate_run(result)
        assert any("Quadrant pcts sum" in w for w in warnings)

    def test_quadrant_sum_near_one_ok(self):
        result = {
            "run_id": "ok_quad",
            "status": "completed",
            "quadrant_stuck_pct": 0.25,
            "quadrant_chaotic_pct": 0.24,
            "quadrant_converged_pct": 0.26,
            "quadrant_healthy_pct": 0.25,
        }
        warnings = validate_run(result)
        assert not any("Quadrant pcts sum" in w for w in warnings)

    def test_nan_in_required_column(self):
        result = {
            "run_id": "nan_run",
            "status": "completed",
            "quadrant_stuck_pct": 0.25,
            "quadrant_chaotic_pct": 0.25,
            "quadrant_converged_pct": 0.25,
            "quadrant_healthy_pct": 0.25,
            "final_rho_bar": float("nan"),
        }
        warnings = validate_run(result)
        assert any("NaN" in w for w in warnings)

    def test_missing_quadrant_keys(self):
        result = {
            "run_id": "missing",
            "status": "completed",
        }
        # Missing quadrant keys default to 0, sum = 0 → warning
        warnings = validate_run(result)
        assert any("Quadrant pcts sum" in w for w in warnings)


# ===========================================================================
# write_results
# ===========================================================================


class TestWriteResults:
    def test_creates_output_files(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path)
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "aggregated.csv").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "errors.log").exists()
        assert (tmp_path / "suite_summary.txt").exists()

    def test_summary_csv_has_all_rows(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path)
        with open(tmp_path / "summary.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(sample_completed_results)

    def test_summary_csv_has_metric_columns(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path)
        with open(tmp_path / "summary.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "final_rho_bar" in row
        assert "steady_state_error" in row

    def test_config_json_has_counts(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path, {"mock": True})
        data = json.loads((tmp_path / "config.json").read_text())
        assert data["num_runs"] == 5
        assert data["num_completed"] == 5
        assert data["cli_args"]["mock"] is True

    def test_errors_log_no_errors(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path)
        content = (tmp_path / "errors.log").read_text()
        assert "No errors" in content

    def test_errors_log_records_failures(self, tmp_path):
        results = [
            {"run_id": "fail_1", "status": "runtime_error: timeout"},
            {"run_id": "ok_1", "status": "completed",
             "quadrant_stuck_pct": 0.25, "quadrant_chaotic_pct": 0.25,
             "quadrant_converged_pct": 0.25, "quadrant_healthy_pct": 0.25},
        ]
        write_results(results, tmp_path)
        content = (tmp_path / "errors.log").read_text()
        assert "fail_1" in content
        assert "runtime_error" in content

    def test_empty_results_handled(self, tmp_path):
        write_results([], tmp_path)
        assert (tmp_path / "summary.csv").exists()

    def test_suite_summary_txt_written(self, tmp_path, sample_completed_results):
        write_results(sample_completed_results, tmp_path)
        content = (tmp_path / "suite_summary.txt").read_text()
        assert "ABLATION SUITE SUMMARY" in content

    def test_creates_output_dir_if_missing(self, tmp_path, sample_completed_results):
        subdir = tmp_path / "nested" / "deep"
        write_results(sample_completed_results, subdir)
        assert (subdir / "summary.csv").exists()


# ===========================================================================
# build_suite_summary
# ===========================================================================


class TestBuildSuiteSummary:
    def test_contains_header(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "ABLATION SUITE SUMMARY" in summary

    def test_contains_total_count(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "Total runs:" in summary
        assert "5" in summary

    def test_contains_completed_count(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "Completed:" in summary

    def test_contains_stability_info(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "Control stable:" in summary
        assert "Behavioral stable:" in summary

    def test_contains_convergence_info(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "Converged (window):" in summary

    def test_contains_mean_metrics(self, sample_completed_results):
        summary = build_suite_summary(sample_completed_results)
        assert "steady_state_error" in summary

    def test_skipped_runs_counted(self):
        results = [
            {"run_id": "skip1", "status": "skipped_unstable"},
            {"run_id": "skip2", "status": "skipped_unstable"},
        ]
        summary = build_suite_summary(results)
        assert "Skipped:" in summary
        assert "2" in summary

    def test_empty_results(self):
        summary = build_suite_summary([])
        assert "Total runs:" in summary
        assert "0" in summary

    def test_temperature_regime_breakdown(self):
        results = [
            {
                "run_id": f"run_{i}", "status": "completed",
                "stochastic_regime": i >= 2,
                "behavioral_stable": True,
                "beta_oscillation_flag": False,
                "rho_oscillation_flag": False,
                "convergence_window_met": True,
                "steady_state_error": 0.05,
                "beta_overshoot": 0.1,
                "empirical_kappa": 0.8,
                "paranoia_rate": 0.1,
                "net_effect": 0.1,
                "final_rho_bar": 0.75,
                "rho_star": 0.8,
                "quadrant_stuck_pct": 0.0,
                "quadrant_chaotic_pct": 0.0,
                "quadrant_converged_pct": 0.5,
                "quadrant_healthy_pct": 0.5,
            }
            for i in range(4)
        ]
        summary = build_suite_summary(results)
        assert "Stochastic" in summary
        assert "Deterministic" in summary
