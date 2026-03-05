"""Tests for scripts/ablation/main.py — CLI entrypoint and suite execution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.ablation.config import BASELINE, DEFAULT_DATASET_PATH, DEFAULT_MEMO_FORMAT
from scripts.ablation.main import SCENARIO_SETS, _print_matrix, _print_progress, parse_args


# ===========================================================================
# parse_args
# ===========================================================================


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["main"]):
            args = parse_args()
        assert args.mock is False
        assert args.dry_run is False
        assert args.groups is None
        assert args.replicates == 3
        assert args.scenario_set == "single"
        assert args.quarterly is False

    def test_mock_flag(self):
        with patch("sys.argv", ["main", "--mock"]):
            args = parse_args()
        assert args.mock is True

    def test_dry_run_flag(self):
        with patch("sys.argv", ["main", "--dry-run"]):
            args = parse_args()
        assert args.dry_run is True

    def test_groups_parsed(self):
        with patch("sys.argv", ["main", "--groups", "gains,quality"]):
            args = parse_args()
        assert args.groups == "gains,quality"

    def test_seed(self):
        with patch("sys.argv", ["main", "--seed", "42"]):
            args = parse_args()
        assert args.seed == 42

    def test_replicates(self):
        with patch("sys.argv", ["main", "--replicates", "5"]):
            args = parse_args()
        assert args.replicates == 5

    def test_scenario_set_quad(self):
        with patch("sys.argv", ["main", "--scenario-set", "quad"]):
            args = parse_args()
        assert args.scenario_set == "quad"

    def test_quarterly_flag(self):
        with patch("sys.argv", ["main", "--quarterly"]):
            args = parse_args()
        assert args.quarterly is True

    def test_dataset_path(self):
        with patch("sys.argv", ["main", "--dataset-path", "/my/data"]):
            args = parse_args()
        assert args.dataset_path == "/my/data"

    def test_memo_format_json(self):
        with patch("sys.argv", ["main", "--memo-format", "json"]):
            args = parse_args()
        assert args.memo_format == "json"

    def test_dataset_path_default(self):
        with patch("sys.argv", ["main"]):
            args = parse_args()
        assert args.dataset_path == DEFAULT_DATASET_PATH

    def test_memo_format_default(self):
        with patch("sys.argv", ["main"]):
            args = parse_args()
        assert args.memo_format == DEFAULT_MEMO_FORMAT

    def test_max_workers(self):
        with patch("sys.argv", ["main", "--max-workers", "8"]):
            args = parse_args()
        assert args.max_workers == 8

    def test_no_plots_flag(self):
        with patch("sys.argv", ["main", "--no-plots"]):
            args = parse_args()
        assert args.no_plots is True

    def test_force_parallel(self):
        with patch("sys.argv", ["main", "--force-parallel"]):
            args = parse_args()
        assert args.force_parallel is True


# ===========================================================================
# SCENARIO_SETS
# ===========================================================================


class TestScenarioSets:
    def test_single(self):
        assert SCENARIO_SETS["single"] == ["neutral"]

    def test_triple(self):
        assert len(SCENARIO_SETS["triple"]) == 3
        assert "neutral" in SCENARIO_SETS["triple"]

    def test_quad(self):
        assert len(SCENARIO_SETS["quad"]) == 4
        assert "conflicted" in SCENARIO_SETS["quad"]


# ===========================================================================
# _print_progress
# ===========================================================================


class TestPrintProgress:
    def test_zero_total(self, capsys):
        _print_progress(0, 0)
        # Should not crash

    def test_partial_progress(self, capsys):
        _print_progress(5, 10)
        # Just check no crash (output goes to stderr)

    def test_complete_progress(self, capsys):
        _print_progress(10, 10)


# ===========================================================================
# _print_matrix
# ===========================================================================


class TestPrintMatrix:
    def test_prints_header(self, capsys):
        matrix = [{
            "run_id": "test_run",
            "group": "gains",
            "param": "Kp",
            "value": "0.15",
            "scenario": "neutral",
            "replicate": 0,
        }]
        _print_matrix(matrix)
        captured = capsys.readouterr()
        assert "run_id" in captured.out
        assert "test_run" in captured.out

    def test_prints_all_rows(self, capsys):
        matrix = [
            {
                "run_id": f"run_{i}",
                "group": "gains",
                "param": "Kp",
                "value": str(0.1 * i),
                "scenario": "neutral",
                "replicate": 0,
            }
            for i in range(3)
        ]
        _print_matrix(matrix)
        captured = capsys.readouterr()
        assert "run_0" in captured.out
        assert "run_1" in captured.out
        assert "run_2" in captured.out
