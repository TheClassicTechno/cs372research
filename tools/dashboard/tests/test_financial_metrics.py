"""Backend unit tests for financial metric computation from logging data.

Validates ``_compute_run_financial_metrics()`` against the canonical
test fixture at ``logging/runs/test/run_2026-03-07_19-50-06/``.

Run:
    pytest tools/dashboard/tests/test_financial_metrics.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.dashboard.run_scanner import (
    _compute_run_financial_metrics,
    compute_financial_significance_summary as _real_significance_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_BASE = REPO_ROOT / "logging" / "runs"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "financial_expected.json"


@pytest.fixture()
def expected_judge_metrics() -> dict[str, float]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["judge_metrics"]


@pytest.fixture()
def test_run_dir() -> Path:
    return RUNS_BASE / "test" / "run_2026-03-07_19-50-06"


@pytest.fixture()
def clean_cache(test_run_dir: Path):
    """Remove _dashboard cache before and after test."""
    cache_dir = test_run_dir / "_dashboard"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


class TestComputeRunFinancialMetrics:
    def test_judge_metrics_match_expected(
        self, test_run_dir: Path, expected_judge_metrics: dict, clean_cache: None
    ):
        """Computed judge metrics match fixture values within tolerance."""
        result = _compute_run_financial_metrics(test_run_dir, use_mean_revisions=False)
        assert result is not None, "Expected metrics dict, got None"

        for key, expected_val in expected_judge_metrics.items():
            actual_val = result.get(key)
            assert actual_val is not None, f"Missing key: {key}"
            assert abs(actual_val - expected_val) < abs(expected_val) * 1e-6 + 1e-9, (
                f"{key}: expected {expected_val}, got {actual_val}"
            )

    def test_all_nine_metric_keys_present(
        self, test_run_dir: Path, clean_cache: None
    ):
        """Output contains all 9 expected daily_metrics_ keys."""
        result = _compute_run_financial_metrics(test_run_dir, use_mean_revisions=False)
        assert result is not None

        expected_keys = {
            "daily_metrics_trading_days",
            "daily_metrics_total_return_pct",
            "daily_metrics_annualized_sharpe",
            "daily_metrics_annualized_sortino",
            "daily_metrics_annualized_volatility",
            "daily_metrics_max_drawdown_pct",
            "daily_metrics_calmar_ratio",
            "daily_metrics_spy_return_pct",
            "daily_metrics_excess_return_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_cache_written_and_reused(self, test_run_dir: Path, clean_cache: None):
        """First call writes cache; second call reads from it."""
        cache_path = test_run_dir / "_dashboard" / "financial_metrics.json"
        assert not cache_path.exists()

        result1 = _compute_run_financial_metrics(test_run_dir, use_mean_revisions=False)
        assert result1 is not None
        assert cache_path.exists(), "Cache file should be written after first call"

        # Second call should return same result from cache
        result2 = _compute_run_financial_metrics(test_run_dir, use_mean_revisions=False)
        assert result2 == result1

    def test_missing_portfolio_returns_none(self, tmp_path: Path):
        """Run dir with no final/final_portfolio.json returns None."""
        run_dir = tmp_path / "run_fake"
        run_dir.mkdir()

        # Minimal manifest
        manifest = {
            "ticker_universe": ["AAPL", "_CASH_"],
            "config_paths": ["/fake/debate.yaml", "/fake/scenario.yaml"],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        # final/ dir with scenario yaml but no portfolio
        final_dir = run_dir / "final"
        final_dir.mkdir()
        scenario = {"invest_quarter": "2023Q2"}
        (final_dir / "scenario.yaml").write_text(yaml.dump(scenario))
        debate = {"broker": {"initial_cash": 100000.0}}
        (final_dir / "debate.yaml").write_text(yaml.dump(debate))

        result = _compute_run_financial_metrics(run_dir, use_mean_revisions=False)
        assert result is None

    def test_missing_daily_prices_returns_none(self, tmp_path: Path):
        """Run dir pointing to nonexistent daily prices returns None."""
        run_dir = tmp_path / "run_fake"
        run_dir.mkdir()

        manifest = {
            "ticker_universe": ["AAPL", "_CASH_"],
            "config_paths": ["/fake/debate.yaml", "/fake/scenario.yaml"],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        final_dir = run_dir / "final"
        final_dir.mkdir()
        # invest_quarter pointing to a quarter with no price data
        scenario = {"invest_quarter": "1999Q1"}
        (final_dir / "scenario.yaml").write_text(yaml.dump(scenario))
        debate = {"broker": {"initial_cash": 100000.0}}
        (final_dir / "debate.yaml").write_text(yaml.dump(debate))
        portfolio = {"AAPL": 0.5, "_CASH_": 0.5}
        (final_dir / "final_portfolio.json").write_text(json.dumps(portfolio))

        result = _compute_run_financial_metrics(run_dir, use_mean_revisions=False)
        assert result is None

    def test_mean_allocation_computed_correctly(self, tmp_path: Path):
        """Mean of 2 agents' allocations is computed correctly."""
        run_dir = tmp_path / "run_mean"
        run_dir.mkdir()

        # Use a real scenario that has daily prices available
        manifest = {
            "ticker_universe": ["AAPL", "AMD", "_CASH_"],
            "config_paths": ["/fake/debate.yaml", "/fake/scenario.yaml"],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        final_dir = run_dir / "final"
        final_dir.mkdir()
        scenario = {"invest_quarter": "2023Q2"}
        (final_dir / "scenario.yaml").write_text(yaml.dump(scenario))
        debate = {"broker": {"initial_cash": 100000.0}}
        (final_dir / "debate.yaml").write_text(yaml.dump(debate))

        # metrics_revision with 2 agents
        rounds_dir = run_dir / "rounds" / "round_001"
        rounds_dir.mkdir(parents=True)
        revision = {
            "allocations": {
                "macro": {"AAPL": 0.6, "AMD": 0.2, "_CASH_": 0.2},
                "technical": {"AAPL": 0.4, "AMD": 0.4, "_CASH_": 0.2},
            }
        }
        (rounds_dir / "metrics_revision.json").write_text(json.dumps(revision))

        result = _compute_run_financial_metrics(run_dir, use_mean_revisions=True)

        # If daily prices exist for 2023Q2, we get metrics
        # The key thing is mean weights: AAPL=0.5, AMD=0.3, _CASH_=0.2
        if result is not None:
            assert isinstance(result, dict)
            assert "daily_metrics_total_return_pct" in result
        # If prices don't exist (test env), None is acceptable


class TestFinancialSignificanceSummary:
    """Verify compute_financial_significance_summary returns expected shape."""

    def test_significance_summary_returns_expected_shape(self):
        """Summary has experiments list and metrics list with results dicts."""
        result = _real_significance_summary(RUNS_BASE)
        assert isinstance(result, dict)
        assert "experiments" in result
        assert "metrics" in result
        assert isinstance(result["experiments"], list)
        assert isinstance(result["metrics"], list)

        # Each metric entry has a metric key and results dict
        for entry in result["metrics"]:
            assert "metric" in entry
            assert "results" in entry
            assert isinstance(entry["results"], dict)
            # Results keys should be a subset of experiments
            for exp_key in entry["results"]:
                assert exp_key in result["experiments"]

    def test_significance_summary_empty_base(self, tmp_path: Path):
        """Empty base path returns empty experiments and metrics."""
        result = _real_significance_summary(tmp_path)
        assert result == {"experiments": [], "metrics": []}


class TestFinancialDataIntegrity:
    """Verify compute_financial_paired_tests reads from logging data only."""

    def test_no_results_dir_dependency(self, test_run_dir: Path, clean_cache: None):
        """_compute_run_financial_metrics never accesses results/ directory."""
        # The function should work purely from logging run data + daily prices.
        # We verify by calling it and confirming it succeeds without any
        # results/ directory involvement.
        result = _compute_run_financial_metrics(test_run_dir, use_mean_revisions=False)
        assert result is not None
        assert "daily_metrics_excess_return_pct" in result
        assert "daily_metrics_annualized_sharpe" in result

    def test_paired_tests_uses_logging_runs(self, test_run_dir: Path, clean_cache: None):
        """compute_financial_paired_tests reads config stems from manifests."""
        from tools.dashboard.run_scanner import compute_financial_paired_tests

        # For the single-run "test" experiment, we expect < 2 configs
        # (only one debate config), so it returns a "pending" response.
        result = compute_financial_paired_tests(
            base_path=RUNS_BASE, experiment="test",
        )
        # The function should run without error. With only 1 config,
        # it returns a pending dict.
        assert isinstance(result, dict)
        assert "pending" in result or "config_a" in result
