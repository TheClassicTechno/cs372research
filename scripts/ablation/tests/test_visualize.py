"""Tests for scripts/ablation/visualize.py — visualization generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ablation.visualize import HAS_MATPLOTLIB, _extract_gain_grid, generate_plots


# ===========================================================================
# _extract_gain_grid
# ===========================================================================


class TestExtractGainGrid:
    def test_filters_gains_group(self):
        results = [
            {"group": "gains", "Kp": 0.1, "Ki": 0.01},
            {"group": "quality", "rho_star": 0.9},
            {"group": "random_gain_samples", "Kp": 0.2, "Ki": 0.05},
        ]
        grid = _extract_gain_grid(results)
        assert len(grid) == 2
        groups = {r["group"] for r in grid}
        assert "quality" not in groups

    def test_includes_high_gain_stress(self):
        results = [
            {"group": "high_gain_stress", "Kp": 0.4},
            {"group": "interactions", "Kp": 0.3},
        ]
        grid = _extract_gain_grid(results)
        assert len(grid) == 2

    def test_empty_results(self):
        assert _extract_gain_grid([]) == []


# ===========================================================================
# generate_plots
# ===========================================================================


class TestGeneratePlots:
    def test_insufficient_data_returns_empty(self, tmp_path):
        results = [{"status": "completed"}, {"status": "completed"}]
        plots = generate_plots(results, tmp_path)
        assert plots == []

    def test_no_completed_runs(self, tmp_path):
        results = [{"status": "skipped_unstable"}] * 5
        plots = generate_plots(results, tmp_path)
        assert plots == []

    def test_returns_list(self, tmp_path, sample_completed_results):
        plots = generate_plots(sample_completed_results, tmp_path)
        assert isinstance(plots, list)

    def test_no_matplotlib_returns_empty(self, tmp_path, sample_completed_results):
        """When matplotlib is unavailable, generate_plots returns []."""
        if not HAS_MATPLOTLIB:
            plots = generate_plots(sample_completed_results, tmp_path)
            assert plots == []

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_creates_plots_dir(self, tmp_path, sample_completed_results):
        for r in sample_completed_results:
            r["group"] = "gains"
        generate_plots(sample_completed_results, tmp_path)
        assert (tmp_path / "plots").exists()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_returns_list_of_strings(self, tmp_path, sample_completed_results):
        for r in sample_completed_results:
            r["group"] = "gains"
        plots = generate_plots(sample_completed_results, tmp_path)
        for p in plots:
            assert isinstance(p, str)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_overshoot_scatter_generated(self, tmp_path, sample_completed_results):
        plots = generate_plots(sample_completed_results, tmp_path)
        overshoot_plots = [p for p in plots if "overshoot" in p]
        assert len(overshoot_plots) >= 1

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_paranoia_scatter_generated(self, tmp_path, sample_completed_results):
        plots = generate_plots(sample_completed_results, tmp_path)
        paranoia_plots = [p for p in plots if "paranoia" in p]
        assert len(paranoia_plots) >= 1
