"""Backend unit tests for debate impact metric correctness.

Validates computed values against known formulas using the canonical
test fixture at ``logging/runs/test/run_2026-03-07_19-50-06/``.

Run:
    pytest tools/dashboard/tests/test_debate_impact_metrics.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.dashboard.run_scanner import (
    compute_collapse_diagnostics,
    compute_debate_impact,
    get_run_detail,
)

RUNS_BASE = Path(__file__).resolve().parents[3] / "logging" / "runs"
PRICES_BASE = Path(__file__).resolve().parents[3] / "data-pipeline" / "daily_prices" / "data"
EXPERIMENT = "test"
RUN_ID = "run_2026-03-07_19-50-06"


class TestDeltaCorrectness:
    def test_delta_pct_matches_formula(self):
        """Verify delta_pct == (pv - 100000) / 100000 * 100 within tolerance."""
        impact = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        for role, entry in impact["agent_deltas"].items():
            for phase_key in ("r1_proposal", "r1_revision", "r2_revision"):
                phase = entry.get(phase_key)
                if phase is None:
                    continue
                expected = (phase["pv"] - 100_000) / 100_000 * 100
                assert abs(phase["delta_pct"] - expected) < 0.01, (
                    f"{role}.{phase_key}: delta_pct={phase['delta_pct']} "
                    f"but expected {expected:.2f}"
                )

    def test_portfolio_values_positive(self):
        """Every computed PV must be > 0."""
        impact = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        for role, entry in impact["agent_deltas"].items():
            for phase_key in ("r1_proposal", "r1_revision", "r2_revision"):
                phase = entry.get(phase_key)
                if phase is None:
                    continue
                assert phase["pv"] > 0, f"{role}.{phase_key}: pv={phase['pv']} <= 0"


class TestSharpe:
    def test_sharpe_reproducible(self):
        """Sharpe computed twice on same portfolio must return identical values."""
        impact1 = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        impact2 = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        for key in impact1.get("sharpe", {}):
            assert impact1["sharpe"][key] == impact2["sharpe"][key], (
                f"Sharpe for {key} not reproducible: "
                f"{impact1['sharpe'][key]} != {impact2['sharpe'][key]}"
            )

    def test_sharpe_in_reasonable_range(self):
        """Sharpe values should be within a reasonable annualized range."""
        impact = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        for key, val in impact.get("sharpe", {}).items():
            if val is None:
                continue
            assert -10.0 < val < 10.0, (
                f"Sharpe {key}={val} out of reasonable range"
            )


class TestDebateAlpha:
    def test_debate_alpha_matches_formula(self):
        """debate_alpha == final_debate_return - mean_proposal_return."""
        impact = compute_debate_impact(RUNS_BASE, EXPERIMENT, RUN_ID, PRICES_BASE)
        s = impact["summary"]
        if s["debate_alpha"] is not None:
            expected = round(s["final_debate_return"] - s["mean_proposal_return"], 2)
            assert abs(s["debate_alpha"] - expected) < 0.01


class TestCollapse:
    def test_collapse_leader_has_max_share(self):
        """collapse_leader must be the agent with the highest collapse_share."""
        result = compute_collapse_diagnostics(RUNS_BASE, EXPERIMENT, RUN_ID)
        for entry in result:
            if entry["collapse_leader"] is None:
                continue
            leader = entry["collapse_leader"]
            leader_share = entry["agents"][leader]["collapse_share"]
            for role, data in entry["agents"].items():
                if data["collapse_share"] is not None:
                    assert leader_share >= data["collapse_share"]

    def test_collapse_shares_sum_to_one(self):
        """When positive movement exists, collapse shares must sum to ~1."""
        result = compute_collapse_diagnostics(RUNS_BASE, EXPERIMENT, RUN_ID)
        for entry in result:
            shares = [
                data["collapse_share"]
                for data in entry["agents"].values()
                if data["collapse_share"] is not None
            ]
            if shares:
                total = sum(shares)
                assert abs(total - 1.0) < 0.01, (
                    f"Round {entry['round']}: shares sum to {total}, expected ~1.0"
                )

    def test_collapse_has_expected_agents(self):
        """Each round should have entries for all 3 agents."""
        result = compute_collapse_diagnostics(RUNS_BASE, EXPERIMENT, RUN_ID)
        assert len(result) >= 1, "No collapse diagnostics computed"
        for entry in result:
            assert len(entry["agents"]) == 3, (
                f"Round {entry['round']}: expected 3 agents, "
                f"got {len(entry['agents'])}"
            )


class TestTickerPerformance:
    def test_ticker_performance_values(self, monkeypatch):
        """pct_change matches (close-open)/open*100."""
        monkeypatch.chdir(Path(__file__).resolve().parents[3])
        detail = get_run_detail(RUNS_BASE, EXPERIMENT, RUN_ID)
        for t in detail.get("ticker_performance", []):
            expected = round((t["close"] - t["open"]) / t["open"] * 100, 2)
            assert abs(t["pct_change"] - expected) < 0.02, (
                f"{t['ticker']}: pct_change={t['pct_change']} expected={expected}"
            )

    def test_ticker_performance_has_tickers(self, monkeypatch):
        """Ticker performance should include the portfolio tickers."""
        monkeypatch.chdir(Path(__file__).resolve().parents[3])
        detail = get_run_detail(RUNS_BASE, EXPERIMENT, RUN_ID)
        tp = detail.get("ticker_performance", [])
        assert len(tp) > 0, "No ticker performance data"
        tickers = {t["ticker"] for t in tp}
        assert "AMD" in tickers, "Missing AMD"


class TestBackfillArtifacts:
    def test_backfill_creates_artifacts(self, tmp_path):
        """compute functions write _dashboard/ artifacts."""
        src = RUNS_BASE / EXPERIMENT / RUN_ID
        dst = tmp_path / EXPERIMENT / RUN_ID
        shutil.copytree(src, dst)
        compute_debate_impact(tmp_path, EXPERIMENT, RUN_ID, PRICES_BASE)
        compute_collapse_diagnostics(tmp_path, EXPERIMENT, RUN_ID)
        dashboard_dir = dst / "_dashboard"
        assert dashboard_dir.is_dir()
        assert (dashboard_dir / "debate_impact.json").exists()
        assert (dashboard_dir / "collapse_diagnostics.json").exists()
