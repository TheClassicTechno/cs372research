"""Unit tests for eval.financial — equity curve, Sharpe, Sortino, drawdown."""

import math

import pytest

from eval.financial import (
    FinancialMetrics,
    build_equity_curve,
    compute_financial_metrics,
    compute_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from models.decision import Decision
from models.log import DecisionPointLog, EpisodeLog
from models.portfolio import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dp(
    idx: int,
    cash_before: float,
    positions_before: dict,
    cash_after: float,
    positions_after: dict,
    prices: dict,
) -> DecisionPointLog:
    return DecisionPointLog(
        case_id=f"ep:dp_{idx}",
        decision_point_idx=idx,
        portfolio_before=PortfolioSnapshot(cash=cash_before, positions=positions_before),
        portfolio_after=PortfolioSnapshot(cash=cash_after, positions=positions_after),
        extracted_decision=Decision(orders=[]),
        case_prices=prices,
    )


def _episode(dps: list[DecisionPointLog], final_prices: dict | None = None) -> EpisodeLog:
    last = dps[-1] if dps else None
    return EpisodeLog(
        episode_id="ep_000",
        agent_id="test",
        decision_point_logs=dps,
        final_portfolio=last.portfolio_after if last else PortfolioSnapshot(cash=100_000, positions={}),
        final_prices=final_prices or (last.case_prices if last else {}),
    )


# ---------------------------------------------------------------------------
# build_equity_curve
# ---------------------------------------------------------------------------

class TestBuildEquityCurve:
    def test_no_decision_points(self):
        ep = _episode([])
        curve = build_equity_curve(ep, 100_000)
        assert curve == [100_000]

    def test_single_decision_hold(self):
        dp = _dp(0, 100_000, {}, 100_000, {}, {"NVDA": 100.0})
        ep = _episode([dp])
        curve = build_equity_curve(ep, 100_000)
        assert curve == [100_000, 100_000]

    def test_buy_then_price_up(self):
        dp0 = _dp(0, 100_000, {}, 90_000, {"NVDA": 100}, {"NVDA": 100.0})
        dp1 = _dp(1, 90_000, {"NVDA": 100}, 90_000, {"NVDA": 100}, {"NVDA": 110.0})
        ep = _episode([dp0, dp1])
        curve = build_equity_curve(ep, 100_000)
        assert curve == [100_000, 100_000, 101_000]

    def test_fallback_to_final_prices(self):
        dp = DecisionPointLog(
            case_id="ep:dp_0",
            decision_point_idx=0,
            portfolio_before=PortfolioSnapshot(cash=100_000, positions={}),
            portfolio_after=PortfolioSnapshot(cash=90_000, positions={"NVDA": 100}),
            extracted_decision=Decision(orders=[]),
        )
        ep = _episode([dp], final_prices={"NVDA": 105.0})
        curve = build_equity_curve(ep, 100_000)
        assert curve == [100_000, 100_500]


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_empty(self):
        assert compute_returns([]) == []
        assert compute_returns([100]) == []

    def test_simple(self):
        rets = compute_returns([100, 110, 99])
        assert len(rets) == 2
        assert pytest.approx(rets[0], abs=1e-9) == 0.1
        assert pytest.approx(rets[1], abs=1e-9) == -0.1


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_none_for_too_few(self):
        assert sharpe_ratio([]) is None
        assert sharpe_ratio([0.05]) is None

    def test_none_for_zero_std(self):
        assert sharpe_ratio([0.05, 0.05, 0.05]) is None

    def test_positive(self):
        result = sharpe_ratio([0.01, 0.02, 0.03])
        assert result is not None
        assert result > 0

    def test_risk_free(self):
        sr_no_rf = sharpe_ratio([0.01, 0.02, 0.03], risk_free=0.0)
        sr_with_rf = sharpe_ratio([0.01, 0.02, 0.03], risk_free=0.01)
        assert sr_no_rf is not None
        assert sr_with_rf is not None
        assert sr_with_rf < sr_no_rf


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_none_for_too_few(self):
        assert sortino_ratio([]) is None
        assert sortino_ratio([0.05]) is None

    def test_none_when_no_downside(self):
        assert sortino_ratio([0.05, 0.10, 0.03]) is None

    def test_negative_returns(self):
        result = sortino_ratio([0.05, -0.02, 0.03, -0.01])
        assert result is not None

    def test_denominator_uses_all_observations(self):
        """Downside deviation divides by N (total), not just count of negatives.

        Sortino & Price (1994): DD = sqrt( sum(min(r-T,0)^2) / N ).
        With returns [0.10, -0.10] and T=0:
            DD = sqrt(0.01 / 2) = sqrt(0.005) ≈ 0.07071
            mean = 0.0, Sortino = 0.0 / 0.07071 = 0.0
        If we incorrectly divided by 1 (count of negatives):
            DD = sqrt(0.01 / 1) = 0.10, Sortino = 0.0
        Both give 0 here, so use an asymmetric example:
        returns [0.10, 0.10, -0.10], T=0:
            mean = 0.0333..., DD = sqrt(0.01 / 3) ≈ 0.05774
            Sortino = 0.0333 / 0.05774 ≈ 0.5774
        If incorrectly divided by 1:
            DD = sqrt(0.01 / 1) = 0.10
            Sortino = 0.0333 / 0.10 = 0.333  (wrong, too low)
        """
        rets = [0.10, 0.10, -0.10]
        result = sortino_ratio(rets, risk_free=0.0)
        assert result is not None
        expected_dd = math.sqrt(0.01 / 3)
        expected = (sum(rets) / 3) / expected_dd
        assert pytest.approx(result, abs=1e-6) == expected


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_empty(self):
        assert max_drawdown([]) == (0.0, 0.0)
        assert max_drawdown([100]) == (0.0, 0.0)

    def test_no_drawdown(self):
        dd_abs, dd_pct = max_drawdown([100, 110, 120])
        assert dd_abs == 0.0
        assert dd_pct == 0.0

    def test_simple_drawdown(self):
        dd_abs, dd_pct = max_drawdown([100, 120, 90, 110])
        assert dd_abs == 30.0
        assert pytest.approx(dd_pct, abs=1e-9) == 30.0 / 120.0

    def test_multiple_drawdowns_picks_max(self):
        dd_abs, dd_pct = max_drawdown([100, 80, 90, 70])
        assert dd_abs == 30.0
        assert pytest.approx(dd_pct, abs=1e-9) == 30.0 / 100.0

    def test_abs_and_pct_tracked_independently(self):
        """Max absolute and max percentage drawdowns may come from
        different peak/trough pairs and must be tracked separately.

        [10, 1, 1000, 500]:
            Peak 10  -> 1:    abs=9,   pct=90%
            Peak 1000 -> 500: abs=500, pct=50%
        Max abs = 500, max pct = 90%.
        """
        dd_abs, dd_pct = max_drawdown([10, 1, 1000, 500])
        assert dd_abs == 500.0
        assert pytest.approx(dd_pct, abs=1e-9) == 0.9


# ---------------------------------------------------------------------------
# compute_financial_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeFinancialMetrics:
    def test_full_pipeline(self):
        dp0 = _dp(0, 100_000, {}, 90_000, {"NVDA": 100}, {"NVDA": 100.0})
        dp1 = _dp(1, 90_000, {"NVDA": 100}, 90_000, {"NVDA": 100}, {"NVDA": 110.0})
        ep = _episode([dp0, dp1])

        metrics = compute_financial_metrics(ep, 100_000)

        assert isinstance(metrics, FinancialMetrics)
        assert len(metrics.equity_curve) == 3
        assert len(metrics.returns) == 2
        assert pytest.approx(metrics.total_return_pct, abs=0.01) == 1.0
        assert metrics.max_drawdown >= 0.0

    def test_hold_only(self):
        dp = _dp(0, 100_000, {}, 100_000, {}, {"NVDA": 100.0})
        ep = _episode([dp])
        metrics = compute_financial_metrics(ep, 100_000)
        assert metrics.total_return_pct == 0.0
        assert metrics.max_drawdown == 0.0
