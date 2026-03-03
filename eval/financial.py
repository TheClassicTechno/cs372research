"""
Financial performance metrics for simulation episodes.

Computes equity curves, risk-adjusted returns, and drawdown statistics
from EpisodeLog data.  Pure functions — no side effects, no LLM calls.

Pipeline context:
    EpisodeLog (with case_prices per DecisionPointLog)
        │
        ▼
    build_equity_curve()  →  list of book values
        │
        ▼
    compute_returns()     →  per-step simple returns
        │
        ├── sharpe_ratio()
        ├── sortino_ratio()
        └── max_drawdown()
                │
                ▼
    FinancialMetrics (Pydantic model for downstream consumers)

    TODO: Ratios and drawdown should likely be computed over daily price rather
    than just decision points to give more realistic values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from models.log import EpisodeLog


class FinancialMetrics(BaseModel):
    """Computed financial performance for one episode."""

    equity_curve: list[float]
    returns: list[float]
    total_return_pct: float
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float | None = None


def _book_value(cash: float, positions: dict[str, int], prices: dict[str, float]) -> float:
    """Mark-to-market portfolio value."""
    return cash + sum(
        qty * prices.get(ticker, 0.0) for ticker, qty in positions.items()
    )


def build_equity_curve(episode_log: EpisodeLog, initial_cash: float) -> list[float]:
    """Build a time series of portfolio book values across decision points.

    The curve starts with *initial_cash* (before any trades) and appends
    the after-trade book value at each decision point.

    Falls back to ``final_prices`` for decision points that lack
    ``case_prices`` (backward compat with logs generated before
    case_prices was added).
    """
    curve: list[float] = [initial_cash]
    fallback_prices = episode_log.final_prices

    for dp in episode_log.decision_point_logs:
        # TODO: Cash should add interest rate here if it isn't added by sim
        prices = dp.case_prices if dp.case_prices else fallback_prices
        bv = _book_value(
            dp.portfolio_after.cash,
            dp.portfolio_after.positions,
            prices,
        )
        curve.append(bv)

    return curve


def compute_returns(equity_curve: list[float]) -> list[float]:
    """Simple per-step returns from an equity curve.

    Returns an empty list if there are fewer than two data points.
    """
    if len(equity_curve) < 2:
        return []
    return [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        if equity_curve[i - 1] != 0.0
        else 0.0
        for i in range(1, len(equity_curve))
    ]


def sharpe_ratio(returns: list[float], risk_free: float = 0.0) -> float | None:
    """Sharpe ratio: (mean excess return) / std(returns).

    Returns None if fewer than 2 returns or zero standard deviation.
    """
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    excess = mean_r - risk_free
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if math.fabs(std) < 1e-12:
        return None
    return excess / std


def sortino_ratio(returns: list[float], risk_free: float = 0.0) -> float | None:
    """Sortino ratio: (mean excess return) / downside deviation.

    Downside deviation (Sortino & Price 1994) is the square root of the
    average squared shortfall across **all** observations — returns above
    the target contribute zero, but are still counted in N.

    Returns None if fewer than 2 returns or zero downside deviation
    (i.e. no returns fall below the target).
    """
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    excess = mean_r - risk_free
    downside_sum = sum(
        (r - risk_free) ** 2 for r in returns if r < risk_free
    )
    if downside_sum == 0.0:
        return None
    downside_dev = math.sqrt(downside_sum / len(returns))
    if downside_dev == 0.0:
        return None
    return excess / downside_dev


def max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Maximum drawdown in absolute and percentage terms.

    Returns (abs_drawdown, pct_drawdown).  Both are non-negative and
    tracked **independently** — the maximum absolute drawdown and the
    maximum percentage drawdown may occur at different points in the
    curve (e.g. a small portfolio losing 90% vs a large portfolio
    losing a bigger dollar amount but smaller fraction).

    Returns (0.0, 0.0) for curves with fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for val in equity_curve[1:]:
        if val > peak:
            peak = val
        dd_abs = peak - val
        dd_pct = dd_abs / peak if peak > 0.0 else 0.0
        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd_abs, max_dd_pct


def compute_financial_metrics(
    episode_log: EpisodeLog,
    initial_cash: float,
) -> FinancialMetrics:
    """Compute all financial metrics for one episode.

    This is the main entry point.  It builds the equity curve from the
    episode's decision point logs and derives all risk-adjusted metrics.
    """
    curve = build_equity_curve(episode_log, initial_cash)
    rets = compute_returns(curve)

    dd_abs, dd_pct = max_drawdown(curve)

    sr = sharpe_ratio(rets)
    so = sortino_ratio(rets)

    total_return_pct = (
        ((curve[-1] - initial_cash) / initial_cash) * 100.0
        if initial_cash > 0.0
        else 0.0
    )

    calmar = None
    if dd_pct > 0.0 and total_return_pct != 0.0:
        calmar = (total_return_pct / 100.0) / dd_pct

    return FinancialMetrics(
        equity_curve=curve,
        returns=rets,
        total_return_pct=total_return_pct,
        sharpe_ratio=sr,
        sortino_ratio=so,
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        calmar_ratio=calmar,
    )
