"""
Financial performance metrics for simulation episodes.

Computes equity curves, risk-adjusted returns, and drawdown statistics
from EpisodeLog data.  Pure functions — no side effects, no LLM calls.

Pipeline context — two granularities:

  1. Decision-point granularity (original):
     EpisodeLog → build_equity_curve() → compute_returns() → metrics

  2. Daily granularity (new):
     allocation weights + daily prices →
       build_daily_equity_curve() → compute_returns() → metrics
     Uses ~60 daily data points per quarter for realistic Sharpe/Sortino/drawdown.

Both produce FinancialMetrics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from models.case import ClosePricePoint
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


# ---------------------------------------------------------------------------
# Daily-granularity metrics (allocation weights + daily prices)
# ---------------------------------------------------------------------------


class DailyFinancialMetrics(BaseModel):
    """Financial metrics computed from daily price data over one quarter."""

    equity_curve: list[float]
    returns: list[float]
    total_return_pct: float
    annualized_sharpe: float | None = None
    annualized_sortino: float | None = None
    annualized_volatility: float | None = None
    max_drawdown_pct: float = 0.0
    calmar_ratio: float | None = None
    trading_days: int = 0
    # SPY benchmark (if available)
    spy_return_pct: float | None = None
    excess_return_pct: float | None = None


def build_daily_equity_curve(
    allocation: dict[str, float],
    initial_value: float,
    daily_prices: dict[str, list[ClosePricePoint]],
) -> list[float]:
    """Build a daily equity curve from allocation weights and daily prices.

    Assumes buy-and-hold at the given weights starting on day 0.
    For each day t, portfolio value = sum over tickers of:
        weight_i * initial_value * (price_t / price_0)

    Tickers in the allocation that lack daily price data are treated as
    earning zero return (their portion stays constant at initial weight).

    Parameters
    ----------
    allocation:
        {ticker: weight} where weights sum to ~1.0.
    initial_value:
        Starting portfolio value (e.g. 100_000).
    daily_prices:
        {ticker: [ClosePricePoint, ...]} sorted by date ascending.

    Returns
    -------
    List of daily portfolio values, starting from day 0.
    """
    if not allocation or initial_value <= 0.0:
        return [initial_value]

    # Find common trading dates across all tickers that have prices
    tickers_with_prices = [
        t for t in allocation if t in daily_prices and daily_prices[t]
    ]

    if not tickers_with_prices:
        return [initial_value]

    # Use dates from the ticker with the most data points as reference
    ref_ticker = max(tickers_with_prices, key=lambda t: len(daily_prices[t]))
    n_days = len(daily_prices[ref_ticker])

    # Build per-ticker price arrays indexed by day
    # For tickers without data, their contribution stays flat
    ticker_prices: dict[str, list[float]] = {}
    for t in tickers_with_prices:
        bars = daily_prices[t]
        ticker_prices[t] = [bar.close for bar in bars]

    curve: list[float] = []
    for day_idx in range(n_days):
        day_value = 0.0
        for t, weight in allocation.items():
            if weight <= 0.0:
                continue
            if t in ticker_prices:
                prices = ticker_prices[t]
                if day_idx < len(prices) and prices[0] > 0.0:
                    day_value += weight * initial_value * (prices[day_idx] / prices[0])
                else:
                    # Beyond available data or zero start price — hold flat
                    day_value += weight * initial_value
            else:
                # No daily data for this ticker — hold flat
                day_value += weight * initial_value
        curve.append(day_value)

    return curve


def compute_daily_financial_metrics(
    allocation: dict[str, float],
    initial_value: float,
    daily_prices: dict[str, list[ClosePricePoint]],
    spy_daily: list[ClosePricePoint] | None = None,
) -> DailyFinancialMetrics | None:
    """Compute daily-granularity financial metrics for an allocation.

    Returns None if insufficient daily price data is available.

    Parameters
    ----------
    allocation:
        {ticker: weight} from the debate output.
    initial_value:
        Starting portfolio value.
    daily_prices:
        {ticker: [ClosePricePoint, ...]} from the MTM case's stock_data.
    spy_daily:
        Optional SPY daily prices for benchmark comparison.
    """
    curve = build_daily_equity_curve(allocation, initial_value, daily_prices)

    if len(curve) < 2:
        return None

    rets = compute_returns(curve)
    trading_days = len(rets)

    if trading_days < 2:
        return None

    total_return_pct = ((curve[-1] - curve[0]) / curve[0]) * 100.0

    # Annualized metrics (scale daily to annual)
    ann_sr = sharpe_ratio(rets)
    if ann_sr is not None:
        ann_sr = ann_sr * math.sqrt(252)

    ann_so = sortino_ratio(rets)
    if ann_so is not None:
        ann_so = ann_so * math.sqrt(252)

    # Annualized volatility
    mean_r = sum(rets) / len(rets)
    variance = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    ann_vol = math.sqrt(variance) * math.sqrt(252) if variance > 0 else None

    _, dd_pct = max_drawdown(curve)

    calmar = None
    if dd_pct > 0.0 and total_return_pct != 0.0:
        # Annualize the quarterly return for calmar
        ann_return = (1 + total_return_pct / 100.0) ** 4 - 1
        calmar = ann_return / dd_pct

    # SPY benchmark
    spy_return_pct = None
    excess_return_pct = None
    if spy_daily and len(spy_daily) >= 2:
        spy_start = spy_daily[0].close
        spy_end = spy_daily[-1].close
        if spy_start > 0.0:
            spy_return_pct = ((spy_end - spy_start) / spy_start) * 100.0
            excess_return_pct = total_return_pct - spy_return_pct

    return DailyFinancialMetrics(
        equity_curve=curve,
        returns=rets,
        total_return_pct=total_return_pct,
        annualized_sharpe=ann_sr,
        annualized_sortino=ann_so,
        annualized_volatility=ann_vol,
        max_drawdown_pct=dd_pct,
        calmar_ratio=calmar,
        trading_days=trading_days,
        spy_return_pct=spy_return_pct,
        excess_return_pct=excess_return_pct,
    )
