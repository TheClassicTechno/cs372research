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


def build_equity_curve(
    episode_log: EpisodeLog,
    initial_cash: float,
    risk_free_rate: float = 0.0,
) -> list[float]:
    """Build a time series of portfolio book values across decision points.

    The curve starts with *initial_cash* (before any trades) and appends
    the after-trade book value at each decision point.

    If *risk_free_rate* is > 0 (annualized, e.g. 0.05), cash earns compounding
    interest between decision points.
    Assumes each step is 1 quarter (0.25 years).

    Falls back to ``final_prices`` for decision points that lack
    ``case_prices`` (backward compat with logs generated before
    case_prices was added).
    """
    curve: list[float] = [initial_cash]
    fallback_prices = episode_log.final_prices

    # Quarterly growth factor: (1 + r)^(1/4)
    q_factor = (1.0 + risk_free_rate) ** 0.25
    interest_cash = 0.0
    for dp in episode_log.decision_point_logs:
        interest_cash = (dp.portfolio_after.cash + interest_cash) * q_factor - dp.portfolio_after.cash
        prices = dp.case_prices if dp.case_prices else fallback_prices
        bv = _book_value(
            dp.portfolio_after.cash + interest_cash,
            dp.portfolio_after.positions,
            prices,
        )
        curve.append(bv)

    print("Total quarterly cash interest", interest_cash)
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
    risk_free_rate: float = 0.0,
) -> FinancialMetrics:
    """Compute all financial metrics for one episode.

    This is the main entry point.  It builds the equity curve from the
    episode's decision point logs and derives all risk-adjusted metrics.
    """
    curve = build_equity_curve(episode_log, initial_cash, risk_free_rate)
    rets = compute_returns(curve)

    # Periodic risk-free rate for Sharpe/Sortino (quarterly)
    rf_q = (1.0 + risk_free_rate) ** 0.25 - 1.0

    dd_abs, dd_pct = max_drawdown(curve)

    sr = sharpe_ratio(rets, risk_free=rf_q)
    so = sortino_ratio(rets, risk_free=rf_q)

    total_return_pct = (
        ((curve[-1] - initial_cash) / initial_cash) * 100.0
        if initial_cash > 0.0
        else 0.0
    )

    calmar = None
    if dd_pct > 0.0 and total_return_pct != 0.0:
        # Annualize the quarterly return for calmar
        ann_return = (1 + total_return_pct / 100.0) ** 4 - 1
        calmar = ann_return / dd_pct

    return FinancialMetrics(
        equity_curve=curve,
        returns=rets,
        total_return_pct=total_return_pct,
        sharpe_ratio=sr,
        sortino_ratio=so,
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct * 100.0,
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
    cagr: float | None = None  # compound annual growth rate
    # SPY benchmark (if available)
    spy_return_pct: float | None = None
    excess_return_pct: float | None = None


def build_daily_equity_curve(
    positions: dict[str, int],
    cash: float,
    initial_value: float,
    daily_prices: dict[str, list[ClosePricePoint]],
    risk_free_rate: float = 0.0,
) -> list[float]:
    """Build a daily equity curve from actual share positions and cash.

    Assumes buy-and-hold at the given weights starting on day 0.
    """
    # Find common trading dates across all tickers that have prices
    tickers_with_prices = [
        t for t in positions if t in daily_prices and daily_prices[t]
    ]

    if not tickers_with_prices:
        return [initial_value]

    # Use dates from the ticker with the most data points as reference
    ref_ticker = max(tickers_with_prices, key=lambda t: len(daily_prices[t]))
    n_days = len(daily_prices[ref_ticker])

    # Daily growth factor: (1 + r)^(1/252)
    d_factor = (1.0 + risk_free_rate) ** (1.0 / 252.0)

    # Build per-ticker price arrays indexed by day
    ticker_prices: dict[str, list[float]] = {}
    for t in positions:
        if t in daily_prices:
            bars = daily_prices[t]
            ticker_prices[t] = [bar.close for bar in bars]

    curve: list[float] = [initial_value]
    interest_wealth = 0.0
    for day_idx in range(n_days):
        if risk_free_rate > 0:
            interest_wealth = (cash + interest_wealth) * d_factor - cash
            
        day_value = cash + interest_wealth
        for t, qty in positions.items():
            if qty == 0:
                continue
            if t in ticker_prices:
                prices = ticker_prices[t]
                if day_idx < len(prices):
                    price = prices[day_idx]
                    if day_idx == n_days - 1:
                        price = round(price, 2)
                    day_value += qty * price
                else:
                    # Beyond available data — hold at last seen price
                    day_value += qty * prices[-1]
            else:
                # No daily data for this ticker — value is 0 (conservative)
                pass
        curve.append(day_value)
    print("Total daily cash interest", interest_wealth, "n days:", n_days)
    return curve


def compute_daily_financial_metrics(
    positions: dict[str, int],
    cash: float,
    initial_value: float,
    daily_prices: dict[str, list[ClosePricePoint]],
    spy_daily: list[ClosePricePoint] | None = None,
    risk_free_rate: float = 0.0,
) -> DailyFinancialMetrics | None:
    """Compute daily-granularity financial metrics for actual positions.

    Returns None if insufficient daily price data is available.

    Parameters
    ----------
    positions:
        {ticker: share_count} resulting from simulation trades.
    cash:
        Cash balance after trades.
    initial_value:
        Starting portfolio value.
    daily_prices:
        {ticker: [ClosePricePoint, ...]} from the MTM case's stock_data.
    spy_daily:
        Optional SPY daily prices for benchmark comparison.
    risk_free_rate:
        Annualized risk-free rate (e.g. 0.05).
    """
    curve = build_daily_equity_curve(
        positions, cash, initial_value, daily_prices, risk_free_rate
    )
    if len(curve) < 2:
        return None

    rets = compute_returns(curve)
    trading_days = len(rets)

    if trading_days < 2:
        return None

    # Daily risk-free rate for Sharpe/Sortino
    rf_d = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    total_return_pct = ((curve[-1] - curve[0]) / curve[0]) * 100.0

    # Annualized metrics (scale daily to annual)
    ann_sr = sharpe_ratio(rets, risk_free=rf_d)
    if ann_sr is not None:
        ann_sr = ann_sr * math.sqrt(252)

    ann_so = sortino_ratio(rets, risk_free=rf_d)
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

    # CAGR: annualize the quarterly return
    cagr_val = None
    if total_return_pct is not None:
        cagr_val = ((1 + total_return_pct / 100.0) ** 4 - 1) * 100.0  # as pct

    return DailyFinancialMetrics(
        equity_curve=curve,
        returns=rets,
        total_return_pct=total_return_pct,
        annualized_sharpe=ann_sr,
        annualized_sortino=ann_so,
        annualized_volatility=ann_vol,
        max_drawdown_pct=dd_pct * 100.0,
        calmar_ratio=calmar,
        trading_days=trading_days,
        cagr=cagr_val,
        spy_return_pct=spy_return_pct,
        excess_return_pct=excess_return_pct,
    )
