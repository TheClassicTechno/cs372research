"""Feature engineering: convert a simulation Case into a debate Observation.

This module lives in the Environment Layer (simulation/) and is responsible
for computing derived features (returns, volatility) from raw case data
before passing an Observation to the Generation Layer.

See documentation/integration_plan.md §5 — ``simulation/feature_engineering.py``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from models.case import Case
from multi_agent.models import MarketState, Observation, PortfolioState


def build_observation(case: Case) -> Observation:
    """Convert a simulation ``Case`` into a debate ``Observation``.

    Feature engineering:
        - prices:     extracted from stock_data[t].current_price
        - returns:    computed from daily_bars (last close vs first close)
        - volatility: std-dev of daily log-returns over the bar window
        - portfolio:  cash + positions (int → float)
        - text_context: flattened from case_data items
        - timestamp:  from case metadata or fallback to now
    """
    prices: dict[str, float] = {}
    returns: dict[str, float] = {}
    volatility: dict[str, float] = {}

    for ticker, sd in case.stock_data.items():
        prices[ticker] = sd.current_price

        # Compute returns and volatility from daily bars.
        bars = sd.daily_bars
        if len(bars) >= 2:
            closes = [b.close for b in bars]
            # Simple return over the interval.
            returns[ticker] = (closes[-1] - closes[0]) / closes[0]
            # Volatility: std-dev of daily log-returns.
            log_rets = [
                math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            if len(log_rets) >= 2:
                mean_lr = sum(log_rets) / len(log_rets)
                variance = sum((lr - mean_lr) ** 2 for lr in log_rets) / (
                    len(log_rets) - 1
                )
                volatility[ticker] = math.sqrt(variance)
            elif len(log_rets) == 1:
                volatility[ticker] = abs(log_rets[0])
            else:
                volatility[ticker] = 0.0
        else:
            returns[ticker] = 0.0
            volatility[ticker] = 0.0

    market_state = MarketState(
        prices=prices,
        returns=returns if returns else None,
        volatility=volatility if volatility else None,
    )

    # Portfolio: int positions → float for debate models.
    portfolio_state = PortfolioState(
        cash=case.portfolio.cash,
        positions={t: float(q) for t, q in case.portfolio.positions.items()},
    )

    # Flatten case_data items into text_context.
    text_parts: list[str] = []
    for item in case.case_data.items:
        label = item.kind.upper() if item.kind != "other" else "INFO"
        text_parts.append(f"[{label}] {item.content}")
    text_context = "\n".join(text_parts) if text_parts else None

    timestamp = case.information_cutoff_timestamp or datetime.now(
        timezone.utc
    ).isoformat()

    return Observation(
        timestamp=timestamp,
        universe=case.tickers,
        market_state=market_state,
        text_context=text_context,
        portfolio_state=portfolio_state,
    )
