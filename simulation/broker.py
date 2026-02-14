"""In-process broker: portfolio management and trade execution.

The broker validates and executes decisions using all-or-nothing semantics.
It tracks portfolio state (cash + positions) and trade history across an episode.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from models.decision import Decision, DecisionResult, ExecutedTrade, Order
from models.portfolio import PortfolioSnapshot
from models.config import BrokerConfig

if TYPE_CHECKING:
    from models.case import Case


class Broker:
    """Stateful broker that validates, executes, and records trades for one episode.

    Instantiate one ``Broker`` per episode. The broker owns the canonical
    portfolio state and produces ``DecisionResult`` objects that flow back to
    the agent via the ``submit_decision`` tool.
    """

    def __init__(self, config: BrokerConfig, tickers: list[str]) -> None:
        self._config = config
        self._allowed_tickers = set(tickers)
        self._cash: float = config.initial_cash
        self._positions: dict[str, int] = {}
        self._trade_history: list[ExecutedTrade] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_portfolio(self) -> PortfolioSnapshot:
        """Return a snapshot of the current portfolio state."""
        return PortfolioSnapshot(cash=self._cash, positions=dict(self._positions))

    def get_trade_history(self) -> list[ExecutedTrade]:
        """Return the full list of executed trades so far."""
        return list(self._trade_history)

    def execute_decision(
        self,
        decision: Decision,
        case: Case,
        agent_id: str,
    ) -> DecisionResult:
        """Validate and execute *decision* against the current portfolio.

        Execution is **all-or-nothing**: if any order fails validation the
        entire decision is rejected with a descriptive message.  When accepted,
        sells are processed before buys so that proceeds are available.

        Returns a ``DecisionResult`` that is fed back to the agent as tool
        output.
        """
        if not decision.orders:
            # Empty orders = hold; accepted with no trades.
            return DecisionResult(
                status="accepted",
                executed_trades=[],
                message="Hold — no orders submitted.",
            )

        # ---- Phase 1: Validate all orders --------------------------------
        rejection = self._validate_orders(decision.orders, case)
        if rejection is not None:
            return DecisionResult(
                status="rejected",
                executed_trades=[],
                message=rejection,
            )

        # ---- Phase 2: Simulate execution (sells then buys) ---------------
        # We work on copies so we can roll back on failure.
        sim_cash = self._cash
        sim_positions = dict(self._positions)

        # Sort: sells first, buys second.
        indexed_orders = list(enumerate(decision.orders))
        sells = [(i, o) for i, o in indexed_orders if o.side == "sell"]
        buys = [(i, o) for i, o in indexed_orders if o.side == "buy"]

        executed: list[ExecutedTrade] = []

        for order_index, order in sells + buys:
            price = self._resolve_price(order.ticker, case)
            trade_value = price * order.quantity

            if order.side == "sell":
                held = sim_positions.get(order.ticker, 0)
                if order.quantity > held:
                    return DecisionResult(
                        status="rejected",
                        executed_trades=[],
                        message=(
                            f"Cannot sell {order.quantity} shares of {order.ticker} — "
                            f"only {held} held."
                        ),
                    )
                sim_positions[order.ticker] = held - order.quantity
                if sim_positions[order.ticker] == 0:
                    del sim_positions[order.ticker]
                sim_cash += trade_value
            else:  # buy
                if trade_value > sim_cash:
                    return DecisionResult(
                        status="rejected",
                        executed_trades=[],
                        message=(
                            f"Insufficient cash to buy {order.quantity} shares of "
                            f"{order.ticker} at ${price:.2f} (cost ${trade_value:.2f}, "
                            f"available ${sim_cash:.2f})."
                        ),
                    )
                sim_cash -= trade_value
                sim_positions[order.ticker] = (
                    sim_positions.get(order.ticker, 0) + order.quantity
                )

            executed.append(
                ExecutedTrade(
                    trade_id=uuid.uuid4().hex[:12],
                    case_id=case.case_id,
                    agent_id=agent_id,
                    order_index=order_index,
                    ticker=order.ticker,
                    side=order.side,
                    quantity=order.quantity,
                    price=price,
                )
            )

        # ---- Phase 3: Commit -------------------------------------------
        self._cash = sim_cash
        self._positions = sim_positions
        self._trade_history.extend(executed)

        return DecisionResult(
            status="accepted",
            executed_trades=executed,
            message=f"Executed {len(executed)} trade(s).",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_orders(self, orders: list[Order], case: Case) -> str | None:
        """Return an error message if any order is invalid, else ``None``."""
        case_tickers = set(case.tickers)
        invalid_tickers = [o.ticker for o in orders if o.ticker not in case_tickers]
        if invalid_tickers:
            return (
                f"Ticker(s) not in case universe: {', '.join(sorted(set(invalid_tickers)))}. "
                f"Allowed: {', '.join(sorted(case_tickers))}."
            )

        not_in_broker = [o.ticker for o in orders if o.ticker not in self._allowed_tickers]
        if not_in_broker:
            return (
                f"Ticker(s) not in broker universe: {', '.join(sorted(set(not_in_broker)))}."
            )

        for order in orders:
            if order.quantity <= 0:
                return f"Order quantity must be positive, got {order.quantity} for {order.ticker}."

        return None

    @staticmethod
    def _resolve_price(ticker: str, case: Case) -> float:
        """Get the execution price for *ticker* from the case's stock data.

        Uses ``current_price`` from the ``StockData`` entry for the ticker.
        """
        stock = case.stock_data.get(ticker)
        if stock is None:
            raise ValueError(f"No stock data for ticker {ticker} in case {case.case_id}")
        return stock.current_price
