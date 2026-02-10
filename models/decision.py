"""Agent output and execution models: Order, Decision, ExecutedTrade, DecisionResult."""

from typing import Literal

from pydantic import BaseModel


class Order(BaseModel):
    """Single order: ticker, side, quantity."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int


class Decision(BaseModel):
    """Buy/sell/hold. Empty orders = hold."""

    orders: list[Order]


class ExecutedTrade(BaseModel):
    """Single executed fill. Produced by the simulation from one Order."""

    trade_id: str
    case_id: str
    agent_id: str
    order_index: int  # Index in Decision.orders that produced this trade
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    commission: float = 0.0


class DecisionResult(BaseModel):
    """Simulation response to submit_decision (tool output to agent).

    Execution is all-or-nothing: either all orders are accepted and executed,
    or the decision is rejected (e.g. if any ticker is not in Case.tickers,
    or execution constraints are violated). When rejected, message must
    explain the reason (e.g. which tickers were not in the case universe).
    """

    status: Literal["accepted", "rejected"]
    executed_trades: list[ExecutedTrade]  # Non-empty only when status is "accepted"
    message: str = ""  # When rejected, must mention reason (e.g. ticker not in Case.tickers)
