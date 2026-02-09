"""Episode-level models."""

from pydantic import BaseModel

from models.decision import ExecutedTrade
from models.portfolio import PortfolioSnapshot


class TradeHistory(BaseModel):
    """List of ExecutedTrade for an episode."""

    agent_id: str
    episode_id: str
    trades: list[ExecutedTrade]


class EpisodeConfig(BaseModel):
    """Episode parameters: episode_id, agent_id, tickers, initial_cash."""

    episode_id: str
    agent_id: str
    tickers: list[str]
    initial_cash: float


class EpisodeResult(BaseModel):
    """Final outcome: trade_history, final_portfolio."""

    episode_id: str
    agent_id: str
    trade_history: TradeHistory
    final_portfolio: PortfolioSnapshot
