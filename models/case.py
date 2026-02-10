"""Case and market data models."""

from typing import Literal

from pydantic import BaseModel

from models.portfolio import PortfolioSnapshot


class PricePoint(BaseModel):
    """One day's price bar (OHLCV) within the interval between decision points."""

    timestamp: str  # Date/datetime for this bar (e.g. YYYY-MM-DD or ISO8601)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class IntervalPriceSummary(BaseModel):
    """Summary of price movement over an interval (e.g. between decision points).

    Optional lighter-weight payload for the agent instead of full daily_bars.
    Simulation may populate this in StockData when passing case to the agent.
    """

    open: float
    close: float
    high: float
    low: float
    volume: float = 0.0


class CaseDataItem(BaseModel):
    """Single item in case information (e.g. earnings report, news headline)."""

    kind: Literal["earnings", "news", "other"] = "other"
    content: str


class CaseData(BaseModel):
    """Variable-length list of information for the case (earnings, news, etc.)."""

    items: list[CaseDataItem] = []


class StockData(BaseModel):
    """Per-ticker price data for the interval between previous and current decision point.

    daily_bars: full series (each bar has timestamp). interval_summary: optional
    single summary for the interval; simulation may set this for a lighter
    agent payload (e.g. in for_agent() use interval_summary when present).
    """

    ticker: str
    current_price: float
    daily_bars: list[PricePoint]  # One bar per day in the interval
    interval_summary: IntervalPriceSummary | None = None  # Optional summary for agent


class Case(BaseModel):
    """Single decision scenario: case data, price statistics, portfolio.

    Wraps CaseData, price statistics (per-ticker), and current portfolio value.
    Metadata (case_id, decision_point_idx, information_cutoff_timestamp) is
    stored for logging/reproducibility but should not be passed to the agent.
    """

    # Agent-facing: passed in the prompt
    case_data: CaseData
    stock_data: dict[str, StockData]
    portfolio: PortfolioSnapshot

    # Metadata: stored but not passed to the agent
    case_id: str  # Required; must uniquely identify this case (e.g. f"{episode_id}:{decision_point_idx}")
    decision_point_idx: int = 0
    information_cutoff_timestamp: str | None = None

    @property
    def tickers(self) -> list[str]:
        """Tickers covered by this case (from stock_data keys)."""
        return list(self.stock_data.keys())

    def for_agent(self) -> dict:
        """Payload to send to the agent (excludes metadata)."""
        return self.model_dump(
            include={"case_data", "stock_data", "portfolio"},
            exclude_none=True,
        )
