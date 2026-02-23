"""Case and market data models."""

from typing import Literal

from pydantic import BaseModel

from models.portfolio import PortfolioSnapshot


class ClosePricePoint(BaseModel):
    """One day's close price within the interval between decision points."""

    timestamp: str  # Date/datetime for this bar (e.g. YYYY-MM-DD or ISO8601)
    close: float


class IntervalPriceSummary(BaseModel):
    """Summary of price movement over an interval (e.g. between decision points).

    Optional lighter-weight payload for the agent instead of full daily_bars.
    Simulation may populate this in StockData when passing case to the agent.
    """

    open: float
    close: float


class CaseDataItem(BaseModel):
    """Single item in case information (e.g. earnings report, news headline).

    impact_score is used for top-N filtering at load time; it is never passed
    to the agent (excluded in for_agent).
    """

    kind: Literal["earnings", "news", "other"] = "other"
    content: str
    impact_score: float | None = None


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
    daily_bars: list[ClosePricePoint]  # One bar per day in the interval
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
    portfolio: PortfolioSnapshot = PortfolioSnapshot(cash=0, positions={})

    # Metadata: stored but not passed to the agent.
    # Defaults allow Case to be loaded from disk as a template (without
    # runtime fields); the simulation runner stamps these via model_copy().
    case_id: str = ""
    decision_point_idx: int = 0
    information_cutoff_timestamp: str | None = None

    @property
    def tickers(self) -> list[str]:
        """Tickers covered by this case (from stock_data keys)."""
        return list(self.stock_data.keys())

    def for_agent(self) -> dict:
        """Payload to send to the agent (excludes metadata and impact_score)."""
        return {
            "case_data": {
                "items": [
                    {"kind": item.kind, "content": item.content}
                    for item in self.case_data.items
                ]
            },
            "stock_data": {k: v.model_dump() for k, v in self.stock_data.items()},
            "portfolio": self.portfolio.model_dump(),
        }
