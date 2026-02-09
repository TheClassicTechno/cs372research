"""Portfolio state models."""

from pydantic import BaseModel


class PortfolioSnapshot(BaseModel):
    """Cash and positions (ticker -> shares) at a decision point.

    Used as an attribute of Case and set dynamically by the simulation
    at each decision point. Other case attributes (case_data, stock_data)
    can be pre-computed or pre-formatted.
    """

    cash: float
    positions: dict[str, int]
