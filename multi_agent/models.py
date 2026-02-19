"""
Pydantic models for the CS372 Multi-Agent Trading System.
Matches the TypeScript types in agents/types.ts for cross-language compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# PEARL LEVELS (for causal claim classification)
# =============================================================================


class PearlLevel(str, Enum):
    """Pearl's Ladder of Causation levels for causal claim classification."""

    L1 = "L1"  # Association: "X is associated with Y"
    L2 = "L2"  # Intervention: "If we do X, Y changes"
    L3 = "L3"  # Counterfactual: "Had X not occurred, Y would..."


# =============================================================================
# OBSERVATION (input from simulator to agents)
# =============================================================================


class MarketState(BaseModel):
    """Snapshot of market data for the universe."""

    prices: dict[str, float]
    returns: Optional[dict[str, float]] = None
    volatility: Optional[dict[str, float]] = None


class PortfolioState(BaseModel):
    """Current portfolio holdings and cash."""

    cash: float
    positions: dict[str, float]
    exposures: Optional[dict[str, float]] = None


class Constraints(BaseModel):
    """Trading constraints / risk limits."""

    max_leverage: Optional[float] = Field(None, alias="maxLeverage")
    max_position_size: Optional[float] = Field(None, alias="maxPositionSize")
    risk_limits: Optional[dict[str, float]] = Field(None, alias="riskLimits")

    model_config = {"populate_by_name": True}


class Observation(BaseModel):
    """
    Input from the simulator at each decision point.
    Contains market state, portfolio state, and optional text context.
    """

    timestamp: str  # ISO 8601
    universe: list[str]  # ticker symbols
    market_state: MarketState
    text_context: Optional[str] = None  # news / earnings snippets
    portfolio_state: PortfolioState
    constraints: Optional[Constraints] = None


# =============================================================================
# ACTION (output from agents to broker)
# =============================================================================


class Order(BaseModel):
    """A single trade order."""

    ticker: str
    side: str  # "buy" | "sell"
    size: float  # shares
    type: str = "market"
    limit_price: Optional[float] = Field(None, alias="limitPrice")

    model_config = {"populate_by_name": True}


class Claim(BaseModel):
    """
    Machine-readable causal claim for T3/CRIT-style evaluation.
    Each claim is classified by Pearl level for reasoning quality scoring.
    """

    claim_text: str
    pearl_level: PearlLevel = PearlLevel.L1
    variables: list[str] = Field(default_factory=list)
    assumptions: Optional[list[str]] = None
    timestamp_dependency: Optional[str] = None
    confidence: float = 0.5


class Action(BaseModel):
    """Output from agents: trading orders + auditable reasoning."""

    orders: list[Order] = Field(default_factory=list)
    justification: str = ""
    confidence: float = 0.5
    claims: list[Claim] = Field(default_factory=list)


# =============================================================================
# DEBATE & TRACE (for multi-agent architectures)
# =============================================================================


class DebateTurn(BaseModel):
    """A single turn in a multi-agent debate."""

    round: int
    agent_id: str
    role: Optional[str] = None
    proposal: Optional[Action] = None
    critique: Optional[str] = None
    objections: Optional[list[str]] = None
    revision: Optional[Action] = None


class AgentTrace(BaseModel):
    """
    Full auditable trace of an agent decision.
    Used by the eval team for T3/CRIT scoring and ablation analysis.
    """

    observation_timestamp: str
    architecture: str  # "single" | "majority_vote" | "debate"
    what_i_saw: str
    hypothesis: str
    decision: str
    risks_or_falsifiers: Optional[str] = None
    strongest_objection: Optional[str] = None
    debate_turns: Optional[list[DebateTurn]] = None
    action: Action
    logged_at: str


# =============================================================================
# PIPELINE OUTPUT (for preprocessing agents)
# =============================================================================


class PipelineOutput(BaseModel):
    """Output from a pipeline preprocessing agent (NewsDigester or DataAnalyst)."""

    agent_type: str  # "news_digest" | "data_analysis"
    summary: str
    key_signals: list[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1
    confidence: float = 0.5
