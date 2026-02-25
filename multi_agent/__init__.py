"""
CS372 Multi-Agent Trading System — LangGraph Orchestrator
Juli's multi-agent debate implementation with:
  - Enriched role-specific agents (macro, value, risk, technical, sentiment)
  - Pipeline preprocessing (NewsDigester, DataAnalyst)
  - Configurable debate rounds with agreeableness knob
  - Adversarial/devil's advocate mode
  - Full LangGraph orchestration
"""

from .models import (
    Observation, Action, Order, Claim, AgentTrace, DebateTurn,
    PearlLevel, MarketState, PortfolioState, PipelineOutput,
)
from .config import DebateConfig, AgentRole
from .runner import MultiAgentRunner
from .majority_vote_runner import MajorityVoteRunner

__all__ = [
    "Observation", "Action", "Order", "Claim", "AgentTrace", "DebateTurn",
    "PearlLevel", "MarketState", "PortfolioState", "PipelineOutput",
    "DebateConfig", "AgentRole", "MultiAgentRunner", "MajorityVoteRunner",
]
