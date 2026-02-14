"""Logging and experiment storage models.

- ``DecisionPointLog`` — per-decision audit with evaluation data.
- ``EpisodeLog`` — full episode audit trail.
- ``SimulationLog`` — run-level log with embedded config for reproducibility.
"""

from __future__ import annotations

from pydantic import BaseModel

from models.config import SimulationConfig
from models.decision import Decision, DecisionResult, ExecutedTrade
from models.portfolio import PortfolioSnapshot


class DecisionPointLog(BaseModel):
    """Per-decision audit with evaluation data.

    Captures the full before/after portfolio state so per-step P&L can be
    computed during evaluation without re-running the simulation.
    """

    case_id: str
    decision_point_idx: int
    portfolio_before: PortfolioSnapshot
    portfolio_after: PortfolioSnapshot
    extracted_decision: Decision
    execution_result: DecisionResult | None = None
    agent_output: dict | str | None = None
    elapsed_seconds: float = 0.0


class EpisodeLog(BaseModel):
    """Full episode audit trail.

    Episode-level identifiers live here; run-level parameters (tickers,
    initial_cash, etc.) are stored once on ``SimulationLog.config``.
    """

    episode_id: str
    agent_id: str
    decision_point_logs: list[DecisionPointLog] = []
    trades: list[ExecutedTrade] = []
    final_portfolio: PortfolioSnapshot | None = None


class SimulationLog(BaseModel):
    """Run-level log with embedded configuration for reproducibility.

    ``run_name`` is derived from the configuration file path by the runner.
    """

    run_name: str
    config: SimulationConfig
    episode_logs: list[EpisodeLog] = []
    errors: list[str] = []
