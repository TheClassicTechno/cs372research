"""Logging and experiment storage models."""

from pydantic import BaseModel

from models.decision import Decision, DecisionResult
from models.episode import EpisodeConfig, EpisodeResult


class DecisionPointLog(BaseModel):
    """Per-decision audit: case_id, agent_output, extracted_decision, execution_result."""

    case_id: str
    decision_point_idx: int
    agent_output: dict | str | None = None
    extracted_decision: Decision
    execution_result: DecisionResult | None = None


class EpisodeLog(BaseModel):
    """Full episode audit: config, decision_point_logs, episode_result."""

    episode_config: EpisodeConfig
    decision_point_logs: list[DecisionPointLog] = []
    episode_result: EpisodeResult | None = None


class SimulationLog(BaseModel):
    """Run-level log: episodes, versions (for reproducibility), errors."""

    run_id: str
    episode_logs: list[EpisodeLog] = []
    errors: list[str] = []
    # Optional version info for reproducibility
    simulation_version: str | None = None
    agent_version: str | None = None
    evaluation_version: str | None = None
