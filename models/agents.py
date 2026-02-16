"""Agent interface models."""

from typing import Any

from pydantic import BaseModel

from models.case import Case
from models.decision import Decision


class AgentInvocation(BaseModel):
    """Input passed when the simulation invokes the agent at a decision point."""

    case: Case
    episode_id: str
    agent_id: str
    steps_remaining: int | None = None  # Optional: how many decision steps remain in the episode


class AgentInvocationResult(BaseModel):
    """Parsed output from the agent."""

    decision: Decision
    raw_output: dict[str, Any] | str | None = None
