"""Abstract base class for agent systems.

Every agent system (single-LLM, multi-node LangGraph, etc.) implements this
protocol so the simulation runner can invoke them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from models.agents import AgentInvocation, AgentInvocationResult
from models.config import AgentConfig


class AgentSystem(ABC):
    """Common interface for pluggable agent architectures.

    Lifecycle:
        1. ``__init__`` — receive agent config.
        2. ``bind_tools`` — called once per case with the ``submit_decision``
           callable so the agent's LLM tool routes to the broker.
        3. ``invoke`` — called once per decision point.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abstractmethod
    def bind_tools(self, submit_decision_fn: Callable[..., Any]) -> None:
        """Bind the ``submit_decision`` tool to the current broker/case context.

        Called before each ``invoke`` so the tool closure captures the correct
        broker state and case reference.
        """

    @abstractmethod
    async def invoke(self, invocation: AgentInvocation) -> AgentInvocationResult:
        """Run the agent for one decision point and return the result.

        The agent may call ``submit_decision`` zero or more times during
        execution. The implementation must extract the final ``Decision``
        and wrap it in an ``AgentInvocationResult``.
        """
