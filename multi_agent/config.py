"""
Configuration for the multi-agent debate system.
All experimental knobs in one place for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    """Available agent roles for the trading desk."""

    MACRO = "macro"
    VALUE = "value"
    RISK = "risk"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    DEVILS_ADVOCATE = "devils_advocate"


# Default debate roster (4 specialist agents)
DEFAULT_ROLES: list[AgentRole] = [
    AgentRole.MACRO,
    AgentRole.VALUE,
    AgentRole.RISK,
    AgentRole.TECHNICAL,
]


@dataclass
class DebateConfig:
    """
    Configuration knobs for the multi-agent debate orchestrator.

    Designed for ablation experiments:
      - roles: which agents participate
      - max_rounds: how many critique-revision cycles
      - agreeableness: sycophancy knob (0=confrontational, 1=agreeable)
      - enable_adversarial: add devil's advocate agent
      - enable_news_pipeline / enable_data_pipeline: preprocessing stages
      - model_name / temperature: LLM settings
      - mock: use mock responses (no API calls)
    """

    # --- Agent roster ---
    roles: list[AgentRole] = field(default_factory=lambda: list(DEFAULT_ROLES))

    # --- Debate structure ---
    # Number of critique-revision cycles (1 = propose -> critique -> revise -> judge)
    max_rounds: int = 1

    # Agreeableness knob: 0.0 = maximally confrontational, 1.0 = sycophantic
    # Default 0.3 = somewhat skeptical (good for research on reducing groupthink)
    agreeableness: float = 0.3

    # Whether to add an explicit devil's advocate agent
    enable_adversarial: bool = False

    # --- Pipeline preprocessing ---
    enable_news_pipeline: bool = True
    enable_data_pipeline: bool = True

    # --- LLM settings ---
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3

    # --- Mock mode (no API calls, deterministic for testing) ---
    mock: bool = False

    # --- Verbose mode (print full debate content to terminal) ---
    verbose: bool = False

    # --- Output ---
    trace_dir: str = "./traces"

    def to_dict(self) -> dict:
        """Serialize config to dict for LangGraph state."""
        return {
            "roles": [r.value for r in self.roles],
            "max_rounds": self.max_rounds,
            "agreeableness": self.agreeableness,
            "enable_adversarial": self.enable_adversarial,
            "enable_news_pipeline": self.enable_news_pipeline,
            "enable_data_pipeline": self.enable_data_pipeline,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "mock": self.mock,
            "verbose": self.verbose,
            "trace_dir": self.trace_dir,
        }
