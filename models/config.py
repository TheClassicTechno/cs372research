"""Simulation configuration models, loaded from YAML.

These live in ``models/`` because they are shared data contracts used by the
simulation runner, agent systems, and evaluation tooling.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for the agent system."""

    agent_system: str = Field(
        description="Registered agent system name, e.g. 'single_llm', 'analyst_pm_graph'."
    )
    llm_provider: str = Field(
        description="LLM provider identifier, e.g. 'openai', 'anthropic'."
    )
    llm_model: str = Field(
        description="Model name, e.g. 'gpt-4o', 'claude-sonnet-4-20250514'."
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of submit_decision retries allowed per case.",
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Optional override for the agent's system prompt.",
    )


class BrokerConfig(BaseModel):
    """Configuration for the in-process broker."""

    initial_cash: float = Field(
        default=100_000.0,
        gt=0,
        description="Starting cash balance for the portfolio.",
    )


class SimulationConfig(BaseModel):
    """Top-level configuration for a simulation run, loaded from YAML.

    The run name is derived from the config file path at load time rather
    than being specified inside the YAML itself.
    """

    dataset_path: str = Field(description="Path to the case dataset (directory or file).")
    top_n_news: int | None = Field(
        default=None,
        ge=1,
        description="If set, filter case_data items to top N by abs(impact_score) at load time. "
        "Items without impact_score are included after scored items. Agent never sees impact_score.",
    )
    agent: AgentConfig = Field(description="Agent system configuration.")
    broker: BrokerConfig = Field(
        default_factory=BrokerConfig,
        description="Broker / execution configuration.",
    )
    tickers: list[str] = Field(description="Universe of ticker symbols for this run.")
    num_episodes: int = Field(
        default=1,
        ge=1,
        description="Number of episodes to run.",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        """Load and validate a ``SimulationConfig`` from a YAML file.

        Raises ``FileNotFoundError`` if the file does not exist and
        ``ValueError`` if the content is not a valid YAML mapping.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected a YAML mapping in {path}, got {type(raw).__name__}."
            )

        return cls(**raw)
