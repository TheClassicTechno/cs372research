"""Simulation configuration models, loaded from YAML.

These live in ``models/`` because they are shared data contracts used by the
simulation runner, agent systems, and evaluation tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


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

    # --- Debate structure (only used by multi_agent_debate agent) ---
    max_rounds: int = Field(
        default=1,
        ge=1,
        description="Number of critique-revision cycles. PID needs >= 2 to intervene between rounds.",
    )
    debate_roles: list[str] | None = Field(
        default=None,
        description="Agent roles for debate, e.g. ['macro', 'value', 'risk']. "
        "If omitted, defaults to ['macro', 'value', 'risk', 'technical'].",
    )
    agreeableness: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sycophancy knob: 0.0 = confrontational, 1.0 = agreeable.",
    )
    enable_adversarial: bool = Field(
        default=False,
        description="Add an explicit devil's advocate agent to the debate.",
    )

    log_system_prompts: bool = Field(
        default=False,
        description="Log the system prompt sent to each agent.",
    )
    log_user_prompts: bool = Field(
        default=False,
        description="Log the full rendered user prompt (includes case data).",
    )
    log_llm_responses: bool = Field(
        default=False,
        description="Log the raw LLM response text.",
    )

    # --- PID controller settings (flat YAML fields) ---
    pid_enabled: bool = Field(
        default=False,
        description="Enable PID controller for debate quality regulation.",
    )
    pid_kp: float = Field(default=0.15, description="PID proportional gain.")
    pid_ki: float = Field(default=0.01, description="PID integral gain.")
    pid_kd: float = Field(default=0.03, description="PID derivative gain.")
    pid_rho_star: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Target reasonableness score for PID controller.",
    )
    pid_initial_beta: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial agreeableness value for PID controller.",
    )
    pid_propose: bool = Field(
        default=False,
        description="Whether PID controls agreeableness during propose phase.",
    )
    pid_critique: bool = Field(
        default=True,
        description="Whether PID controls agreeableness during critique phase.",
    )
    pid_revise: bool = Field(
        default=True,
        description="Whether PID controls agreeableness during revise phase.",
    )

    # --- PID logging ---
    pid_log_metrics: bool = Field(
        default=True,
        description="Log scalar PID metrics (rho_bar, beta, JS, gains, etc.) each round.",
    )
    pid_log_llm_calls: bool = Field(
        default=False,
        description="Log full CRIT LLM prompts and responses each round.",
    )

    # --- Prompt logging (modular prompt path) ---
    prompt_logging: dict = Field(
        default_factory=dict,
        description="Prompt build logging config. Keys: enabled, log_rendered_prompt, "
        "log_selected_blocks, log_beta_bucket, max_prompt_log_chars.",
    )

    # --- Memo / allocation mode ---
    allocation_mode: bool = Field(
        default=False,
        description="When True, agents output allocation weights instead of orders. "
        "Auto-set by SimulationConfig when case_format='memo'.",
    )
    skip_pipeline: bool = Field(
        default=False,
        description="When True, skip news_digest and data_analysis pipeline nodes. "
        "Auto-set by SimulationConfig when case_format='memo'.",
    )


class AllocationConstraints(BaseModel):
    """Constraints on portfolio allocation weights (memo mode)."""

    max_weight: float = Field(
        default=0.40,
        gt=0.0,
        le=1.0,
        description="Maximum weight per ticker (default 40%).",
    )
    min_holdings: int = Field(
        default=3,
        ge=1,
        description="Minimum number of tickers with non-zero weight.",
    )
    fully_invested: bool = Field(
        default=True,
        description="Weights must sum to 1.0 (no cash reserve).",
    )
    max_tickers: int = Field(
        default=10,
        ge=1,
        description="Maximum number of tickers in the allocation universe.",
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
    quarters: list[str] | None = Field(
        default=None,
        description="If set, only load cases matching these quarters (e.g. ['Q1', 'Q3']). "
        "Matched against the case filename (e.g. '2025_Q1.json' matches 'Q1'). "
        "If omitted, all quarters are loaded.",
    )
    merge_tickers: bool = Field(
        default=False,
        description="If true, merge single-ticker cases from the same quarter into "
        "one multi-ticker case. Requires tickers to have matching quarter files.",
    )
    num_episodes: int = Field(
        default=1,
        ge=1,
        description="Number of episodes to run.",
    )

    # --- Memo / allocation mode ---
    case_format: Literal["legacy", "memo"] = Field(
        default="legacy",
        description="Case format: 'legacy' for earnings/news/price cases, "
        "'memo' for quarterly memo-based allocation.",
    )
    memo_format: Literal["text", "json"] = Field(
        default="text",
        description="Memo payload format: 'text' for .txt memo, 'json' for snapshot JSON.",
    )
    invest_quarter: str | None = Field(
        default=None,
        description="Invest quarter for memo mode, e.g. '2025Q1'. "
        "Agents see the prior quarter's data and allocate for this quarter.",
    )
    allocation_constraints: AllocationConstraints = Field(
        default_factory=AllocationConstraints,
        description="Allocation weight constraints (memo mode only).",
    )

    @model_validator(mode="after")
    def _wire_memo_mode(self) -> SimulationConfig:
        """Auto-set agent flags when case_format is 'memo'."""
        if self.case_format == "memo":
            self.agent.allocation_mode = True
            self.agent.skip_pipeline = True
            if self.invest_quarter is None:
                raise ValueError(
                    "invest_quarter is required when case_format='memo'."
                )
            if len(self.tickers) > self.allocation_constraints.max_tickers:
                raise ValueError(
                    f"Too many tickers ({len(self.tickers)}) for allocation mode "
                    f"(max {self.allocation_constraints.max_tickers})."
                )
        return self

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
