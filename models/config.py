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
    role_llms: dict[str, dict[str, str]] | None = Field(
        default=None,
        description="Optional per-role LLM overrides. "
        "Format: {role: {provider: 'openai'|'anthropic'|'google', model: '<model-name>'}}. "
        "Example: {'macro': {'provider': 'openai', 'model': 'gpt-5-mini'}, "
        "'risk': {'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}}.",
    )
    phase_llms: dict[str, dict[str, str]] | None = Field(
        default=None,
        description="Optional per-phase LLM overrides (takes priority over role_llms). "
        "Format: {phase: {provider: '...', model: '...'}} where phase is "
        "'propose', 'critique', 'revise', or 'judge'. "
        "Example: {'judge': {'provider': 'openai', 'model': 'gpt-5'}}.",
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
    propose_only: bool = Field(
        default=False,
        description="If true, skips critique and revise phases, running only the propose phase (0 rounds of debate).",
    )
    judge_type: str = Field(
        default="llm",
        description="Type of judge to use: 'llm' (default) or 'average' (simple unweighted average).",
    )
    debate_roles: list[str] | None = Field(
        default=None,
        description="Agent roles for debate, e.g. ['macro', 'value', 'risk']. "
        "If omitted, defaults to ['macro', 'value', 'risk', 'technical'].",
    )
    enable_adversarial: bool = Field(
        default=False,
        description="Add an explicit devil's advocate agent to the debate.",
    )
    parallel_agents: bool = Field(
        default=True,
        description="Run debate agents in parallel via LangGraph fan-out. "
        "Set to false for sequential execution (easier to debug).",
    )
    no_rate_limit: bool = Field(
        default=False,
        description="Disable stagger entirely. All parallel LLM calls fire at once.",
    )
    llm_stagger_ms: int = Field(
        default=200,
        ge=0,
        description="Milliseconds between parallel LLM call starts. "
        "Default 200ms spreads 4 calls over 600ms to avoid 429 bursts.",
    )
    max_concurrent_llm: int = Field(
        default=0,
        ge=0,
        description="Max concurrent LLM calls. 0 = unlimited.",
    )

    log_tokens: bool = Field(
        default=False,
        description="Print per-request token counts (prompt, completion, total) to console.",
    )
    log_rendered_prompts: bool = Field(
        default=False,
        description="Log the full rendered system + user prompts sent to the LLM "
        "via the debate.prompts logger. Useful for debugging prompt quality.",
    )
    log_prompt_manifest: bool = Field(
        default=False,
        description="Log prompt file names once per round (compact manifest). "
        "Shows system block files, phase templates, and snapshot identifier.",
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
        description="Initial beta value for PID tone controller.",
    )
    pid_epsilon: float = Field(
        default=0.001,
        gt=0.0,
        le=1.0,
        description="JS divergence convergence tolerance. Debate stops early "
        "when JS <= epsilon. Default 0.001 (tight — agents must nearly agree).",
    )
    convergence_window: int = Field(
        default=2,
        ge=1,
        description="Consecutive stable rounds (converged quadrant + JS < epsilon + "
        "rho_bar plateau) before early termination.",
    )
    delta_rho: float = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description="Rho-bar plateau tolerance for convergence detection. "
        "Termination requires |rho_bar(t) - rho_bar(t-1)| < delta_rho.",
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

    # --- Structured debate logging ---
    logging_mode: str = Field(
        default="off",
        description="Debate logging mode: 'standard' (artifacts only), "
        "'debug' (artifacts + prompts), 'off' (disabled).",
    )
    experiment_name: str | None = Field(
        default=None,
        description="Experiment name for logging directory. "
        "Defaults to config filename stem if not set.",
    )

    # --- Console display ---
    console_display: bool = Field(
        default=True,
        description="Enable Rich-formatted terminal display during debates. "
        "Set to false for minimal plain-text logging.",
    )

    # --- Event logging (logging_v2 causal trace system) ---
    event_logging: bool = Field(
        default=False,
        description="Enable logging_v2 append-only event stream. "
        "Produces events.jsonl alongside the standard debate log.",
    )
    event_logging_store_full_text: bool = Field(
        default=True,
        description="Store full prompt/response text in event log. "
        "Set to false for compact logs (hashes only).",
    )

    # --- Agent profiles (new unified system) ---
    agents: dict[str, str] | None = Field(
        default=None,
        description="Mapping of role name to agent profile name. "
        "E.g. {'macro': 'macro_diverse', 'value': 'value_diverse'}. "
        "When set, replaces debate_roles + prompt_profile + prompt_file_overrides.",
    )
    judge_profile: str = Field(
        default="judge_standard",
        description="Agent profile name for the judge.",
    )

    # --- Prompt block/section ordering (for ablation experiments) ---
    system_prompt_block_order: list[str] | None = Field(
        default=None,
        description="Order of system prompt blocks. "
        "Default: [causal_contract, role_system, phase_preamble, tone].",
    )
    user_prompt_section_order: list[str] | None = Field(
        default=None,
        description="Order of user prompt sections. "
        "Default: [preamble, context, agent_data, task, scaffolding, output_format].",
    )
    prompt_file_overrides: dict[str, str] | None = Field(
        default=None,
        description="Override which .txt file to load for a given block/section name. "
        "Keys: 'causal_contract', 'role_<rolename>', 'proposal_template', etc. "
        "Values: filename relative to multi_agent/prompts/ directory.",
    )

    # --- Prompt profile (per-agent prompt composition) ---
    prompt_profile: str = Field(
        default="default",
        description="Prompt profile name (e.g. 'default', 'minimal', 'diverse_agents').",
    )
    role_overrides: dict | None = Field(
        default=None,
        description="Per-role prompt profile overrides. Keys are role names, "
        "values are dicts with 'system_blocks' and/or 'user_sections' lists.",
    )

    # --- Intervention engine (intra-round retry on acute failures) ---
    intervention_config: dict | None = Field(
        default=None,
        description="Intervention engine config. Contains 'enabled' and 'rules' dict.",
    )

    # --- CRIT configuration ---
    crit_llm_model: str = Field(
        default="gpt-5-mini",
        description="LLM model to use for CRIT scoring calls. "
        "Separate from llm_model so CRIT can use a stronger model.",
    )
    crit_system_template: str = Field(
        default="crit_system_enumerated.jinja",
        description="CRIT system prompt template filename (in eval/crit/prompts/).",
    )
    crit_user_template: str = Field(
        default="crit_user_master.jinja",
        description="CRIT user prompt template filename (in eval/crit/prompts/).",
    )

    @model_validator(mode="after")
    def _validate_propose_only(self) -> AgentConfig:
        if self.propose_only and self.max_rounds != 1:
            raise ValueError("When propose_only is true, max_rounds must be 1.")
        return self

    # --- Runtime metadata (set by run_simulation.py, not in YAML) ---
    run_command: str | None = Field(
        default=None,
        description="Effective CLI command used to launch this run. "
        "Set automatically by run_simulation.py.",
    )
    config_paths: list[str] = Field(
        default_factory=list,
        description="Config file path(s) used for this run (agents, scenario). "
        "Set automatically by run_simulation.py.",
    )

    # --- Constraints (populated from SimulationConfig, not set in YAML) ---
    sector_config: dict | None = Field(
        default=None,
        description="Sector constraints forwarded from SimulationConfig. "
        "Contains 'sectors', 'sector_limits', 'agent_sector_permissions'. "
        "Not intended to be set directly in agent YAML configs.",
    )
    allocation_constraints: dict | None = Field(
        default=None,
        description="Allocation weight constraints forwarded from SimulationConfig. "
        "Contains 'max_weight', 'min_holdings', 'fully_invested'.",
    )


class SectorLimit(BaseModel):
    """Min/max exposure bound for a single sector."""

    min: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum sector exposure.")
    max: float = Field(default=1.0, ge=0.0, le=1.0, description="Maximum sector exposure.")

    @model_validator(mode="after")
    def _min_le_max(self) -> SectorLimit:
        if self.min > self.max:
            raise ValueError(f"Sector limit min ({self.min}) > max ({self.max})")
        return self


class AllocationConstraints(BaseModel):
    """Constraints on portfolio allocation weights (memo mode)."""

    max_weight: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Maximum weight per ticker (default 100%).",
    )
    min_holdings: int = Field(
        default=1,
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
    output_dir: str = Field(
        default="results",
        description="Top-level directory for simulation output."
    )
    top_n_news: int | None = Field(
        default=None,
        ge=1,
        description="If set, filter case_data items to top N by abs(impact_score) at load time. "
        "Items without impact_score are included after scored items. Agent never sees impact_score.",
    )
    debate_setup: AgentConfig = Field(description="Agent system configuration.")

    # --- Top-level agent identity (copied into debate_setup by validator) ---
    agents: dict[str, str] | None = Field(
        default=None,
        description="Top-level role→profile mapping. Copied into debate_setup.agents by validator.",
    )
    judge_profile: str | None = Field(
        default=None,
        description="Top-level judge profile name. Copied into debate_setup.judge_profile by validator.",
    )
    broker: BrokerConfig = Field(
        default_factory=BrokerConfig,
        description="Broker / execution configuration.",
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Universe of ticker symbols for this run. "
        "Required after scenario merge, but optional in debate-only configs.",
    )
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
    memo_format: Literal["text", "json"] = Field(
        default="text",
        description="Memo payload format: 'text' for .txt memo, 'json' for snapshot JSON.",
    )
    invest_quarter: str | None = Field(
        default=None,
        description="Invest quarter for memo mode, e.g. '2025Q1'. "
        "Agents see the prior quarter's data and allocate for this quarter.",
    )
    memo_override_path: str | None = Field(
        default=None,
        description="If set, load the memo from this path instead of the default "
        "memo_data/ directory. Used for scenario-specific memo caching.",
    )
    use_cash_virtual_ticker: bool = Field(
        default=False,
        description="If true, adds a virtual '_CASH_' ticker to the universe with a fixed price of 1.0.",
    )
    allocation_constraints: AllocationConstraints = Field(
        default_factory=AllocationConstraints,
        description="Allocation weight constraints (memo mode only).",
    )

    # --- Sector configuration (optional) ---
    sectors: dict[str, list[str]] | None = Field(
        default=None,
        description="Mapping of sector name to list of tickers. "
        "If provided, every ticker must belong to exactly one sector.",
    )
    sector_limits: dict[str, SectorLimit] | None = Field(
        default=None,
        description="Per-sector min/max exposure limits. "
        "Requires 'sectors' to be defined.",
    )
    agent_sector_permissions: dict[str, list[str]] | None = Field(
        default=None,
        description="Per-role allowed sectors. Use ['*'] for all sectors. "
        "Requires 'sectors' to be defined.",
    )
    max_sector_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum portfolio weight for any single sector. "
        "Requires 'sectors' to be defined.",
    )

    @model_validator(mode="after")
    def _copy_top_level_agent_identity(self) -> SimulationConfig:
        """Copy top-level agents/judge_profile into debate_setup so AgentConfig carries them."""
        if self.agents is not None:
            self.debate_setup.agents = self.agents
        if self.judge_profile is not None:
            self.debate_setup.judge_profile = self.judge_profile
        return self

    @model_validator(mode="after")
    def _validate_sectors(self) -> SimulationConfig:
        """Validate sector configuration and pack into debate_setup.sector_config."""
        if self.sectors is None:
            if self.sector_limits is not None:
                raise ValueError("sector_limits requires 'sectors' to be defined.")
            if self.agent_sector_permissions is not None:
                raise ValueError("agent_sector_permissions requires 'sectors' to be defined.")
            if self.max_sector_weight is not None:
                raise ValueError("max_sector_weight requires 'sectors' to be defined.")
            return self

        # Every ticker must be in exactly one sector
        ticker_to_sector: dict[str, str] = {}
        for sector, tickers in self.sectors.items():
            for t in tickers:
                if t in ticker_to_sector:
                    raise ValueError(
                        f"Ticker {t} appears in multiple sectors: "
                        f"'{ticker_to_sector[t]}' and '{sector}'"
                    )
                ticker_to_sector[t] = sector

        # All config tickers must appear in sector map
        missing = [t for t in self.tickers if t not in ticker_to_sector]
        if missing:
            raise ValueError(f"Tickers missing from sector mapping: {missing}")

        # All sector tickers must be in config tickers
        config_tickers = set(self.tickers)
        for sector, tickers in self.sectors.items():
            unknown = [t for t in tickers if t not in config_tickers]
            if unknown:
                raise ValueError(
                    f"Sector '{sector}' references unknown tickers: {unknown}"
                )

        # Validate sector_limits
        if self.sector_limits is not None:
            valid_sectors = set(self.sectors.keys())
            for sector_name in self.sector_limits:
                if sector_name not in valid_sectors:
                    raise ValueError(
                        f"sector_limits references unknown sector: '{sector_name}'"
                    )
            min_sum = sum(sl.min for sl in self.sector_limits.values())
            if min_sum > 1.0 + 1e-8:
                raise ValueError(
                    f"Sector min limits sum to {min_sum:.2f} > 1.0 — infeasible"
                )
            max_sum = sum(sl.max for sl in self.sector_limits.values())
            if max_sum < 1.0 - 1e-8:
                raise ValueError(
                    f"Sector max limits sum to {max_sum:.2f} < 1.0 — "
                    f"cannot reach fully invested"
                )

        # Validate agent_sector_permissions
        valid_roles = {"macro", "value", "risk", "technical", "sentiment", "devils_advocate"}
        if self.agent_sector_permissions is not None:
            valid_sectors = set(self.sectors.keys())
            for role, allowed in self.agent_sector_permissions.items():
                if role not in valid_roles:
                    raise ValueError(
                        f"agent_sector_permissions references unknown role: '{role}'"
                    )
                if isinstance(allowed, list) and allowed != ["*"]:
                    for s in allowed:
                        if s not in valid_sectors:
                            raise ValueError(
                                f"Role '{role}' references unknown sector: '{s}'"
                            )

        # Pack into debate_setup.sector_config for the debate system
        self.debate_setup.sector_config = {
            "sectors": self.sectors,
            "sector_limits": (
                {k: v.model_dump() for k, v in self.sector_limits.items()}
                if self.sector_limits
                else None
            ),
            "agent_sector_permissions": self.agent_sector_permissions,
            "max_sector_weight": self.max_sector_weight,
        }

        return self

    @model_validator(mode="after")
    def _validate_memo_mode(self) -> SimulationConfig:
        """Validate memo/allocation mode constraints.

        Skips ticker/quarter checks when they are absent (debate-only config
        loaded before scenario merge).  run_simulation.py calls
        ``validate_ready()`` after merging to enforce these at runtime.
        """
        if not self.tickers or self.invest_quarter is None:
            return self
        if len(self.tickers) > self.allocation_constraints.max_tickers:
            raise ValueError(
                f"Too many tickers ({len(self.tickers)}) for allocation mode "
                f"(max {self.allocation_constraints.max_tickers})."
            )
        ac = self.allocation_constraints
        if ac.max_weight * ac.min_holdings < 1.0 - 1e-8:
            raise ValueError(
                f"Impossible allocation constraints: max_weight ({ac.max_weight}) * "
                f"min_holdings ({ac.min_holdings}) = {ac.max_weight * ac.min_holdings:.2f} < 1.0. "
                f"Cannot satisfy both constraints simultaneously."
            )
        return self

    def validate_ready(self) -> None:
        """Raise if the config is missing fields required for a simulation run.

        Call after scenario merge to enforce tickers + invest_quarter.
        """
        if not self.tickers:
            raise ValueError("tickers must not be empty.")
        if self.invest_quarter is None:
            raise ValueError("invest_quarter is required.")

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
