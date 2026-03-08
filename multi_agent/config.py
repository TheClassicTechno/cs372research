"""
Configuration for the multi-agent debate system.
All experimental knobs in one place for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from eval.PID.types import PIDConfig


class AgentRole:
    """Built-in role constants. Any string is a valid role."""

    MACRO = "macro"
    VALUE = "value"
    RISK = "risk"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    DEVILS_ADVOCATE = "devils_advocate"


# Default debate roster (4 specialist agents)
DEFAULT_ROLES: list[str] = [
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
      - enable_adversarial: add devil's advocate agent
      - model_name / temperature: LLM settings
      - mock: use mock responses (no API calls)

    The system operates in allocation mode: agents output portfolio
    allocation weights (memo-based) rather than buy/sell orders.
    Pipeline preprocessing (news_digest, data_analysis) is skipped;
    memo content provides the financial context directly.

    PID controller can be configured either by passing a PIDConfig
    object directly (pid_config), or by setting pid_enabled=True with
    flat gain fields (pid_kp, pid_ki, pid_kd, pid_rho_star).
    """

    # --- Agent roster ---
    roles: list[str] = field(default_factory=lambda: list(DEFAULT_ROLES))

    # --- Debate structure ---
    # Number of critique-revision cycles (1 = propose -> critique -> revise -> judge)
    max_rounds: int = 1

    # Skip critique and revise phases (propose -> judge only)
    propose_only: bool = False

    # Type of judge ('llm' or 'average')
    judge_type: str = "llm"

    # Whether to add an explicit devil's advocate agent
    enable_adversarial: bool = False

    # --- LLM settings ---
    llm_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    # Optional per-role overrides:
    # {role: {"provider": "openai"|"anthropic"|"google", "model": "<name>"}}
    role_llms: dict[str, dict[str, str]] = field(default_factory=dict)
    # Optional per-phase overrides (takes priority over role_llms):
    # {phase: {"provider": "...", "model": "..."}} where phase is
    # "propose", "critique", "revise", or "judge"
    phase_llms: dict[str, dict[str, str]] = field(default_factory=dict)
    temperature: float = 0.3

    # --- Parallel agents (per-agent LangGraph nodes for concurrent LLM calls) ---
    parallel_agents: bool = True

    # --- Rate limiting ---
    no_rate_limit: bool = False   # Disable stagger entirely (all calls fire at once)
    llm_stagger_ms: int = 500    # Milliseconds between parallel LLM call starts
    max_concurrent_llm: int = 0  # Max concurrent LLM calls (0 = unlimited)

    # --- Mock mode (no API calls, deterministic for testing) ---
    mock: bool = False

    # --- Verbose mode (print full debate content to terminal) ---
    verbose: bool = False

    # log_tokens: print per-request token counts to console
    log_tokens: bool = False
    # log_rendered_prompts: log full system + user prompts via debate.prompts logger
    log_rendered_prompts: bool = False
    # log_prompt_manifest: log prompt file names once per round (compact)
    log_prompt_manifest: bool = False

    # --- Output ---
    trace_dir: str = "./traces"

    # --- PID controller ---
    # Option 1: Pass a PIDConfig object directly (used by tests / programmatic API).
    pid_config: Any = None  # PIDConfig | None (Any to avoid import at module level)

    # Option 2: Flat YAML fields (used by DebateAgentSystem adapter).
    # When _pid_enabled_flag is True AND pid_config is None, __post_init__
    # constructs PIDConfig from these flat fields.
    _pid_enabled_flag: bool = False
    pid_kp: float = 0.15
    pid_ki: float = 0.01
    pid_kd: float = 0.03
    pid_rho_star: float = 0.8

    initial_beta: float = 0.5

    # --- PID convergence ---
    pid_epsilon: float = 0.001  # JS divergence convergence tolerance
    convergence_window: int = 2  # consecutive stable rounds required for early stop
    delta_rho: float = 0.02      # rho_bar plateau tolerance for convergence

    # --- PID logging ---
    pid_log_metrics: bool = True
    pid_log_llm_calls: bool = False

    # --- Prompt logging (modular prompt path only) ---
    # Controls logging of prompt build details when PID modular prompts are active.
    # Keys: enabled, log_rendered_prompt, log_selected_blocks,
    #        log_beta_bucket, max_prompt_log_chars
    prompt_logging: dict = field(default_factory=dict)

    # --- Structured debate logging ---
    logging_mode: str = "off"  # "standard" | "debug" | "off"
    experiment_name: str | None = None  # defaults to config filename stem

    # --- Console display ---
    console_display: bool = True  # Rich-formatted terminal output (set False for minimal logs)

    # Override which .txt file to load for a given block/section name.
    # Keys: "causal_contract", "role_<rolename>", "phase_preamble_critique",
    #        "phase_preamble_revise", "proposal_template", "critique_template",
    #        "revision_template", "judge_template", etc.
    # Values: filename relative to multi_agent/prompts/ directory.
    prompt_file_overrides: dict[str, str] = field(default_factory=dict)

    # --- Prompt profile (per-agent prompt composition) ---
    prompt_profile: str = "default"  # profile name (e.g. "default", "minimal", "diverse_agents")
    role_overrides: dict = field(default_factory=dict)  # per-role profile overrides

    # --- Agent profiles (new unified system) ---
    # When set, these replace prompt_profile + prompt_file_overrides + role_overrides.
    agent_profiles: dict = field(default_factory=dict)  # {role: loaded_profile_dict}
    agent_profile_names: dict = field(default_factory=dict)  # {role: profile_name_str}
    judge_profile: dict = field(default_factory=dict)   # loaded judge profile dict

    # --- CRIT configuration ---
    crit_model_name: str = "gpt-5-mini"  # LLM model for CRIT scoring (separate from debate model)
    crit_system_template: str = "crit_system_enumerated.jinja"
    crit_user_template: str = "crit_user_master.jinja"

    # --- Runtime metadata (set by run_simulation.py, not in YAML) ---
    run_command: str | None = None
    config_paths: list[str] = field(default_factory=list)

    # --- Sector constraints (optional, forwarded from SimulationConfig) ---
    sector_config: dict | None = None  # {sectors, sector_limits, agent_sector_permissions}

    # --- Allocation constraints (optional, forwarded from SimulationConfig) ---
    allocation_constraints: dict | None = None  # {max_weight, min_holdings, fully_invested}

    @property
    def evaluation_mode(self) -> str:
        """Return 'in_loop' if PID is enabled, 'post_hoc' otherwise."""
        return "in_loop" if self.pid_config is not None else "post_hoc"

    @property
    def pid_enabled(self) -> bool:
        """Whether the PID controller is active for this debate."""
        return self.pid_config is not None

    def __post_init__(self) -> None:
        """Validate config values after initialization."""
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if not (0.0 <= self.initial_beta <= 1.0):
            raise ValueError(
                f"initial_beta must be in [0, 1], got {self.initial_beta}"
            )
        if not self.roles:
            raise ValueError("roles must not be empty")
        if self.pid_kp < 0 or self.pid_ki < 0 or self.pid_kd < 0:
            raise ValueError(
                f"PID gains must be non-negative, got Kp={self.pid_kp}, "
                f"Ki={self.pid_ki}, Kd={self.pid_kd}"
            )
        if not (0.0 <= self.pid_rho_star <= 1.0):
            raise ValueError(
                f"pid_rho_star must be in [0, 1], got {self.pid_rho_star}"
            )

        # If flat YAML fields request PID but no PIDConfig was provided,
        # construct it here so the adapter never imports from eval/.
        if self._pid_enabled_flag and self.pid_config is None:
            from eval.PID.types import PIDConfig, PIDGains

            self.pid_config = PIDConfig(
                gains=PIDGains(Kp=self.pid_kp, Ki=self.pid_ki, Kd=self.pid_kd),
                rho_star=self.pid_rho_star,
                epsilon=self.pid_epsilon,
            )

    def to_dict(self) -> dict:
        """Serialize config to dict for LangGraph state."""
        return {
            "roles": list(self.roles),
            "max_rounds": self.max_rounds,
            "propose_only": self.propose_only,
            "judge_type": self.judge_type,
            "enable_adversarial": self.enable_adversarial,
            "model_name": self.model_name,
            "llm_provider": self.llm_provider,
            "role_llms": self.role_llms,
            "phase_llms": self.phase_llms,
            "temperature": self.temperature,
            "parallel_agents": self.parallel_agents,
            "mock": self.mock,
            "no_rate_limit": self.no_rate_limit,
            "llm_stagger_ms": self.llm_stagger_ms,
            "max_concurrent_llm": self.max_concurrent_llm,
            "verbose": self.verbose,
            "log_tokens": self.log_tokens,
            "log_rendered_prompts": self.log_rendered_prompts,
            "log_prompt_manifest": self.log_prompt_manifest,
            "trace_dir": self.trace_dir,
            "pid_enabled": self.pid_enabled,
            "prompt_logging": self.prompt_logging,
            "convergence_window": self.convergence_window,
            "delta_rho": self.delta_rho,
            "allocation_mode": True,
            "skip_pipeline": True,
            "logging_mode": self.logging_mode,
            "console_display": self.console_display,
            "prompt_file_overrides": self.prompt_file_overrides,
            "prompt_profile": self.prompt_profile,
            "role_overrides": self.role_overrides,
            "agent_profiles": self.agent_profiles,
            "judge_profile": self.judge_profile,
            "crit_system_template": self.crit_system_template,
            "crit_user_template": self.crit_user_template,
            "sector_config": self.sector_config,
            "allocation_constraints": self.allocation_constraints,
        }
