"""
Configuration for the multi-agent debate system.
All experimental knobs in one place for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from eval.PID.types import PIDConfig


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

    PID controller can be configured either by passing a PIDConfig
    object directly (pid_config), or by setting pid_enabled=True with
    flat gain fields (pid_kp, pid_ki, pid_kd, pid_rho_star).
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

    # --- Parallel agents (per-agent LangGraph nodes for concurrent LLM calls) ---
    parallel_agents: bool = True

    # --- Mock mode (no API calls, deterministic for testing) ---
    mock: bool = False

    # --- Verbose mode (print full debate content to terminal) ---
    verbose: bool = False

    # --- Logging levels for LLM calls ---
    # log_system_prompts: print the system prompt sent to each agent
    log_system_prompts: bool = False
    # log_user_prompts: print the full rendered user prompt (includes case data)
    log_user_prompts: bool = False
    # log_llm_responses: print the raw LLM response text
    log_llm_responses: bool = False

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

    # --- Per-phase PID intervention toggles ---
    # Controls which debate phases use PID's beta_new as agreeableness.
    # pid_propose: proposals not strongly affected by agreeableness (default off)
    # pid_critique: critique tone is most sensitive to agreeableness (default on)
    # pid_revise: revision deference is also sensitive (default on)
    pid_propose: bool = False
    pid_critique: bool = True
    pid_revise: bool = True

    # --- PID logging ---
    pid_log_metrics: bool = True
    pid_log_llm_calls: bool = False

    # --- Prompt logging (modular prompt path only) ---
    # Controls logging of prompt build details when PID modular prompts are active.
    # Keys: enabled, log_rendered_prompt, log_selected_blocks,
    #        log_beta_bucket, max_prompt_log_chars
    prompt_logging: dict = field(default_factory=dict)

    # --- Memo / allocation mode ---
    allocation_mode: bool = False
    skip_pipeline: bool = False

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
        if not (0.0 <= self.agreeableness <= 1.0):
            raise ValueError(
                f"agreeableness must be in [0, 1], got {self.agreeableness}"
            )
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

        # If flat YAML fields request PID but no PIDConfig was provided,
        # construct it here so the adapter never imports from eval/.
        if self._pid_enabled_flag and self.pid_config is None:
            from eval.PID.types import PIDConfig, PIDGains

            self.pid_config = PIDConfig(
                gains=PIDGains(Kp=self.pid_kp, Ki=self.pid_ki, Kd=self.pid_kd),
                rho_star=self.pid_rho_star,
            )

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
            "parallel_agents": self.parallel_agents,
            "mock": self.mock,
            "verbose": self.verbose,
            "log_system_prompts": self.log_system_prompts,
            "log_user_prompts": self.log_user_prompts,
            "log_llm_responses": self.log_llm_responses,
            "trace_dir": self.trace_dir,
            "pid_enabled": self.pid_enabled,
            "pid_propose": self.pid_propose,
            "pid_critique": self.pid_critique,
            "pid_revise": self.pid_revise,
            "prompt_logging": self.prompt_logging,
            "allocation_mode": self.allocation_mode,
            "skip_pipeline": self.skip_pipeline,
        }
