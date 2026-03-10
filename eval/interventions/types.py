"""Type definitions for the intervention framework.

Defines the data structures used by InterventionEngine and rules:
  - InterventionContext: snapshot of debate state available to rules
  - InterventionRule: a single intervention trigger
  - InterventionResult: output when a rule fires
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InterventionContext:
    """Snapshot of debate state available to rules."""

    round_num: int
    stage: str  # "post_revision" | "post_crit"
    retry_count: int  # how many retries already attempted this stage
    state: dict  # full LangGraph state

    # Pre-computed metrics (populated by runner before calling engine)
    js_proposal: float | None = None  # JS across proposals (pre-critique)
    js_revision: float | None = None  # JS across revisions (post-revise)
    ov_revision: float | None = None  # evidence overlap post-revise
    rho_bar: float | None = None  # aggregate CRIT score (post_crit only)
    agent_crit_scores: dict | None = None  # per-agent CritResult (post_crit only)
    pid_result: object | None = None  # PIDStepResult (post_crit only)

    # Intervention history from prior rounds and stages
    intervention_history: list[dict] = field(default_factory=list)


@dataclass
class InterventionResult:
    """Output when a rule fires."""

    rule_name: str
    action: str  # "retry_revision" | "retry_critique_revision" | "log_only"
    nudge_text: str  # prompt modifier injected into retry
    metrics: dict  # rule-specific diagnostic data (for logging)
    severity: str  # "warning" | "critical"
    target_roles: list[str] | None = None  # if set, nudge only these roles


@dataclass
class InterventionRule:
    """A single intervention trigger."""

    name: str  # e.g. "js_collapse", "low_rho", "pillar_collapse"
    stage: str  # "post_revision" | "post_crit"
    max_retries: int  # bound on retry attempts
    evaluate: Callable[[InterventionContext], InterventionResult | None]
