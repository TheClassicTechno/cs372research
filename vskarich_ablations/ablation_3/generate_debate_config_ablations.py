#!/usr/bin/env python3
"""Generate ablation 3 debate config YAML files.

Ablation 3 evaluates PID + Intervention (Premature Convergence Guard):
  - 5 conditions testing PID, JS collapse intervention, and reasoning
    quality intervention in various combinations.
  - Base: best from ablation_1 (no_risk: macro+technical+value, intense
    enriched profiles, conservative PID).

Conditions:
  C1: PID only
  C2: PID + JS collapse intervention (post_revision)
  C3: PID + JS collapse + reasoning quality intervention (post_revision + post_crit)
  C4: Intervention only: JS + reasoning (no PID)
  C5: Intervention only: reasoning only (no PID)

Usage:
    python vskarich_ablations/ablation_3/generate_debate_config_ablations.py
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Target directory — wiped and regenerated on every run
# ---------------------------------------------------------------------------
TARGET_DIR = Path(__file__).resolve().parent / "debate_configs"


# ---------------------------------------------------------------------------
# YAML template fragments
# ---------------------------------------------------------------------------

_HEADER = """\
# ===========================================================================
# Ablation 3 — {title}
# Condition: {cond_name}
# Pipeline:  {pipeline}
# ===========================================================================
"""

_DATA = """\
# --- Data source ---
case_format: memo
dataset_path: data-pipeline/final_snapshots
memo_format: text
num_episodes: 5
"""

_AGENTS = """\

# --- Agent profiles (intense enriched) ---
agents:
  macro: macro_enriched_intense
  technical: technical_enriched_intense
  value: value_enriched_intense
  risk: risk_enriched_intense

judge_profile: judge_standard
"""

_DEBATE_OPEN = """\

# --- Debate setup ---
debate_setup:

  # LLM configuration
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-5-mini            # debate agent model
  crit_llm_model: gpt-5            # CRIT auditor model (higher capability)
  temperature: 0.3

  # Debate structure
  max_rounds: 2
  max_retries: 3
  propose_only: false

  # Logging & parallelism
  logging_mode: debug
  parallel_agents: true
  llm_stagger_ms: 500
  max_concurrent_llm: 3
  experiment_name: vskarich_ablation_3
"""

_PID_ENABLED = """\

  # --- PID controller (inter-round quality regulation) ---
  # Adjusts critique intensity (beta) based on CRIT reasoning scores (rho).
  # Conservative tuning: slow integral, moderate derivative damping.
  pid_enabled: true
  pid_kp: 0.20                     # proportional gain
  pid_ki: 0.015                    # integral gain (slow accumulation)
  pid_kd: 0.10                     # derivative gain (damping)
  pid_rho_star: 0.83               # target reasoning quality
  pid_initial_beta: 0.35           # starting critique intensity
  pid_epsilon: 0.02                # deadband around rho_star
  pid_gamma_beta: 0.35             # beta clamp range [initial - gamma, initial + gamma]
"""

_PID_DISABLED = """\

  # --- PID controller ---
  pid_enabled: false                # no inter-round beta adjustment
"""

_INTERVENTION_JS_COLLAPSE_ON = """\
      # JS Collapse Detection (post_revision)
      # Fires when JS_revision / JS_proposal < threshold, indicating agents
      # converged toward identical allocations during the revision phase.
      js_collapse:
        enabled: true
        threshold: 0.4              # collapse_ratio below this triggers retry
        min_js_proposal: 0.10       # skip when proposals already near-consensus
        max_retries: 2              # max retry attempts for this rule
"""

_INTERVENTION_JS_COLLAPSE_OFF = """\
      # JS Collapse Detection (post_revision) — disabled for this condition
      js_collapse:
        enabled: false
        threshold: 0.4
        min_js_proposal: 0.10
        max_retries: 2
"""


# ---------------------------------------------------------------------------
# JS Collapse nudge prompt text — per-agent intervention prompts
# Combined structure: HARD_DIRECTIVE + GENERAL_HEADER + ROLE_REMINDER
#                     + REVISION_CONSTRAINTS
# ---------------------------------------------------------------------------

_JS_NUDGE_DIRECTIVE = """\
You must maintain independent reasoning. Do not converge with \
other agents unless critiques logically invalidate your thesis."""

_JS_NUDGE_HEADER = """\
⚠️ DEBATE DIVERSITY PROTOCOL ACTIVATED

Rapid consensus formation has been detected in the debate.

In complex financial analysis, early consensus often indicates
premature convergence or sycophancy rather than strong evidence.

To maintain analytical rigor:

• Defensible dissent is valuable and encouraged.
• You should not abandon a well-supported thesis solely because
  other agents appear to agree.
• Rapid agreement requires explicit justification.

Agreement without independent reasoning is considered a debate failure.

Your objective is rigorous reasoning, not agreement."""

_JS_NUDGE_CONSTRAINTS = """\
Before finalizing your revision:

1. Re-examine your original thesis.
2. Identify whether critiques genuinely invalidate it.
3. Defend at least one element of your original reasoning unless
   it has been clearly disproven.
4. Identify one vulnerability in the emerging consensus view.
5. Explain which agent you disagree with most and why.
6. List ONE reason why another agent's argument is still flawed."""

_JS_NUDGE_ROLE: dict[str, str] = {
    "macro": """\
MACRO INTERVENTION REMINDER

Your responsibility is to analyze macro regime dynamics.

Do NOT anchor on individual stock narratives if they conflict \
with macro conditions.

Re-check:

• interest rate regime
• inflation trajectory
• liquidity conditions
• credit spreads
• global growth signals

If other agents ignored macro constraints, you should challenge them.

A portfolio that ignores the macro regime is structurally fragile.""",

    "value": """\
VALUE INTERVENTION REMINDER

Your responsibility is valuation discipline.

You should NOT abandon valuation arguments simply because other \
agents prefer momentum or macro narratives.

Re-evaluate:

• earnings quality
• valuation multiples
• margin sustainability
• balance sheet strength

If consensus is forming around expensive growth stocks, your job \
is to ask whether the valuation assumptions are justified.

Consensus does not make an asset undervalued.""",

    "technical": """\
TECHNICAL INTERVENTION REMINDER

Your role is to analyze price structure and market behavior.

You should NOT abandon a technical signal simply because \
fundamental narratives sound persuasive.

Re-check:

• trend structure
• momentum persistence
• support/resistance levels
• volatility regimes
• market breadth

If fundamentals contradict price action, you must explain why.

Price often reflects information before narratives catch up.""",

    "risk": """\
RISK INTERVENTION REMINDER

Your role is to protect the portfolio from tail risk and \
concentration risk.

Do NOT converge toward high-return narratives without analyzing:

• volatility regimes
• correlation structure
• drawdown risk
• diversification

If multiple agents are converging on the same assets, evaluate \
whether the portfolio is becoming fragile.

Consensus allocations often increase systemic risk.""",
}


def _indent_block(text: str, spaces: int = 12) -> str:
    """Indent text for YAML block scalar.  Empty lines stay empty."""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else ""
        for line in text.split("\n")
    )


def _build_js_nudge_yaml(roles: list[str]) -> str:
    """Build the nudge_prompts YAML block for js_collapse intervention."""
    lines: list[str] = ["        nudge_prompts:"]
    for role in roles:
        combined = (
            _JS_NUDGE_DIRECTIVE + "\n\n"
            + _JS_NUDGE_HEADER + "\n\n"
            + _JS_NUDGE_ROLE[role] + "\n\n"
            + _JS_NUDGE_CONSTRAINTS
        )
        lines.append(f"          {role}: |")
        lines.append(_indent_block(combined))
    return "\n".join(lines)


_INTERVENTION_REASONING_ON = """\
      # Reasoning Quality (post_crit, per-agent, pillar-aware)
      # Fires when any agent's rho_i or pillar score falls below threshold.
      # Targeted nudge: only weak agent(s) receive pillar-specific remediation.
      reasoning_quality:
        enabled: true
        rho_threshold: 0.5          # agent rho below this triggers the rule
        max_retries: 1              # post-crit retries are expensive (re-run revise + CRIT)

        # Per-agent pillar thresholds — single source of truth.
        # Every agent x every pillar must be listed; unlisted = not checked.
        agent_pillar_thresholds:
          macro:
            logical_validity: 0.40
            evidential_support: 0.40
            alternative_consideration: 0.40
            causal_alignment: 0.50  # macro must demonstrate causal mechanisms
          technical:
            logical_validity: 0.50  # technical must maintain logical rigor
            evidential_support: 0.40
            alternative_consideration: 0.40
            causal_alignment: 0.40
          value:
            logical_validity: 0.40
            evidential_support: 0.45  # value must cite valuation evidence
            alternative_consideration: 0.40
            causal_alignment: 0.40
          risk:
            logical_validity: 0.40
            evidential_support: 0.40
            alternative_consideration: 0.45  # risk must consider tail scenarios
            causal_alignment: 0.40
"""

_INTERVENTION_REASONING_OFF = """\
      # Reasoning Quality (post_crit) — disabled for this condition
      reasoning_quality:
        enabled: false
        rho_threshold: 0.5
        max_retries: 1
        agent_pillar_thresholds:
          macro:
            logical_validity: 0.40
            evidential_support: 0.40
            alternative_consideration: 0.40
            causal_alignment: 0.50
          technical:
            logical_validity: 0.50
            evidential_support: 0.40
            alternative_consideration: 0.40
            causal_alignment: 0.40
          value:
            logical_validity: 0.40
            evidential_support: 0.45
            alternative_consideration: 0.40
            causal_alignment: 0.40
          risk:
            logical_validity: 0.40
            evidential_support: 0.40
            alternative_consideration: 0.45
            causal_alignment: 0.40
"""

_BROKER = """\

# --- Broker ---
broker:
  initial_cash: 100000.0
"""


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "name": "c1_pid_only",
        "title": "PID Only",
        "pipeline": "propose -> critique -> revise -> CRIT -> PID -> judge",
        "pid": True,
        "js_collapse": False,
        "reasoning": False,
    },
    {
        "name": "c2_pid_js_intervention",
        "title": "PID + JS Collapse Intervention",
        "pipeline": "propose -> critique -> revise -> [IE post_revision] -> CRIT -> PID -> judge",
        "pid": True,
        "js_collapse": True,
        "reasoning": False,
    },
    {
        "name": "c3_pid_full_intervention",
        "title": "PID + JS Collapse + Reasoning Quality Intervention",
        "pipeline": "propose -> critique -> revise -> [IE post_revision] -> CRIT -> PID -> [IE post_crit] -> judge",
        "pid": True,
        "js_collapse": True,
        "reasoning": True,
    },
    {
        "name": "c4_intervention_js_reasoning",
        "title": "Intervention Only (JS + Reasoning, no PID)",
        "pipeline": "propose -> critique -> revise -> [IE post_revision] -> CRIT -> [IE post_crit] -> judge",
        "pid": False,
        "js_collapse": True,
        "reasoning": True,
    },
    {
        "name": "c5_intervention_reasoning_only",
        "title": "Intervention Only (Reasoning, no PID)",
        "pipeline": "propose -> critique -> revise -> CRIT -> [IE post_crit] -> judge",
        "pid": False,
        "js_collapse": False,
        "reasoning": True,
    },
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_yaml(cond: dict) -> str:
    """Build a formatted YAML string for a condition."""
    parts: list[str] = []

    # Header
    parts.append(_HEADER.format(
        title=cond["title"],
        cond_name=cond["name"],
        pipeline=cond["pipeline"],
    ))

    # Data + agents
    parts.append(_DATA)
    parts.append(_AGENTS)

    # Debate setup opening
    parts.append(_DEBATE_OPEN)

    # PID section
    if cond["pid"]:
        parts.append(_PID_ENABLED)
    else:
        parts.append(_PID_DISABLED)

    # Intervention config — always present, per-rule enabled/disabled
    parts.append("")
    parts.append("  # --- Intervention engine (intra-round retry on acute failures) ---")
    parts.append("  intervention_config:")
    parts.append("    enabled: true")
    parts.append("")
    parts.append("    # Portfolio revision limits applied to all intervention retries.")
    parts.append("    revision_limits:")
    parts.append("      max_changed_tickers: 2")
    parts.append("      max_change_pct: 5")
    parts.append("")
    parts.append("    rules:")

    if cond["js_collapse"]:
        parts.append(_INTERVENTION_JS_COLLAPSE_ON.rstrip())
        parts.append(_build_js_nudge_yaml(["macro", "technical", "value", "risk"]))
    else:
        parts.append(_INTERVENTION_JS_COLLAPSE_OFF.rstrip())

    parts.append("")  # blank line between rules

    if cond["reasoning"]:
        parts.append(_INTERVENTION_REASONING_ON.rstrip())
    else:
        parts.append(_INTERVENTION_REASONING_OFF.rstrip())

    parts.append("")

    # Broker
    parts.append(_BROKER)

    return "\n".join(parts)


def generate() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)

    print(f"Generating {len(CONDITIONS)} config files in {TARGET_DIR}/")

    for cond in CONDITIONS:
        filename = f"debate_{cond['name']}_r2_t0.3.yaml"
        content = _build_yaml(cond)

        out_path = TARGET_DIR / filename
        out_path.write_text(content)
        print(f"  {filename}")

    print(f"Done. {len(CONDITIONS)} files written.")


if __name__ == "__main__":
    generate()
