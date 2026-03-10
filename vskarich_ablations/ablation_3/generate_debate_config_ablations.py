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

# --- Agent profiles (intense enriched, no risk agent) ---
agents:
  macro: macro_enriched_intense
  technical: technical_enriched_intense
  value: value_enriched_intense

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
    parts.append("    rules:")

    if cond["js_collapse"]:
        parts.append(_INTERVENTION_JS_COLLAPSE_ON.rstrip())
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
