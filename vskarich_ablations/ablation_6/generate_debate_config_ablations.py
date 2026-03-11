#!/usr/bin/env python3
"""Generate ablation 6 debate config YAML files.

Ablation 6 — JS Collapse Intervention: Strict vs Soft Nudge

Isolates nudge prompt style and collapse threshold as experimental axes:
  - **strict**: nudge text + PORTFOLIO REVISION LIMITS block + YOUR ORIGINAL PROPOSAL data
  - **soft**: nudge text only — no constraints, no proposal data
  - **threshold 0.8**: fires on >20% JS drop (sensitive)
  - **threshold 0.65**: fires on >35% JS drop (permissive)

Design:
  - 3 agent subsets × 2 nudge types × 2 thresholds = 12 configs
  - 3 scenarios (reused from ablation_3)
  - Total runs: 12 × 3 = 36

Agent subsets:
  - 2-agent: macro, technical
  - 3-agent-value: macro, technical, value
  - 3-agent-risk: macro, technical, risk

Usage:
    python vskarich_ablations/ablation_6/generate_debate_config_ablations.py
"""

from __future__ import annotations

import shutil
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
# Ablation 6 — JS Collapse Intervention: {title}
# Config: {config_name}
# Pipeline:  {pipeline}
# ===========================================================================
"""

_DATA = """\
# --- Data source ---
case_format: memo
dataset_path: data-pipeline/final_snapshots
memo_format: text
num_episodes: 2
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
  experiment_name: vskarich_ablation_6
"""

_PID_DISABLED = """\

  # --- PID controller ---
  pid_enabled: false                # no inter-round beta adjustment
"""

_BROKER = """\

# --- Broker ---
broker:
  initial_cash: 100000.0
"""

# ---------------------------------------------------------------------------
# JS Collapse nudge prompt text — per-agent intervention prompts
# Reused from ablation_3
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


# ---------------------------------------------------------------------------
# Agent subsets
# ---------------------------------------------------------------------------

AGENT_SUBSETS = [
    {
        "name": "2agent",
        "agents": {
            "macro": "macro_enriched_intense",
            "technical": "technical_enriched_intense",
        },
    },
    {
        "name": "3agent_value",
        "agents": {
            "macro": "macro_enriched_intense",
            "technical": "technical_enriched_intense",
            "value": "value_enriched_intense",
        },
    },
    {
        "name": "3agent_risk",
        "agents": {
            "macro": "macro_enriched_intense",
            "technical": "technical_enriched_intense",
            "risk": "risk_enriched_intense",
        },
    },
]

# ---------------------------------------------------------------------------
# Nudge types
# ---------------------------------------------------------------------------

NUDGE_TYPES = [
    {"name": "strict", "has_revision_limits": True},
    {"name": "soft",   "has_revision_limits": False},
]

# ---------------------------------------------------------------------------
# Collapse thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = [0.8, 0.65]


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

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


def _build_agents_yaml(agents: dict[str, str]) -> str:
    """Build the agents YAML block for a given agent subset."""
    lines = [
        "",
        "# --- Agent profiles (intense enriched) ---",
        "agents:",
    ]
    for role, profile in agents.items():
        lines.append(f"  {role}: {profile}")
    lines.append("")
    lines.append("judge_profile: judge_standard")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

PIPELINE = "propose -> critique -> revise -> [IE post_revision] -> CRIT -> judge"


def _build_yaml(subset: dict, nudge: dict, threshold: float) -> str:
    """Build a formatted YAML string for one (subset, nudge, threshold) combination."""
    parts: list[str] = []

    thresh_tag = str(threshold).replace(".", "")  # e.g. "08" or "065"
    config_name = f"{subset['name']}_{nudge['name']}_th{thresh_tag}"
    roles = list(subset["agents"].keys())

    title = f"{subset['name']} / {nudge['name']} nudge / threshold {threshold}"

    # Header
    parts.append(_HEADER.format(
        title=title,
        config_name=config_name,
        pipeline=PIPELINE,
    ))

    # Data
    parts.append(_DATA)

    # Agents
    parts.append(_build_agents_yaml(subset["agents"]))

    # Debate setup
    parts.append(_DEBATE_OPEN)

    # PID disabled
    parts.append(_PID_DISABLED)

    # Intervention config
    parts.append("")
    parts.append("  # --- Intervention engine (intra-round retry on acute failures) ---")
    parts.append("  intervention_config:")
    parts.append("    enabled: true")

    # Strict: include revision_limits; Soft: omit entirely
    if nudge["has_revision_limits"]:
        parts.append("")
        parts.append("    # Portfolio revision limits applied to all intervention retries.")
        parts.append("    revision_limits:")
        parts.append("      max_changed_tickers: 2")
        parts.append("      max_change_pct: 5")

    parts.append("")
    parts.append("    rules:")

    # JS Collapse — always enabled, parameterised threshold, max_retries 1
    parts.append("      # JS Collapse Detection (post_revision)")
    parts.append("      # Fires when JS_revision / JS_proposal < threshold, indicating agents")
    parts.append("      # converged toward identical allocations during the revision phase.")
    parts.append("      js_collapse:")
    parts.append("        enabled: true")
    parts.append(f"        threshold: {threshold}              # collapse_ratio below this triggers retry")
    parts.append("        min_js_proposal: 0.10       # skip when proposals already near-consensus")
    parts.append("        max_retries: 1              # max retry attempts for this rule")
    parts.append(_build_js_nudge_yaml(roles))

    parts.append("")

    # Reasoning quality — disabled
    parts.append("      # Reasoning Quality (post_crit) — disabled for this ablation")
    parts.append("      reasoning_quality:")
    parts.append("        enabled: false")
    parts.append("        rho_threshold: 0.5")
    parts.append("        max_retries: 1")

    parts.append("")

    # Broker
    parts.append(_BROKER)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)

    configs = [
        (subset, nudge, threshold)
        for subset in AGENT_SUBSETS
        for nudge in NUDGE_TYPES
        for threshold in THRESHOLDS
    ]

    print(f"Generating {len(configs)} config files in {TARGET_DIR}/")

    for subset, nudge, threshold in configs:
        thresh_tag = str(threshold).replace(".", "")  # e.g. "08" or "065"
        filename = f"debate_{subset['name']}_{nudge['name']}_th{thresh_tag}_r2_t0.3.yaml"
        content = _build_yaml(subset, nudge, threshold)

        out_path = TARGET_DIR / filename
        out_path.write_text(content)
        print(f"  {filename}")

    print(f"Done. {len(configs)} files written.")


if __name__ == "__main__":
    generate()
