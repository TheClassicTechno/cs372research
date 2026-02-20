"""
Enriched role prompts and debate prompts for the multi-agent trading system.

Prompts are loaded from .txt template files in this package directory and
rendered via Jinja2.  The public API (constants + functions) is identical to
the former single-file ``prompts.py`` so that all existing imports work
unchanged.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..config import AgentRole
from ..models import Observation

# ---------------------------------------------------------------------------
# Jinja2 environment — templates live next to this __init__.py
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
)


def _load(name: str) -> str:
    """Return the raw text of a template file (no rendering)."""
    return (_TEMPLATE_DIR / name).read_text()


# =============================================================================
# ANTI-FAILURE-MODE RULES (injected into every agent prompt)
# =============================================================================

CAUSAL_CLAIM_FORMAT: str = _load("causal_claim_format.txt")
FORCED_UNCERTAINTY: str = _load("forced_uncertainty.txt")
TRAP_AWARENESS: str = _load("trap_awareness.txt")
JSON_OUTPUT_INSTRUCTIONS: str = _load("json_output_instructions.txt")

# =============================================================================
# ENRICHED ROLE PROMPTS
# =============================================================================

ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    AgentRole.MACRO: _load("role_macro.txt"),
    AgentRole.VALUE: _load("role_value.txt"),
    AgentRole.RISK: _load("role_risk.txt"),
    AgentRole.TECHNICAL: _load("role_technical.txt"),
    AgentRole.SENTIMENT: _load("role_sentiment.txt"),
    AgentRole.DEVILS_ADVOCATE: _load("role_devils_advocate.txt"),
}

# =============================================================================
# PIPELINE AGENT PROMPTS
# =============================================================================

NEWS_DIGEST_SYSTEM_PROMPT: str = _load("news_digest_system.txt")
DATA_ANALYSIS_SYSTEM_PROMPT: str = _load("data_analysis_system.txt")

# =============================================================================
# AGREEABLENESS MODIFIER (injected into critique prompts)
# =============================================================================

_AGREEABLENESS_TEMPLATES = {
    "confrontational": _load("agreeableness_confrontational.txt"),
    "skeptical": _load("agreeableness_skeptical.txt"),
    "balanced": _load("agreeableness_balanced.txt"),
    "collaborative": _load("agreeableness_collaborative.txt"),
    "agreeable": _load("agreeableness_agreeable.txt"),
}


def get_agreeableness_modifier(agreeableness: float) -> str:
    """
    Generate a system prompt modifier based on the agreeableness knob.

    agreeableness=0.0 -> maximally confrontational (fights every point)
    agreeableness=0.5 -> balanced (critiques on merit)
    agreeableness=1.0 -> maximally agreeable/sycophantic (finds consensus)

    This is a key experimental variable for RQ3 (does debate reduce sycophancy?).
    """
    if agreeableness < 0.2:
        return _AGREEABLENESS_TEMPLATES["confrontational"]
    elif agreeableness < 0.4:
        return _AGREEABLENESS_TEMPLATES["skeptical"]
    elif agreeableness < 0.6:
        return _AGREEABLENESS_TEMPLATES["balanced"]
    elif agreeableness < 0.8:
        return _AGREEABLENESS_TEMPLATES["collaborative"]
    else:
        return _AGREEABLENESS_TEMPLATES["agreeable"]


# =============================================================================
# OBSERVATION CONTEXT BUILDER
# =============================================================================


def build_observation_context(
    obs: Observation,
    pipeline_context: str = "",
) -> str:
    """Build the market context string from an Observation, optionally enriched
    with pipeline preprocessing output."""
    prices_str = ", ".join(
        f"{t}: ${p:.2f}" for t, p in obs.market_state.prices.items()
    )

    returns_str = "N/A"
    if obs.market_state.returns:
        returns_str = ", ".join(
            f"{t}: {r * 100:.2f}%"
            for t, r in obs.market_state.returns.items()
        )

    vol_str = ""
    if obs.market_state.volatility:
        vol_str = "\n- Volatility: " + ", ".join(
            f"{t}: {v:.4f}" for t, v in obs.market_state.volatility.items()
        )

    portfolio_str = (
        f"Cash: ${obs.portfolio_state.cash:.2f}, "
        f"Positions: {obs.portfolio_state.positions}"
    )

    context = f"""## Market Observation
- Timestamp: {obs.timestamp}
- Universe: {', '.join(obs.universe)}
- Prices: {prices_str}
- Returns: {returns_str}{vol_str}
- Portfolio: {portfolio_str}"""

    if obs.text_context:
        context += f"\n- News/Context: {obs.text_context}"

    if obs.constraints:
        context += (
            f"\n- Constraints: max_leverage={obs.constraints.max_leverage}, "
            f"max_position_size={obs.constraints.max_position_size}"
        )

    if pipeline_context:
        context += f"\n\n## Pre-Processed Intelligence\n{pipeline_context}"

    return context


# =============================================================================
# DEBATE PHASE PROMPTS (rendered via Jinja2 templates)
# =============================================================================


def build_proposal_user_prompt(context: str) -> str:
    """User prompt sent to each role agent for their initial proposal."""
    tmpl = _env.get_template("proposal.txt")
    return tmpl.render(
        context=context,
        causal_claim_format=CAUSAL_CLAIM_FORMAT,
        forced_uncertainty=FORCED_UNCERTAINTY,
        trap_awareness=TRAP_AWARENESS,
        json_output_instructions=JSON_OUTPUT_INSTRUCTIONS,
    )


def build_critique_prompt(
    role: str,
    context: str,
    all_proposals: list[dict],
    my_proposal: str,
    agreeableness: float = 0.3,
) -> str:
    """Build critique prompt for a role agent in the debate."""
    others = [p for p in all_proposals if p["role"] != role]
    others_text = "\n\n".join(
        f"### {p['role'].upper()} agent proposed:\n{p['proposal']}"
        for p in others
    )

    agreeableness_mod = get_agreeableness_modifier(agreeableness)

    tmpl = _env.get_template("critique.txt")
    return tmpl.render(
        role=role.upper(),
        agreeableness_mod=agreeableness_mod,
        context=context,
        my_proposal=my_proposal,
        others_text=others_text,
    )


def build_revision_prompt(
    role: str,
    context: str,
    my_proposal: str,
    critiques_received: list[dict],
    agreeableness: float = 0.3,
) -> str:
    """Build revision prompt for a role agent after receiving critiques."""
    critiques_text = "\n".join(
        f"- [{c['from_role'].upper()}]: {c['objection']}"
        + (f" | Falsifier: {c.get('falsifier', 'N/A')}" if c.get("falsifier") else "")
        for c in critiques_received
    )

    if not critiques_text:
        critiques_text = "(No critiques targeted at you this round.)"

    tmpl = _env.get_template("revision.txt")
    return tmpl.render(
        role=role.upper(),
        context=context,
        my_proposal=my_proposal,
        critiques_text=critiques_text,
        causal_claim_format=CAUSAL_CLAIM_FORMAT,
        forced_uncertainty=FORCED_UNCERTAINTY,
    )


def build_judge_prompt(
    context: str,
    revisions: list[dict],
    all_critiques_text: str,
    strongest_disagreements: str = "",
) -> str:
    """Build the judge/aggregator prompt for final decision."""
    revisions_text = "\n\n".join(
        f"### {r['role'].upper()} (confidence: {r['confidence']:.2f})\n{r['action']}"
        for r in revisions
    )

    disagreements_section = ""
    if strongest_disagreements:
        disagreements_section = (
            f"\n## Strongest Disagreements (preserved for audit)\n"
            f"{strongest_disagreements}"
        )

    tmpl = _env.get_template("judge.txt")
    return tmpl.render(
        context=context,
        revisions_text=revisions_text,
        all_critiques_text=all_critiques_text,
        disagreements_section=disagreements_section,
        causal_claim_format=CAUSAL_CLAIM_FORMAT,
    )
