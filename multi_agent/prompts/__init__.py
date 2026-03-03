"""
Enriched role prompts and debate prompts for the multi-agent trading system.

Prompts are loaded from .txt template files in this package directory and
rendered via Jinja2.  The public API (constants + functions) is identical to
the former single-file ``prompts.py`` so that all existing imports work
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import re

from jinja2 import Environment, FileSystemLoader

from ..config import AgentRole

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


_DEFAULT_SECTION_ORDER = [
    "preamble", "context", "agent_data", "task", "scaffolding", "output_format",
]

_SECTION_RE = re.compile(r"^---SECTION:\s*(\w+)---\s*$", re.MULTILINE)


def _load_sectioned_template(name: str) -> dict[str, str]:
    """Split a template file by ``---SECTION: name---`` delimiters.

    Returns ``{section_name: raw_template_text}``.
    If the file has no section markers the entire content is returned
    under the key ``"_unsectioned"``.
    """
    raw = (_TEMPLATE_DIR / name).read_text()
    markers = list(_SECTION_RE.finditer(raw))
    if not markers:
        return {"_unsectioned": raw}

    sections: dict[str, str] = {}
    for i, m in enumerate(markers):
        section_name = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        sections[section_name] = raw[start:end].strip("\n")
    return sections


def _assemble_user_prompt(
    sections: dict[str, str],
    order: list[str],
    template_vars: dict,
) -> str:
    """Render each section with Jinja2 and join in the specified order.

    Sections not present in *sections* dict are silently skipped.
    """
    parts: list[str] = []
    for section_name in order:
        raw_tmpl = sections.get(section_name)
        if raw_tmpl is None:
            continue
        tmpl = _env.from_string(raw_tmpl)
        rendered = tmpl.render(**template_vars).strip()
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


# =============================================================================
# ANTI-FAILURE-MODE RULES (injected into every agent prompt)
# =============================================================================

CAUSAL_CLAIM_FORMAT: str = _load("scaffolding/causal_claim_format.txt")
FORCED_UNCERTAINTY: str = _load("scaffolding/forced_uncertainty.txt")
TRAP_AWARENESS: str = _load("scaffolding/trap_awareness.txt")
JSON_OUTPUT_INSTRUCTIONS: str = _load("output_format/json_output_instructions.txt")
ALLOCATION_OUTPUT_INSTRUCTIONS: str = _load("output_format/allocation_output_instructions.txt")

# =============================================================================
# ENRICHED ROLE PROMPTS
# =============================================================================

ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    AgentRole.MACRO: _load("roles/macro.txt"),
    AgentRole.VALUE: _load("roles/value.txt"),
    AgentRole.RISK: _load("roles/risk.txt"),
    AgentRole.TECHNICAL: _load("roles/technical.txt"),
    AgentRole.SENTIMENT: _load("roles/sentiment.txt"),
    AgentRole.DEVILS_ADVOCATE: _load("roles/devils_advocate.txt"),
}

# =============================================================================
# SYSTEM-LEVEL CAUSAL CONTRACT (optional, gated by use_system_causal_contract)
# =============================================================================

SYSTEM_CAUSAL_CONTRACT: str = _load("system_contract/system_causal_contract.txt")

ROLE_SYSTEM_PROMPTS_SLIM: dict[str, str] = {
    AgentRole.MACRO: _load("roles/macro_slim.txt"),
    AgentRole.VALUE: _load("roles/value_slim.txt"),
    AgentRole.RISK: _load("roles/risk_slim.txt"),
    AgentRole.TECHNICAL: _load("roles/technical_slim.txt"),
    AgentRole.SENTIMENT: _load("roles/sentiment_slim.txt"),
    AgentRole.DEVILS_ADVOCATE: _load("roles/devils_advocate_slim.txt"),
}


def get_role_prompts(use_causal_contract: bool = False) -> dict[str, str]:
    """Return the appropriate role prompt dict based on causal contract flag."""
    return ROLE_SYSTEM_PROMPTS_SLIM if use_causal_contract else ROLE_SYSTEM_PROMPTS

# =============================================================================
# AGREEABLENESS MODIFIER (injected into critique prompts)
# =============================================================================

_AGREEABLENESS_TEMPLATES = {
    "confrontational": _load("agreeableness/confrontational.txt"),
    "skeptical": _load("agreeableness/skeptical.txt"),
    "balanced": _load("agreeableness/balanced.txt"),
    "collaborative": _load("agreeableness/collaborative.txt"),
    "agreeable": _load("agreeableness/agreeable.txt"),
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
# DEBATE PHASE PROMPTS (rendered via Jinja2 templates)
# =============================================================================


def build_proposal_user_prompt(
    context: str,
    use_system_causal_contract: bool = False,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
) -> str:
    """User prompt sent to each role agent for their initial proposal."""
    causal = "" if use_system_causal_contract else CAUSAL_CLAIM_FORMAT
    uncertainty = "" if use_system_causal_contract else FORCED_UNCERTAINTY
    traps = "" if use_system_causal_contract else TRAP_AWARENESS

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("proposal_template", "phases/proposal_allocation.txt")

    template_vars = {
        "context": context,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
        "trap_awareness": traps,
        "json_output_instructions": JSON_OUTPUT_INSTRUCTIONS,
        "allocation_output_instructions": ALLOCATION_OUTPUT_INSTRUCTIONS,
    }

    order = section_order if section_order is not None else _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        # Template has no section markers — render as a single template (legacy)
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars)


def build_critique_prompt(
    role: str,
    context: str,
    all_proposals: list[dict],
    my_proposal: str,
    agreeableness: float = 0.3,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
) -> str:
    """Build critique prompt for a role agent in the debate."""
    others = [p for p in all_proposals if p["role"] != role]
    others_text = "\n\n".join(
        f"### {p['role'].upper()} agent proposed:\n{p['proposal']}"
        for p in others
    )

    agreeableness_mod = get_agreeableness_modifier(agreeableness)

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("critique_template", "phases/critique_allocation.txt")

    template_vars = {
        "role": role.upper(),
        "agreeableness_mod": agreeableness_mod,
        "context": context,
        "my_proposal": my_proposal,
        "others_text": others_text,
    }

    order = section_order if section_order is not None else _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars)


def build_revision_prompt(
    role: str,
    context: str,
    my_proposal: str,
    critiques_received: list[dict],
    agreeableness: float = 0.3,
    use_system_causal_contract: bool = False,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
) -> str:
    """Build revision prompt for a role agent after receiving critiques."""
    critiques_text = "\n".join(
        f"- [{c['from_role'].upper()}]: {c['objection']}"
        + (f" | Falsifier: {c.get('falsifier', 'N/A')}" if c.get("falsifier") else "")
        for c in critiques_received
    )

    if not critiques_text:
        critiques_text = "(No critiques targeted at you this round.)"

    causal = "" if use_system_causal_contract else CAUSAL_CLAIM_FORMAT
    uncertainty = "" if use_system_causal_contract else FORCED_UNCERTAINTY

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("revision_template", "phases/revision_allocation.txt")

    template_vars = {
        "role": role.upper(),
        "context": context,
        "my_proposal": my_proposal,
        "critiques_text": critiques_text,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
    }

    order = section_order if section_order is not None else _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars)


def build_judge_prompt(
    context: str,
    revisions: list[dict],
    all_critiques_text: str,
    strongest_disagreements: str = "",
    use_system_causal_contract: bool = False,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
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

    causal = "" if use_system_causal_contract else CAUSAL_CLAIM_FORMAT

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("judge_template", "phases/judge_allocation.txt")

    template_vars = {
        "context": context,
        "revisions_text": revisions_text,
        "all_critiques_text": all_critiques_text,
        "disagreements_section": disagreements_section,
        "causal_claim_format": causal,
    }

    order = section_order if section_order is not None else _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars)
