"""
Enriched role prompts and debate prompts for the multi-agent trading system.

Prompts are loaded from .txt template files in this package directory and
rendered via Jinja2.  The public API (constants + functions) is identical to
the former single-file ``prompts.py`` so that all existing imports work
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import re

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import AgentRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 environment — templates live next to this __init__.py
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)

# ---------------------------------------------------------------------------
# Module catalog — auto-discovery of prompt modules from directory structure
# ---------------------------------------------------------------------------

_SUBDIR_PREFIX: dict[str, str] = {
    "roles": "role",
    "scaffolding": "scaffolding",
    "output_format": "output",
    "tone": "tone",
    "system_contract": "contract",
    "phases": "phase",
}

_MODULE_ALIASES: dict[str, str] = {
    "scaffolding_causal": "scaffolding/causal_claim_format.txt",
    "scaffolding_uncertainty": "scaffolding/forced_uncertainty.txt",
    "scaffolding_traps": "scaffolding/trap_awareness.txt",
    "output_allocation": "output_format/allocation_output_slim.txt",
    "output_json": "output_format/json_output_instructions.txt",
    "causal_contract": "system_contract/system_causal_contract.txt",
}


def _build_module_catalog() -> dict[str, str]:
    """Auto-discover prompt modules from directory structure.

    Convention: {subdir}/{file}.txt -> {prefix}_{filestem}
    """
    catalog: dict[str, str] = {}
    for subdir, prefix in _SUBDIR_PREFIX.items():
        dir_path = _TEMPLATE_DIR / subdir
        if not dir_path.is_dir():
            continue
        for txt_file in dir_path.glob("*.txt"):
            module_name = f"{prefix}_{txt_file.stem}"
            catalog[module_name] = str(txt_file.relative_to(_TEMPLATE_DIR))
    # Apply short aliases for common modules
    catalog.update(_MODULE_ALIASES)
    return catalog


MODULE_CATALOG: dict[str, str] = _build_module_catalog()


def load_module(name: str, overrides: dict[str, str] | None = None) -> str:
    """Load a prompt module by symbolic name.

    Checks prompt_file_overrides first, then MODULE_CATALOG.
    Returns empty string if not found.
    """
    if overrides and name in overrides:
        path = _TEMPLATE_DIR / overrides[name]
        return path.read_text() if path.exists() else ""
    rel_path = MODULE_CATALOG.get(name)
    if rel_path:
        return (_TEMPLATE_DIR / rel_path).read_text()
    return ""


# ---------------------------------------------------------------------------
# Prompt profile loading + resolution
# ---------------------------------------------------------------------------


def load_prompt_profile(name: str) -> dict:
    """Load a prompt profile YAML file by name."""
    path = _TEMPLATE_DIR / "profiles" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt profile not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def resolve_prompt_profile(config: dict, role: str, phase: str) -> dict:
    """Resolve the prompt profile for a given agent role and phase.

    Returns phase-specific ``system_blocks`` and ``user_sections`` lists.
    Priority: role_overrides[role] keys > base profile keys.
    """
    profile_name = config.get("prompt_profile", "default")
    base = load_prompt_profile(profile_name)
    role_overrides = config.get("role_overrides", {}).get(role, {})
    merged = {**base, **role_overrides}

    # Extract phase-specific lists
    sys_blocks = merged.get("system_blocks", {})
    usr_sections = merged.get("user_sections", {})

    return {
        "system_blocks": sys_blocks.get(phase, []),
        "user_sections": usr_sections.get(phase, []),
    }


# ---------------------------------------------------------------------------
# Template variable registry (for CI validation)
# ---------------------------------------------------------------------------

TEMPLATE_VARS: dict[str, set[str]] = {
    "propose": {
        "context", "causal_claim_format", "forced_uncertainty",
        "trap_awareness", "json_output_instructions",
        "allocation_output_instructions", "sector_constraints",
    },
    "critique": {"role", "context", "my_proposal", "others_text", "sector_constraints"},
    "revise": {
        "role", "context", "my_proposal", "critiques_text",
        "causal_claim_format", "forced_uncertainty", "sector_constraints",
    },
    "judge": {
        "context", "revisions_text", "all_critiques_text",
        "disagreements_section", "causal_claim_format", "sector_constraints",
    },
}

CRIT_TEMPLATE_VARS: dict[str, set[str]] = {
    "crit_system": {"agent_role"},
    "crit_user": {"round", "agent_role", "proposal",
                  "critiques_received", "revised_argument"},
}


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
    prompt_file_overrides: dict[str, str] | None = None,
) -> str:
    """Render each section with Jinja2 and join in the specified order.

    For each name in *order*:
      - If it exists in *sections* dict, render with Jinja2 (template section).
      - Otherwise, try MODULE_CATALOG (static module text, no Jinja rendering).
      - If neither, skip silently.
    """
    parts: list[str] = []
    skipped: list[str] = []
    for section_name in order:
        raw_tmpl = sections.get(section_name)
        if raw_tmpl is not None:
            # Template section — render with Jinja2
            tmpl = _env.from_string(raw_tmpl)
            rendered = tmpl.render(**template_vars).strip()
            if rendered:
                parts.append(rendered)
            else:
                skipped.append(section_name)
        else:
            # Not a template section — try MODULE_CATALOG
            content = load_module(section_name, prompt_file_overrides)
            if content:
                parts.append(content.strip())
            else:
                skipped.append(section_name)
    if skipped:
        logger.warning(
            "User prompt sections declared but not found: %s", skipped
        )
    return "\n\n".join(parts)


# =============================================================================
# ANTI-FAILURE-MODE RULES (injected into every agent prompt)
# =============================================================================

CAUSAL_CLAIM_FORMAT: str = _load("scaffolding/causal_claim_format.txt")
FORCED_UNCERTAINTY: str = _load("scaffolding/forced_uncertainty.txt")
TRAP_AWARENESS: str = _load("scaffolding/trap_awareness.txt")
JSON_OUTPUT_INSTRUCTIONS: str = _load("output_format/json_output_instructions.txt")
ALLOCATION_OUTPUT_INSTRUCTIONS: str = _load("output_format/allocation_output_slim.txt")

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
# SYSTEM-LEVEL CAUSAL CONTRACT
# =============================================================================

SYSTEM_CAUSAL_CONTRACT: str = _load("system_contract/system_causal_contract.txt")


def get_role_prompts() -> dict[str, str]:
    """Return the full role prompt dict."""
    return ROLE_SYSTEM_PROMPTS

# =============================================================================
# DEBATE PHASE PROMPTS (rendered via Jinja2 templates)
# =============================================================================


def build_proposal_user_prompt(
    context: str,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
    user_sections: list[str] | None = None,
    sector_constraints: str = "",
) -> str:
    """User prompt sent to each role agent for their initial proposal."""
    causal = CAUSAL_CLAIM_FORMAT
    uncertainty = FORCED_UNCERTAINTY
    traps = TRAP_AWARENESS

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("proposal_template", "phases/proposal_allocation.txt")

    # Allow overriding the allocation output instructions via prompt_file_overrides
    alloc_instructions = (
        load_module("output_allocation", overrides)
        if "output_allocation" in overrides
        else ALLOCATION_OUTPUT_INSTRUCTIONS
    )

    template_vars = {
        "context": context,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
        "trap_awareness": traps,
        "json_output_instructions": JSON_OUTPUT_INSTRUCTIONS,
        "allocation_output_instructions": alloc_instructions,
        "sector_constraints": sector_constraints,
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        # Template has no section markers — render as a single template (legacy)
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides)


def build_critique_prompt(
    role: str,
    context: str,
    all_proposals: list[dict],
    my_proposal: str,
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
    user_sections: list[str] | None = None,
    sector_constraints: str = "",
) -> str:
    """Build critique user prompt for a role agent in the debate.

    Tone/agreeableness is now handled by the system prompt via
    ``PromptRegistry.build()`` — not injected into the user prompt.
    """
    others = [p for p in all_proposals if p["role"] != role]
    others_text = "\n\n".join(
        f"### {p['role'].upper()} agent proposed:\n{p['proposal']}"
        for p in others
    )

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("critique_template", "phases/critique_allocation.txt")

    template_vars = {
        "role": role.upper(),
        "context": context,
        "my_proposal": my_proposal,
        "others_text": others_text,
        "sector_constraints": sector_constraints,
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides)


def build_revision_prompt(
    role: str,
    context: str,
    my_proposal: str,
    critiques_received: list[dict],
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
    user_sections: list[str] | None = None,
    sector_constraints: str = "",
) -> str:
    """Build revision user prompt for a role agent after receiving critiques."""
    critiques_text = "\n".join(
        f"- [{c['from_role'].upper()}]: {c['objection']}"
        + (f" | Falsifier: {c.get('falsifier', 'N/A')}" if c.get("falsifier") else "")
        for c in critiques_received
    )

    if not critiques_text:
        critiques_text = "(No critiques targeted at you this round.)"

    causal = CAUSAL_CLAIM_FORMAT
    uncertainty = FORCED_UNCERTAINTY

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("revision_template", "phases/revision_allocation.txt")

    template_vars = {
        "role": role.upper(),
        "context": context,
        "my_proposal": my_proposal,
        "critiques_text": critiques_text,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
        "sector_constraints": sector_constraints,
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides)


def build_judge_prompt(
    context: str,
    revisions: list[dict],
    all_critiques_text: str,
    strongest_disagreements: str = "",
    section_order: list[str] | None = None,
    prompt_file_overrides: dict[str, str] | None = None,
    allocation_mode: bool = True,  # kept for backward compat, always True
    user_sections: list[str] | None = None,
    sector_constraints: str = "",
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

    causal = CAUSAL_CLAIM_FORMAT

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("judge_template", "phases/judge_allocation.txt")

    template_vars = {
        "context": context,
        "revisions_text": revisions_text,
        "all_critiques_text": all_critiques_text,
        "disagreements_section": disagreements_section,
        "causal_claim_format": causal,
        "sector_constraints": sector_constraints,
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides)
