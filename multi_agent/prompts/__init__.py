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
    "output_allocation": "output_format/allocation_output_instructions.txt",
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


# ---------------------------------------------------------------------------
# Allocation constraint defaults — match AllocationConstraints model defaults
# ---------------------------------------------------------------------------

_DEFAULT_MAX_WEIGHT = 1.0
_DEFAULT_MIN_HOLDINGS = 1


def _render_alloc_instructions(
    raw_text: str, constraints: dict | None = None,
) -> str:
    """Replace ``__MARKER__`` placeholders in allocation instruction text.

    Args:
        raw_text: Raw instruction text with ``__MAX_WEIGHT__``, etc.
        constraints: Dict with ``max_weight`` / ``min_holdings`` keys
            (typically from ``config["allocation_constraints"]``).

    Returns:
        Instruction text with actual constraint values substituted.
    """
    c = constraints or {}
    max_w = c.get("max_weight", _DEFAULT_MAX_WEIGHT)
    min_h = c.get("min_holdings", _DEFAULT_MIN_HOLDINGS)
    text = raw_text
    text = text.replace("__MAX_WEIGHT__", f"{max_w:.2f}")
    text = text.replace("__MAX_WEIGHT_PCT__", str(int(round(max_w * 100))))
    text = text.replace("__MIN_HOLDINGS__", str(min_h))
    return text


_DEFAULT_SECTION_ORDER = [
    "preamble", "context", "agent_data", "task", "scaffolding", "output_format",
]

_SECTION_RE = re.compile(r"^---SECTION:\s*(\w+)---\s*$", re.MULTILINE)

# Regex to find Jinja2 variable references like {{ var_name }}
_VAR_REF_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Vars that are legitimately empty in some configurations — no warning needed.
_OPTIONAL_VARS = frozenset({
    "sector_constraints",
    "disagreements_section",
    "max_weight",
    "max_weight_pct",
    "min_holdings",
})


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


def _warn_empty_template_vars(
    section_name: str,
    raw_template: str,
    template_vars: dict,
    phase: str = "",
) -> None:
    """Warn loudly when a template section references an empty variable.

    Scans the raw Jinja2 template for ``{{ var_name }}`` references and
    checks whether the corresponding value in *template_vars* is falsy
    (empty string, empty list, etc.).  Skips vars in ``_OPTIONAL_VARS``.
    """
    referenced = set(_VAR_REF_RE.findall(raw_template))
    for var_name in sorted(referenced):
        if var_name in _OPTIONAL_VARS:
            continue
        val = template_vars.get(var_name)
        # val is None → StrictUndefined will already crash.
        # val is falsy but present → silent blank in rendered prompt.
        if val is not None and not val:
            tag = f" (phase={phase})" if phase else ""
            logger.warning(
                "EMPTY TEMPLATE VAR: '%s' is empty in section '%s'%s. "
                "The rendered prompt will have a blank where agent data should be.",
                var_name, section_name, tag,
            )


def _assemble_user_prompt(
    sections: dict[str, str],
    order: list[str],
    template_vars: dict,
    prompt_file_overrides: dict[str, str] | None = None,
    phase: str = "",
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
            # Check for empty vars BEFORE rendering
            _warn_empty_template_vars(section_name, raw_tmpl, template_vars, phase)
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
# Raw templates with __MARKER__ placeholders — rendered with defaults for
# backward-compat module-level constants; re-rendered per-call with actual
# config values inside the builder functions.
_RAW_ALLOC_INSTRUCTIONS: str = _load("output_format/allocation_output_instructions.txt")
_RAW_ALLOC_INSTRUCTIONS_ENUM: str = _load("output_format/allocation_output_instructions_enumerated.txt")
ALLOCATION_OUTPUT_INSTRUCTIONS: str = _render_alloc_instructions(_RAW_ALLOC_INSTRUCTIONS)
ALLOCATION_OUTPUT_INSTRUCTIONS_ENUMERATED: str = _render_alloc_instructions(_RAW_ALLOC_INSTRUCTIONS_ENUM)

# =============================================================================
# ENRICHED ROLE PROMPTS
# =============================================================================

ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    "macro": _load("roles/macro.txt"),
    "value": _load("roles/value.txt"),
    "risk": _load("roles/risk.txt"),
    "technical": _load("roles/technical.txt"),
    "sentiment": _load("roles/sentiment.txt"),
    "devils_advocate": _load("roles/devils_advocate.txt"),
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
    allocation_constraints: dict | None = None,
) -> str:
    """User prompt sent to each role agent for their initial proposal."""
    causal = CAUSAL_CLAIM_FORMAT
    uncertainty = FORCED_UNCERTAINTY
    traps = TRAP_AWARENESS

    overrides = prompt_file_overrides or {}
    template_name = overrides.get("proposal_template", "phases/proposal_allocation.txt")

    # Render allocation instruction text with actual constraint values
    ac = allocation_constraints
    if "output_allocation" in overrides:
        raw_alloc = load_module("output_allocation", overrides)
    else:
        raw_alloc = _RAW_ALLOC_INSTRUCTIONS
    alloc_instructions = _render_alloc_instructions(raw_alloc, ac)
    alloc_instructions_enum = _render_alloc_instructions(_RAW_ALLOC_INSTRUCTIONS_ENUM, ac)

    # Compute display values for Jinja phase template vars
    _ac = ac or {}
    max_weight_val = _ac.get("max_weight", _DEFAULT_MAX_WEIGHT)
    min_holdings_val = _ac.get("min_holdings", _DEFAULT_MIN_HOLDINGS)

    template_vars = {
        "context": context,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
        "trap_awareness": traps,
        "json_output_instructions": JSON_OUTPUT_INSTRUCTIONS,
        "allocation_output_instructions": alloc_instructions,
        "allocation_output_instructions_enumerated": alloc_instructions_enum,
        "sector_constraints": sector_constraints,
        "max_weight": f"{max_weight_val:.2f}",
        "max_weight_pct": str(int(round(max_weight_val * 100))),
        "min_holdings": str(min_holdings_val),
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        # Template has no section markers — render as a single template (legacy)
        _warn_empty_template_vars("_unsectioned", (_TEMPLATE_DIR / template_name).read_text(), template_vars, "propose")
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides, phase="propose")


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
    my_proposal_v2: str = "",
    others_text_v2: str = "",
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
        "my_proposal_v2": my_proposal_v2,
        "others_text": others_text,
        "others_text_v2": others_text_v2,
        "sector_constraints": sector_constraints,
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        _warn_empty_template_vars("_unsectioned", (_TEMPLATE_DIR / template_name).read_text(), template_vars, "critique")
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides, phase="critique")


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
    my_proposal_v2: str = "",
    critiques_text_v2: str = "",
    allocation_constraints: dict | None = None,
    intervention_nudge: str = "",
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

    # Compute constraint display values for Jinja phase template vars
    _ac = allocation_constraints or {}
    max_weight_val = _ac.get("max_weight", _DEFAULT_MAX_WEIGHT)
    min_holdings_val = _ac.get("min_holdings", _DEFAULT_MIN_HOLDINGS)

    template_vars = {
        "role": role.upper(),
        "context": context,
        "my_proposal": my_proposal,
        "my_proposal_v2": my_proposal_v2,
        "critiques_text": critiques_text,
        "critiques_text_v2": critiques_text_v2,
        "causal_claim_format": causal,
        "forced_uncertainty": uncertainty,
        "sector_constraints": sector_constraints,
        "max_weight": f"{max_weight_val:.2f}",
        "max_weight_pct": str(int(round(max_weight_val * 100))),
        "min_holdings": str(min_holdings_val),
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        _warn_empty_template_vars("_unsectioned", (_TEMPLATE_DIR / template_name).read_text(), template_vars, "revise")
        tmpl = _env.get_template(template_name)
        rendered = tmpl.render(**template_vars)
    else:
        rendered = _assemble_user_prompt(sections, order, template_vars, overrides, phase="revise")

    # Prepend intervention nudge if present (ephemeral, for retry only)
    if intervention_nudge:
        rendered = f"### INTERVENTION NOTICE\n\n{intervention_nudge}\n\n---\n\n{rendered}"

    return rendered


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
    allocation_constraints: dict | None = None,
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

    # Compute constraint display values for Jinja phase template vars
    _ac = allocation_constraints or {}
    max_weight_val = _ac.get("max_weight", _DEFAULT_MAX_WEIGHT)

    template_vars = {
        "context": context,
        "revisions_text": revisions_text,
        "all_critiques_text": all_critiques_text,
        "disagreements_section": disagreements_section,
        "causal_claim_format": causal,
        "sector_constraints": sector_constraints,
        "max_weight": f"{max_weight_val:.2f}",
        "max_weight_pct": str(int(round(max_weight_val * 100))),
    }

    order = user_sections or section_order or _DEFAULT_SECTION_ORDER
    sections = _load_sectioned_template(template_name)

    if "_unsectioned" in sections:
        _warn_empty_template_vars("_unsectioned", (_TEMPLATE_DIR / template_name).read_text(), template_vars, "judge")
        tmpl = _env.get_template(template_name)
        return tmpl.render(**template_vars)

    return _assemble_user_prompt(sections, order, template_vars, overrides, phase="judge")
