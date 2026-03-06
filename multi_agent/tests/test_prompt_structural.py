"""L2 structural tests for rendered debate prompts.

Validates:
  - No unresolved Jinja2 variables remain after rendering (all four phases).
  - Role identity text appears in system prompts for each role.
  - Rendered prompt lengths meet minimum character thresholds.
  - Every variable declared in TEMPLATE_VARS[phase] appears in the template file.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from multi_agent.prompts import (
    build_proposal_user_prompt,
    build_critique_prompt,
    build_revision_prompt,
    build_judge_prompt,
    TEMPLATE_VARS,
    resolve_prompt_profile,
    _TEMPLATE_DIR,
)
from multi_agent.prompts.registry import (
    PromptRegistry,
    reset_registry_cache,
)

# ---------------------------------------------------------------------------
# Shared mock context for rendering
# ---------------------------------------------------------------------------

_CONTEXT = "## Portfolio\n- Cash: $100,000\n- Universe: AAPL, MSFT\n\nMemo text."

_MY_PROPOSAL = (
    '{"allocation": {"AAPL": 0.6, "MSFT": 0.4}, '
    '"justification": "Growth", "confidence": 0.8, '
    '"risks_or_falsifiers": "Downturn", '
    '"claims": [{"claim_text": "AAPL up", "pearl_level": "L1", '
    '"variables": ["AAPL"], "assumptions": [], "confidence": 0.7}]}'
)

_ALL_PROPOSALS = [
    {"role": "macro", "proposal": _MY_PROPOSAL},
    {"role": "value", "proposal": _MY_PROPOSAL},
    {"role": "risk", "proposal": _MY_PROPOSAL},
    {"role": "technical", "proposal": _MY_PROPOSAL},
]

_CRITIQUES_RECEIVED = [
    {
        "from_role": "value",
        "objection": "Too concentrated in AAPL.",
        "falsifier": "AAPL earnings beat by >10%",
    },
]

_REVISIONS = [
    {"role": "macro", "action": _MY_PROPOSAL, "confidence": 0.8},
    {"role": "value", "action": _MY_PROPOSAL, "confidence": 0.7},
]

_ALL_CRITIQUES_TEXT = "- [VALUE]: Too concentrated in AAPL."


def _render_user(phase: str, role: str = "macro") -> str:
    """Helper: render the user prompt for a given phase."""
    if phase == "propose":
        return build_proposal_user_prompt(_CONTEXT)
    elif phase == "critique":
        return build_critique_prompt(
            role=role,
            context=_CONTEXT,
            all_proposals=_ALL_PROPOSALS,
            my_proposal=_MY_PROPOSAL,
        )
    elif phase == "revise":
        return build_revision_prompt(
            role=role,
            context=_CONTEXT,
            my_proposal=_MY_PROPOSAL,
            critiques_received=_CRITIQUES_RECEIVED,
        )
    elif phase == "judge":
        return build_judge_prompt(
            context=_CONTEXT,
            revisions=_REVISIONS,
            all_critiques_text=_ALL_CRITIQUES_TEXT,
        )
    raise ValueError(f"Unknown phase: {phase}")


def _render_system(role: str, phase: str) -> str:
    """Helper: render the system prompt for a given role and phase."""
    config = {"prompt_profile": "default"}
    profile = resolve_prompt_profile(config, role, phase)
    registry = PromptRegistry()
    result = registry.build(
        role=role,
        phase=phase,
        beta=0.5 if phase in ("critique", "revise") else None,
        user_prompt="",
        block_order=profile["system_blocks"],
    )
    return result.system_prompt


# ---------------------------------------------------------------------------
# 1. No unresolved Jinja2 variables
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNoUnresolvedJinjaVars:
    """Rendered prompts must have no leftover {{ ... }} placeholders."""

    def setup_method(self):
        reset_registry_cache()

    @pytest.mark.parametrize("phase", ["propose", "critique", "revise", "judge"])
    def test_user_prompt_no_jinja_vars(self, phase):
        rendered = _render_user(phase)
        assert "{{" not in rendered, (
            f"Unresolved Jinja2 variable in {phase} user prompt: "
            f"{rendered[:300]}"
        )

    @pytest.mark.parametrize("phase", ["propose", "critique", "revise", "judge"])
    def test_system_prompt_no_jinja_vars(self, phase):
        role = "macro" if phase != "judge" else "judge"
        rendered = _render_system(role, phase)
        assert "{{" not in rendered, (
            f"Unresolved Jinja2 variable in {phase} system prompt: "
            f"{rendered[:300]}"
        )


# ---------------------------------------------------------------------------
# 2. Role identity appears in system prompt
# ---------------------------------------------------------------------------

_ROLE_IDENTITY_MARKERS = {
    "macro": "MACRO",
    "value": "VALUE",
    "risk": "RISK",
    "technical": "TECHNICAL",
}


@pytest.mark.fast
class TestRoleIdentityInSystemPrompt:
    """Each role's system prompt must contain its role identity marker."""

    def setup_method(self):
        reset_registry_cache()

    @pytest.mark.parametrize("role", ["macro", "value", "risk", "technical"])
    def test_role_identity_in_propose(self, role):
        sys_prompt = _render_system(role, "propose")
        marker = _ROLE_IDENTITY_MARKERS[role]
        assert marker in sys_prompt.upper(), (
            f"Role identity '{marker}' not found in {role} propose system prompt"
        )

    @pytest.mark.parametrize("role", ["macro", "value", "risk", "technical"])
    def test_role_identity_in_critique(self, role):
        sys_prompt = _render_system(role, "critique")
        marker = _ROLE_IDENTITY_MARKERS[role]
        assert marker in sys_prompt.upper(), (
            f"Role identity '{marker}' not found in {role} critique system prompt"
        )


# ---------------------------------------------------------------------------
# 3. Minimum length thresholds
# ---------------------------------------------------------------------------

_MIN_LENGTHS = {
    "propose": 500,
    "critique": 200,
    "revise": 200,
    "judge": 300,
}


@pytest.mark.fast
class TestMinimumPromptLength:
    """Rendered user prompts must meet minimum character thresholds."""

    @pytest.mark.parametrize("phase,min_chars", list(_MIN_LENGTHS.items()))
    def test_user_prompt_min_length(self, phase, min_chars):
        rendered = _render_user(phase)
        assert len(rendered) >= min_chars, (
            f"{phase} user prompt too short: {len(rendered)} chars < {min_chars} min. "
            f"Preview: {rendered[:200]}"
        )


# ---------------------------------------------------------------------------
# 4. Template variable coverage
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTemplateVarCoverage:
    """Every var in TEMPLATE_VARS[phase] must appear in the template file."""

    # Map phase to the default template file
    _PHASE_TEMPLATES = {
        "propose": "phases/proposal_allocation.txt",
        "critique": "phases/critique_allocation.txt",
        "revise": "phases/revision_allocation.txt",
        "judge": "phases/judge_allocation.txt",
    }

    @pytest.mark.parametrize("phase", ["propose", "critique", "revise", "judge"])
    def test_all_declared_vars_in_template(self, phase):
        """Every var in TEMPLATE_VARS[phase] must appear in the template file
        as a Jinja2 reference ({{ var }}) or in the rendered output.

        Note: some vars (e.g. json_output_instructions in propose) are
        passed to template_vars for backward compatibility but not
        referenced by the current allocation template.  These are excluded
        via a known-unused allowlist to keep the test actionable.
        """
        # Vars that are passed to template_vars but intentionally unused
        # by the current allocation-mode templates (legacy order-mode vars).
        _KNOWN_UNUSED = {
            "propose": {"json_output_instructions"},
        }

        template_path = _TEMPLATE_DIR / self._PHASE_TEMPLATES[phase]
        raw_text = template_path.read_text()
        rendered = _render_user(phase)
        allowed_unused = _KNOWN_UNUSED.get(phase, set())
        missing = []
        for var in TEMPLATE_VARS[phase]:
            if var in allowed_unused:
                continue
            in_template = var in raw_text
            in_rendered = var in rendered
            if not in_template and not in_rendered:
                missing.append(var)
        assert not missing, (
            f"TEMPLATE_VARS['{phase}'] declares vars not found in template "
            f"or rendered output for {self._PHASE_TEMPLATES[phase]}: {missing}"
        )
