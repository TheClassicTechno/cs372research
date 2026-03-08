"""Tests for the prompt profile system: MODULE_CATALOG, profiles, StrictUndefined."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from jinja2 import UndefinedError


# ---------------------------------------------------------------------------
# MODULE_CATALOG auto-discovery
# ---------------------------------------------------------------------------


class TestModuleCatalog:
    """Verify auto-discovery populates MODULE_CATALOG correctly."""

    def test_catalog_is_populated(self):
        from multi_agent.prompts import MODULE_CATALOG
        assert len(MODULE_CATALOG) > 0

    def test_auto_generated_names_present(self):
        from multi_agent.prompts import MODULE_CATALOG
        # Spot-check a few auto-generated names
        assert "role_macro" in MODULE_CATALOG
        assert "role_risk" in MODULE_CATALOG
        assert "scaffolding_causal_claim_format" in MODULE_CATALOG
        assert "tone_critique_adversarial" in MODULE_CATALOG

    def test_aliases_present(self):
        from multi_agent.prompts import MODULE_CATALOG
        assert "scaffolding_causal" in MODULE_CATALOG
        assert "scaffolding_uncertainty" in MODULE_CATALOG
        assert "scaffolding_traps" in MODULE_CATALOG
        assert "output_allocation" in MODULE_CATALOG
        assert "output_json" in MODULE_CATALOG
        assert "causal_contract" in MODULE_CATALOG

    def test_alias_points_to_same_file_as_auto_name(self):
        from multi_agent.prompts import MODULE_CATALOG
        # scaffolding_causal alias should point to same file as scaffolding_causal_claim_format
        assert MODULE_CATALOG["scaffolding_causal"] == MODULE_CATALOG["scaffolding_causal_claim_format"]

    def test_all_catalog_entries_point_to_existing_files(self):
        from multi_agent.prompts import MODULE_CATALOG, _TEMPLATE_DIR
        for name, rel_path in MODULE_CATALOG.items():
            full_path = _TEMPLATE_DIR / rel_path
            assert full_path.exists(), f"MODULE_CATALOG[{name!r}] points to missing file: {full_path}"


# ---------------------------------------------------------------------------
# load_module
# ---------------------------------------------------------------------------


class TestLoadModule:
    """Verify load_module resolves from catalog and overrides."""

    def test_load_known_module(self):
        from multi_agent.prompts import load_module
        content = load_module("scaffolding_causal")
        assert len(content) > 0

    def test_load_unknown_module_returns_empty(self):
        from multi_agent.prompts import load_module
        assert load_module("nonexistent_module_xyz") == ""

    def test_override_takes_precedence(self):
        from multi_agent.prompts import load_module
        # Point scaffolding_causal at a different file via overrides
        overrides = {"scaffolding_causal": "scaffolding/forced_uncertainty.txt"}
        content = load_module("scaffolding_causal", overrides)
        from multi_agent.prompts import FORCED_UNCERTAINTY
        assert content == FORCED_UNCERTAINTY


# ---------------------------------------------------------------------------
# Profile loading + resolution
# ---------------------------------------------------------------------------


class TestProfileLoading:
    """Verify profile YAML loading and resolution."""

    def test_load_default_profile(self):
        from multi_agent.prompts import load_prompt_profile
        profile = load_prompt_profile("default")
        assert "system_blocks" in profile
        assert "user_sections" in profile
        # Per-phase format: dicts keyed by phase
        assert isinstance(profile["system_blocks"], dict)
        assert isinstance(profile["user_sections"], dict)
        assert "propose" in profile["system_blocks"]
        assert "critique" in profile["system_blocks"]

    def test_load_minimal_profile(self):
        from multi_agent.prompts import load_prompt_profile
        profile = load_prompt_profile("minimal")
        assert "role_system" in profile["system_blocks"]["propose"]
        assert "tone" not in profile["system_blocks"]["propose"]

    def test_load_no_scaffold_profile(self):
        from multi_agent.prompts import load_prompt_profile
        profile = load_prompt_profile("no_scaffold")
        assert "tone" in profile["system_blocks"]["critique"]

    def test_load_nonexistent_profile_raises(self):
        from multi_agent.prompts import load_prompt_profile
        with pytest.raises(FileNotFoundError):
            load_prompt_profile("does_not_exist_xyz")

    def test_resolve_with_profile(self):
        from multi_agent.prompts import resolve_prompt_profile
        config = {"prompt_profile": "default"}
        result = resolve_prompt_profile(config, "macro", "propose")
        assert "system_blocks" in result
        assert "user_sections" in result
        # Returns phase-specific lists
        assert isinstance(result["system_blocks"], list)
        assert isinstance(result["user_sections"], list)

    def test_resolve_with_role_override(self):
        from multi_agent.prompts import resolve_prompt_profile
        config = {
            "prompt_profile": "default",
            "role_overrides": {
                "risk": {
                    "user_sections": {
                        "propose": ["context", "task", "output_allocation"],
                    },
                }
            }
        }
        result = resolve_prompt_profile(config, "risk", "propose")
        # role_overrides should replace user_sections for that phase
        assert result["user_sections"] == ["context", "task", "output_allocation"]
        # system_blocks should still come from default profile
        assert isinstance(result["system_blocks"], list)

    def test_resolve_role_override_does_not_affect_other_roles(self):
        from multi_agent.prompts import resolve_prompt_profile
        config = {
            "prompt_profile": "default",
            "role_overrides": {
                "risk": {
                    "user_sections": {
                        "propose": ["context", "task"],
                    },
                }
            }
        }
        result = resolve_prompt_profile(config, "macro", "propose")
        # macro should get the full default profile, not risk's override
        assert len(result["user_sections"]) > 2


# ---------------------------------------------------------------------------
# _assemble_user_prompt with MODULE_CATALOG fallback
# ---------------------------------------------------------------------------


class TestAssembleUserPromptCatalogFallback:
    """Verify _assemble_user_prompt falls back to MODULE_CATALOG."""

    def test_template_section_rendered(self):
        from multi_agent.prompts import _assemble_user_prompt
        sections = {"context": "Context: {{ context }}"}
        order = ["context"]
        template_vars = {"context": "hello world"}
        result = _assemble_user_prompt(sections, order, template_vars)
        assert "hello world" in result

    def test_catalog_fallback_for_unknown_section(self):
        from multi_agent.prompts import _assemble_user_prompt
        sections = {"context": "Context: {{ context }}"}
        # output_allocation is not in sections but is in MODULE_CATALOG
        order = ["context", "output_allocation"]
        template_vars = {"context": "test"}
        result = _assemble_user_prompt(sections, order, template_vars)
        assert "test" in result
        # output_allocation module content should be appended
        assert len(result) > len("Context: test")

    def test_unknown_section_and_module_skipped(self):
        from multi_agent.prompts import _assemble_user_prompt
        sections = {"context": "Context: {{ context }}"}
        order = ["context", "totally_nonexistent_xyz"]
        template_vars = {"context": "test"}
        result = _assemble_user_prompt(sections, order, template_vars)
        assert result == "Context: test"


# ---------------------------------------------------------------------------
# StrictUndefined
# ---------------------------------------------------------------------------


class TestStrictUndefined:
    """Verify StrictUndefined catches missing template variables."""

    def test_missing_var_raises(self):
        from multi_agent.prompts import _env
        tmpl = _env.from_string("Hello {{ missing_var }}")
        with pytest.raises(UndefinedError):
            tmpl.render()

    def test_present_var_works(self):
        from multi_agent.prompts import _env
        tmpl = _env.from_string("Hello {{ name }}")
        assert tmpl.render(name="world") == "Hello world"


# ---------------------------------------------------------------------------
# TEMPLATE_VARS registry
# ---------------------------------------------------------------------------


class TestTemplateVarsRegistry:
    """Verify TEMPLATE_VARS and CRIT_TEMPLATE_VARS are defined."""

    def test_template_vars_keys(self):
        from multi_agent.prompts import TEMPLATE_VARS
        assert set(TEMPLATE_VARS.keys()) == {"propose", "critique", "revise", "judge"}

    def test_crit_template_vars_keys(self):
        from multi_agent.prompts import CRIT_TEMPLATE_VARS
        assert "crit_system" in CRIT_TEMPLATE_VARS
        assert "crit_user" in CRIT_TEMPLATE_VARS

    def test_propose_vars_complete(self):
        from multi_agent.prompts import TEMPLATE_VARS
        propose_vars = TEMPLATE_VARS["propose"]
        assert "context" in propose_vars
        assert "causal_claim_format" in propose_vars


# ---------------------------------------------------------------------------
# build_*_user_prompt user_sections parameter
# ---------------------------------------------------------------------------


class TestBuildFunctionUserSections:
    """Verify user_sections param is accepted and used by build functions."""

    def test_proposal_accepts_user_sections(self):
        from multi_agent.prompts import build_proposal_user_prompt
        # Just verify it doesn't raise with user_sections param
        result = build_proposal_user_prompt(
            context="test context",
            user_sections=["context", "task"],
        )
        assert isinstance(result, str)

    def test_critique_accepts_user_sections(self):
        from multi_agent.prompts import build_critique_prompt
        result = build_critique_prompt(
            role="macro",
            context="test context",
            all_proposals=[{"role": "value", "proposal": "{}"}],
            my_proposal="{}",
            user_sections=["context", "task"],
        )
        assert isinstance(result, str)

    def test_revision_accepts_user_sections(self):
        from multi_agent.prompts import build_revision_prompt
        result = build_revision_prompt(
            role="macro",
            context="test context",
            my_proposal="{}",
            critiques_received=[],
            user_sections=["context", "task"],
        )
        assert isinstance(result, str)

    def test_judge_accepts_user_sections(self):
        from multi_agent.prompts import build_judge_prompt
        result = build_judge_prompt(
            context="test context",
            revisions=[
                {"role": "macro", "action": "{}", "confidence": 0.8},
            ],
            all_critiques_text="none",
            user_sections=["context", "task"],
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# CRIT template configurability
# ---------------------------------------------------------------------------


class TestCritTemplateConfig:
    """Verify CRIT render_crit_prompts accepts custom template names."""

    def test_render_crit_prompts_default_templates(self):
        from eval.crit.prompts import render_crit_prompts
        bundle = {
            "round": 1,
            "agent_role": "macro",
            "proposal": "test proposal",
            "critiques_received": "none",
            "revised_argument": "test revision",
        }
        system, user = render_crit_prompts(bundle)
        assert len(system) > 0
        assert len(user) > 0

    def test_render_crit_prompts_accepts_template_params(self):
        from eval.crit.prompts import render_crit_prompts
        bundle = {
            "round": 1,
            "agent_role": "macro",
            "proposal": "test proposal",
            "critiques_received": "none",
            "revised_argument": "test revision",
        }
        # Default templates should work when passed explicitly
        system, user = render_crit_prompts(
            bundle,
            system_template="crit_system.jinja",
            user_template="crit_user.jinja",
        )
        assert len(system) > 0
        assert len(user) > 0


# ---------------------------------------------------------------------------
# DebateConfig new fields
# ---------------------------------------------------------------------------


class TestDebateConfigNewFields:
    """Verify new config fields have correct defaults and serialize."""

    def test_default_values(self):
        from multi_agent.config import DebateConfig
        cfg = DebateConfig()
        assert cfg.prompt_profile == "default"
        assert cfg.role_overrides == {}
        assert cfg.crit_system_template == "crit_system_enumerated.jinja"
        assert cfg.crit_user_template == "crit_user_enumerated.jinja"

    def test_to_dict_includes_new_fields(self):
        from multi_agent.config import DebateConfig
        cfg = DebateConfig(prompt_profile="minimal")
        d = cfg.to_dict()
        assert d["prompt_profile"] == "minimal"
        assert "role_overrides" in d
        assert "crit_system_template" in d
        assert "crit_user_template" in d


# ---------------------------------------------------------------------------
# CritScorer template params
# ---------------------------------------------------------------------------


class TestCritScorerTemplateParams:
    """Verify CritScorer accepts template name params."""

    def test_scorer_init_with_defaults(self):
        from eval.crit.scorer import CritScorer
        scorer = CritScorer(llm_fn=lambda s, u: "{}")
        assert scorer._crit_system_template == "crit_system_enumerated.jinja"
        assert scorer._crit_user_template == "crit_user_enumerated.jinja"

    def test_scorer_init_with_custom_templates(self):
        from eval.crit.scorer import CritScorer
        scorer = CritScorer(
            llm_fn=lambda s, u: "{}",
            crit_system_template="custom_system.jinja",
            crit_user_template="custom_user.jinja",
        )
        assert scorer._crit_system_template == "custom_system.jinja"
        assert scorer._crit_user_template == "custom_user.jinja"
