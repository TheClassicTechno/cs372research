"""Comprehensive tests for configurable prompt block ordering, section ordering,
prompt file overrides, and the prompts directory reorganization.

Tests cover:
  1. System prompt block reordering via PromptRegistry.build(block_order=...)
  2. User prompt section reordering via build_*_prompt(section_order=...)
  3. Prompt file overrides (selecting alternate .txt files)
  4. Default ordering matches original output
  5. Unknown/missing sections/blocks are silently skipped
  6. _load_sectioned_template() parsing
  7. _assemble_user_prompt() rendering and ordering
  8. DebateConfig and AgentConfig carry ordering fields
  9. Prompts directory structure (files in correct subdirectories)
"""

import textwrap
from pathlib import Path

import pytest

from multi_agent.config import DebateConfig
from multi_agent.prompts import (
    _assemble_user_prompt,
    _DEFAULT_SECTION_ORDER,
    _load_sectioned_template,
    build_critique_prompt,
    build_judge_prompt,
    build_proposal_user_prompt,
    build_revision_prompt,
    CAUSAL_CLAIM_FORMAT,
)
from multi_agent.prompts.registry import (
    _DEFAULT_BLOCK_ORDER,
    _load_prompt_file,
    PromptRegistry,
    reset_registry_cache,
)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "multi_agent" / "prompts"


# =============================================================================
# Directory structure validation
# =============================================================================

class TestPromptsDirectoryStructure:
    """Verify that prompt files live in the expected subdirectories."""

    EXPECTED_SUBDIRS = [
        "roles", "phases", "scaffolding", "output_format",
        "tone", "agreeableness", "pipeline", "system_contract",
    ]

    def test_all_subdirectories_exist(self):
        for subdir in self.EXPECTED_SUBDIRS:
            path = PROMPT_DIR / subdir
            assert path.is_dir(), f"Expected subdirectory missing: {subdir}/"

    @pytest.mark.parametrize("path", [
        "roles/macro.txt", "roles/macro_slim.txt",
        "roles/value.txt", "roles/risk.txt",
        "roles/technical.txt", "roles/sentiment.txt",
        "roles/devils_advocate.txt",
    ])
    def test_role_files_in_roles_dir(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    @pytest.mark.parametrize("path", [
        "phases/proposal.txt", "phases/proposal_allocation.txt",
        "phases/critique.txt", "phases/critique_allocation.txt",
        "phases/revision.txt", "phases/revision_allocation.txt",
        "phases/judge.txt", "phases/judge_allocation.txt",
    ])
    def test_phase_templates_in_phases_dir(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    @pytest.mark.parametrize("path", [
        "scaffolding/causal_claim_format.txt",
        "scaffolding/forced_uncertainty.txt",
        "scaffolding/trap_awareness.txt",
    ])
    def test_scaffolding_files(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    @pytest.mark.parametrize("path", [
        "output_format/json_output_instructions.txt",
        "output_format/allocation_output_instructions.txt",
    ])
    def test_output_format_files(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    @pytest.mark.parametrize("path", [
        "agreeableness/confrontational.txt",
        "agreeableness/skeptical.txt",
        "agreeableness/balanced.txt",
        "agreeableness/collaborative.txt",
        "agreeableness/agreeable.txt",
    ])
    def test_agreeableness_files(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    @pytest.mark.parametrize("path", [
        "pipeline/news_digest_system.txt",
        "pipeline/data_analysis_system.txt",
    ])
    def test_pipeline_files(self, path):
        assert (PROMPT_DIR / path).exists(), f"Missing: {path}"

    def test_system_contract_file(self):
        assert (PROMPT_DIR / "system_contract/system_causal_contract.txt").exists()

    def test_no_stale_files_in_root(self):
        """Verify no .txt files remain in the prompts root directory."""
        root_txts = list(PROMPT_DIR.glob("*.txt"))
        assert root_txts == [], (
            f"Stale .txt files in prompts root (should be in subdirs): "
            f"{[f.name for f in root_txts]}"
        )


# =============================================================================
# _load_sectioned_template
# =============================================================================

class TestLoadSectionedTemplate:
    """Test parsing of ---SECTION: name--- delimiters in template files."""

    def test_proposal_allocation_has_sections(self):
        sections = _load_sectioned_template("phases/proposal_allocation.txt")
        assert "_unsectioned" not in sections
        assert "context" in sections
        assert "task" in sections
        assert "scaffolding" in sections
        assert "output_format" in sections

    def test_critique_has_all_sections(self):
        sections = _load_sectioned_template("phases/critique.txt")
        assert "preamble" in sections
        assert "context" in sections
        assert "agent_data" in sections
        assert "task" in sections
        assert "output_format" in sections

    def test_critique_allocation_has_all_sections(self):
        sections = _load_sectioned_template("phases/critique_allocation.txt")
        assert "preamble" in sections
        assert "context" in sections
        assert "agent_data" in sections
        assert "task" in sections
        assert "output_format" in sections

    def test_revision_has_sections(self):
        sections = _load_sectioned_template("phases/revision.txt")
        assert "preamble" in sections
        assert "context" in sections
        assert "agent_data" in sections
        assert "task" in sections
        assert "scaffolding" in sections
        assert "output_format" in sections

    def test_judge_has_sections(self):
        sections = _load_sectioned_template("phases/judge.txt")
        assert "preamble" in sections
        assert "context" in sections
        assert "agent_data" in sections
        assert "task" in sections
        assert "output_format" in sections

    def test_proposal_has_sections(self):
        sections = _load_sectioned_template("phases/proposal.txt")
        assert "context" in sections
        # proposal.txt has no preamble section (it's just context + scaffolding + output)
        assert "scaffolding" in sections
        assert "output_format" in sections

    def test_sections_contain_template_variables(self):
        """Section content should contain Jinja2 variables."""
        sections = _load_sectioned_template("phases/critique.txt")
        assert "{{ role }}" in sections["preamble"]
        assert "{{ context }}" in sections["context"]
        assert "{{ my_proposal }}" in sections["agent_data"]


# =============================================================================
# _assemble_user_prompt
# =============================================================================

class TestAssembleUserPrompt:
    """Test section-based prompt assembly with custom ordering."""

    def _make_sections(self):
        return {
            "preamble": "PREAMBLE: {{ role }}",
            "context": "CONTEXT: {{ data }}",
            "task": "TASK: do the thing",
            "scaffolding": "SCAFFOLDING: be careful",
            "output_format": "OUTPUT: JSON",
        }

    def test_default_order(self):
        sections = self._make_sections()
        result = _assemble_user_prompt(
            sections,
            ["preamble", "context", "task", "scaffolding", "output_format"],
            {"role": "MACRO", "data": "prices"},
        )
        lines = result.split("\n\n")
        assert lines[0] == "PREAMBLE: MACRO"
        assert lines[1] == "CONTEXT: prices"
        assert lines[2] == "TASK: do the thing"

    def test_reversed_order(self):
        sections = self._make_sections()
        result = _assemble_user_prompt(
            sections,
            ["output_format", "scaffolding", "task", "context", "preamble"],
            {"role": "RISK", "data": "vol"},
        )
        lines = result.split("\n\n")
        assert lines[0] == "OUTPUT: JSON"
        assert lines[-1] == "PREAMBLE: RISK"

    def test_subset_order(self):
        """Only include specified sections, skip others."""
        sections = self._make_sections()
        result = _assemble_user_prompt(
            sections,
            ["task", "output_format"],
            {"role": "X", "data": "Y"},
        )
        assert "TASK: do the thing" in result
        assert "OUTPUT: JSON" in result
        assert "PREAMBLE" not in result
        assert "CONTEXT" not in result

    def test_unknown_section_silently_skipped(self):
        sections = self._make_sections()
        result = _assemble_user_prompt(
            sections,
            ["nonexistent_section", "task"],
            {"role": "X", "data": "Y"},
        )
        assert "TASK: do the thing" in result

    def test_empty_rendered_section_skipped(self):
        sections = {"a": "{{ val }}", "b": "B content"}
        result = _assemble_user_prompt(sections, ["a", "b"], {"val": ""})
        # Empty rendered section should be skipped
        assert result == "B content"


# =============================================================================
# User prompt section ordering (build_*_prompt functions)
# =============================================================================

class TestUserPromptSectionOrdering:
    """Test that builder functions respect custom section_order."""

    def test_proposal_default_order_contains_context(self):
        result = build_proposal_user_prompt("MY CONTEXT DATA")
        assert "MY CONTEXT DATA" in result

    def test_proposal_allocation_default_order(self):
        result = build_proposal_user_prompt("MY CONTEXT", allocation_mode=True)
        assert "MY CONTEXT" in result
        assert "Evidence citation rules" in result

    def test_critique_custom_section_order_output_first(self):
        """Put output_format before task — output instructions appear first."""
        result = build_critique_prompt(
            role="macro",
            context="CTX",
            all_proposals=[{"role": "value", "proposal": "buy"}],
            my_proposal="my prop",
            section_order=["output_format", "preamble", "context", "agent_data", "task"],
        )
        # output_format should come before preamble
        output_pos = result.find("JSON")
        preamble_pos = result.find("MACRO")
        assert output_pos < preamble_pos, "output_format should appear before preamble"

    def test_revision_custom_section_order(self):
        result = build_revision_prompt(
            role="risk",
            context="CTX",
            my_proposal="orig",
            critiques_received=[{"from_role": "macro", "objection": "too risky"}],
            section_order=["scaffolding", "task", "preamble", "context", "agent_data", "output_format"],
        )
        # scaffolding should appear before the task section
        if CAUSAL_CLAIM_FORMAT:
            scaff_pos = result.find("Causal Claim")
            task_pos = result.find("Your Task")
            assert scaff_pos < task_pos, "scaffolding should appear before task"

    def test_judge_custom_section_order(self):
        result = build_judge_prompt(
            context="CTX",
            revisions=[{"role": "macro", "action": "buy", "confidence": 0.8}],
            all_critiques_text="critique text",
            section_order=["context", "preamble", "agent_data", "task", "output_format", "scaffolding"],
        )
        # context should appear before preamble
        ctx_pos = result.find("CTX")
        preamble_pos = result.find("JUDGE")
        assert ctx_pos < preamble_pos

    def test_missing_section_for_phase_skipped(self):
        """Proposal templates don't have 'preamble' — should be silently skipped."""
        result = build_proposal_user_prompt(
            "CTX",
            section_order=["preamble", "context", "scaffolding", "output_format"],
        )
        assert "CTX" in result


# =============================================================================
# Prompt file overrides (user prompt templates)
# =============================================================================

class TestPromptFileOverrides:
    """Test selecting alternate template files via prompt_file_overrides."""

    def test_critique_template_override_to_allocation(self):
        """Override critique template to use the allocation variant."""
        result_normal = build_critique_prompt(
            "macro", "CTX",
            [{"role": "value", "proposal": "buy"}],
            "my prop",
        )
        result_override = build_critique_prompt(
            "macro", "CTX",
            [{"role": "value", "proposal": "buy"}],
            "my prop",
            prompt_file_overrides={"critique_template": "phases/critique_allocation.txt"},
        )
        # Allocation template mentions "portfolio allocation"
        assert "portfolio allocation" in result_override
        # Normal template should not
        assert "portfolio allocation" not in result_normal

    def test_proposal_template_override(self):
        result = build_proposal_user_prompt(
            "CTX",
            allocation_mode=False,
            prompt_file_overrides={"proposal_template": "phases/proposal_allocation.txt"},
        )
        # Should use allocation template even though allocation_mode=False
        assert "Evidence citation rules" in result

    def test_revision_template_override(self):
        result = build_revision_prompt(
            "macro", "CTX", "orig",
            [{"from_role": "value", "objection": "bad"}],
            prompt_file_overrides={"revision_template": "phases/revision_allocation.txt"},
        )
        assert "portfolio allocation" in result.lower() or "allocation universe" in result

    def test_judge_template_override(self):
        result = build_judge_prompt(
            "CTX",
            [{"role": "macro", "action": "buy", "confidence": 0.8}],
            "critique text",
            prompt_file_overrides={"judge_template": "phases/judge_allocation.txt"},
        )
        assert "portfolio allocation" in result.lower() or "allocation universe" in result


# =============================================================================
# System prompt block ordering (PromptRegistry.build)
# =============================================================================

class TestSystemPromptBlockOrdering:
    """Test custom block_order in PromptRegistry.build()."""

    def setup_method(self):
        reset_registry_cache()
        self.registry = PromptRegistry()

    def test_default_block_order_constant(self):
        assert _DEFAULT_BLOCK_ORDER == ["causal_contract", "role_system", "phase_preamble", "tone"]

    def test_default_order_critique_with_causal_contract(self):
        """Default order: causal_contract → role_system → phase_preamble → tone."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
        )
        assert "causal_contract" in result.blocks_used
        assert "role_system" in result.blocks_used
        assert "phase_preamble" in result.blocks_used
        assert "tone" in result.blocks_used
        # Verify order in blocks_used
        cc_idx = result.blocks_used.index("causal_contract")
        rs_idx = result.blocks_used.index("role_system")
        pp_idx = result.blocks_used.index("phase_preamble")
        t_idx = result.blocks_used.index("tone")
        assert cc_idx < rs_idx < pp_idx < t_idx

    def test_tone_first_block_order(self):
        """Custom order: tone → role_system → phase_preamble → causal_contract."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
            block_order=["tone", "role_system", "phase_preamble", "causal_contract"],
        )
        # Tone should appear before role_system in the output
        tone_idx = result.blocks_used.index("tone")
        rs_idx = result.blocks_used.index("role_system")
        assert tone_idx < rs_idx, "tone should come before role_system"

    def test_role_system_only_block_order(self):
        """Block order with only role_system — others skipped."""
        result = self.registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
            block_order=["role_system"],
        )
        assert result.blocks_used == ["role_system"]
        assert "tone" not in result.blocks_used

    def test_unknown_block_name_skipped(self):
        """Unknown block names in order are silently skipped."""
        result = self.registry.build(
            role="macro", phase="propose",
            block_order=["nonexistent_block", "role_system"],
        )
        assert "role_system" in result.blocks_used
        assert "nonexistent_block" not in result.blocks_used

    def test_propose_phase_no_tone_regardless_of_order(self):
        """Propose phase has no tone block, even if requested in order."""
        result = self.registry.build(
            role="macro", phase="propose", beta=0.9,
            block_order=["tone", "role_system"],
        )
        assert "tone" not in result.blocks_used
        assert "role_system" in result.blocks_used

    def test_judge_phase_uses_judge_system_block(self):
        """Judge phase maps role_system to judge_system content."""
        result = self.registry.build(
            role="judge", phase="judge",
            block_order=["role_system"],
        )
        assert "Judge" in result.system_prompt

    def test_block_order_none_uses_default(self):
        """block_order=None should use the default order."""
        result_default = self.registry.build(
            role="macro", phase="critique", beta=0.5,
            use_system_causal_contract=True,
            block_order=None,
        )
        result_explicit = self.registry.build(
            role="macro", phase="critique", beta=0.5,
            use_system_causal_contract=True,
            block_order=_DEFAULT_BLOCK_ORDER,
        )
        assert result_default.blocks_used == result_explicit.blocks_used


# =============================================================================
# System prompt file overrides (PromptRegistry.build)
# =============================================================================

class TestSystemPromptFileOverrides:
    """Test prompt_file_overrides in PromptRegistry.build()."""

    def setup_method(self):
        reset_registry_cache()
        self.registry = PromptRegistry()

    def test_role_file_override(self):
        """Override role_macro to use slim variant."""
        result_normal = self.registry.build(role="macro", phase="propose")
        result_override = self.registry.build(
            role="macro", phase="propose",
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        # Both should have role_system block but with different content
        assert "role_system" in result_normal.blocks_used
        assert "role_system" in result_override.blocks_used
        # Slim version should be shorter
        assert len(result_override.system_prompt) <= len(result_normal.system_prompt)

    def test_causal_contract_file_override(self):
        """Override causal_contract to a different file."""
        # Use a role file as the override just to test the mechanism
        result = self.registry.build(
            role="macro", phase="critique", beta=0.5,
            use_system_causal_contract=True,
            prompt_file_overrides={"causal_contract": "scaffolding/causal_claim_format.txt"},
        )
        assert "causal_contract" in result.blocks_used

    def test_nonexistent_override_file_returns_empty(self):
        """Override pointing to nonexistent file → block is empty, skipped."""
        result = self.registry.build(
            role="macro", phase="propose",
            prompt_file_overrides={"role_macro": "nonexistent_file.txt"},
        )
        # role_system should not be in blocks_used since content is empty
        assert "role_system" not in result.blocks_used


# =============================================================================
# _load_prompt_file
# =============================================================================

class TestLoadPromptFile:
    """Test the registry's file loader."""

    def test_load_from_subdirectory(self):
        content = _load_prompt_file("roles/macro.txt")
        assert len(content) > 0
        assert "MACRO" in content.upper() or "macro" in content.lower()

    def test_load_from_tone_subdirectory(self):
        content = _load_prompt_file("tone/critique_adversarial.txt")
        assert len(content) > 0

    def test_load_nonexistent_returns_empty(self):
        content = _load_prompt_file("this_does_not_exist.txt")
        assert content == ""


# =============================================================================
# DebateConfig ordering fields
# =============================================================================

class TestDebateConfigOrderingFields:
    """Test that DebateConfig carries ordering and override fields."""

    def test_default_system_prompt_block_order(self):
        cfg = DebateConfig()
        assert cfg.system_prompt_block_order == [
            "causal_contract", "role_system", "phase_preamble", "tone"
        ]

    def test_default_user_prompt_section_order(self):
        cfg = DebateConfig()
        assert cfg.user_prompt_section_order == [
            "preamble", "context", "agent_data", "task", "scaffolding", "output_format"
        ]

    def test_default_prompt_file_overrides_empty(self):
        cfg = DebateConfig()
        assert cfg.prompt_file_overrides == {}

    def test_custom_ordering_in_constructor(self):
        cfg = DebateConfig(
            system_prompt_block_order=["tone", "role_system"],
            user_prompt_section_order=["output_format", "task"],
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        assert cfg.system_prompt_block_order == ["tone", "role_system"]
        assert cfg.user_prompt_section_order == ["output_format", "task"]
        assert cfg.prompt_file_overrides == {"role_macro": "roles/macro_slim.txt"}

    def test_to_dict_includes_ordering_fields(self):
        cfg = DebateConfig(
            system_prompt_block_order=["tone", "role_system"],
            user_prompt_section_order=["task", "context"],
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        d = cfg.to_dict()
        assert d["system_prompt_block_order"] == ["tone", "role_system"]
        assert d["user_prompt_section_order"] == ["task", "context"]
        assert d["prompt_file_overrides"] == {"role_macro": "roles/macro_slim.txt"}


# =============================================================================
# AgentConfig ordering fields
# =============================================================================

class TestAgentConfigOrderingFields:
    """Test that AgentConfig carries optional ordering fields."""

    def test_defaults_are_none(self):
        from models.config import AgentConfig
        cfg = AgentConfig(
            agent_system="multi_agent_debate",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        assert cfg.system_prompt_block_order is None
        assert cfg.user_prompt_section_order is None
        assert cfg.prompt_file_overrides is None

    def test_custom_values(self):
        from models.config import AgentConfig
        cfg = AgentConfig(
            agent_system="multi_agent_debate",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            system_prompt_block_order=["tone", "role_system"],
            user_prompt_section_order=["task", "context"],
            prompt_file_overrides={"role_macro": "roles/macro_slim.txt"},
        )
        assert cfg.system_prompt_block_order == ["tone", "role_system"]
        assert cfg.user_prompt_section_order == ["task", "context"]
        assert cfg.prompt_file_overrides == {"role_macro": "roles/macro_slim.txt"}


# =============================================================================
# Default section order matches _DEFAULT_SECTION_ORDER constant
# =============================================================================

class TestDefaultOrderConstants:
    """Ensure default order constants are consistent."""

    def test_default_section_order_value(self):
        assert _DEFAULT_SECTION_ORDER == [
            "preamble", "context", "agent_data", "task", "scaffolding", "output_format"
        ]

    def test_default_block_order_value(self):
        assert _DEFAULT_BLOCK_ORDER == [
            "causal_contract", "role_system", "phase_preamble", "tone"
        ]


# =============================================================================
# End-to-end: ordering + overrides together
# =============================================================================

class TestOrderingAndOverridesCombined:
    """Test using both custom ordering and file overrides simultaneously."""

    def setup_method(self):
        reset_registry_cache()

    def test_system_and_user_custom_ordering_together(self):
        """Custom block order + custom section order in the same build."""
        registry = PromptRegistry()
        result = registry.build(
            role="macro", phase="critique", beta=0.9,
            use_system_causal_contract=True,
            block_order=["tone", "phase_preamble", "role_system", "causal_contract"],
        )
        # Verify tone comes first
        assert result.blocks_used[0] == "tone"
        assert result.blocks_used[-1] == "causal_contract"

    def test_user_prompt_override_plus_reorder(self):
        """Use allocation template with custom section ordering."""
        result = build_critique_prompt(
            "macro", "CTX",
            [{"role": "value", "proposal": "buy"}],
            "my prop",
            prompt_file_overrides={"critique_template": "phases/critique_allocation.txt"},
            section_order=["output_format", "task", "agent_data", "context", "preamble"],
        )
        # Output format should come before preamble
        output_pos = result.find("JSON")
        preamble_pos = result.find("MACRO")
        assert output_pos < preamble_pos

    def test_full_pipeline_with_debate_config(self):
        """DebateConfig → to_dict() → use in builder functions."""
        cfg = DebateConfig(
            system_prompt_block_order=["role_system", "tone", "phase_preamble"],
            user_prompt_section_order=["task", "context", "preamble", "output_format"],
            prompt_file_overrides={"proposal_template": "phases/proposal_allocation.txt"},
        )
        d = cfg.to_dict()

        result = build_proposal_user_prompt(
            "MY CONTEXT",
            section_order=d.get("user_prompt_section_order"),
            prompt_file_overrides=d.get("prompt_file_overrides"),
        )
        # Task section (evidence citation) should come before context
        task_pos = result.find("Evidence citation")
        ctx_pos = result.find("MY CONTEXT")
        assert task_pos < ctx_pos, "task should appear before context with custom order"
