"""Tests that ALL revision prompt templates include the critique_responses schema.

Every revision template must contain:
1. The critique_responses JSON array in the output_format section
2. The mandatory instruction about including one entry per critique
3. The required fields: from_agent, target_claim, disposition, justification
4. revision_notes still present (after critique_responses)
"""

import pytest

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "multi_agent" / "prompts" / "phases"

# Every revision prompt template in the codebase
REVISION_TEMPLATES = sorted(PROMPTS_DIR.glob("revision_allocation*.txt"))


class TestAllRevisionTemplatesHaveCritiqueResponses:
    """Ensure critique_responses schema is present in every revision template."""

    @pytest.fixture(params=[t.name for t in REVISION_TEMPLATES], ids=[t.stem for t in REVISION_TEMPLATES])
    def template_content(self, request):
        path = PROMPTS_DIR / request.param
        return path.name, path.read_text()

    def test_critique_responses_array_present(self, template_content):
        name, content = template_content
        assert '"critique_responses"' in content, (
            f"{name} is missing the critique_responses array in its output schema"
        )

    def test_mandatory_instruction_present(self, template_content):
        name, content = template_content
        assert "Omitting a critique is not allowed" in content, (
            f"{name} is missing the mandatory critique_responses instruction"
        )

    def test_from_agent_field_present(self, template_content):
        name, content = template_content
        assert '"from_agent"' in content, (
            f"{name} is missing the from_agent field in critique_responses schema"
        )

    def test_target_claim_field_present(self, template_content):
        name, content = template_content
        assert '"target_claim"' in content, (
            f"{name} is missing the target_claim field in critique_responses schema"
        )

    def test_disposition_field_present(self, template_content):
        name, content = template_content
        assert '"disposition"' in content, (
            f"{name} is missing the disposition field in critique_responses schema"
        )

    def test_justification_field_present(self, template_content):
        name, content = template_content
        assert '"justification"' in content, (
            f"{name} is missing the justification field in critique_responses schema"
        )

    def test_revision_notes_still_present(self, template_content):
        name, content = template_content
        assert '"revision_notes"' in content, (
            f"{name} is missing the revision_notes field"
        )

    def test_critique_responses_before_revision_notes_in_json_schema(self, template_content):
        """critique_responses must appear before the final revision_notes in the JSON schema."""
        name, content = template_content
        cr_pos = content.index('"critique_responses"')
        # Use rfind to get the last occurrence (the one in the JSON schema,
        # not the earlier guideline reference like 'explain why in revision_notes')
        rn_pos = content.rfind('"revision_notes"')
        assert cr_pos < rn_pos, (
            f"{name}: critique_responses must appear before revision_notes in the output schema"
        )


class TestRevisionTemplateCompleteness:
    """Verify we haven't missed any revision templates."""

    def test_all_expected_templates_exist(self):
        expected = [
            "revision_allocation.txt",
            "revision_allocation_with_enumeration.txt",
            "revision_allocation_with_enumeration_macro_enriched.txt",
            "revision_allocation_with_enumeration_risk_enriched.txt",
            "revision_allocation_with_enumeration_value_enriched.txt",
            "revision_allocation_with_enumeration_technical_enriched.txt",
            "revision_allocation_with_enumeration_macro_enriched_v2.txt",
            "revision_allocation_with_enumeration_risk_enriched_v2.txt",
            "revision_allocation_with_enumeration_value_enriched_v2.txt",
            "revision_allocation_with_enumeration_technical_enriched_v2.txt",
        ]
        actual = {t.name for t in REVISION_TEMPLATES}
        for name in expected:
            assert name in actual, f"Expected revision template {name} not found"

    def test_no_revision_template_without_critique_responses(self):
        """Every revision_allocation*.txt must have critique_responses."""
        missing = []
        for t in REVISION_TEMPLATES:
            content = t.read_text()
            if '"critique_responses"' not in content:
                missing.append(t.name)
        assert not missing, (
            f"Revision templates missing critique_responses: {missing}"
        )


class TestV2AndNonV2Consistency:
    """The v2 and non-v2 enriched templates must have the same critique_responses schema."""

    ENRICHED_PAIRS = [
        ("revision_allocation_with_enumeration_macro_enriched.txt",
         "revision_allocation_with_enumeration_macro_enriched_v2.txt"),
        ("revision_allocation_with_enumeration_risk_enriched.txt",
         "revision_allocation_with_enumeration_risk_enriched_v2.txt"),
        ("revision_allocation_with_enumeration_value_enriched.txt",
         "revision_allocation_with_enumeration_value_enriched_v2.txt"),
        ("revision_allocation_with_enumeration_technical_enriched.txt",
         "revision_allocation_with_enumeration_technical_enriched_v2.txt"),
    ]

    @pytest.fixture(params=ENRICHED_PAIRS, ids=[p[0].split("_enriched")[0].split("enumeration_")[1] for p in ENRICHED_PAIRS])
    def pair_contents(self, request):
        base_name, v2_name = request.param
        base = (PROMPTS_DIR / base_name).read_text()
        v2 = (PROMPTS_DIR / v2_name).read_text()
        return base_name, base, v2

    def test_both_have_critique_responses(self, pair_contents):
        base_name, base, v2 = pair_contents
        assert '"critique_responses"' in base
        assert '"critique_responses"' in v2

    def test_both_have_mandatory_instruction(self, pair_contents):
        base_name, base, v2 = pair_contents
        assert "Omitting a critique is not allowed" in base
        assert "Omitting a critique is not allowed" in v2

    def test_both_have_same_disposition_values(self, pair_contents):
        """Both should offer accept | rebut as disposition options."""
        base_name, base, v2 = pair_contents
        assert "accept | rebut" in base
        assert "accept | rebut" in v2
