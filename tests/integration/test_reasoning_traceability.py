"""Integration tests for reasoning traceability through the CRIT pipeline.

Verifies that:
1. Claims → evidence_ids linkage is preserved through normalization
2. Positions → supporting_claims linkage is preserved
3. Structured reasoning appears in rendered CRIT prompts
"""

import json

import pytest

from multi_agent.runner import (
    build_reasoning_bundle,
    _normalize_claims,
    _normalize_position_rationale,
    _extract_reasoning,
)
from eval.evidence import normalize_evidence_id
from eval.crit.prompts import render_crit_prompts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLAIMS = [
    {
        "claim_id": "C1",
        "claim_text": "Rising rates pressure growth equities [L1-10Y]",
        "claim_type": "macro",
        "reasoning_type": "causal",
        "evidence": ["[L1-10Y]", "[L1-FF: Fed Funds: 5.08%]"],
        "assumptions": ["Fed holds rates steady"],
        "falsifiers": ["Rate cuts announced"],
        "impacts_positions": ["AAPL", "NVDA"],
        "confidence": 0.8,
    },
    {
        "claim_id": "C2",
        "claim_text": "NVDA AI capex cycle supports revenue [NVDA-F3]",
        "claim_type": "firm",
        "reasoning_type": "observational",
        "evidence": ["[NVDA-F3]", "[NVDA-F3]"],  # Duplicate to test dedup
        "assumptions": ["AI demand persists"],
        "falsifiers": ["Capex cuts"],
        "impacts_positions": ["NVDA"],
        "confidence": 0.7,
    },
]

POSITION_RATIONALE = [
    {
        "ticker": "AAPL",
        "weight": 0.4,
        "supported_by_claims": ["C1"],
        "explanation": "Rate sensitivity drives underweight.",
    },
    {
        "ticker": "NVDA",
        "weight": 0.6,
        "supported_by_claims": ["C1", "C2"],
        "explanation": "AI capex cycle offsets rate pressure.",
    },
]


def _make_state():
    """Build a state with rich structured claims and position rationale."""
    prop_action = {
        "allocation": {"AAPL": 0.5, "NVDA": 0.5},
        "thesis": "Macro rates pressure growth but AI cycle supports NVDA.",
        "portfolio_rationale": "Macro rates pressure growth but AI cycle supports NVDA.",
        "justification": "Macro rates pressure growth but AI cycle supports NVDA.",
        "position_rationale": POSITION_RATIONALE,
        "confidence": 0.75,
        "claims": CLAIMS,
        "risks_or_falsifiers": ["Broad market selloff invalidates both positions."],
    }
    rev_action = {
        "allocation": {"AAPL": 0.4, "NVDA": 0.6},
        "thesis": "Revised: reduced AAPL on rate sensitivity.",
        "portfolio_rationale": "Revised: reduced AAPL on rate sensitivity.",
        "justification": "Revised: reduced AAPL on rate sensitivity.",
        "position_rationale": POSITION_RATIONALE,
        "confidence": 0.8,
        "claims": CLAIMS,
        "risks_or_falsifiers": ["Fed pivot would invalidate rate thesis."],
    }
    return {
        "proposals": [
            {"role": "macro", "action_dict": prop_action, "raw_response": "raw proposal text"},
        ],
        "revisions": [
            {
                "role": "macro",
                "action_dict": rev_action,
                "revision_notes": "Accepted critique K1: reduced AAPL weight.",
                "raw_response": "raw revision text",
            },
        ],
        "critiques": [],
    }


# ---------------------------------------------------------------------------
# Test: Claim → evidence traceability
# ---------------------------------------------------------------------------

class TestClaimEvidenceTraceability:
    def test_evidence_ids_normalized(self):
        """evidence_ids should be clean IDs, stripped of brackets and colon text."""
        normalized = _normalize_claims(CLAIMS, normalize_evidence_id)
        c1 = normalized[0]
        assert "L1-10Y" in c1["evidence_ids"]
        assert "L1-FF" in c1["evidence_ids"]
        # Colon-delimited text should be stripped
        assert not any(":" in eid for eid in c1["evidence_ids"])

    def test_evidence_ids_deduplicated(self):
        """Duplicate evidence entries should produce unique evidence_ids."""
        normalized = _normalize_claims(CLAIMS, normalize_evidence_id)
        c2 = normalized[1]
        assert c2["evidence_ids"].count("NVDA-F3") == 1

    def test_empty_claims_dropped(self):
        """Claims with empty claim_text should be dropped."""
        claims_with_empty = CLAIMS + [{"claim_id": "C3", "claim_text": ""}]
        normalized = _normalize_claims(claims_with_empty, normalize_evidence_id)
        assert len(normalized) == 2

    def test_canonical_fields_present(self):
        """All canonical fields should be present on normalized claims."""
        normalized = _normalize_claims(CLAIMS, normalize_evidence_id)
        for claim in normalized:
            assert "claim_id" in claim
            assert "claim_text" in claim
            assert "claim_type" in claim
            assert "reasoning_type" in claim
            assert "evidence" in claim
            assert "evidence_ids" in claim
            assert "assumptions" in claim
            assert "falsifiers" in claim
            assert "impacts_positions" in claim
            assert "confidence" in claim


# ---------------------------------------------------------------------------
# Test: Position → supporting_claims traceability
# ---------------------------------------------------------------------------

class TestPositionClaimTraceability:
    def test_supporting_claims_mapped(self):
        """supported_by_claims should be mapped to supporting_claims."""
        normalized = _normalize_position_rationale(POSITION_RATIONALE)
        aapl = normalized[0]
        assert aapl["supporting_claims"] == ["C1"]
        nvda = normalized[1]
        assert nvda["supporting_claims"] == ["C1", "C2"]

    def test_canonical_fields_present(self):
        """All canonical fields should be present."""
        normalized = _normalize_position_rationale(POSITION_RATIONALE)
        for pos in normalized:
            assert "ticker" in pos
            assert "weight" in pos
            assert "supporting_claims" in pos
            assert "explanation" in pos


# ---------------------------------------------------------------------------
# Test: Full bundle → CRIT prompt rendering
# ---------------------------------------------------------------------------

class TestBundleToCritPrompt:
    def test_structured_reasoning_in_rendered_prompt(self):
        """The rendered CRIT prompt should contain structured reasoning JSON."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert bundle is not None

        _, user_prompt = render_crit_prompts(
            bundle,
            system_template="crit_system_enumerated.jinja",
            user_template="crit_user_master.jinja",
        )

        # Structured reasoning should appear in the rendered prompt
        assert "claims" in user_prompt
        assert "evidence_ids" in user_prompt
        assert "supporting_claims" in user_prompt
        assert "impacts_positions" in user_prompt
        assert "risks_or_falsifiers" in user_prompt

    def test_revision_notes_in_rendered_prompt(self):
        """revision_notes should appear in the revised argument section."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        _, user_prompt = render_crit_prompts(
            bundle,
            system_template="crit_system_enumerated.jinja",
            user_template="crit_user_master.jinja",
        )

        assert "Accepted critique K1" in user_prompt

    def test_raw_response_as_sibling(self):
        """raw_response should be a sibling field, not inside reasoning."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        prop = bundle["proposal"]
        rev = bundle["revised_argument"]

        assert isinstance(prop["reasoning"], dict)
        assert isinstance(prop["raw_response"], str)
        assert isinstance(rev["reasoning"], dict)
        assert isinstance(rev["raw_response"], str)

    def test_extract_reasoning_includes_all_fields(self):
        """_extract_reasoning should produce all canonical fields."""
        action_dict = {
            "claims": CLAIMS,
            "position_rationale": POSITION_RATIONALE,
            "thesis": "Test thesis",
            "confidence": 0.8,
            "risks_or_falsifiers": ["Risk 1"],
        }
        reasoning = _extract_reasoning(
            action_dict, normalize_evidence_id,
            revision_notes="Accepted K1",
        )

        assert "claims" in reasoning
        assert "position_rationale" in reasoning
        assert "thesis" in reasoning
        assert "confidence" in reasoning
        assert "risks_or_falsifiers" in reasoning
        assert "revision_notes" in reasoning
        assert reasoning["revision_notes"] == "Accepted K1"
        assert len(reasoning["claims"]) == 2
        assert len(reasoning["position_rationale"]) == 2
