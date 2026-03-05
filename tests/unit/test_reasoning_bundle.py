"""Unit tests for build_reasoning_bundle — CRIT input pipeline integrity.

These tests ensure that:
1. Revision raw_response is passed through to CRIT bundles (Fix 1)
2. Evidence citations are extracted from text when no structured field exists (Fix 2)
3. The bundle structure matches what CRIT expects
"""

import pytest

from multi_agent.runner import build_reasoning_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    proposal_raw="proposal raw text",
    revision_raw="revision raw text",
    justification="test justification",
    revision_justification="revised justification",
    proposal_citations=None,
    revision_citations=None,
    critiques=None,
):
    """Build a minimal debate state dict for one agent."""
    prop_action = {
        "allocation": {"AAPL": 0.5, "NVDA": 0.5},
        "justification": justification,
        "confidence": 0.8,
        "claims": [],
    }
    if proposal_citations is not None:
        prop_action["evidence_citations"] = proposal_citations

    rev_action = {
        "allocation": {"AAPL": 0.4, "NVDA": 0.6},
        "justification": revision_justification,
        "confidence": 0.85,
        "claims": [],
    }
    if revision_citations is not None:
        rev_action["evidence_citations"] = revision_citations

    state = {
        "proposals": [
            {"role": "macro", "action_dict": prop_action, "raw_response": proposal_raw},
        ],
        "revisions": [
            {"role": "macro", "action_dict": rev_action, "revision_notes": "notes", "raw_response": revision_raw},
        ],
        "critiques": critiques or [],
    }
    return state


# ---------------------------------------------------------------------------
# Fix 1: Revision raw_response must flow into CRIT bundle
# ---------------------------------------------------------------------------

class TestRevisionRawResponse:
    """Ensure revision raw_response is never lost in the CRIT bundle."""

    def test_revision_reasoning_contains_raw_response(self):
        """revised_argument.reasoning must contain the full LLM raw response."""
        state = _make_state(revision_raw="This is the full revision reasoning with analysis.")
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle is not None
        assert bundle["revised_argument"]["reasoning"] == "This is the full revision reasoning with analysis."

    def test_revision_reasoning_not_empty(self):
        """revised_argument.reasoning must NOT be empty when raw_response exists."""
        state = _make_state(revision_raw='{"allocation": {"AAPL": 0.5}, "justification": "test"}')
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["revised_argument"]["reasoning"] != ""

    def test_proposal_reasoning_contains_raw_response(self):
        """proposal.reasoning must contain the full LLM raw response."""
        state = _make_state(proposal_raw="Full proposal reasoning here.")
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["reasoning"] == "Full proposal reasoning here."

    def test_both_reasoning_fields_populated(self):
        """Both proposal and revised_argument must have non-empty reasoning."""
        state = _make_state(
            proposal_raw="Proposal analysis of AAPL [AAPL-RET60].",
            revision_raw="Revised analysis after critique [NVDA-F3].",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["reasoning"] != ""
        assert bundle["revised_argument"]["reasoning"] != ""
        assert "Proposal analysis" in bundle["proposal"]["reasoning"]
        assert "Revised analysis" in bundle["revised_argument"]["reasoning"]

    def test_revision_falls_back_to_proposal_when_no_revision(self):
        """When no revision exists, bundle uses proposal as fallback."""
        state = _make_state(proposal_raw="Only proposal exists.")
        state["revisions"] = []  # No revisions
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["revised_argument"]["reasoning"] == "Only proposal exists."


# ---------------------------------------------------------------------------
# Fix 2: Evidence citations extracted from text
# ---------------------------------------------------------------------------

class TestEvidenceCitationExtraction:
    """Ensure evidence citations are extracted from raw text when no structured field exists."""

    def test_citations_extracted_from_proposal_text(self):
        """Citations like [AAPL-RET60] in raw_response should appear in evidence_citations."""
        state = _make_state(
            proposal_raw="AAPL shows strong momentum [AAPL-RET60] and low vol [AAPL-VOL20].",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        citation_ids = {c["evidence_id"] for c in bundle["proposal"]["evidence_citations"]}
        assert "AAPL-RET60" in citation_ids
        assert "AAPL-VOL20" in citation_ids

    def test_citations_extracted_from_revision_text(self):
        """Citations in revision raw_response should appear in revised_argument.evidence_citations."""
        state = _make_state(
            revision_raw="Revised: NVDA is strong [NVDA-F3] with macro support [L1-VIX].",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        citation_ids = {c["evidence_id"] for c in bundle["revised_argument"]["evidence_citations"]}
        assert "NVDA-F3" in citation_ids
        assert "L1-VIX" in citation_ids

    def test_citations_not_empty_when_text_has_brackets(self):
        """evidence_citations must NOT be [] when raw text contains bracket citations."""
        state = _make_state(
            proposal_raw="Energy is overweight [XOM-RET60] due to [L1-FF] environment.",
            revision_raw="Maintained energy [XOM-RET60] and added [CVX-F1] support.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert len(bundle["proposal"]["evidence_citations"]) > 0
        assert len(bundle["revised_argument"]["evidence_citations"]) > 0

    def test_no_citations_when_text_has_no_brackets(self):
        """evidence_citations should be [] when raw text has no bracket IDs."""
        state = _make_state(
            proposal_raw="AAPL looks good based on general analysis.",
            revision_raw="Maintained allocation after reviewing critiques.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["evidence_citations"] == []
        assert bundle["revised_argument"]["evidence_citations"] == []

    def test_structured_citations_take_precedence(self):
        """If action_dict has evidence_citations, those should be used (not text extraction)."""
        structured = [{"evidence_id": "AAPL-BETA"}]
        state = _make_state(
            proposal_raw="Text also has [AAPL-RET60] but structured should win.",
            proposal_citations=structured,
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        citation_ids = {c["evidence_id"] for c in bundle["proposal"]["evidence_citations"]}
        # Structured citation should be present
        assert "AAPL-BETA" in citation_ids
        # Text-extracted citation should NOT be present (structured takes precedence)
        assert "AAPL-RET60" not in citation_ids

    def test_l1_macro_citations_extracted(self):
        """L1-* macro citations should be extracted."""
        state = _make_state(
            proposal_raw="Rising rates [L1-10Y] with high vol [L1-VIX] and fed funds [L1-FF].",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        citation_ids = {c["evidence_id"] for c in bundle["proposal"]["evidence_citations"]}
        assert "L1-10Y" in citation_ids
        assert "L1-VIX" in citation_ids
        assert "L1-FF" in citation_ids

    def test_evidence_enrichment_with_memo_lookup(self):
        """Citations should be enriched with memo text when lookup is provided."""
        state = _make_state(
            proposal_raw="AAPL momentum [AAPL-RET60] is strong.",
        )
        memo_lookup = {"AAPL-RET60": "60-day return: +15.3%"}
        bundle = build_reasoning_bundle(state, "macro", 1, memo_lookup)

        cite = bundle["proposal"]["evidence_citations"][0]
        assert cite["evidence_id"] == "AAPL-RET60"
        assert cite["evidence_text"] == "60-day return: +15.3%"

    def test_missing_evidence_marked(self):
        """Citations not in memo lookup should get MISSING_EVIDENCE marker."""
        state = _make_state(
            proposal_raw="Citing [FAKE-ID1] which doesn't exist in memo.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        cite = bundle["proposal"]["evidence_citations"][0]
        assert cite["evidence_id"] == "FAKE-ID1"
        assert cite["evidence_text"] == "MISSING_EVIDENCE"


# ---------------------------------------------------------------------------
# Bundle structure integrity
# ---------------------------------------------------------------------------

class TestBundleStructure:
    """Ensure the CRIT bundle has all required fields."""

    def test_bundle_has_all_required_keys(self):
        """Bundle must have round, agent_role, proposal, critiques_received, revised_argument."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 3, {})

        assert bundle["round"] == 3
        assert bundle["agent_role"] == "macro"
        assert "proposal" in bundle
        assert "critiques_received" in bundle
        assert "revised_argument" in bundle

    def test_proposal_has_all_fields(self):
        """Proposal sub-dict must have thesis, portfolio_allocation, reasoning, evidence_citations."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        prop = bundle["proposal"]
        assert "thesis" in prop
        assert "portfolio_allocation" in prop
        assert "reasoning" in prop
        assert "evidence_citations" in prop

    def test_revised_argument_has_all_fields(self):
        """Revised argument sub-dict must have thesis, portfolio_allocation, reasoning, evidence_citations."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        rev = bundle["revised_argument"]
        assert "thesis" in rev
        assert "portfolio_allocation" in rev
        assert "reasoning" in rev
        assert "evidence_citations" in rev

    def test_returns_none_for_unknown_role(self):
        """Bundle returns None when the requested role has no proposal."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "nonexistent_role", 1, {})

        assert bundle is None

    def test_critiques_filtered_to_target_role(self):
        """Only critiques targeting this agent should appear in the bundle."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {"target_role": "macro", "objection": "Overweight energy"},
                        {"target_role": "risk", "objection": "Not for macro"},
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert len(bundle["critiques_received"]) == 1
        assert bundle["critiques_received"][0]["from_role"] == "value"
        assert bundle["critiques_received"][0]["critique_text"] == "Overweight energy"
