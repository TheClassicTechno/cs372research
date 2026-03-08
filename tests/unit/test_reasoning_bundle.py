"""Unit tests for build_reasoning_bundle — CRIT input pipeline integrity.

These tests ensure that:
1. Revision raw_response is passed through to CRIT bundles (Fix 1)
2. Evidence citations are extracted from text when no structured field exists (Fix 2)
3. The bundle structure matches what CRIT expects
4. Enriched agent output format (portfolio_rationale, claims[].evidence) is handled
"""

import pytest

from multi_agent.runner import build_reasoning_bundle, _extract_thesis, _extract_citations


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
    proposal_claims=None,
    revision_claims=None,
    critiques=None,
):
    """Build a minimal debate state dict for one agent (base format)."""
    prop_action = {
        "allocation": {"AAPL": 0.5, "NVDA": 0.5},
        "justification": justification,
        "confidence": 0.8,
        "claims": proposal_claims or [],
    }
    if proposal_citations is not None:
        prop_action["evidence_citations"] = proposal_citations

    rev_action = {
        "allocation": {"AAPL": 0.4, "NVDA": 0.6},
        "justification": revision_justification,
        "confidence": 0.85,
        "claims": revision_claims or [],
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


def _make_enriched_state(
    proposal_raw="proposal raw text",
    revision_raw="revision raw text",
    portfolio_rationale="enriched portfolio rationale",
    revision_portfolio_rationale="revised enriched rationale",
    proposal_claims=None,
    revision_claims=None,
    critiques=None,
):
    """Build a minimal debate state dict using enriched agent output format.

    Enriched agents output ``portfolio_rationale`` (not ``justification``)
    and embed evidence inside ``claims[].evidence`` (not top-level
    ``evidence_citations``).
    """
    prop_action = {
        "allocation": {"AAPL": 0.5, "NVDA": 0.5},
        "portfolio_rationale": portfolio_rationale,
        "confidence": 0.8,
        "claims": proposal_claims or [],
    }

    rev_action = {
        "allocation": {"AAPL": 0.4, "NVDA": 0.6},
        "portfolio_rationale": revision_portfolio_rationale,
        "confidence": 0.85,
        "claims": revision_claims or [],
    }

    state = {
        "proposals": [
            {"role": "macro", "action_dict": prop_action, "raw_response": proposal_raw},
        ],
        "revisions": [
            {"role": "macro", "action_dict": rev_action, "raw_response": revision_raw},
        ],
        "critiques": critiques or [],
    }
    return state


# ---------------------------------------------------------------------------
# Fix 1: Revision raw_response must flow into CRIT bundle
# ---------------------------------------------------------------------------

class TestRevisionRawResponse:
    """Ensure revision raw_response is never lost in the CRIT bundle."""

    def test_revision_raw_response_contains_raw_text(self):
        """revised_argument.raw_response must contain the full LLM raw response."""
        state = _make_state(revision_raw="This is the full revision reasoning with analysis.")
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle is not None
        assert bundle["revised_argument"]["raw_response"] == "This is the full revision reasoning with analysis."

    def test_revision_raw_response_not_empty(self):
        """revised_argument.raw_response must NOT be empty when raw_response exists."""
        state = _make_state(revision_raw='{"allocation": {"AAPL": 0.5}, "justification": "test"}')
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["revised_argument"]["raw_response"] != ""

    def test_proposal_raw_response_contains_raw_text(self):
        """proposal.raw_response must contain the full LLM raw response."""
        state = _make_state(proposal_raw="Full proposal reasoning here.")
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["raw_response"] == "Full proposal reasoning here."

    def test_both_raw_response_fields_populated(self):
        """Both proposal and revised_argument must have non-empty raw_response."""
        state = _make_state(
            proposal_raw="Proposal analysis of AAPL [AAPL-RET60].",
            revision_raw="Revised analysis after critique [NVDA-F3].",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["raw_response"] != ""
        assert bundle["revised_argument"]["raw_response"] != ""
        assert "Proposal analysis" in bundle["proposal"]["raw_response"]
        assert "Revised analysis" in bundle["revised_argument"]["raw_response"]

    def test_revision_falls_back_to_proposal_when_no_revision(self):
        """When no revision exists, bundle uses proposal as fallback."""
        state = _make_state(proposal_raw="Only proposal exists.")
        state["revisions"] = []  # No revisions
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["revised_argument"]["raw_response"] == "Only proposal exists."

    def test_reasoning_is_structured_dict(self):
        """reasoning must be a structured dict (not a raw string)."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert isinstance(bundle["proposal"]["reasoning"], dict)
        assert isinstance(bundle["revised_argument"]["reasoning"], dict)
        assert "claims" in bundle["proposal"]["reasoning"]
        assert "thesis" in bundle["proposal"]["reasoning"]


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
        """Proposal sub-dict must have portfolio_allocation, reasoning, raw_response, evidence_citations."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        prop = bundle["proposal"]
        assert "portfolio_allocation" in prop
        assert "reasoning" in prop
        assert "raw_response" in prop
        assert "evidence_citations" in prop
        assert "thesis" in prop["reasoning"]

    def test_revised_argument_has_all_fields(self):
        """Revised argument sub-dict must have portfolio_allocation, reasoning, raw_response, evidence_citations."""
        state = _make_state()
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        rev = bundle["revised_argument"]
        assert "portfolio_allocation" in rev
        assert "reasoning" in rev
        assert "raw_response" in rev
        assert "evidence_citations" in rev
        assert "thesis" in rev["reasoning"]

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


# ---------------------------------------------------------------------------
# Critique evidence extraction (counter_evidence field)
# ---------------------------------------------------------------------------

class TestCritiqueEvidence:
    """Critique templates output ``counter_evidence``, not ``evidence_citations``.

    These tests ensure critique evidence is extracted from the correct field
    and converted to structured format for CRIT evaluation.
    """

    def test_counter_evidence_extracted(self):
        """CRITICAL: counter_evidence from critique output must appear in bundle."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Ignores valuation",
                            "counter_evidence": ["[AAPL-RET60]", "[L1-VIX]"],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        ev_ids = {c["evidence_id"] for c in crit["evidence_citations"]}
        assert "AAPL-RET60" in ev_ids, "counter_evidence not extracted into critique bundle"
        assert "L1-VIX" in ev_ids

    def test_counter_evidence_brackets_stripped(self):
        """Bracketed evidence IDs in counter_evidence have brackets stripped."""
        state = _make_state(
            critiques=[
                {
                    "role": "risk",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Too risky",
                            "counter_evidence": ["[NVDA-DVOL60]"],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        assert crit["evidence_citations"][0]["evidence_id"] == "NVDA-DVOL60"

    def test_counter_evidence_enriched_with_memo(self):
        """counter_evidence citations should be enriched with memo text."""
        memo_lookup = {"AAPL-RET60": "60-day return: +15.3%"}
        state = _make_state(
            critiques=[
                {
                    "role": "technical",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Momentum is weak",
                            "counter_evidence": ["[AAPL-RET60]"],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, memo_lookup)

        crit = bundle["critiques_received"][0]
        assert crit["evidence_citations"][0]["evidence_text"] == "60-day return: +15.3%"

    def test_empty_counter_evidence(self):
        """Empty counter_evidence produces empty evidence_citations."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Weak thesis",
                            "counter_evidence": [],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        assert crit["evidence_citations"] == []

    def test_missing_counter_evidence_field(self):
        """Critiques without counter_evidence field produce empty citations."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "No evidence field at all",
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        assert crit["evidence_citations"] == []

    def test_legacy_evidence_citations_still_works(self):
        """If a critique has evidence_citations (legacy), those should be used."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Legacy format",
                            "evidence_citations": [{"evidence_id": "AAPL-BETA"}],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        assert crit["evidence_citations"][0]["evidence_id"] == "AAPL-BETA"

    def test_counter_evidence_preferred_over_evidence_citations(self):
        """counter_evidence takes precedence when both fields exist."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Both fields present",
                            "counter_evidence": ["[AAPL-RET60]"],
                            "evidence_citations": [{"evidence_id": "OLD-ID"}],
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        crit = bundle["critiques_received"][0]
        ev_ids = {c["evidence_id"] for c in crit["evidence_citations"]}
        assert "AAPL-RET60" in ev_ids
        assert "OLD-ID" not in ev_ids


# ---------------------------------------------------------------------------
# Inline evidence expansion
# ---------------------------------------------------------------------------

class TestExpandEvidenceIdsInline:
    """Direct unit tests for expand_evidence_ids_inline()."""

    def test_single_id_expanded(self):
        from eval.evidence import expand_evidence_ids_inline
        text = "Strong momentum [AAPL-RET60] observed."
        lookup = {"AAPL-RET60": "60-day return: +15.3%"}
        result = expand_evidence_ids_inline(text, lookup)
        assert result == "Strong momentum [AAPL-RET60: 60-day return: +15.3%] observed."

    def test_unknown_id_unchanged(self):
        from eval.evidence import expand_evidence_ids_inline
        text = "Citing [FAKE-ID1] here."
        result = expand_evidence_ids_inline(text, {})
        assert result == "Citing [FAKE-ID1] here."

    def test_multiple_ids_expanded(self):
        from eval.evidence import expand_evidence_ids_inline
        text = "Returns [AAPL-RET60] and vol [AAPL-VOL20] look good."
        lookup = {"AAPL-RET60": "60-day return: +15.3%", "AAPL-VOL20": "20-day vol: 0.22"}
        result = expand_evidence_ids_inline(text, lookup)
        assert "[AAPL-RET60: 60-day return: +15.3%]" in result
        assert "[AAPL-VOL20: 20-day vol: 0.22]" in result

    def test_mixed_known_unknown(self):
        from eval.evidence import expand_evidence_ids_inline
        text = "Known [AAPL-RET60] and unknown [FAKE-ID1]."
        lookup = {"AAPL-RET60": "60-day return: +15.3%"}
        result = expand_evidence_ids_inline(text, lookup)
        assert "[AAPL-RET60: 60-day return: +15.3%]" in result
        assert "[FAKE-ID1]" in result

    def test_empty_text(self):
        from eval.evidence import expand_evidence_ids_inline
        assert expand_evidence_ids_inline("", {"AAPL-RET60": "x"}) == ""

    def test_no_ids_in_text(self):
        from eval.evidence import expand_evidence_ids_inline
        text = "No evidence IDs here."
        assert expand_evidence_ids_inline(text, {"AAPL-RET60": "x"}) == text


class TestInlineEvidenceExpansion:
    """Test that build_reasoning_bundle expands evidence IDs inline in text fields."""

    MEMO_LOOKUP = {
        "AAPL-RET60": "60-day return: +15.3%",
        "NVDA-F3": "Filing: revenue up 20% YoY",
        "L1-VIX": "VIX: 17.35",
    }

    def test_thesis_evidence_expanded_inline(self):
        """[AAPL-RET60] in thesis becomes [AAPL-RET60: 60-day return: +15.3%]."""
        state = _make_state(justification="AAPL momentum is strong [AAPL-RET60].")
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        assert "[AAPL-RET60: 60-day return: +15.3%]" in bundle["proposal"]["reasoning"]["thesis"]

    def test_raw_response_evidence_expanded_inline(self):
        """Evidence IDs in raw_response field are expanded inline."""
        state = _make_state(proposal_raw="Analysis shows [NVDA-F3] supports overweight.")
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        assert "[NVDA-F3: Filing: revenue up 20% YoY]" in bundle["proposal"]["raw_response"]

    def test_revised_thesis_expanded(self):
        """Revised argument thesis is also expanded."""
        state = _make_state(revision_justification="After review [L1-VIX] supports risk-on.")
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        assert "[L1-VIX: VIX: 17.35]" in bundle["revised_argument"]["reasoning"]["thesis"]

    def test_revised_raw_response_expanded(self):
        """Revised argument raw_response is also expanded."""
        state = _make_state(revision_raw="Revised view: [AAPL-RET60] and [NVDA-F3].")
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        assert "[AAPL-RET60: 60-day return: +15.3%]" in bundle["revised_argument"]["raw_response"]
        assert "[NVDA-F3: Filing: revenue up 20% YoY]" in bundle["revised_argument"]["raw_response"]

    def test_critique_text_evidence_expanded_inline(self):
        """Evidence IDs in critique_text are expanded inline."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Ignores low vol [L1-VIX] environment.",
                        },
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        crit_text = bundle["critiques_received"][0]["critique_text"]
        assert "[L1-VIX: VIX: 17.35]" in crit_text

    def test_unknown_ids_left_unchanged(self):
        """IDs not in lookup remain as [FAKE-ID]."""
        state = _make_state(justification="Citing [FAKE-ID] with no lookup entry.")
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        assert "[FAKE-ID]" in bundle["proposal"]["reasoning"]["thesis"]

    def test_multiple_ids_expanded(self):
        """Multiple IDs in one string all get expanded."""
        state = _make_state(
            proposal_raw="Returns [AAPL-RET60] with vol [L1-VIX] and filings [NVDA-F3]."
        )
        bundle = build_reasoning_bundle(state, "macro", 1, self.MEMO_LOOKUP)

        raw_resp = bundle["proposal"]["raw_response"]
        assert "[AAPL-RET60: 60-day return: +15.3%]" in raw_resp
        assert "[L1-VIX: VIX: 17.35]" in raw_resp
        assert "[NVDA-F3: Filing: revenue up 20% YoY]" in raw_resp


# ---------------------------------------------------------------------------
# Enriched agent format — portfolio_rationale + claims[].evidence
# ---------------------------------------------------------------------------

ENRICHED_CLAIMS = [
    {
        "claim_id": "C1",
        "claim_text": "Rising rates pressure growth equities",
        "evidence": ["[AAPL-RET60]", "[L1-VIX]"],
        "variables": ["rates", "growth"],
        "assumptions": ["Fed holds"],
        "falsifiers": ["Rate cuts"],
        "impacts_positions": ["AAPL"],
        "confidence": 0.8,
    },
    {
        "claim_id": "C2",
        "claim_text": "NVDA AI capex cycle supports revenue",
        "evidence": ["[NVDA-F3]"],
        "variables": ["capex"],
        "assumptions": ["AI demand persists"],
        "falsifiers": ["Capex cuts"],
        "impacts_positions": ["NVDA"],
        "confidence": 0.7,
    },
]


class TestExtractThesis:
    """Unit tests for _extract_thesis helper.

    Priority order: thesis → portfolio_rationale → justification → empty.
    """

    def test_thesis_field_preferred(self):
        """Canonical thesis field takes highest priority."""
        assert _extract_thesis({"thesis": "canonical thesis"}) == "canonical thesis"

    def test_thesis_over_portfolio_rationale(self):
        """thesis takes precedence over portfolio_rationale."""
        action = {"thesis": "canonical", "portfolio_rationale": "enriched"}
        assert _extract_thesis(action) == "canonical"

    def test_portfolio_rationale_over_justification(self):
        """portfolio_rationale takes precedence over justification."""
        action = {"portfolio_rationale": "enriched", "justification": "base"}
        assert _extract_thesis(action) == "enriched"

    def test_justification_fallback(self):
        """justification used when no thesis or portfolio_rationale."""
        assert _extract_thesis({"justification": "base thesis"}) == "base thesis"

    def test_empty_string_fallback(self):
        """Returns empty string when no field exists."""
        assert _extract_thesis({}) == ""

    def test_empty_thesis_falls_through(self):
        """Empty thesis falls through to portfolio_rationale."""
        action = {"thesis": "", "portfolio_rationale": "enriched"}
        assert _extract_thesis(action) == "enriched"

    def test_empty_all_falls_through(self):
        """Empty thesis and portfolio_rationale falls through to justification."""
        action = {"thesis": "", "portfolio_rationale": "", "justification": "base"}
        assert _extract_thesis(action) == "base"


class TestExtractCitations:
    """Unit tests for _extract_citations helper."""

    def test_top_level_citations_preferred(self):
        """Top-level evidence_citations takes priority over claims."""
        from eval.evidence import extract_evidence_ids
        action = {
            "evidence_citations": [{"evidence_id": "AAPL-BETA"}],
            "claims": ENRICHED_CLAIMS,
        }
        result = _extract_citations(action, "", extract_evidence_ids)
        assert len(result) == 1
        assert result[0]["evidence_id"] == "AAPL-BETA"

    def test_claims_evidence_extracted(self):
        """Evidence from claims[].evidence is extracted when no top-level citations."""
        from eval.evidence import extract_evidence_ids
        action = {"claims": ENRICHED_CLAIMS}
        result = _extract_citations(action, "", extract_evidence_ids)
        ids = {c["evidence_id"] for c in result}
        assert "AAPL-RET60" in ids
        assert "L1-VIX" in ids
        assert "NVDA-F3" in ids

    def test_claims_evidence_deduped(self):
        """Duplicate evidence IDs across claims are deduplicated."""
        from eval.evidence import extract_evidence_ids
        claims = [
            {"claim_id": "C1", "evidence": ["[AAPL-RET60]"]},
            {"claim_id": "C2", "evidence": ["[AAPL-RET60]", "[NVDA-F3]"]},
        ]
        result = _extract_citations({"claims": claims}, "", extract_evidence_ids)
        ids = [c["evidence_id"] for c in result]
        assert ids.count("AAPL-RET60") == 1

    def test_brackets_stripped_from_evidence(self):
        """Bracket-wrapped evidence IDs like '[AAPL-RET60]' have brackets stripped."""
        from eval.evidence import extract_evidence_ids
        claims = [{"claim_id": "C1", "evidence": ["[AAPL-RET60]"]}]
        result = _extract_citations({"claims": claims}, "", extract_evidence_ids)
        assert result[0]["evidence_id"] == "AAPL-RET60"

    def test_bare_evidence_ids_accepted(self):
        """Bare evidence IDs like 'AAPL-RET60' (no brackets) are accepted."""
        from eval.evidence import extract_evidence_ids
        claims = [{"claim_id": "C1", "evidence": ["AAPL-RET60"]}]
        result = _extract_citations({"claims": claims}, "", extract_evidence_ids)
        assert result[0]["evidence_id"] == "AAPL-RET60"

    def test_regex_fallback_when_no_claims(self):
        """Falls back to regex extraction when no claims and no top-level citations."""
        from eval.evidence import extract_evidence_ids
        raw = "AAPL momentum [AAPL-RET60] and macro [L1-VIX]."
        result = _extract_citations({"claims": []}, raw, extract_evidence_ids)
        ids = {c["evidence_id"] for c in result}
        assert "AAPL-RET60" in ids
        assert "L1-VIX" in ids

    def test_empty_claims_falls_to_regex(self):
        """Empty claims array (no evidence) falls through to regex."""
        from eval.evidence import extract_evidence_ids
        claims = [{"claim_id": "C1", "evidence": []}]
        raw = "Citing [NVDA-F3] here."
        result = _extract_citations({"claims": claims}, raw, extract_evidence_ids)
        ids = {c["evidence_id"] for c in result}
        assert "NVDA-F3" in ids


class TestEnrichedFormatBundle:
    """Integration tests: build_reasoning_bundle with enriched agent output format.

    These tests would have caught the bug where enriched agents using
    portfolio_rationale + claims[].evidence produced empty thesis and
    degraded evidence citations in CRIT bundles.
    """

    def test_enriched_thesis_not_empty(self):
        """CRITICAL: thesis must NOT be empty when portfolio_rationale exists."""
        state = _make_enriched_state(
            portfolio_rationale="Value thesis: AAPL underpriced relative to earnings.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["proposal"]["reasoning"]["thesis"] != ""
        assert "AAPL underpriced" in bundle["proposal"]["reasoning"]["thesis"]

    def test_enriched_revision_thesis_not_empty(self):
        """CRITICAL: revised thesis must NOT be empty for enriched agents."""
        state = _make_enriched_state(
            revision_portfolio_rationale="Revised: maintained AAPL on valuation grounds.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert bundle["revised_argument"]["reasoning"]["thesis"] != ""
        assert "maintained AAPL" in bundle["revised_argument"]["reasoning"]["thesis"]

    def test_enriched_evidence_from_claims(self):
        """CRITICAL: evidence must be extracted from claims[].evidence for enriched agents."""
        state = _make_enriched_state(
            proposal_claims=ENRICHED_CLAIMS,
            revision_claims=ENRICHED_CLAIMS,
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        prop_ids = {c["evidence_id"] for c in bundle["proposal"]["evidence_citations"]}
        rev_ids = {c["evidence_id"] for c in bundle["revised_argument"]["evidence_citations"]}

        assert "AAPL-RET60" in prop_ids, "Enriched claims evidence must appear in proposal citations"
        assert "L1-VIX" in prop_ids
        assert "NVDA-F3" in prop_ids
        assert "AAPL-RET60" in rev_ids, "Enriched claims evidence must appear in revision citations"

    def test_enriched_evidence_not_empty(self):
        """CRITICAL: evidence_citations must NOT be empty when claims have evidence."""
        state = _make_enriched_state(
            proposal_claims=ENRICHED_CLAIMS,
            revision_claims=ENRICHED_CLAIMS,
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})

        assert len(bundle["proposal"]["evidence_citations"]) > 0, \
            "Proposal evidence_citations empty despite claims having evidence"
        assert len(bundle["revised_argument"]["evidence_citations"]) > 0, \
            "Revision evidence_citations empty despite claims having evidence"

    def test_enriched_evidence_enriched_with_memo(self):
        """Evidence from claims should be enriched with memo text."""
        memo_lookup = {"AAPL-RET60": "60-day return: +15.3%", "L1-VIX": "VIX: 17.35"}
        state = _make_enriched_state(proposal_claims=ENRICHED_CLAIMS)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_lookup)

        cite_map = {c["evidence_id"]: c for c in bundle["proposal"]["evidence_citations"]}
        assert cite_map["AAPL-RET60"]["evidence_text"] == "60-day return: +15.3%"
        assert cite_map["L1-VIX"]["evidence_text"] == "VIX: 17.35"

    def test_enriched_thesis_expanded_inline(self):
        """Evidence IDs in enriched thesis are expanded inline."""
        memo_lookup = {"AAPL-RET60": "60-day return: +15.3%"}
        state = _make_enriched_state(
            portfolio_rationale="AAPL momentum [AAPL-RET60] supports overweight.",
        )
        bundle = build_reasoning_bundle(state, "macro", 1, memo_lookup)

        assert "[AAPL-RET60: 60-day return: +15.3%]" in bundle["proposal"]["reasoning"]["thesis"]

    def test_no_justification_field_in_enriched(self):
        """Enriched state has no 'justification' key — thesis must still be populated."""
        state = _make_enriched_state()
        # Verify our fixture doesn't accidentally include justification
        prop_action = state["proposals"][0]["action_dict"]
        assert "justification" not in prop_action, "Test fixture should not have justification"

        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert bundle["proposal"]["reasoning"]["thesis"] != "", \
            "Thesis empty — build_reasoning_bundle ignores portfolio_rationale"

    def test_no_evidence_citations_field_in_enriched(self):
        """Enriched state has no top-level 'evidence_citations' — claims must be used."""
        state = _make_enriched_state(proposal_claims=ENRICHED_CLAIMS)
        prop_action = state["proposals"][0]["action_dict"]
        assert "evidence_citations" not in prop_action, \
            "Test fixture should not have top-level evidence_citations"

        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert len(bundle["proposal"]["evidence_citations"]) >= 3, \
            "Evidence not extracted from claims — build_reasoning_bundle ignores claims[].evidence"


# ---------------------------------------------------------------------------
# Warning emission — silent failure prevention
# ---------------------------------------------------------------------------

class TestWarningsEmitted:
    """Verify that warnings fire when data is missing or falling through.

    These tests ensure that field-name mismatches between agent output
    and build_reasoning_bundle are LOUD, not silent.
    """

    def test_warns_on_empty_thesis(self, caplog):
        """WARNING must fire when thesis is empty (no justification or portfolio_rationale)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            _extract_thesis({}, role="macro", phase="propose")
        assert any("thesis is EMPTY" in r.message for r in caplog.records), \
            "No warning emitted for empty thesis — silent failure"

    def test_no_warning_when_justification_exists(self, caplog):
        """No warning when justification is present."""
        import logging
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            _extract_thesis({"justification": "valid"}, role="macro", phase="propose")
        assert not any("thesis is EMPTY" in r.message for r in caplog.records)

    def test_no_warning_when_portfolio_rationale_exists(self, caplog):
        """No warning when portfolio_rationale is present (enriched format)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            _extract_thesis({"portfolio_rationale": "valid"}, role="macro", phase="propose")
        assert not any("thesis is EMPTY" in r.message for r in caplog.records)

    def test_warns_on_regex_fallback(self, caplog):
        """WARNING must fire when evidence extraction falls to regex."""
        import logging
        from eval.evidence import extract_evidence_ids
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            _extract_citations(
                {"claims": []}, "Some text [AAPL-RET60].", extract_evidence_ids,
                role="macro", phase="propose",
            )
        assert any("fell back to regex" in r.message for r in caplog.records), \
            "No warning emitted for regex fallback — silent failure"

    def test_warns_on_no_evidence_anywhere(self, caplog):
        """WARNING must fire when no evidence is found anywhere."""
        import logging
        from eval.evidence import extract_evidence_ids
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            _extract_citations(
                {"claims": []}, "No bracket IDs here.", extract_evidence_ids,
                role="value", phase="revise",
            )
        assert any("No evidence citations found anywhere" in r.message for r in caplog.records), \
            "No warning emitted when evidence is completely missing — silent failure"

    def test_warns_on_missing_critique_evidence(self, caplog):
        """WARNING must fire when critique has no counter_evidence."""
        import logging
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "No evidence field at all",
                        },
                    ],
                },
            ],
        )
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            build_reasoning_bundle(state, "macro", 1, {})
        assert any("no counter_evidence" in r.message for r in caplog.records), \
            "No warning emitted when critique has no evidence — silent failure"

    def test_no_warning_when_critique_has_counter_evidence(self, caplog):
        """No warning when critique has counter_evidence."""
        import logging
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {
                            "target_role": "macro",
                            "objection": "Has evidence",
                            "counter_evidence": ["[AAPL-RET60]"],
                        },
                    ],
                },
            ],
        )
        with caplog.at_level(logging.WARNING, logger="multi_agent.runner"):
            build_reasoning_bundle(state, "macro", 1, {})
        assert not any("no counter_evidence" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Critique target_role normalization (case + suffix)
# ---------------------------------------------------------------------------

class TestCritiqueTargetRoleNormalization:
    """LLM outputs vary in casing and may add suffixes like 'agent'.

    The filter must match regardless of case or trailing words.
    """

    def test_uppercase_target_role_matched(self):
        """target_role='MACRO' should match role='macro'."""
        state = _make_state(
            critiques=[
                {
                    "role": "risk",
                    "critiques": [
                        {"target_role": "MACRO", "objection": "Uppercase target"},
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert len(bundle["critiques_received"]) == 1
        assert bundle["critiques_received"][0]["critique_text"] == "Uppercase target"

    def test_target_role_with_agent_suffix_matched(self):
        """target_role='MACRO agent' should match role='macro'."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {"target_role": "MACRO agent", "objection": "Suffix target"},
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert len(bundle["critiques_received"]) == 1
        assert bundle["critiques_received"][0]["critique_text"] == "Suffix target"

    def test_mixed_case_target_role_matched(self):
        """target_role='Macro' should match role='macro'."""
        state = _make_state(
            critiques=[
                {
                    "role": "technical",
                    "critiques": [
                        {"target_role": "Macro", "objection": "Mixed case"},
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert len(bundle["critiques_received"]) == 1

    def test_non_matching_still_excluded(self):
        """target_role='RISK' should NOT match role='macro'."""
        state = _make_state(
            critiques=[
                {
                    "role": "value",
                    "critiques": [
                        {"target_role": "RISK", "objection": "Not for macro"},
                    ],
                },
            ],
        )
        bundle = build_reasoning_bundle(state, "macro", 1, {})
        assert len(bundle["critiques_received"]) == 0
