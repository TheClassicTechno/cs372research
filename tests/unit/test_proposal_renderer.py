"""Tests for multi_agent.graph.proposal_renderer."""

import pytest

from multi_agent.graph.proposal_renderer import (
    render_critiques_received,
    render_others_proposals,
    render_previous_proposal,
)


FULL_ACTION_DICT = {
    "allocation": {"AAPL": 0.25, "NVDA": 0.20, "JPM": 0.15, "XOM": 0.10, "SLB": 0.30},
    "justification": "Tech rebound thesis with energy hedge.",
    "confidence": 0.72,
    "risks_or_falsifiers": "Fed pivot fails to materialize; inflation re-accelerates.",
    "claims": [
        {
            "claim_id": "C1",
            "claim_text": "Inflation will peak within 2 quarters",
            "pearl_level": "L2",
            "variables": ["CPI", "oil_price"],
            "assumptions": ["supply constraints ease"],
            "confidence": 0.70,
        },
        {
            "claim_id": "C2",
            "claim_text": "Tech earnings rebound in H2",
            "pearl_level": "L1",
            "variables": ["earnings_growth"],
            "assumptions": ["demand expansion"],
            "confidence": 0.65,
        },
    ],
}

ACTION_DICT_NO_CLAIM_IDS = {
    "allocation": {"AAPL": 0.50, "JPM": 0.50},
    "justification": "Simple barbell.",
    "confidence": 0.60,
    "risks_or_falsifiers": "Concentration risk.",
    "claims": [
        {
            "claim_text": "Tech leads recovery",
            "pearl_level": "L1",
            "variables": ["earnings"],
            "assumptions": ["no recession"],
            "confidence": 0.55,
        },
        {
            "claim_text": "Financials benefit from rate hikes",
            "pearl_level": "L2",
            "variables": ["fed_funds"],
            "assumptions": ["credit quality holds"],
            "confidence": 0.60,
        },
    ],
}


class TestRenderPreviousProposal:
    def test_full_output_has_all_sections(self):
        result = render_previous_proposal(FULL_ACTION_DICT)
        assert "## Previous Portfolio Allocation" in result
        assert "## Previous Thesis" in result
        assert "## Previous Claims" in result
        assert "## Previous Risks / Falsifiers" in result
        assert "## Previous Confidence" in result

    def test_allocation_sorted_by_ticker(self):
        result = render_previous_proposal(FULL_ACTION_DICT)
        lines = result.split("\n")
        alloc_start = next(i for i, l in enumerate(lines) if "Previous Portfolio" in l)
        alloc_lines = []
        for l in lines[alloc_start + 1:]:
            if l.startswith("##") or l == "":
                break
            alloc_lines.append(l)
        tickers = [l.split(":")[0] for l in alloc_lines]
        assert tickers == sorted(tickers)

    def test_claim_ids_used_when_present(self):
        result = render_previous_proposal(FULL_ACTION_DICT)
        assert "C1: Inflation will peak" in result
        assert "C2: Tech earnings rebound" in result

    def test_claim_ids_generated_when_missing(self):
        result = render_previous_proposal(ACTION_DICT_NO_CLAIM_IDS)
        assert "C1: Tech leads recovery" in result
        assert "C2: Financials benefit" in result

    def test_claim_metadata_rendered(self):
        result = render_previous_proposal(FULL_ACTION_DICT)
        assert "Pearl Level: L2" in result
        assert "Variables: CPI, oil_price" in result
        assert "Assumptions: supply constraints ease" in result
        assert "Confidence: 0.7" in result

    def test_empty_dict_returns_fallback(self):
        assert render_previous_proposal({}) == "(No previous proposal available.)"

    def test_none_returns_fallback(self):
        assert render_previous_proposal(None) == "(No previous proposal available.)"

    def test_missing_claims_still_renders(self):
        d = {"allocation": {"AAPL": 1.0}, "justification": "All in."}
        result = render_previous_proposal(d)
        assert "## Previous Portfolio Allocation" in result
        assert "## Previous Thesis" in result
        assert "## Previous Claims" not in result

    def test_deterministic(self):
        """Same input always produces same output."""
        r1 = render_previous_proposal(FULL_ACTION_DICT)
        r2 = render_previous_proposal(FULL_ACTION_DICT)
        assert r1 == r2


# ---- Source entries for render_others_proposals tests ----

SOURCE_ENTRIES = [
    {
        "role": "macro",
        "action_dict": {
            "allocation": {"AAPL": 0.50, "NVDA": 0.50},
            "justification": "Macro thesis.",
        },
    },
    {
        "role": "value",
        "action_dict": {
            "allocation": {"JPM": 0.60, "XOM": 0.40},
            "justification": "Value thesis.",
        },
    },
    {
        "role": "risk",
        "action_dict": {
            "allocation": {"SLB": 1.0},
            "justification": "Risk thesis.",
        },
    },
]


class TestRenderOthersProposals:
    def test_excludes_own_role(self):
        result = render_others_proposals(SOURCE_ENTRIES, "macro")
        assert "MACRO agent proposed" not in result
        assert "VALUE agent proposed" in result
        assert "RISK agent proposed" in result

    def test_renders_each_via_render_previous_proposal(self):
        result = render_others_proposals(SOURCE_ENTRIES, "risk")
        # Should contain rendered allocation sections from other agents
        assert "Previous Portfolio Allocation" in result
        assert "Previous Thesis" in result

    def test_empty_source_returns_fallback(self):
        result = render_others_proposals([], "macro")
        assert result == "(No other proposals available.)"

    def test_single_agent_excludes_self(self):
        source = [{"role": "macro", "action_dict": {"allocation": {"AAPL": 1.0}}}]
        result = render_others_proposals(source, "macro")
        assert result == "(No other proposals available.)"


# ---- Critique data for render_critiques_received tests ----

FULL_CRITIQUES = [
    {
        "from_role": "value",
        "target_claim": "C2",
        "objection": "Tech earnings rebound is overly optimistic.",
        "counter_evidence": ["[NVDA-RET60]", "[AAPL-VOL60]"],
        "portfolio_implication": "Overweight in tech is risky.",
        "suggested_adjustment": "Reduce NVDA to 0.10.",
        "falsifier": "Q2 earnings beat consensus by >10%.",
        "objection_confidence": 0.8,
    },
    {
        "from_role": "risk",
        "target_claim": "C1",
        "objection": "Inflation may not peak.",
        "counter_evidence": ["[L1-CPI]"],
        "portfolio_implication": "Duration exposure too high.",
        "suggested_adjustment": "Add energy hedge.",
        "falsifier": "CPI falls below 3%.",
        "objection_confidence": 0.65,
    },
]

MINIMAL_CRITIQUES = [
    {
        "from_role": "macro",
        "objection": "Concentration risk is too high.",
    },
]


class TestRenderCritiquesReceived:
    def test_full_fields(self):
        result = render_critiques_received(FULL_CRITIQUES)
        assert "Critique 1 (from VALUE)" in result
        assert "Target claim: C2" in result
        assert "Objection: Tech earnings rebound is overly optimistic." in result
        assert "[NVDA-RET60]" in result
        assert "Portfolio implication: Overweight in tech is risky." in result
        assert "Suggested adjustment: Reduce NVDA to 0.10." in result
        assert "Falsifier: Q2 earnings beat consensus by >10%." in result
        assert "Objection confidence: 0.8" in result

        assert "Critique 2 (from RISK)" in result
        assert "Target claim: C1" in result

    def test_minimal_fields_backward_compat(self):
        result = render_critiques_received(MINIMAL_CRITIQUES)
        assert "Critique 1 (from MACRO)" in result
        assert "Objection: Concentration risk is too high." in result
        # Should not have optional fields
        assert "Target claim" not in result
        assert "Counter-evidence" not in result
        assert "Portfolio implication" not in result

    def test_empty_list(self):
        result = render_critiques_received([])
        assert result == "(No critiques targeted at you this round.)"

    def test_missing_from_role_crashes(self):
        """Critique without from_role must crash, not silently say 'unknown'."""
        with pytest.raises(KeyError, match="from_role"):
            render_critiques_received([{"objection": "bad"}])


# ---------------------------------------------------------------------------
# Enriched format — portfolio_rationale, position_rationale
# ---------------------------------------------------------------------------

ENRICHED_ACTION_DICT = {
    "allocation": {"AAPL": 0.40, "NVDA": 0.35, "JPM": 0.25},
    "portfolio_rationale": "Overweight tech on datacenter demand; financials hedge.",
    "position_rationale": [
        {
            "ticker": "AAPL",
            "weight": 0.40,
            "explanation": "Services revenue growth supports premium valuation.",
            "supported_by_claims": ["C1", "C3"],
        },
        {
            "ticker": "NVDA",
            "weight": 0.35,
            "explanation": "Datacenter GPU demand accelerating.",
            "supported_by_claims": ["C2"],
        },
        {
            "ticker": "JPM",
            "weight": 0.25,
            "explanation": "NII benefiting from higher rates.",
            "supported_by_claims": ["C1"],
        },
    ],
    "confidence": 0.78,
    "claims": [
        {
            "claim_id": "C1",
            "claim_text": "Higher-for-longer rates benefit financials.",
            "pearl_level": "L1",
            "evidence": ["[L1-FF]", "[JPM-NII]"],
        },
        {
            "claim_id": "C2",
            "claim_text": "AI infrastructure buildout accelerating.",
            "pearl_level": "L2",
            "evidence": ["[NVDA-F1]", "[L1-CAPEX]"],
        },
    ],
    "risks_or_falsifiers": [
        "Fed pivots to rate cuts before H2 — removes financials tailwind.",
        "AI capex cycle stalls on ROI concerns.",
    ],
}


class TestRenderEnrichedProposal:
    def test_portfolio_rationale_used_as_thesis(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "## Previous Thesis" in result
        assert "Overweight tech on datacenter demand" in result

    def test_portfolio_rationale_preferred_over_justification(self):
        """When both exist, portfolio_rationale wins."""
        both = {
            "allocation": {"AAPL": 1.0},
            "justification": "Old thesis.",
            "portfolio_rationale": "Enriched thesis.",
        }
        result = render_previous_proposal(both)
        assert "Enriched thesis." in result

    def test_justification_fallback_when_no_portfolio_rationale(self):
        """Base agents use justification — still works."""
        base = {"allocation": {"AAPL": 1.0}, "justification": "Base thesis."}
        result = render_previous_proposal(base)
        assert "Base thesis." in result

    def test_position_rationale_rendered(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "## Previous Position Rationale" in result
        assert "AAPL (0.4)" in result
        assert "Services revenue growth" in result
        assert "Supported by claims: C1, C3" in result

    def test_position_rationale_all_tickers(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "NVDA (0.35)" in result
        assert "JPM (0.25)" in result

    def test_risks_as_list_rendered(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "## Previous Risks / Falsifiers" in result
        assert "- Fed pivots to rate cuts" in result
        assert "- AI capex cycle stalls" in result

    def test_enriched_claims_with_evidence(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "[L1-FF]" in result
        assert "[NVDA-F1]" in result

    def test_all_enriched_sections_present(self):
        result = render_previous_proposal(ENRICHED_ACTION_DICT)
        assert "## Previous Portfolio Allocation" in result
        assert "## Previous Thesis" in result
        assert "## Previous Position Rationale" in result
        assert "## Previous Claims" in result
        assert "## Previous Risks / Falsifiers" in result
        assert "## Previous Confidence" in result


class TestRenderOthersProposalsCrashOnMissing:
    def test_missing_role_crashes(self):
        """Entry without 'role' must crash, not silently say 'unknown'."""
        entries = [{"action_dict": {"allocation": {"AAPL": 1.0}}}]
        with pytest.raises(KeyError, match="role"):
            render_others_proposals(entries, "macro")

    def test_enriched_proposals_rendered(self):
        """Enriched proposals flow through render_others correctly."""
        entries = [
            {"role": "technical", "action_dict": ENRICHED_ACTION_DICT},
            {"role": "macro", "action_dict": {"allocation": {"JPM": 1.0}, "justification": "Rates."}},
        ]
        result = render_others_proposals(entries, "value")
        assert "TECHNICAL agent proposed" in result
        assert "Overweight tech on datacenter demand" in result
        assert "Position Rationale" in result
        assert "MACRO agent proposed" in result
