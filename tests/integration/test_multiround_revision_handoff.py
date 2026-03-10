"""Integration test for multi-round revision handoff through CRIT.

Verifies that when revisions accumulate across rounds (via operator.add),
build_reasoning_bundle always picks the LATEST revision for each role,
and the resulting CRIT prompt reflects round N data — not stale round 1 data.

Regression test for the bug where a forward scan on state["revisions"]
always grabbed the first entry (round 1), causing CRIT to evaluate stale
or broken reasoning even when later rounds produced valid output.
"""

import json

import pytest

from multi_agent.runner import build_reasoning_bundle, _extract_reasoning
from eval.crit.prompts import render_crit_prompts


# ---------------------------------------------------------------------------
# Fixtures — simulate operator.add accumulation across 2 rounds
# ---------------------------------------------------------------------------

ROUND1_CLAIMS = [
    {
        "claim_id": "C1",
        "claim_text": "Inflation regime favors energy [L1-CPI]",
        "claim_type": "macro",
        "reasoning_type": "causal",
        "evidence": ["[L1-CPI]"],
        "assumptions": ["CPI stays elevated"],
        "falsifiers": ["Deflation shock"],
        "impacts_positions": ["XOM", "CVX"],
        "confidence": 0.7,
    },
]

ROUND2_CLAIMS = [
    {
        "claim_id": "C1",
        "claim_text": "Inflation regime favors energy but rate hikes cap upside [L1-CPI] [L1-10Y]",
        "claim_type": "macro",
        "reasoning_type": "causal",
        "evidence": ["[L1-CPI]", "[L1-10Y]"],
        "assumptions": ["CPI stays elevated", "Fed follows through on hikes"],
        "falsifiers": ["Deflation shock", "Fed pivot"],
        "impacts_positions": ["XOM", "CVX", "UNH"],
        "confidence": 0.75,
    },
    {
        "claim_id": "C2",
        "claim_text": "Defensive healthcare outperforms in tightening cycles [L1-FF]",
        "claim_type": "sector",
        "reasoning_type": "causal",
        "evidence": ["[L1-FF]"],
        "assumptions": ["Tightening continues"],
        "falsifiers": ["Recession reverses policy"],
        "impacts_positions": ["UNH", "JNJ"],
        "confidence": 0.65,
    },
]

ROUND2_POSITION_RATIONALE = [
    {
        "ticker": "XOM",
        "weight": 0.25,
        "supported_by_claims": ["C1"],
        "explanation": "Energy benefits from inflation but capped by rate pressure.",
    },
    {
        "ticker": "UNH",
        "weight": 0.20,
        "supported_by_claims": ["C1", "C2"],
        "explanation": "Defensive healthcare hedge against tightening.",
    },
]


def _make_multi_round_state():
    """Build state simulating 2 rounds of operator.add accumulation.

    Round 1: macro agent has a weak revision (simulating a parse failure
    that produced empty claims — the exact bug scenario).
    Round 2: macro agent has a strong, valid revision.

    The revisions list contains both, in chronological order.
    """
    proposal = {
        "role": "macro",
        "action_dict": {
            "allocation": {"XOM": 0.3, "CVX": 0.3, "UNH": 0.2, "JNJ": 0.2},
            "portfolio_rationale": "Energy-heavy proposal based on inflation regime.",
            "confidence": 0.7,
            "claims": ROUND1_CLAIMS,
            "position_rationale": [],
            "risks_or_falsifiers": ["Deflation shock."],
        },
        "raw_response": "raw proposal with [L1-CPI] evidence",
    }

    round1_revision = {
        "role": "macro",
        "action_dict": {
            # Simulates parse failure: empty claims, empty rationale
            "allocation": {"XOM": 0.3, "CVX": 0.3, "UNH": 0.2, "JNJ": 0.2},
            "portfolio_rationale": "",
            "confidence": 0.5,
            "claims": [],
            "position_rationale": [],
            "risks_or_falsifiers": [],
            "critique_responses": [],
        },
        "revision_notes": "",
        "raw_response": "round 1 broken raw",
    }

    round2_revision = {
        "role": "macro",
        "action_dict": {
            "allocation": {"XOM": 0.25, "CVX": 0.15, "UNH": 0.20, "JNJ": 0.15, "WMT": 0.10, "AAPL": 0.15},
            "portfolio_rationale": "Rebalanced: accepted rate-pressure critique, added healthcare hedge.",
            "confidence": 0.75,
            "claims": ROUND2_CLAIMS,
            "position_rationale": ROUND2_POSITION_RATIONALE,
            "risks_or_falsifiers": ["Fed pivot invalidates tightening thesis."],
            "critique_responses": [
                "Accepted risk agent critique: reduced XOM/CVX concentration.",
                "Added healthcare hedge per sector rotation argument.",
            ],
        },
        "revision_notes": "Accepted risk agent critique on energy concentration. Added C2 for healthcare.",
        "raw_response": "round 2 valid raw with [L1-CPI] [L1-10Y] [L1-FF] evidence",
    }

    # Critiques targeting macro from round 1 (used in round 2 revision)
    critiques = [
        {
            "role": "risk",
            "self_critique": "Risk model may overweight tail events.",
            "critiques": [
                {
                    "target_role": "MACRO",
                    "target_claim": "C1",
                    "objection": "Energy concentration creates drawdown risk.",
                    "counter_evidence": ["[L1-VIX]"],
                    "portfolio_implication": "30% sector bet exceeds risk budget.",
                    "suggested_adjustment": "Cap energy at 20%, reallocate to defensives.",
                    "falsifier": "VIX spike above 30 triggers forced liquidation.",
                    "objection_confidence": 0.8,
                },
            ],
        },
        {
            "role": "value",
            "self_critique": "Valuation multiples may lag in inflationary regime.",
            "critiques": [
                {
                    "target_role": "MACRO",
                    "target_claim": "C1",
                    "objection": "XOM/CVX multiples already reflect commodity premium.",
                    "counter_evidence": ["[XOM-F3]"],
                    "portfolio_implication": "Overpaying for energy names limits upside.",
                    "suggested_adjustment": "Reduce XOM weight; consider mid-cap energy with better multiples.",
                    "falsifier": "Commodity super-cycle reprices sector multiples higher.",
                    "objection_confidence": 0.7,
                },
            ],
        },
    ]

    return {
        "proposals": [proposal],
        # operator.add accumulation: round 1 + round 2
        "revisions": [round1_revision, round2_revision],
        "critiques": critiques,
    }


# ---------------------------------------------------------------------------
# Tests: latest revision is selected
# ---------------------------------------------------------------------------

class TestMultiRoundHandoff:
    """Verify build_reasoning_bundle picks round 2 revision, not round 1."""

    def test_round2_allocation_in_bundle(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        alloc = bundle["revised_argument"]["portfolio_allocation"]
        assert alloc["XOM"] == 0.25
        assert alloc["UNH"] == 0.20
        assert "WMT" in alloc  # Only in round 2

    def test_round2_claims_in_bundle(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        claims = bundle["revised_argument"]["reasoning"]["claims"]
        assert len(claims) == 2  # Round 2 has C1 + C2; round 1 had 0
        claim_ids = [c["claim_id"] for c in claims]
        assert "C1" in claim_ids
        assert "C2" in claim_ids

    def test_round2_thesis_in_bundle(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        thesis = bundle["revised_argument"]["reasoning"]["thesis"]
        assert "healthcare hedge" in thesis

    def test_round1_broken_data_excluded(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        raw = bundle["revised_argument"]["raw_response"]
        assert "round 1 broken" not in raw
        thesis = bundle["revised_argument"]["reasoning"]["thesis"]
        assert thesis != ""  # Round 1 had empty thesis

    def test_revision_notes_from_round2(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        notes = bundle["revised_argument"]["reasoning"].get("revision_notes", "")
        assert "energy concentration" in notes

    def test_round2_evidence_ids_present(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        evidence = bundle["revised_argument"]["evidence_citations"]
        # Round 2 claims reference L1-CPI, L1-10Y, L1-FF
        evidence_flat = [e if isinstance(e, str) else str(e) for e in evidence]
        evidence_text = " ".join(evidence_flat)
        assert "L1-CPI" in evidence_text
        assert "L1-10Y" in evidence_text

    def test_critiques_passed_through(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        crits = bundle["critiques_received"]
        assert len(crits) == 2
        sources = {c["from_role"] for c in crits}
        assert "risk" in sources
        assert "value" in sources


# ---------------------------------------------------------------------------
# Tests: CRIT prompt renders round 2 data, not round 1
# ---------------------------------------------------------------------------

class TestMultiRoundCritRendering:
    """Verify the rendered CRIT prompt reflects round 2, not stale round 1."""

    def test_crit_prompt_contains_round2_claims(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        _, user_prompt = render_crit_prompts(
            bundle,
            system_template="crit_system_enumerated.jinja",
            user_template="crit_user_master.jinja",
        )
        # Round 2 claim C2 about healthcare should appear
        assert "healthcare" in user_prompt.lower()
        # Round 2 has 2 claims; round 1 had 0
        assert "C2" in user_prompt

    def test_crit_prompt_contains_round2_revision_notes(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        _, user_prompt = render_crit_prompts(
            bundle,
            system_template="crit_system_enumerated.jinja",
            user_template="crit_user_master.jinja",
        )
        assert "energy concentration" in user_prompt

    def test_crit_prompt_has_nonempty_reasoning(self):
        """The CRIT prompt must not have empty claims (the round 1 bug)."""
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        _, user_prompt = render_crit_prompts(
            bundle,
            system_template="crit_system_enumerated.jinja",
            user_template="crit_user_master.jinja",
        )
        # Parse the JSON embedded in the prompt to verify claims are populated
        assert '"claims": []' not in user_prompt


# ---------------------------------------------------------------------------
# Tests: proposal is stable across rounds
# ---------------------------------------------------------------------------

class TestProposalStability:
    """Proposal should remain the original round 1 proposal in all rounds."""

    def test_proposal_unchanged_in_round2_bundle(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        prop_alloc = bundle["proposal"]["portfolio_allocation"]
        assert prop_alloc["XOM"] == 0.3
        assert prop_alloc["CVX"] == 0.3

    def test_proposal_raw_response_preserved(self):
        state = _make_multi_round_state()
        bundle = build_reasoning_bundle(state, "macro", 2, {})
        assert "raw proposal" in bundle["proposal"]["raw_response"]
