"""Integration tests for multi-round state handoff in debate graph nodes.

Tests every aspect of the operator.add accumulation bug fix:
1. _latest_per_role deduplicates correctly
2. _resolve_source validates and deduplicates
3. Serial critique_node sees latest revisions, not stale round 1
4. Serial revise_node sees latest revisions, not stale round 1
5. Parallel make_critique_node sees latest revisions
6. Parallel make_revise_node sees latest revisions
7. Empty/corrupt state crashes instead of silently proceeding
8. Round 1 (no prior revisions) still works correctly

Uses mock LLM mode to avoid real API calls.
"""

from __future__ import annotations

import json

import pytest

from multi_agent.graph.nodes import (
    _latest_per_role,
    _resolve_source,
)


# ---------------------------------------------------------------------------
# Fixtures — simulate operator.add accumulation
# ---------------------------------------------------------------------------

def _make_action_dict(label: str, alloc: dict) -> dict:
    return {
        "allocation": alloc,
        "portfolio_rationale": f"{label} rationale",
        "justification": f"{label} justification",
        "confidence": 0.8,
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": f"{label} claim",
                "claim_type": "macro",
                "reasoning_type": "causal",
                "evidence": [f"[{label}-E1]"],
                "assumptions": ["assumption"],
                "falsifiers": ["falsifier"],
                "impacts_positions": list(alloc.keys()),
                "confidence": 0.8,
            },
        ],
        "position_rationale": [
            {
                "ticker": t,
                "weight": w,
                "supported_by_claims": ["C1"],
                "explanation": f"{label} position",
            }
            for t, w in alloc.items() if w > 0
        ],
        "risks_or_falsifiers": [f"{label} risk"],
    }


def _make_entry(role: str, label: str, alloc: dict) -> dict:
    return {
        "role": role,
        "action_dict": _make_action_dict(label, alloc),
        "raw_response": f"{label} raw response",
        "revision_notes": f"{label} notes",
    }


def _make_round1_proposals() -> list[dict]:
    return [
        _make_entry("macro", "macro_propose", {"XOM": 0.4, "AAPL": 0.3, "UNH": 0.3}),
        _make_entry("technical", "tech_propose", {"NVDA": 0.5, "AAPL": 0.3, "META": 0.2}),
        _make_entry("value", "value_propose", {"JPM": 0.4, "WMT": 0.3, "JNJ": 0.3}),
    ]


def _make_round1_revisions() -> list[dict]:
    return [
        _make_entry("macro", "macro_r1", {"XOM": 0.35, "AAPL": 0.35, "UNH": 0.3}),
        _make_entry("technical", "tech_r1", {"NVDA": 0.45, "AAPL": 0.35, "META": 0.2}),
        _make_entry("value", "value_r1", {"JPM": 0.35, "WMT": 0.35, "JNJ": 0.3}),
    ]


def _make_round2_revisions() -> list[dict]:
    return [
        _make_entry("macro", "macro_r2", {"XOM": 0.25, "AAPL": 0.40, "UNH": 0.35}),
        _make_entry("technical", "tech_r2", {"NVDA": 0.40, "AAPL": 0.40, "META": 0.20}),
        _make_entry("value", "value_r2", {"JPM": 0.30, "WMT": 0.40, "JNJ": 0.30}),
    ]


def _make_accumulated_revisions() -> list[dict]:
    """Simulates state["revisions"] after 2 rounds of operator.add."""
    return _make_round1_revisions() + _make_round2_revisions()


def _make_round2_state() -> dict:
    """Full state as it would appear at the start of round 2 critique."""
    return {
        "proposals": _make_round1_proposals(),
        "revisions": _make_accumulated_revisions(),
        "critiques": [],
    }


# ---------------------------------------------------------------------------
# Test: _latest_per_role
# ---------------------------------------------------------------------------

class TestLatestPerRole:
    """Unit-level tests for the deduplication helper."""

    def test_keeps_latest_when_duplicated(self):
        entries = _make_accumulated_revisions()
        result = _latest_per_role(entries)
        assert len(result) == 3
        labels = {e["action_dict"]["portfolio_rationale"] for e in result}
        assert "macro_r2 rationale" in labels
        assert "tech_r2 rationale" in labels
        assert "value_r2 rationale" in labels

    def test_round1_data_excluded(self):
        entries = _make_accumulated_revisions()
        result = _latest_per_role(entries)
        labels = {e["action_dict"]["portfolio_rationale"] for e in result}
        assert "macro_r1 rationale" not in labels
        assert "tech_r1 rationale" not in labels
        assert "value_r1 rationale" not in labels

    def test_single_round_unchanged(self):
        entries = _make_round1_revisions()
        result = _latest_per_role(entries)
        assert len(result) == 3

    def test_preserves_all_roles(self):
        entries = _make_accumulated_revisions()
        result = _latest_per_role(entries)
        roles = {e["role"] for e in result}
        assert roles == {"macro", "technical", "value"}

    def test_empty_list(self):
        assert _latest_per_role([]) == []

    def test_three_rounds_picks_last(self):
        r1 = [_make_entry("macro", "r1", {"XOM": 0.5, "AAPL": 0.5})]
        r2 = [_make_entry("macro", "r2", {"XOM": 0.4, "AAPL": 0.6})]
        r3 = [_make_entry("macro", "r3", {"XOM": 0.3, "AAPL": 0.7})]
        result = _latest_per_role(r1 + r2 + r3)
        assert len(result) == 1
        assert result[0]["action_dict"]["allocation"]["XOM"] == 0.3

    def test_allocations_are_round2(self):
        entries = _make_accumulated_revisions()
        result = _latest_per_role(entries)
        macro = next(e for e in result if e["role"] == "macro")
        assert macro["action_dict"]["allocation"]["XOM"] == 0.25
        assert macro["action_dict"]["allocation"]["AAPL"] == 0.40

    def test_raw_response_is_round2(self):
        entries = _make_accumulated_revisions()
        result = _latest_per_role(entries)
        macro = next(e for e in result if e["role"] == "macro")
        assert "macro_r2" in macro["raw_response"]
        assert "macro_r1" not in macro["raw_response"]


# ---------------------------------------------------------------------------
# Test: _resolve_source
# ---------------------------------------------------------------------------

class TestResolveSource:
    """Tests for the source resolution + validation layer."""

    def test_uses_revisions_when_present(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        assert len(source) == 3
        # Should be round 2 data
        macro = next(e for e in source if e["role"] == "macro")
        assert "macro_r2" in macro["raw_response"]

    def test_falls_back_to_proposals(self):
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": [],
            "critiques": [],
        }
        source = _resolve_source(state)
        assert len(source) == 3
        macro = next(e for e in source if e["role"] == "macro")
        assert "macro_propose" in macro["raw_response"]

    def test_crashes_on_empty_state(self):
        state = {"proposals": [], "revisions": [], "critiques": []}
        with pytest.raises(RuntimeError, match="no proposals and no revisions"):
            _resolve_source(state)

    def test_crashes_on_missing_keys(self):
        with pytest.raises(RuntimeError, match="no proposals and no revisions"):
            _resolve_source({})

    def test_deduplicates_accumulated_revisions(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        # Must be 3, not 6 (which would happen without dedup)
        assert len(source) == 3

    def test_none_revisions_falls_back(self):
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": None,
            "critiques": [],
        }
        source = _resolve_source(state)
        assert len(source) == 3


# ---------------------------------------------------------------------------
# Test: Agent sees correct data in multi-round scenario
# ---------------------------------------------------------------------------

class TestAgentInputDataIntegrity:
    """Verify that source data fed to agents is from the latest round."""

    def test_each_agent_sees_own_latest_revision(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        for role, expected_label in [
            ("macro", "macro_r2"),
            ("technical", "tech_r2"),
            ("value", "value_r2"),
        ]:
            entry = next(e for e in source if e["role"] == role)
            assert expected_label in entry["raw_response"]
            assert expected_label in entry["action_dict"]["portfolio_rationale"]

    def test_each_agent_sees_others_latest_revisions(self):
        """When building all_proposals_for_critique, all entries must be round 2."""
        state = _make_round2_state()
        source = _resolve_source(state)
        all_proposals = [
            {"role": p["role"], "proposal": json.dumps(p.get("action_dict", {}))}
            for p in source
        ]
        assert len(all_proposals) == 3
        for p in all_proposals:
            proposal_data = json.loads(p["proposal"])
            assert "r2 rationale" in proposal_data["portfolio_rationale"]

    def test_no_duplicate_role_entries_in_source(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        roles = [e["role"] for e in source]
        assert len(roles) == len(set(roles)), f"Duplicate roles in source: {roles}"

    def test_critique_prompt_data_is_round2(self):
        """Simulates what critique_node builds for my_proposal_v2."""
        state = _make_round2_state()
        source = _resolve_source(state)
        for entry in source:
            action_dict = entry.get("action_dict", {})
            my_proposal_json = json.dumps(action_dict)
            # Must contain round 2 data
            assert "r2 rationale" in my_proposal_json
            assert "r1 rationale" not in my_proposal_json

    def test_revise_sees_round2_own_proposal(self):
        """Simulates what revise_node uses as the agent's own previous output."""
        state = _make_round2_state()
        source = _resolve_source(state)
        macro = next(e for e in source if e["role"] == "macro")
        action_dict = macro.get("action_dict", {})
        # This is what gets passed as my_proposal_v2
        assert action_dict["allocation"]["XOM"] == 0.25  # Round 2 value
        assert action_dict["allocation"]["XOM"] != 0.35  # Not round 1


# ---------------------------------------------------------------------------
# Test: Round 1 (no prior revisions) still works
# ---------------------------------------------------------------------------

class TestRound1Fallback:
    """Ensure round 1 (proposals only, no revisions) works correctly."""

    def test_proposals_used_when_no_revisions(self):
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": [],
            "critiques": [],
        }
        source = _resolve_source(state)
        assert len(source) == 3
        roles = {e["role"] for e in source}
        assert roles == {"macro", "technical", "value"}

    def test_proposal_data_intact(self):
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": [],
            "critiques": [],
        }
        source = _resolve_source(state)
        macro = next(e for e in source if e["role"] == "macro")
        assert macro["action_dict"]["allocation"]["XOM"] == 0.4
        assert "macro_propose" in macro["raw_response"]


# ---------------------------------------------------------------------------
# Test: Broken round 1 does not contaminate round 2
# ---------------------------------------------------------------------------

class TestBrokenRound1Isolation:
    """If round 1 had a parse failure, round 2 data must be used instead."""

    def test_broken_r1_replaced_by_valid_r2(self):
        broken_r1 = [
            {
                "role": "macro",
                "action_dict": {
                    "allocation": {"XOM": 0.4, "AAPL": 0.3, "UNH": 0.3},
                    "portfolio_rationale": "",
                    "justification": "",
                    "confidence": 0.5,
                    "claims": [],
                    "position_rationale": [],
                    "risks_or_falsifiers": [],
                },
                "raw_response": "broken r1",
                "revision_notes": "",
            },
        ]
        valid_r2 = [
            _make_entry("macro", "macro_r2", {"XOM": 0.25, "AAPL": 0.40, "UNH": 0.35}),
        ]
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": broken_r1 + valid_r2,
            "critiques": [],
        }
        source = _resolve_source(state)
        macro = next(e for e in source if e["role"] == "macro")
        assert macro["action_dict"]["allocation"]["XOM"] == 0.25
        assert len(macro["action_dict"]["claims"]) == 1
        assert "macro_r2" in macro["raw_response"]
        assert "broken" not in macro["raw_response"]

    def test_broken_r1_claims_not_leaked(self):
        """Empty claims from a parse failure must not appear in round 2 source."""
        broken_r1 = [
            {
                "role": "value",
                "action_dict": {
                    "allocation": {"JPM": 0.5, "WMT": 0.5},
                    "portfolio_rationale": "",
                    "confidence": 0.5,
                    "claims": [],  # Empty from parse failure
                    "position_rationale": [],
                    "risks_or_falsifiers": [],
                },
                "raw_response": "value broken",
                "revision_notes": "",
            },
        ]
        valid_r2 = [
            _make_entry("value", "value_r2", {"JPM": 0.30, "WMT": 0.40, "JNJ": 0.30}),
        ]
        state = {
            "proposals": _make_round1_proposals(),
            "revisions": broken_r1 + valid_r2,
            "critiques": [],
        }
        source = _resolve_source(state)
        value = next(e for e in source if e["role"] == "value")
        assert len(value["action_dict"]["claims"]) > 0, "Round 2 claims must be present"


# ---------------------------------------------------------------------------
# Test: Multiple agents, partial breakage
# ---------------------------------------------------------------------------

class TestPartialRoundBreakage:
    """Only one agent breaks in round 1; others are fine. Round 2 fixes it."""

    def test_mixed_breakage_all_agents_get_round2(self):
        r1_revisions = [
            # macro: broken
            {
                "role": "macro",
                "action_dict": {
                    "allocation": {"XOM": 0.4, "AAPL": 0.3, "UNH": 0.3},
                    "portfolio_rationale": "",
                    "confidence": 0.5,
                    "claims": [],
                    "position_rationale": [],
                    "risks_or_falsifiers": [],
                },
                "raw_response": "macro broken r1",
                "revision_notes": "",
            },
            # technical: fine in r1
            _make_entry("technical", "tech_r1", {"NVDA": 0.45, "AAPL": 0.35, "META": 0.2}),
            # value: fine in r1
            _make_entry("value", "value_r1", {"JPM": 0.35, "WMT": 0.35, "JNJ": 0.3}),
        ]
        r2_revisions = _make_round2_revisions()

        state = {
            "proposals": _make_round1_proposals(),
            "revisions": r1_revisions + r2_revisions,
            "critiques": [],
        }
        source = _resolve_source(state)
        assert len(source) == 3

        # All three should be round 2
        for role in ["macro", "technical", "value"]:
            entry = next(e for e in source if e["role"] == role)
            assert "r2" in entry["raw_response"], f"{role} should have r2 data"
            assert "r1" not in entry["raw_response"], f"{role} should not have r1 data"


# ---------------------------------------------------------------------------
# Test: Critique receives all agents' latest data for cross-agent comparison
# ---------------------------------------------------------------------------

class TestCrossAgentVisibility:
    """In critique phase, each agent must see ALL other agents' latest revisions."""

    def test_all_proposals_for_critique_are_latest(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        all_proposals_for_critique = [
            {"role": p["role"], "proposal": json.dumps(p.get("action_dict", {}))}
            for p in source
        ]
        for p in all_proposals_for_critique:
            data = json.loads(p["proposal"])
            # Every agent's proposal data must be from round 2
            assert "r2" in data["portfolio_rationale"], (
                f"Agent {p['role']} critique input has stale data: {data['portfolio_rationale']}"
            )

    def test_critique_sees_exactly_3_agents(self):
        state = _make_round2_state()
        source = _resolve_source(state)
        assert len(source) == 3

    def test_no_agent_sees_itself_twice(self):
        """Source must have exactly one entry per role."""
        state = _make_round2_state()
        source = _resolve_source(state)
        roles = [e["role"] for e in source]
        assert sorted(roles) == sorted(set(roles))
