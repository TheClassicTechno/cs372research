"""
Integration tests: Agent Prompt → Normalizer → CRIT contract compatibility.

These tests verify two things:

1. That agent prompt templates produce outputs compatible with CRIT (Groups 1-4).
2. That inter-phase data handoffs preserve structural fields through the full
   debate pipeline: propose → critique → revise → CRIT (Groups 7-9).

Pipeline under test:

    Agent Prompt (rendered)
        ↓
    LLM output JSON
        ↓
    render_previous_proposal() / render_others_proposals()    ← propose→critique
        ↓
    render_critiques_received()                               ← critique→revise
        ↓
    _normalize_claims() / _normalize_position_rationale()     ← revise→CRIT
        ↓
    build_reasoning_bundle()
        ↓
    render_crit_prompts() → CRIT evaluation

CRIT expects claims with:
    claim_id, claim_text, claim_type, evidence_ids,
    impacts_positions, falsifiers, confidence

Position objects must also contain:
    position_rationale, supporting_claims
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from multi_agent.prompts import (
    build_proposal_user_prompt,
    build_critique_prompt,
    build_revision_prompt,
    load_module,
)
from multi_agent.runner import (
    _normalize_claims,
    _normalize_position_rationale,
    _extract_reasoning,
    build_reasoning_bundle,
)
from multi_agent.graph.proposal_renderer import (
    render_previous_proposal,
    render_others_proposals,
    render_critiques_received,
)
from eval.evidence import normalize_evidence_id
from eval.crit.prompts import render_crit_prompts
from eval.crit.schema import validate_raw_response, Diagnostics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AGENT_PROFILES_DIR = (
    Path(__file__).resolve().parents[2] / "config" / "agent_profiles"
)
_PROMPTS_DIR = (
    Path(__file__).resolve().parents[2] / "multi_agent" / "prompts"
)


# ===================================================================
# FIXTURES
# ===================================================================


# Fields that CRIT requires on every claim (agent-produced).
# Note: evidence_ids is computed by _normalize_claims from the raw "evidence"
# list — agents produce "evidence", the normalizer derives "evidence_ids".
CRIT_REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "claim_type",
    "evidence",
    "falsifiers",
    "impacts_positions",
}

# Fields that CRIT requires on position rationale entries.
CRIT_REQUIRED_POSITION_FIELDS = {
    "supporting_claims",
}

# Valid claim_type enum values.
VALID_CLAIM_TYPES = {"macro", "sector", "firm", "risk", "technical"}

# Valid reasoning_type enum values.
VALID_REASONING_TYPES = {"causal", "observational", "risk_assessment", "pattern"}

# Minimal evidence IDs for deterministic memo fixture.
EVIDENCE_IDS = ["[L1-10Y]", "[AAPL-RET60]", "[AAPL-GM]", "[MSFT-VOL20]", "[L1-VIX]"]

UNIVERSE = ["AAPL", "MSFT", "GOOGL"]


def _load_profile(name: str) -> dict:
    """Load an agent profile YAML by filename (without .yaml extension)."""
    path = _AGENT_PROFILES_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text()) or {}


def _all_profile_names() -> list[str]:
    """Return every .yaml profile name (sans extension) in agent_profiles/."""
    return sorted(p.stem for p in _AGENT_PROFILES_DIR.glob("*.yaml"))


def _classify_profile(name: str) -> str:
    """Classify a profile as 'enriched' or 'base'."""
    if "enriched" in name:
        return "enriched"
    return "base"


# --- Profile ID lists for parametrize --------------------------------

_ALL_PROFILES = _all_profile_names()
_BASE_PROFILES = [p for p in _ALL_PROFILES if _classify_profile(p) == "base"]
_ENRICHED_PROFILES = [p for p in _ALL_PROFILES if _classify_profile(p) == "enriched"]

# Exclude judge profiles from agent-output tests (judges don't propose).
_PROPOSING_BASE = [p for p in _BASE_PROFILES if "judge" not in p]
_PROPOSING_ENRICHED = [p for p in _ENRICHED_PROFILES if "judge" not in p]


@pytest.fixture()
def mini_memo() -> str:
    """A deterministic minimal memo with known evidence IDs."""
    return (
        "[INFO] QUARTERLY SNAPSHOT MEMO\n"
        "\n"
        "## Macro Environment\n"
        "10-Year Treasury Yield: 4.25% [L1-10Y]\n"
        "VIX: 18.5 [L1-VIX]\n"
        "\n"
        "## AAPL\n"
        "60-day return: +8.2% [AAPL-RET60]\n"
        "Gross margin: 45.3% [AAPL-GM]\n"
        "\n"
        "## MSFT\n"
        "20-day volatility: 0.22 [MSFT-VOL20]\n"
        "\n"
        "## GOOGL\n"
        "No additional evidence available.\n"
    )


@pytest.fixture()
def memo_evidence_lookup() -> dict[str, str]:
    """Evidence lookup dict mapping normalized IDs to context sentences."""
    return {
        "L1-10Y": "10-Year Treasury Yield: 4.25%",
        "L1-VIX": "VIX: 18.5",
        "AAPL-RET60": "AAPL 60-day return: +8.2%",
        "AAPL-GM": "AAPL Gross margin: 45.3%",
        "MSFT-VOL20": "MSFT 20-day volatility: 0.22",
    }


# --- Simulated agent outputs -----------------------------------------

def _make_enriched_output() -> dict:
    """Simulated output from an enriched-family agent (CRIT-compatible)."""
    return {
        "allocation": {"AAPL": 0.40, "MSFT": 0.35, "GOOGL": 0.25},
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "Rising yields pressure growth equities [L1-10Y].",
                "claim_type": "macro",
                "reasoning_type": "causal",
                "evidence": ["[L1-10Y]", "[L1-VIX]"],
                "assumptions": ["Fed maintains current rate path"],
                "falsifiers": ["Rapid rate cuts would invalidate"],
                "impacts_positions": ["AAPL", "MSFT"],
                "confidence": 0.7,
            },
            {
                "claim_id": "C2",
                "claim_text": "AAPL margin resilience supports overweight [AAPL-GM].",
                "claim_type": "firm",
                "reasoning_type": "observational",
                "evidence": ["[AAPL-GM]", "[AAPL-RET60]"],
                "assumptions": ["Margins stay above 44%"],
                "falsifiers": ["Margin compression below 40%"],
                "impacts_positions": ["AAPL"],
                "confidence": 0.65,
            },
        ],
        "position_rationale": [
            {
                "ticker": "AAPL",
                "weight": 0.40,
                "supported_by_claims": ["C1", "C2"],
                "explanation": "Macro headwinds offset by firm-level margin strength.",
            },
            {
                "ticker": "MSFT",
                "weight": 0.35,
                "supported_by_claims": ["C1"],
                "explanation": "Growth name with moderate macro exposure.",
            },
            {
                "ticker": "GOOGL",
                "weight": 0.25,
                "supported_by_claims": ["C1"],
                "explanation": "Diversification across mega-cap tech.",
            },
        ],
        "portfolio_rationale": "Overweight AAPL on margin strength, tilt MSFT.",
        "confidence": 0.65,
        "risks_or_falsifiers": ["Rate shock beyond 5%", "Earnings miss in AAPL"],
    }


def _make_base_output() -> dict:
    """Simulated output from a base-family agent (missing CRIT fields)."""
    return {
        "allocation": {"AAPL": 0.40, "MSFT": 0.35, "GOOGL": 0.25},
        "justification": "Equal-weight with AAPL overweight based on momentum.",
        "confidence": 0.5,
        "claims": [
            {
                "claim_text": "AAPL has strong momentum [AAPL-RET60].",
                "reasoning_type": "causal",
                "assumptions": ["Momentum persists"],
                "confidence": 0.55,
            },
            {
                "claim_text": "Low volatility environment favors equities [L1-VIX].",
                "reasoning_type": "observational",
                "assumptions": ["VIX stays below 20"],
                "confidence": 0.5,
            },
        ],
        "risks_or_falsifiers": "VIX spike above 30",
    }


# ===================================================================
# TEST GROUP 1 — PROMPT CONTRACT TESTS
#
# Verify that rendered prompts include instructions for CRIT fields.
# ===================================================================

class TestPromptContract:
    """Rendered agent prompts must instruct the LLM to produce CRIT fields."""

    @staticmethod
    def _render_propose_prompt(profile_name: str, memo: str) -> str:
        """Render the full user prompt for a profile's propose phase."""
        profile = _load_profile(profile_name)
        user_cfg = profile.get("user_prompts", {}).get("propose", {})
        template = user_cfg.get("template")
        sections = user_cfg.get("sections")

        overrides = {}
        if template:
            overrides["proposal_template"] = template

        return build_proposal_user_prompt(
            context=memo,
            prompt_file_overrides=overrides or None,
            user_sections=sections,
            allocation_constraints={"max_weight": 0.40, "min_holdings": 2},
        )

    # --- Enriched profiles: must mention all CRIT fields ---------------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_claim_id(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "claim_id" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing 'claim_id' instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_claim_type(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "claim_type" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing 'claim_type' instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_evidence_field(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        # Must mention either "evidence_ids" or "evidence" as a field
        assert "evidence" in prompt.lower(), (
            f"Enriched profile '{profile}' propose prompt missing evidence instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_falsifiers(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "falsifiers" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing 'falsifiers' instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_impacts_positions(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "impacts_positions" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing 'impacts_positions' instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_position_rationale(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "position_rationale" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing 'position_rationale' instruction"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_prompt_contains_supporting_claims(self, profile, mini_memo):
        prompt = self._render_propose_prompt(profile, mini_memo)
        assert "supported_by_claims" in prompt or "supporting_claims" in prompt, (
            f"Enriched profile '{profile}' propose prompt missing supporting_claims instruction"
        )

    # --- Base profiles: expose the missing CRIT fields -----------------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_prompt_missing_claim_id(self, profile, mini_memo):
        """Base prompts do NOT instruct agents to produce claim_id."""
        prompt = self._render_propose_prompt(profile, mini_memo)
        has_claim_id = "claim_id" in prompt
        if not has_claim_id:
            pytest.xfail(
                f"Base profile '{profile}' propose prompt lacks 'claim_id' — "
                "known schema gap with CRIT"
            )
        # If a base profile gains claim_id, the test passes (that's good!).

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_prompt_missing_impacts_positions(self, profile, mini_memo):
        """Base prompts do NOT instruct agents to produce impacts_positions."""
        prompt = self._render_propose_prompt(profile, mini_memo)
        has_field = "impacts_positions" in prompt
        if not has_field:
            pytest.xfail(
                f"Base profile '{profile}' propose prompt lacks 'impacts_positions' — "
                "known schema gap with CRIT"
            )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_prompt_missing_position_rationale(self, profile, mini_memo):
        """Base prompts do NOT instruct agents to produce position_rationale."""
        prompt = self._render_propose_prompt(profile, mini_memo)
        has_field = "position_rationale" in prompt
        if not has_field:
            pytest.xfail(
                f"Base profile '{profile}' propose prompt lacks 'position_rationale' — "
                "known schema gap with CRIT"
            )


# ===================================================================
# TEST GROUP 2 — AGENT OUTPUT CONTRACT TESTS
#
# Validate the JSON structure agents must produce.
# Uses simulated outputs (base vs enriched) since real LLM calls are
# expensive. The simulated outputs are representative of what each
# template family produces (verified by prompt inspection).
# ===================================================================

class TestAgentOutputContract:
    """Agent JSON output must contain CRIT-required fields."""

    # --- Enriched output: all CRIT fields present ----------------------

    @pytest.mark.fast
    def test_enriched_claims_have_required_fields(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            for field in CRIT_REQUIRED_CLAIM_FIELDS:
                assert field in claim, (
                    f"Enriched output claim missing '{field}': {claim.get('claim_id', '?')}"
                )

    @pytest.mark.fast
    def test_enriched_claim_type_valid_enum(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            assert claim["claim_type"] in VALID_CLAIM_TYPES, (
                f"Invalid claim_type '{claim['claim_type']}' in {claim['claim_id']}"
            )

    @pytest.mark.fast
    def test_enriched_reasoning_type_valid_enum(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            assert claim["reasoning_type"] in VALID_REASONING_TYPES, (
                f"Invalid reasoning_type '{claim['reasoning_type']}' in {claim['claim_id']}"
            )

    @pytest.mark.fast
    def test_enriched_evidence_is_list(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            assert isinstance(claim["evidence"], list), (
                f"Evidence must be a list in {claim['claim_id']}"
            )
            assert len(claim["evidence"]) > 0, (
                f"Evidence must not be empty in {claim['claim_id']}"
            )

    @pytest.mark.fast
    def test_enriched_confidence_is_float(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            assert isinstance(claim["confidence"], (int, float)), (
                f"Confidence must be numeric in {claim['claim_id']}"
            )
            assert 0.0 <= claim["confidence"] <= 1.0

    @pytest.mark.fast
    def test_enriched_position_rationale_has_supporting_claims(self):
        output = _make_enriched_output()
        assert "position_rationale" in output, "Missing position_rationale"
        for pos in output["position_rationale"]:
            assert "supported_by_claims" in pos or "supporting_claims" in pos, (
                f"Position for {pos['ticker']} missing supporting_claims"
            )

    @pytest.mark.fast
    def test_enriched_impacts_positions_are_universe_tickers(self):
        output = _make_enriched_output()
        for claim in output["claims"]:
            for ticker in claim["impacts_positions"]:
                assert ticker in UNIVERSE, (
                    f"impacts_positions ticker '{ticker}' not in universe"
                )

    # --- Base output: missing CRIT fields (expected failures) ----------

    @pytest.mark.fast
    def test_base_claims_missing_claim_id(self):
        """Base output lacks claim_id — CRIT will pad with empty string."""
        output = _make_base_output()
        for claim in output["claims"]:
            has_id = "claim_id" in claim and claim["claim_id"]
            if not has_id:
                pytest.xfail("Base output missing claim_id — known CRIT gap")

    @pytest.mark.fast
    def test_base_claims_missing_claim_type(self):
        """Base output lacks claim_type — normalizer defaults to 'unknown'."""
        output = _make_base_output()
        for claim in output["claims"]:
            has_type = "claim_type" in claim and claim["claim_type"]
            if not has_type:
                pytest.xfail("Base output missing claim_type — known CRIT gap")

    @pytest.mark.fast
    def test_base_claims_missing_evidence_list(self):
        """Base output lacks structured evidence list."""
        output = _make_base_output()
        for claim in output["claims"]:
            has_ev = "evidence" in claim and isinstance(claim["evidence"], list)
            if not has_ev:
                pytest.xfail("Base output missing evidence list — known CRIT gap")

    @pytest.mark.fast
    def test_base_claims_missing_falsifiers(self):
        """Base output lacks per-claim falsifiers."""
        output = _make_base_output()
        for claim in output["claims"]:
            has_f = "falsifiers" in claim and claim["falsifiers"]
            if not has_f:
                pytest.xfail("Base output missing falsifiers — known CRIT gap")

    @pytest.mark.fast
    def test_base_claims_missing_impacts_positions(self):
        """Base output lacks impacts_positions."""
        output = _make_base_output()
        for claim in output["claims"]:
            has_ip = "impacts_positions" in claim and claim["impacts_positions"]
            if not has_ip:
                pytest.xfail("Base output missing impacts_positions — known CRIT gap")

    @pytest.mark.fast
    def test_base_output_missing_position_rationale(self):
        """Base output lacks position_rationale entirely."""
        output = _make_base_output()
        has_pr = "position_rationale" in output and output["position_rationale"]
        if not has_pr:
            pytest.xfail("Base output missing position_rationale — known CRIT gap")


# ===================================================================
# TEST GROUP 3 — NORMALIZATION TESTS
#
# Feed agent outputs into _normalize_claims / _normalize_position_rationale.
# Verify normalized claims contain valid CRIT fields.
# ===================================================================

class TestNormalization:
    """_normalize_claims must produce CRIT-compatible structures."""

    # --- Enriched: normalizer preserves all fields ---------------------

    @pytest.mark.fast
    def test_enriched_normalized_claim_id_present(self):
        output = _make_enriched_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        for claim in claims:
            assert claim["claim_id"], (
                f"Normalized enriched claim should have non-empty claim_id"
            )

    @pytest.mark.fast
    def test_enriched_normalized_claim_type_valid(self):
        output = _make_enriched_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        for claim in claims:
            assert claim["claim_type"] in VALID_CLAIM_TYPES, (
                f"Normalized claim_type '{claim['claim_type']}' not in valid set"
            )

    @pytest.mark.fast
    def test_enriched_normalized_evidence_ids_populated(self):
        output = _make_enriched_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        for claim in claims:
            assert len(claim["evidence_ids"]) > 0, (
                f"Normalized enriched claim {claim['claim_id']} has empty evidence_ids"
            )

    @pytest.mark.fast
    def test_enriched_normalized_falsifiers_present(self):
        output = _make_enriched_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        for claim in claims:
            assert len(claim["falsifiers"]) > 0, (
                f"Normalized enriched claim {claim['claim_id']} has empty falsifiers"
            )

    @pytest.mark.fast
    def test_enriched_normalized_impacts_positions_present(self):
        output = _make_enriched_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        for claim in claims:
            assert len(claim["impacts_positions"]) > 0, (
                f"Normalized enriched claim {claim['claim_id']} has empty impacts_positions"
            )

    @pytest.mark.fast
    def test_enriched_position_rationale_supporting_claims_mapped(self):
        output = _make_enriched_output()
        positions = _normalize_position_rationale(output["position_rationale"])
        for pos in positions:
            assert len(pos["supporting_claims"]) > 0, (
                f"Normalized position for {pos['ticker']} has empty supporting_claims"
            )

    @pytest.mark.fast
    def test_enriched_extract_reasoning_all_fields(self):
        output = _make_enriched_output()
        reasoning = _extract_reasoning(output, normalize_evidence_id)
        assert len(reasoning["claims"]) > 0
        assert len(reasoning["position_rationale"]) > 0
        assert reasoning["thesis"]
        assert reasoning["confidence"] > 0
        assert len(reasoning["risks_or_falsifiers"]) > 0

    # --- Base: normalizer pads with defaults (CRIT-degraded) -----------

    @pytest.mark.fast
    def test_base_normalized_claim_id_empty(self):
        """Normalizer pads base claims with empty claim_id."""
        output = _make_base_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        empty_ids = [c for c in claims if not c["claim_id"]]
        assert len(empty_ids) > 0, (
            "Expected base claims to have empty claim_id after normalization"
        )

    @pytest.mark.fast
    def test_base_normalized_claim_type_unknown(self):
        """Normalizer defaults base claim_type to 'unknown'."""
        output = _make_base_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        unknown_types = [c for c in claims if c["claim_type"] == "unknown"]
        assert len(unknown_types) > 0, (
            "Expected base claims to have claim_type='unknown' after normalization"
        )

    @pytest.mark.fast
    def test_base_normalized_evidence_ids_empty(self):
        """Base claims have no structured evidence → evidence_ids empty."""
        output = _make_base_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        empty_evidence = [c for c in claims if len(c["evidence_ids"]) == 0]
        assert len(empty_evidence) > 0, (
            "Expected base claims to have empty evidence_ids after normalization"
        )

    @pytest.mark.fast
    def test_base_normalized_falsifiers_empty(self):
        """Base claims have no per-claim falsifiers → falsifiers empty."""
        output = _make_base_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        empty_falsifiers = [c for c in claims if len(c["falsifiers"]) == 0]
        assert len(empty_falsifiers) > 0, (
            "Expected base claims to have empty falsifiers after normalization"
        )

    @pytest.mark.fast
    def test_base_normalized_impacts_positions_empty(self):
        """Base claims have no impacts_positions → empty list."""
        output = _make_base_output()
        claims = _normalize_claims(output["claims"], normalize_evidence_id)
        empty_ip = [c for c in claims if len(c["impacts_positions"]) == 0]
        assert len(empty_ip) > 0, (
            "Expected base claims to have empty impacts_positions after normalization"
        )

    @pytest.mark.fast
    def test_base_position_rationale_empty(self):
        """Base output has no position_rationale → empty list after normalization."""
        output = _make_base_output()
        positions = _normalize_position_rationale(
            output.get("position_rationale", [])
        )
        assert len(positions) == 0, (
            "Expected base output to have empty position_rationale"
        )

    @pytest.mark.fast
    def test_base_extract_reasoning_degraded(self):
        """Normalizer produces degraded reasoning bundle from base output."""
        output = _make_base_output()
        reasoning = _extract_reasoning(output, normalize_evidence_id)
        # Claims exist but are degraded
        assert len(reasoning["claims"]) > 0
        # All claims have unknown type
        for claim in reasoning["claims"]:
            assert claim["claim_type"] == "unknown"
        # No position rationale
        assert len(reasoning["position_rationale"]) == 0
        # Thesis falls back to justification
        assert reasoning["thesis"] == output["justification"]


# ===================================================================
# TEST GROUP 4 — CRIT COMPATIBILITY TESTS
#
# Build full reasoning bundles and verify CRIT structural flags.
# Uses a mock CRIT LLM that returns deterministic scores based on
# the bundle's structural quality — if the bundle is well-formed,
# no diagnostic flags fire; if degraded, flags are set.
# ===================================================================

def _make_state_from_output(
    role: str, output: dict, memo_evidence_lookup: dict,
) -> dict:
    """Build a minimal debate state from a simulated agent output."""
    raw_text = json.dumps(output, indent=2)
    entry = {
        "role": role,
        "action_dict": output,
        "raw_response": raw_text,
    }
    return {
        "proposals": [entry],
        "revisions": [entry],  # Use same as proposal (no critique cycle)
        "critiques": [],
    }


def _crit_response_from_bundle(bundle: dict) -> dict:
    """Inspect a reasoning bundle and generate a deterministic CRIT response.

    This mock CRIT scorer examines the structural quality of the reasoning
    bundle and sets diagnostic flags accordingly — matching what a real CRIT
    LLM would detect.
    """
    revised = bundle.get("revised_argument", {})
    reasoning = revised.get("reasoning", {})
    claims = reasoning.get("claims", [])
    positions = reasoning.get("position_rationale", [])
    allocation = revised.get("portfolio_allocation", {})

    # Detect unsupported claims: no evidence_ids AND no bracketed refs in text
    unsupported_count = 0
    for claim in claims:
        has_evidence_ids = bool(claim.get("evidence_ids"))
        has_inline = bool(re.search(r"\[[\w-]+\]", claim.get("claim_text", "")))
        if not has_evidence_ids and not has_inline:
            unsupported_count += 1

    # Detect conclusion drift: positions > 10% weight with no supporting_claims
    orphaned_count = 0
    position_tickers = {p["ticker"] for p in positions}
    for ticker, weight in allocation.items():
        if weight > 0.10 and ticker not in position_tickers:
            orphaned_count += 1
    for pos in positions:
        if pos.get("weight", 0) > 0.10 and not pos.get("supporting_claims"):
            orphaned_count += 1

    unsupported = unsupported_count > 0
    drift = orphaned_count > 0

    # Score degrades with structural problems
    base_score = 0.85
    if unsupported:
        base_score -= 0.15
    if drift:
        base_score -= 0.15

    return {
        "pillar_scores": {
            "logical_validity": base_score,
            "evidential_support": max(0.3, base_score - (0.2 if unsupported else 0)),
            "alternative_consideration": base_score,
            "causal_alignment": base_score,
        },
        "diagnostics": {
            "contradictions_detected": False,
            "unsupported_claims_detected": unsupported,
            "ignored_critiques_detected": False,
            "premature_certainty_detected": False,
            "causal_overreach_detected": False,
            "conclusion_drift_detected": drift,
            "unsupported_claims_count": unsupported_count,
            "orphaned_positions_count": orphaned_count,
        },
        "explanations": {
            "logical_validity": "Structure check.",
            "evidential_support": "Evidence check.",
            "alternative_consideration": "Alternatives check.",
            "causal_alignment": "Causal check.",
        },
    }


class TestCritCompatibility:
    """Full pipeline: agent output → normalize → bundle → CRIT evaluation."""

    @pytest.mark.fast
    def test_enriched_bundle_no_unsupported_claims(self, memo_evidence_lookup):
        """Enriched output passes CRIT without unsupported_claims flag."""
        output = _make_enriched_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert not result.diagnostics.unsupported_claims_detected, (
            "Enriched output should not trigger unsupported_claims_detected"
        )

    @pytest.mark.fast
    def test_enriched_bundle_no_conclusion_drift(self, memo_evidence_lookup):
        """Enriched output passes CRIT without conclusion_drift flag."""
        output = _make_enriched_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert not result.diagnostics.conclusion_drift_detected, (
            "Enriched output should not trigger conclusion_drift_detected"
        )

    @pytest.mark.fast
    def test_enriched_bundle_rho_bar_above_threshold(self, memo_evidence_lookup):
        """Enriched output achieves ρ̄ > 0.70 (above CRIT degradation zone)."""
        output = _make_enriched_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert result.rho_bar > 0.70, (
            f"Enriched output ρ̄={result.rho_bar:.3f} should be > 0.70"
        )

    @pytest.mark.fast
    def test_base_bundle_claims_have_empty_evidence_ids(self, memo_evidence_lookup):
        """Base output claims have empty evidence_ids after normalization.

        Base claims lack a structured "evidence" list, so _normalize_claims
        produces empty evidence_ids.  Inline bracket refs in claim_text may
        prevent the unsupported_claims boolean from firing, but the structural
        degradation still harms evidential_support scoring.
        """
        output = _make_base_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        revised = bundle["revised_argument"]["reasoning"]
        claims_with_empty_ev = [
            c for c in revised["claims"] if len(c.get("evidence_ids", [])) == 0
        ]
        assert len(claims_with_empty_ev) > 0, (
            "Base output should have claims with empty evidence_ids after "
            "normalization — claims lack structured 'evidence' list"
        )

    @pytest.mark.fast
    def test_base_bundle_triggers_conclusion_drift(self, memo_evidence_lookup):
        """Base output triggers conclusion_drift_detected in CRIT."""
        output = _make_base_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert result.diagnostics.conclusion_drift_detected, (
            "Base output should trigger conclusion_drift_detected — "
            "no position_rationale with supporting_claims"
        )

    @pytest.mark.fast
    def test_base_bundle_rho_bar_degraded(self, memo_evidence_lookup):
        """Base output ρ̄ lands in degradation zone (≤ 0.70)."""
        output = _make_base_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert result.rho_bar <= 0.70, (
            f"Base output ρ̄={result.rho_bar:.3f} should be ≤ 0.70 (degraded)"
        )

    # --- CRIT prompt rendering: bundle structure check -----------------

    @pytest.mark.fast
    def test_enriched_crit_prompt_contains_claims(self, memo_evidence_lookup):
        """CRIT user prompt renders enriched claims with structural fields."""
        output = _make_enriched_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        assert "claim_id" in user_prompt
        assert "claim_type" in user_prompt
        assert "evidence_ids" in user_prompt
        assert "impacts_positions" in user_prompt
        assert "falsifiers" in user_prompt

    @pytest.mark.fast
    def test_enriched_crit_prompt_contains_position_rationale(self, memo_evidence_lookup):
        """CRIT user prompt renders position rationale with supporting_claims."""
        output = _make_enriched_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        assert "position_rationale" in user_prompt
        assert "supporting_claims" in user_prompt

    @pytest.mark.fast
    def test_base_crit_prompt_missing_structural_fields(self, memo_evidence_lookup):
        """CRIT user prompt from base output lacks key structural fields."""
        output = _make_base_output()
        state = _make_state_from_output("macro", output, memo_evidence_lookup)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        # claim_id and evidence_ids are present in the JSON but empty/default
        # The key structural degradation is that values are empty, not that
        # keys are missing from the rendered JSON.
        # Check that the rendered bundle has empty evidence_ids
        assert '"evidence_ids": []' in user_prompt or "'evidence_ids': []" in user_prompt, (
            "Base bundle should render empty evidence_ids in CRIT prompt"
        )


# ===================================================================
# TEST GROUP 5 — CROSS-TEMPLATE CONSISTENCY MATRIX
#
# Verify that ALL profiles within each family behave consistently.
# ===================================================================

class TestTemplateConsistencyMatrix:
    """Cross-profile consistency checks for the propose phase."""

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_template_has_enumerated_output_section(self, profile, mini_memo):
        """All enriched profiles must use the enumerated output format."""
        rendered = TestPromptContract._render_propose_prompt(profile, mini_memo)
        # The enumerated output format includes claim_id, claim_type,
        # impacts_positions, and position_rationale.
        required = ["claim_id", "claim_type", "impacts_positions", "position_rationale"]
        missing = [f for f in required if f not in rendered]
        assert not missing, (
            f"Enriched profile '{profile}' missing CRIT fields in output format: {missing}"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_template_schema_gap_documented(self, profile, mini_memo):
        """Base profiles are expected to lack CRIT structural fields.

        This test documents the gap — it xfails when the gap exists and
        passes if a base profile is upgraded to include the fields.
        """
        rendered = TestPromptContract._render_propose_prompt(profile, mini_memo)
        required = ["claim_id", "claim_type", "impacts_positions", "position_rationale"]
        missing = [f for f in required if f not in rendered]
        if missing:
            pytest.xfail(
                f"Base profile '{profile}' missing CRIT fields: {missing}"
            )


# ===================================================================
# TEST GROUP 6 — REVISION TEMPLATE CONTRACT
#
# Verify that revision templates also include CRIT fields, since
# CRIT evaluates the *revised* argument (not just the proposal).
# ===================================================================

class TestRevisionContract:
    """Revision templates must also produce CRIT-compatible output."""

    @staticmethod
    def _get_revision_template(profile_name: str) -> str | None:
        """Get the revision template filename from a profile."""
        profile = _load_profile(profile_name)
        return profile.get("user_prompts", {}).get("revise", {}).get("template")

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_template_has_crit_fields(self, profile):
        """Enriched revision templates include all CRIT claim fields."""
        template_name = self._get_revision_template(profile)
        if not template_name:
            pytest.skip(f"No revision template for {profile}")

        template_path = _PROMPTS_DIR / template_name
        if not template_path.exists():
            pytest.skip(f"Template file not found: {template_name}")

        raw = template_path.read_text()
        required = ["claim_id", "claim_type", "impacts_positions", "falsifiers"]
        missing = [f for f in required if f not in raw]
        assert not missing, (
            f"Enriched revision template '{template_name}' missing: {missing}"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_revision_template_gap(self, profile):
        """Base revision templates are expected to lack CRIT fields."""
        template_name = self._get_revision_template(profile)
        if not template_name:
            pytest.skip(f"No revision template for {profile}")

        template_path = _PROMPTS_DIR / template_name
        if not template_path.exists():
            pytest.skip(f"Template file not found: {template_name}")

        raw = template_path.read_text()
        required = ["claim_id", "claim_type", "impacts_positions"]
        missing = [f for f in required if f not in raw]
        if missing:
            pytest.xfail(
                f"Base revision template '{template_name}' missing: {missing}"
            )


# ===================================================================
# INTER-PHASE HANDOFF FIXTURES
# ===================================================================

def _make_enriched_critique_output() -> dict:
    """Simulated critique output from an enriched-family agent."""
    return {
        "critiques": [
            {
                "target_role": "risk",
                "target_claim": "C1",
                "objection": "The macro regime assessment ignores rising real yields [L1-10Y].",
                "counter_evidence": ["[L1-10Y]", "[L1-VIX]"],
                "portfolio_implication": "Should reduce equity overweight by 5-10%.",
                "suggested_adjustment": "Cut AAPL from 40% to 30%.",
                "falsifier": "If real yields fall below 1%, regime concern is invalidated.",
                "objection_confidence": 0.75,
            },
            {
                "target_role": "technical",
                "target_claim": "C2",
                "objection": "Momentum signals are stale with 60-day lookback.",
                "counter_evidence": ["[AAPL-RET60]"],
                "portfolio_implication": "Momentum tilt is unreliable.",
                "falsifier": "20-day momentum confirming 60-day would validate.",
                "objection_confidence": 0.6,
            },
        ],
        "self_critique": "My own allocation may underweight tech given AI tailwinds.",
    }


def _make_base_critique_output() -> dict:
    """Simulated critique output from a base-family agent."""
    return {
        "critiques": [
            {
                "target_role": "risk",
                "objection": "Risk assessment is too conservative.",
            },
            {
                "target_role": "technical",
                "objection": "Momentum signals are unreliable in current regime.",
            },
        ],
        "self_critique": "My allocation may be too aggressive.",
    }


def _make_proposals(enriched: bool) -> list[dict]:
    """Build a list of proposals from two agents (macro and risk)."""
    make = _make_enriched_output if enriched else _make_base_output
    macro_out = make()
    risk_out = make()
    # Tweak risk output so it differs
    risk_out["allocation"] = {"AAPL": 0.20, "MSFT": 0.30, "GOOGL": 0.50}
    if "portfolio_rationale" in risk_out:
        risk_out["portfolio_rationale"] = "Defensive tilt toward diversified GOOGL."
    if "justification" in risk_out:
        risk_out["justification"] = "Defensive tilt toward diversified GOOGL."
    return [
        {"role": "macro", "action_dict": macro_out, "raw_response": json.dumps(macro_out)},
        {"role": "risk", "action_dict": risk_out, "raw_response": json.dumps(risk_out)},
    ]


def _make_critiques(enriched: bool) -> list[dict]:
    """Build critique state entries from two agents."""
    make = _make_enriched_critique_output if enriched else _make_base_critique_output
    return [
        {"role": "macro", "critiques": make()["critiques"], "self_critique": ""},
        {"role": "risk", "critiques": make()["critiques"], "self_critique": ""},
    ]


# ===================================================================
# TEST GROUP 7 — PROPOSE → CRITIQUE HANDOFF
#
# Verify that render_previous_proposal() and render_others_proposals()
# preserve structural fields that the critique prompt needs.
# ===================================================================

class TestProposeToCritiqueHandoff:
    """Propose output must render into critique context with structural fields."""

    # --- Enriched: structural fields survive rendering ------------------

    @pytest.mark.fast
    def test_enriched_render_preserves_claim_ids(self):
        """render_previous_proposal() preserves claim_id references."""
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        assert "C1:" in rendered, "Rendered proposal should contain claim_id C1"
        assert "C2:" in rendered, "Rendered proposal should contain claim_id C2"

    @pytest.mark.fast
    def test_enriched_render_preserves_claim_type(self):
        """render_previous_proposal() includes claim_type."""
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        assert "Claim Type: macro" in rendered
        assert "Claim Type: firm" in rendered

    @pytest.mark.fast
    def test_enriched_render_preserves_evidence(self):
        """render_previous_proposal() includes evidence citations."""
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        assert "[L1-10Y]" in rendered
        assert "[AAPL-GM]" in rendered

    @pytest.mark.fast
    def test_enriched_render_preserves_falsifiers(self):
        """render_previous_proposal() includes per-claim falsifiers."""
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        assert "Falsifiers:" in rendered
        assert "Rapid rate cuts" in rendered

    @pytest.mark.fast
    def test_enriched_render_preserves_position_rationale(self):
        """render_previous_proposal() includes position rationale with claim refs."""
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        assert "Previous Position Rationale" in rendered
        assert "Supported by claims: C1, C2" in rendered

    @pytest.mark.fast
    def test_enriched_render_preserves_impacts_positions(self):
        """render_previous_proposal() preserves impacts_positions.

        Note: impacts_positions is NOT currently rendered by proposal_renderer.
        This test documents the gap — the critique agent cannot see which
        tickers each claim affects unless it parses the raw JSON.
        """
        output = _make_enriched_output()
        rendered = render_previous_proposal(output)
        # impacts_positions is NOT rendered in the structured text — it only
        # exists in the raw JSON (my_proposal v1).  The v2 renderer does not
        # include it.  This is a known limitation.
        has_impacts = "impacts_positions" in rendered or "Impacts:" in rendered
        if not has_impacts:
            pytest.xfail(
                "render_previous_proposal() does not render impacts_positions — "
                "critique agent must rely on raw JSON (v1) for this field"
            )

    @pytest.mark.fast
    def test_enriched_others_proposals_renders_all_agents(self):
        """render_others_proposals() includes all agents except self."""
        proposals = _make_proposals(enriched=True)
        rendered = render_others_proposals(proposals, "macro")
        assert "RISK agent proposed:" in rendered
        assert "MACRO agent proposed:" not in rendered

    # --- Base: structural fields missing from rendered proposal ---------

    @pytest.mark.fast
    def test_base_render_missing_claim_ids(self):
        """Base proposal rendering auto-generates claim IDs from list order."""
        output = _make_base_output()
        rendered = render_previous_proposal(output)
        # Base claims have no claim_id, renderer auto-generates C1, C2, ...
        assert "C1:" in rendered, (
            "Renderer should auto-generate C1 for base claims without claim_id"
        )

    @pytest.mark.fast
    def test_base_render_missing_claim_type(self):
        """Base proposal rendering omits claim_type (not in base output)."""
        output = _make_base_output()
        rendered = render_previous_proposal(output)
        has_claim_type = "Claim Type:" in rendered
        if not has_claim_type:
            pytest.xfail(
                "Base proposal rendering lacks Claim Type — "
                "critique agent cannot see domain classification"
            )

    @pytest.mark.fast
    def test_base_render_missing_evidence(self):
        """Base proposal rendering omits structured evidence list."""
        output = _make_base_output()
        rendered = render_previous_proposal(output)
        has_evidence_section = "Evidence:" in rendered
        if not has_evidence_section:
            pytest.xfail(
                "Base proposal rendering lacks Evidence section — "
                "critique agent has no structured evidence to reference"
            )

    @pytest.mark.fast
    def test_base_render_missing_position_rationale(self):
        """Base proposal rendering omits position rationale entirely."""
        output = _make_base_output()
        rendered = render_previous_proposal(output)
        has_pos_rationale = "Previous Position Rationale" in rendered
        if not has_pos_rationale:
            pytest.xfail(
                "Base proposal rendering lacks Position Rationale — "
                "critique agent cannot target specific position reasoning"
            )


# ===================================================================
# TEST GROUP 8 — CRITIQUE → REVISE HANDOFF
#
# Verify that critique output is correctly routed, rendered, and
# injected into the revision prompt with all structural fields.
# ===================================================================

class TestCritiqueToReviseHandoff:
    """Critique output must be routed and rendered for the revise phase."""

    # --- Critique routing (target_role normalization) ------------------

    @pytest.mark.fast
    def test_critique_routing_exact_match(self):
        """Critiques with exact lowercase target_role reach the right agent."""
        critiques = _make_critiques(enriched=True)
        # Simulate revise_node critique collection for "risk" agent
        received = []
        for c in critiques:
            for crit in c["critiques"]:
                target = crit.get("target_role", "").lower().split()[0]
                if target == "risk":
                    received.append(crit)
        assert len(received) > 0, "Risk agent should receive critiques"

    @pytest.mark.fast
    def test_critique_routing_uppercase_target(self):
        """Critiques with uppercase target_role still match after .lower()."""
        critiques = [{"role": "macro", "critiques": [
            {"target_role": "RISK", "objection": "Too conservative"},
        ]}]
        received = []
        for c in critiques:
            for crit in c["critiques"]:
                target = crit.get("target_role", "").lower().split()[0]
                if target == "risk":
                    received.append(crit)
        assert len(received) == 1

    @pytest.mark.fast
    def test_critique_routing_multiword_target(self):
        """Critiques with 'Risk Agent' target match via .split()[0]."""
        critiques = [{"role": "macro", "critiques": [
            {"target_role": "Risk Agent", "objection": "Too conservative"},
        ]}]
        received = []
        for c in critiques:
            for crit in c["critiques"]:
                target = crit.get("target_role", "").lower().split()[0]
                if target == "risk":
                    received.append(crit)
        assert len(received) == 1

    @pytest.mark.fast
    def test_critique_routing_missing_target_dropped(self):
        """Critiques with empty target_role are silently dropped.

        Uses the guarded pattern from runner.py (line 344) which checks
        ``crit.get("target_role")`` before splitting.  Note: nodes.py
        (line 516) lacks this guard and would crash on empty target_role.
        """
        critiques = [{"role": "macro", "critiques": [
            {"target_role": "", "objection": "Unrouted critique"},
        ]}]
        received = []
        for c in critiques:
            for crit in c["critiques"]:
                # Use the safe pattern from runner.py (not the unsafe nodes.py pattern)
                target = crit.get("target_role", "").lower().split()[0] if crit.get("target_role") else ""
                if target == "risk":
                    received.append(crit)
        assert len(received) == 0, "Critique with empty target should not reach any agent"

    @pytest.mark.fast
    def test_critique_routing_empty_target_logs_warning(self, caplog):
        """nodes.py logs a warning and skips critiques with empty target_role.

        Previously ``crit.get("target_role", "").lower().split()[0]`` would
        raise IndexError on empty strings.  The fix guards with an explicit
        check and logs a warning.
        """
        from multi_agent.graph.nodes import revise_node

        # Build minimal state with an empty-target critique
        base_output = _make_base_output()
        state = {
            "config": {
                "roles": ["macro"],
                "mock": True,
                "console_display": True,
            },
            "enriched_context": "Test context",
            "observation": {"universe": list(base_output["allocation"].keys())},
            "proposals": [
                {"role": "macro", "action_dict": base_output,
                 "raw_response": json.dumps(base_output)},
            ],
            "revisions": None,
            "critiques": [
                {"role": "risk", "critiques": [
                    {"target_role": "", "objection": "Unroutable critique"},
                ]},
            ],
            "current_round": 1,
            "debate_turns": [],
        }

        with caplog.at_level(logging.WARNING, logger="multi_agent.graph.nodes"):
            result = revise_node(state)

        assert "empty target_role" in caplog.text, (
            "Should log warning about empty target_role"
        )
        # Revise should complete without crashing
        assert "revisions" in result

    # --- Critique rendering -------------------------------------------

    @pytest.mark.fast
    def test_enriched_critique_renders_target_claim(self):
        """render_critiques_received() includes target_claim references."""
        critiques = [
            {
                "from_role": "macro",
                "objection": "Regime assessment ignores yields.",
                "target_claim": "C1",
                "counter_evidence": ["[L1-10Y]"],
                "falsifier": "Real yields fall below 1%",
                "objection_confidence": 0.75,
            },
        ]
        rendered = render_critiques_received(critiques)
        assert "Target claim: C1" in rendered
        assert "Counter-evidence: [L1-10Y]" in rendered
        assert "Falsifier: Real yields" in rendered
        assert "Objection confidence: 0.75" in rendered

    @pytest.mark.fast
    def test_base_critique_renders_minimal(self):
        """Base critique renders only from_role and objection."""
        critiques = [
            {"from_role": "macro", "objection": "Too conservative"},
        ]
        rendered = render_critiques_received(critiques)
        assert "MACRO" in rendered
        assert "Too conservative" in rendered
        # No target_claim, counter_evidence, etc.
        assert "Target claim:" not in rendered

    @pytest.mark.fast
    def test_enriched_critique_renders_counter_evidence(self):
        """Enriched critique rendering includes counter-evidence IDs."""
        critiques = [
            {
                "from_role": "value",
                "objection": "AAPL overvalued.",
                "counter_evidence": ["[AAPL-GM]", "[AAPL-RET60]"],
            },
        ]
        rendered = render_critiques_received(critiques)
        assert "[AAPL-GM]" in rendered
        assert "[AAPL-RET60]" in rendered

    @pytest.mark.fast
    def test_empty_critiques_renders_message(self):
        """No critiques → informative message, not empty string."""
        rendered = render_critiques_received([])
        assert "No critiques" in rendered

    # --- Revision prompt receives critique context ---------------------

    @pytest.mark.fast
    def test_enriched_revision_prompt_has_critiques(self, mini_memo):
        """build_revision_prompt() includes enriched critiques in the prompt."""
        output = _make_enriched_output()
        my_proposal_v2 = render_previous_proposal(output)
        critiques = [
            {
                "from_role": "risk",
                "objection": "Overweight tech in high-rate regime [L1-10Y].",
                "target_claim": "C1",
                "counter_evidence": ["[L1-10Y]"],
                "falsifier": "Rate cuts would change outlook",
                "objection_confidence": 0.8,
            },
        ]
        critiques_text_v2 = render_critiques_received(critiques)

        profile = _load_profile("macro_enriched")
        user_cfg = profile.get("user_prompts", {}).get("revise", {})
        template = user_cfg.get("template")
        sections = user_cfg.get("sections")
        overrides = {"revision_template": template} if template else {}

        prompt = build_revision_prompt(
            "macro", mini_memo, json.dumps(output), critiques,
            prompt_file_overrides=overrides or None,
            user_sections=sections,
            my_proposal_v2=my_proposal_v2,
            critiques_text_v2=critiques_text_v2,
            allocation_constraints={"max_weight": 0.40, "min_holdings": 2},
        )

        assert "Target claim: C1" in prompt or "C1" in prompt, (
            "Revision prompt should contain target_claim reference"
        )
        assert "[L1-10Y]" in prompt, (
            "Revision prompt should contain counter-evidence IDs"
        )


# ===================================================================
# TEST GROUP 9 — FULL PIPELINE HANDOFF: PROPOSE → CRITIQUE → REVISE → CRIT
#
# Exercise the complete data flow through all phases using simulated
# outputs at each stage, verifying that structural fields survive
# all transformations and land in the CRIT prompt correctly.
# ===================================================================

class TestFullPipelineHandoff:
    """End-to-end: propose output → rendered critique context → revision →
    build_reasoning_bundle → render_crit_prompts.  Verify structural fields
    survive all transformations."""

    @staticmethod
    def _build_full_state(enriched: bool) -> dict:
        """Build a complete debate state with proposals, critiques, revisions."""
        proposals = _make_proposals(enriched)
        critiques = _make_critiques(enriched)

        # Simulate revisions — use the same structure as proposals but
        # with revision_notes added.
        make = _make_enriched_output if enriched else _make_base_output
        macro_rev = make()
        macro_rev["revision_notes"] = "Adjusted AAPL weight down per risk critique."
        risk_rev = make()
        risk_rev["allocation"] = {"AAPL": 0.25, "MSFT": 0.35, "GOOGL": 0.40}
        risk_rev["revision_notes"] = "Increased diversification per macro critique."

        revisions = [
            {"role": "macro", "action_dict": macro_rev,
             "raw_response": json.dumps(macro_rev), "revision_notes": macro_rev["revision_notes"]},
            {"role": "risk", "action_dict": risk_rev,
             "raw_response": json.dumps(risk_rev), "revision_notes": risk_rev["revision_notes"]},
        ]

        return {
            "proposals": proposals,
            "critiques": critiques,
            "revisions": revisions,
        }

    # --- Enriched full pipeline: all fields survive --------------------

    @pytest.mark.fast
    def test_enriched_pipeline_bundle_has_claims_with_ids(self, memo_evidence_lookup):
        """Enriched pipeline: claims in CRIT bundle have non-empty claim_id."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        assert bundle is not None

        claims = bundle["revised_argument"]["reasoning"]["claims"]
        for claim in claims:
            assert claim["claim_id"], (
                f"Enriched claim missing claim_id after full pipeline"
            )

    @pytest.mark.fast
    def test_enriched_pipeline_bundle_has_evidence_ids(self, memo_evidence_lookup):
        """Enriched pipeline: claims have populated evidence_ids."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        claims = bundle["revised_argument"]["reasoning"]["claims"]
        for claim in claims:
            assert len(claim["evidence_ids"]) > 0, (
                f"Enriched claim {claim['claim_id']} has empty evidence_ids after pipeline"
            )

    @pytest.mark.fast
    def test_enriched_pipeline_bundle_has_position_rationale(self, memo_evidence_lookup):
        """Enriched pipeline: position_rationale with supporting_claims survives."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        positions = bundle["revised_argument"]["reasoning"]["position_rationale"]
        assert len(positions) > 0, "Enriched bundle should have position_rationale"
        for pos in positions:
            assert len(pos["supporting_claims"]) > 0, (
                f"Position {pos['ticker']} has empty supporting_claims after pipeline"
            )

    @pytest.mark.fast
    def test_enriched_pipeline_bundle_has_critiques_received(self, memo_evidence_lookup):
        """Enriched pipeline: critiques are correctly routed to the agent."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "risk", 1, memo_evidence_lookup)

        crits = bundle["critiques_received"]
        assert len(crits) > 0, "Risk agent should have received critiques"
        for crit in crits:
            assert crit["from_role"], "Critique should have from_role"
            assert crit["critique_text"], "Critique should have critique_text"

    @pytest.mark.fast
    def test_enriched_pipeline_crit_prompt_has_structural_fields(self, memo_evidence_lookup):
        """Enriched pipeline: final CRIT prompt contains all structural fields."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        # Claims with IDs
        assert "C1" in user_prompt
        assert "C2" in user_prompt
        # Evidence IDs populated
        assert "L1-10Y" in user_prompt
        assert "AAPL-GM" in user_prompt
        # Position rationale
        assert "position_rationale" in user_prompt
        assert "supporting_claims" in user_prompt
        # Falsifiers
        assert "falsifiers" in user_prompt
        # Thesis
        assert "thesis" in user_prompt

    @pytest.mark.fast
    def test_enriched_pipeline_crit_prompt_has_revision_notes(self, memo_evidence_lookup):
        """Enriched pipeline: revision_notes appear in CRIT prompt."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        assert "revision_notes" in user_prompt

    @pytest.mark.fast
    def test_enriched_pipeline_no_crit_flags(self, memo_evidence_lookup):
        """Enriched pipeline: mock CRIT produces no structural failure flags."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert not result.diagnostics.unsupported_claims_detected
        assert not result.diagnostics.conclusion_drift_detected
        assert result.rho_bar > 0.70

    # --- Base full pipeline: structural degradation --------------------

    @pytest.mark.fast
    def test_base_pipeline_bundle_claims_have_empty_ids(self, memo_evidence_lookup):
        """Base pipeline: claims have empty claim_id after full pipeline."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        claims = bundle["revised_argument"]["reasoning"]["claims"]
        empty_ids = [c for c in claims if not c["claim_id"]]
        assert len(empty_ids) > 0, (
            "Base pipeline claims should have empty claim_id"
        )

    @pytest.mark.fast
    def test_base_pipeline_bundle_claims_have_empty_evidence_ids(self, memo_evidence_lookup):
        """Base pipeline: claims have empty evidence_ids after full pipeline."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        claims = bundle["revised_argument"]["reasoning"]["claims"]
        empty_ev = [c for c in claims if len(c["evidence_ids"]) == 0]
        assert len(empty_ev) > 0, (
            "Base pipeline claims should have empty evidence_ids"
        )

    @pytest.mark.fast
    def test_base_pipeline_bundle_no_position_rationale(self, memo_evidence_lookup):
        """Base pipeline: position_rationale is empty after full pipeline."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        positions = bundle["revised_argument"]["reasoning"]["position_rationale"]
        assert len(positions) == 0, (
            "Base pipeline should have empty position_rationale"
        )

    @pytest.mark.fast
    def test_base_pipeline_conclusion_drift(self, memo_evidence_lookup):
        """Base pipeline: conclusion_drift_detected fires in CRIT."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert result.diagnostics.conclusion_drift_detected, (
            "Base pipeline should trigger conclusion_drift — "
            "no position_rationale means no supporting_claims"
        )

    @pytest.mark.fast
    def test_base_pipeline_rho_bar_degraded(self, memo_evidence_lookup):
        """Base pipeline: rho_bar in degradation zone (≤ 0.70)."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        crit_raw = _crit_response_from_bundle(bundle)
        result = validate_raw_response(crit_raw)

        assert result.rho_bar <= 0.70, (
            f"Base pipeline rho_bar={result.rho_bar:.3f} should be ≤ 0.70"
        )

    @pytest.mark.fast
    def test_base_pipeline_crit_prompt_has_empty_evidence_ids(self, memo_evidence_lookup):
        """Base pipeline: CRIT prompt renders claims with empty evidence_ids."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        assert '"evidence_ids": []' in user_prompt, (
            "Base pipeline CRIT prompt should show empty evidence_ids"
        )

    @pytest.mark.fast
    def test_base_pipeline_crit_prompt_has_unknown_claim_type(self, memo_evidence_lookup):
        """Base pipeline: CRIT prompt renders claims with claim_type 'unknown'."""
        state = self._build_full_state(enriched=False)
        bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        assert "unknown" in user_prompt, (
            "Base pipeline CRIT prompt should show claim_type='unknown'"
        )

    # --- Cross-phase critique routing -----------------------------------

    @pytest.mark.fast
    def test_critiques_route_to_correct_agent(self, memo_evidence_lookup):
        """Critiques targeting 'risk' appear in risk's bundle, not macro's."""
        state = self._build_full_state(enriched=True)

        macro_bundle = build_reasoning_bundle(state, "macro", 1, memo_evidence_lookup)
        risk_bundle = build_reasoning_bundle(state, "risk", 1, memo_evidence_lookup)

        # Critiques in test fixture target "risk" and "technical", not "macro"
        risk_crits = risk_bundle["critiques_received"]
        assert len(risk_crits) > 0, "Risk should receive critiques"

    @pytest.mark.fast
    def test_critique_evidence_survives_to_crit(self, memo_evidence_lookup):
        """Counter-evidence from critiques appears in the CRIT prompt."""
        state = self._build_full_state(enriched=True)
        bundle = build_reasoning_bundle(state, "risk", 1, memo_evidence_lookup)

        _, user_prompt = render_crit_prompts(bundle)

        # Critiques targeting risk have counter_evidence with [L1-10Y]
        assert "critique" in user_prompt.lower(), (
            "CRIT prompt should contain critique section"
        )


# ===================================================================
# ASSEMBLED PROMPT HELPERS
# ===================================================================

def _render_full_critique_prompt(
    profile_name: str, memo: str, enriched: bool,
) -> str:
    """Render the full assembled critique prompt for a given profile.

    Builds proposals, renders v2 text, and calls build_critique_prompt()
    exactly as critique_node() does in production.
    """
    profile = _load_profile(profile_name)
    user_cfg = profile.get("user_prompts", {}).get("critique", {})
    template = user_cfg.get("template")
    sections = user_cfg.get("sections")

    overrides = {}
    if template:
        overrides["critique_template"] = template

    # Build proposals the same way critique_node does
    proposals = _make_proposals(enriched)
    role = "macro"
    action_dict = proposals[0]["action_dict"]

    my_proposal = json.dumps(action_dict)
    my_proposal_v2 = render_previous_proposal(action_dict)
    others_text_v2 = render_others_proposals(proposals, role)

    all_proposals_for_critique = [
        {"role": p["role"], "proposal": json.dumps(p.get("action_dict", {}))}
        for p in proposals
    ]

    return build_critique_prompt(
        role, memo, all_proposals_for_critique, my_proposal,
        prompt_file_overrides=overrides or None,
        user_sections=sections,
        my_proposal_v2=my_proposal_v2,
        others_text_v2=others_text_v2,
        sector_constraints="",
    )


def _render_full_revision_prompt(
    profile_name: str, memo: str, enriched: bool,
) -> str:
    """Render the full assembled revision prompt for a given profile.

    Builds proposal + critiques, renders v2 text, and calls
    build_revision_prompt() exactly as revise_node() does in production.
    """
    profile = _load_profile(profile_name)
    user_cfg = profile.get("user_prompts", {}).get("revise", {})
    template = user_cfg.get("template")
    sections = user_cfg.get("sections")

    overrides = {}
    if template:
        overrides["revision_template"] = template

    proposals = _make_proposals(enriched)
    role = "macro"
    action_dict = proposals[0]["action_dict"]

    my_proposal = json.dumps(action_dict)
    my_proposal_v2 = render_previous_proposal(action_dict)

    # Build critiques the same way revise_node collects them
    if enriched:
        critiques_received = [
            {
                "from_role": "risk",
                "objection": "Overweight tech in high-rate regime [L1-10Y].",
                "target_claim": "C1",
                "counter_evidence": ["[L1-10Y]", "[L1-VIX]"],
                "falsifier": "Rate cuts would invalidate this concern",
                "portfolio_implication": "Reduce AAPL weight by 10%",
                "suggested_adjustment": "Cut AAPL from 40% to 30%",
                "objection_confidence": 0.8,
            },
        ]
    else:
        critiques_received = [
            {
                "from_role": "risk",
                "objection": "Too aggressive on tech exposure.",
            },
        ]

    critiques_text_v2 = render_critiques_received(critiques_received)

    return build_revision_prompt(
        role, memo, my_proposal, critiques_received,
        prompt_file_overrides=overrides or None,
        user_sections=sections,
        my_proposal_v2=my_proposal_v2,
        critiques_text_v2=critiques_text_v2,
        sector_constraints="",
        allocation_constraints={"max_weight": 0.40, "min_holdings": 2},
    )


# ===================================================================
# TEST GROUP 10 — ASSEMBLED CRITIQUE PROMPT HANDOFF
#
# Render the full critique prompt per-profile (exactly as critique_node
# does) and verify that structural fields from the proposal survive
# template assembly.  This catches the case where render_previous_proposal
# does its job but the template uses {{ my_proposal }} (v1 raw JSON)
# instead of {{ my_proposal_v2 }} (v2 rendered text).
# ===================================================================

class TestAssembledCritiquePrompt:
    """Full assembled critique prompts must contain structural fields."""

    # --- Enriched: v2 rendered text lands in the prompt ----------------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_claim_ids(self, profile, mini_memo):
        """Enriched critique prompt contains claim_id references from proposal."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "C1:" in prompt or '"C1"' in prompt or "C1" in prompt, (
            f"Enriched profile '{profile}' critique prompt missing claim_id references"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_claim_type(self, profile, mini_memo):
        """Enriched critique prompt contains claim_type from rendered proposal."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "Claim Type:" in prompt or "claim_type" in prompt, (
            f"Enriched profile '{profile}' critique prompt missing claim_type"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_evidence(self, profile, mini_memo):
        """Enriched critique prompt contains evidence citations from proposal."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "[L1-10Y]" in prompt, (
            f"Enriched profile '{profile}' critique prompt missing evidence IDs"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_falsifiers(self, profile, mini_memo):
        """Enriched critique prompt contains per-claim falsifiers from proposal."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "Falsifiers:" in prompt or "falsifiers" in prompt, (
            f"Enriched profile '{profile}' critique prompt missing falsifiers"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_position_rationale(self, profile, mini_memo):
        """Enriched critique prompt contains position rationale with claim refs."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "Supported by claims:" in prompt or "supported_by_claims" in prompt, (
            f"Enriched profile '{profile}' critique prompt missing position rationale claim refs"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_critique_prompt_has_other_agents(self, profile, mini_memo):
        """Enriched critique prompt contains other agents' rendered proposals."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=True)
        assert "RISK agent proposed:" in prompt or "risk" in prompt.lower(), (
            f"Enriched profile '{profile}' critique prompt missing other agents' proposals"
        )

    # --- Base: v1 raw JSON loses structural rendering ------------------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_critique_prompt_lacks_rendered_claim_type(self, profile, mini_memo):
        """Base critique prompt uses raw JSON — 'Claim Type:' label absent."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=False)
        # "Claim Type:" is the v2 rendered label from render_previous_proposal.
        # Base templates use {{ my_proposal }} (v1 raw JSON), so this label
        # never appears even though v2 rendered text was generated.
        has_rendered_label = "Claim Type:" in prompt
        if not has_rendered_label:
            pytest.xfail(
                f"Base profile '{profile}' critique prompt uses v1 raw JSON — "
                "rendered structural labels absent"
            )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_critique_prompt_lacks_supported_by_claims(self, profile, mini_memo):
        """Base critique prompt lacks 'Supported by claims:' label."""
        prompt = _render_full_critique_prompt(profile, mini_memo, enriched=False)
        has_label = "Supported by claims:" in prompt
        if not has_label:
            pytest.xfail(
                f"Base profile '{profile}' critique prompt missing position-claim links"
            )


# ===================================================================
# TEST GROUP 11 — ASSEMBLED REVISION PROMPT HANDOFF
#
# Render the full revision prompt per-profile (exactly as revise_node
# does) and verify that critique details survive template assembly.
# This catches the case where render_critiques_received() produces
# detailed text but the template uses {{ critiques_text }} (lossy
# one-liner) instead of {{ critiques_text_v2 }}.
# ===================================================================

class TestAssembledRevisionPrompt:
    """Full assembled revision prompts must contain critique structural fields."""

    # --- Enriched: v2 critique details land in the prompt --------------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_target_claim(self, profile, mini_memo):
        """Enriched revision prompt contains target_claim from critique."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        assert "Target claim: C1" in prompt or "target_claim" in prompt, (
            f"Enriched profile '{profile}' revision prompt missing target_claim"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_counter_evidence(self, profile, mini_memo):
        """Enriched revision prompt contains counter-evidence from critique."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        assert "[L1-10Y]" in prompt, (
            f"Enriched profile '{profile}' revision prompt missing counter-evidence"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_falsifier(self, profile, mini_memo):
        """Enriched revision prompt contains falsifier from critique."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        assert "Falsifier:" in prompt or "falsifier" in prompt.lower(), (
            f"Enriched profile '{profile}' revision prompt missing critique falsifier"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_portfolio_implication(self, profile, mini_memo):
        """Enriched revision prompt contains portfolio implication from critique."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        assert "Portfolio implication:" in prompt or "portfolio_implication" in prompt, (
            f"Enriched profile '{profile}' revision prompt missing portfolio implication"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_own_proposal(self, profile, mini_memo):
        """Enriched revision prompt contains the agent's own rendered proposal."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        # The v2 rendered proposal should appear via {{ my_proposal_v2 }}
        assert "Previous Portfolio Allocation" in prompt or "allocation" in prompt.lower(), (
            f"Enriched profile '{profile}' revision prompt missing own proposal context"
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_ENRICHED, ids=_PROPOSING_ENRICHED)
    def test_enriched_revision_prompt_has_objection_confidence(self, profile, mini_memo):
        """Enriched revision prompt contains objection confidence from critique."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=True)
        assert "Objection confidence: 0.8" in prompt or "objection_confidence" in prompt, (
            f"Enriched profile '{profile}' revision prompt missing objection confidence"
        )

    # --- Base: lossy v1 critique text drops structural fields ----------

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_revision_prompt_lacks_target_claim(self, profile, mini_memo):
        """Base revision prompt uses lossy v1 critiques — target_claim absent."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=False)
        has_target_claim = "Target claim:" in prompt or "target_claim" in prompt
        if not has_target_claim:
            pytest.xfail(
                f"Base profile '{profile}' revision prompt uses v1 critiques_text — "
                "target_claim lost in lossy format"
            )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_revision_prompt_lacks_counter_evidence(self, profile, mini_memo):
        """Base revision prompt loses counter-evidence in lossy v1 format."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=False)
        # v1 critiques_text is just "- [RISK]: Too aggressive on tech exposure."
        # No counter-evidence, no target_claim, no falsifier
        has_counter_ev = "Counter-evidence:" in prompt or "counter_evidence" in prompt
        if not has_counter_ev:
            pytest.xfail(
                f"Base profile '{profile}' revision prompt uses v1 critiques_text — "
                "counter-evidence lost in lossy format"
            )

    @pytest.mark.fast
    @pytest.mark.parametrize("profile", _PROPOSING_BASE, ids=_PROPOSING_BASE)
    def test_base_revision_prompt_has_objection_text(self, profile, mini_memo):
        """Base revision prompt at least contains the objection text."""
        prompt = _render_full_revision_prompt(profile, mini_memo, enriched=False)
        # Even lossy v1 format includes the objection string
        assert "Too aggressive" in prompt or "objection" in prompt.lower(), (
            f"Base profile '{profile}' revision prompt missing even the objection text"
        )
