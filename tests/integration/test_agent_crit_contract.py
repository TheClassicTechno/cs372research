"""
Integration tests: Agent Prompt → Normalizer → CRIT contract compatibility.

These tests verify that agent prompt templates produce outputs compatible with
the CRIT reasoning evaluator.  The test matrix covers both the "base" template
family (standard/diverse/slim) and the "enriched" template family, exposing
the known schema gap where base templates lack CRIT-required fields.

Pipeline under test:

    Agent Prompt (rendered)
        ↓
    LLM output JSON
        ↓
    _normalize_claims() / _normalize_position_rationale()
        ↓
    build_reasoning_bundle()
        ↓
    CRIT evaluation  (render_crit_prompts → CritScorer)

CRIT expects claims with:
    claim_id, claim_text, claim_type, evidence_ids,
    impacts_positions, falsifiers, confidence

Position objects must also contain:
    position_rationale, supporting_claims
"""

from __future__ import annotations

import json
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
    load_module,
)
from multi_agent.runner import (
    _normalize_claims,
    _normalize_position_rationale,
    _extract_reasoning,
    build_reasoning_bundle,
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
