"""Unit tests for eval.evidence — evidence span extraction and overlap."""

import pytest

from eval.evidence import (
    compute_mean_overlap,
    extract_agent_evidence_spans,
    extract_evidence_spans,
    normalize_variable,
)


# ---------------------------------------------------------------------------
# normalize_variable
# ---------------------------------------------------------------------------

class TestNormalizeVariable:
    def test_lowercase(self):
        assert normalize_variable("NVDA_Revenue") == "nvdarevenue"

    def test_strip_underscores(self):
        assert normalize_variable("fed_rate") == "fedrate"

    def test_strip_hyphens(self):
        assert normalize_variable("nvda-revenue") == "nvdarevenue"

    def test_strip_whitespace(self):
        assert normalize_variable("  nvda revenue  ") == "nvdarevenue"

    def test_mixed(self):
        assert normalize_variable("AI_Infrastructure-demand") == "aiinfrastructuredemand"

    def test_empty(self):
        assert normalize_variable("") == ""

    def test_already_clean(self):
        assert normalize_variable("fedrate") == "fedrate"


# ---------------------------------------------------------------------------
# extract_evidence_spans
# ---------------------------------------------------------------------------

class TestExtractEvidenceSpans:
    def test_from_claims_variables(self):
        dec = {
            "claims": [
                {"claim_text": "something", "variables": ["fed_rate", "GDP_growth"]},
                {"claim_text": "another", "variables": ["NVDA_datacenter_revenue"]},
            ]
        }
        spans = extract_evidence_spans(dec)
        assert spans == {"fedrate", "gdpgrowth", "nvdadatacenterrevenue"}

    def test_from_action_dict_claims(self):
        dec = {
            "action_dict": {
                "claims": [
                    {"claim_text": "x", "variables": ["inflation", "p_e_ratio"]},
                ]
            }
        }
        spans = extract_evidence_spans(dec)
        assert spans == {"inflation", "peratio"}

    def test_fallback_to_claim_text(self):
        dec = {
            "claims": [
                {"claim_text": "The fed rate is rising rapidly"},
            ]
        }
        spans = extract_evidence_spans(dec)
        assert "the" in spans or "fed" in spans
        assert all(len(s) > 2 for s in spans)

    def test_empty_decision(self):
        assert extract_evidence_spans({}) == set()

    def test_no_variables_no_text(self):
        dec = {"claims": [{"variables": [], "claim_text": ""}]}
        assert extract_evidence_spans(dec) == set()


# ---------------------------------------------------------------------------
# extract_agent_evidence_spans
# ---------------------------------------------------------------------------

class TestExtractAgentEvidenceSpans:
    def test_multiple_agents(self):
        decisions = [
            {"role": "macro", "claims": [{"variables": ["fed_rate", "gdp"]}]},
            {"role": "value", "claims": [{"variables": ["pe_ratio", "earnings"]}]},
        ]
        result = extract_agent_evidence_spans(decisions)
        assert "macro" in result
        assert "value" in result
        assert "fedrate" in result["macro"]
        assert "peratio" in result["value"]

    def test_same_role_merges(self):
        decisions = [
            {"role": "macro", "claims": [{"variables": ["fed_rate"]}]},
            {"role": "macro", "claims": [{"variables": ["gdp"]}]},
        ]
        result = extract_agent_evidence_spans(decisions)
        assert result["macro"] == {"fedrate", "gdp"}

    def test_empty_list(self):
        assert extract_agent_evidence_spans([]) == {}


# ---------------------------------------------------------------------------
# compute_mean_overlap
# ---------------------------------------------------------------------------

class TestComputeMeanOverlap:
    def test_identical_sets(self):
        sets = {
            "a": {"x", "y", "z"},
            "b": {"x", "y", "z"},
        }
        assert pytest.approx(compute_mean_overlap(sets)) == 1.0

    def test_disjoint_sets(self):
        sets = {
            "a": {"x", "y"},
            "b": {"m", "n"},
        }
        assert compute_mean_overlap(sets) == 0.0

    def test_partial_overlap(self):
        sets = {
            "a": {"x", "y", "z"},
            "b": {"x", "y", "w"},
        }
        ov = compute_mean_overlap(sets)
        assert pytest.approx(ov, abs=1e-9) == 2.0 / 4.0

    def test_three_agents(self):
        sets = {
            "macro": {"fedrate", "gdpgrowth", "nvdadatacenterrevenue", "inflation"},
            "value": {"nvdadatacenterrevenue", "peratio", "earningsgrowth", "freecashflow"},
            "risk": {"volatilityregime", "nvdadatacenterrevenue", "concentrationrisk"},
        }
        ov = compute_mean_overlap(sets)
        assert 0.1 < ov < 0.3

    def test_fewer_than_two_agents(self):
        assert compute_mean_overlap({}) == None
        assert compute_mean_overlap({"a": {"x"}}) == None

    def test_empty_evidence_ignored(self):
        sets = {
            "a": {"x", "y"},
            "b": set(),
        }
        assert compute_mean_overlap(sets) == 0.0
