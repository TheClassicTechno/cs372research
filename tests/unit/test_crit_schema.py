"""Unit tests for CRIT schema validation."""

import pytest
from pydantic import ValidationError

from eval.crit.schema import (
    CritResult,
    Diagnostics,
    Explanations,
    PillarScores,
    validate_raw_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    ic=0.8, es=0.7, ta=0.9, ci=0.6,
    contradictions=False, unsupported=False, drift=False, overreach=False,
):
    """Build a valid raw CRIT response dict."""
    return {
        "pillar_scores": {
            "internal_consistency": ic,
            "evidence_support": es,
            "trace_alignment": ta,
            "causal_integrity": ci,
        },
        "diagnostics": {
            "contradictions_detected": contradictions,
            "unsupported_claims_detected": unsupported,
            "conclusion_drift_detected": drift,
            "causal_overreach_detected": overreach,
        },
        "explanations": {
            "internal_consistency": "No contradictions found.",
            "evidence_support": "Most claims are supported.",
            "trace_alignment": "Decision follows from reasoning.",
            "causal_integrity": "Some causal leaps noted.",
        },
    }


# ---------------------------------------------------------------------------
# PillarScores tests
# ---------------------------------------------------------------------------

class TestPillarScores:
    def test_valid_scores(self):
        ps = PillarScores(
            internal_consistency=0.8,
            evidence_support=0.7,
            trace_alignment=0.9,
            causal_integrity=0.6,
        )
        assert ps.internal_consistency == 0.8
        assert ps.evidence_support == 0.7

    def test_boundary_zero(self):
        ps = PillarScores(
            internal_consistency=0.0,
            evidence_support=0.0,
            trace_alignment=0.0,
            causal_integrity=0.0,
        )
        assert ps.internal_consistency == 0.0

    def test_boundary_one(self):
        ps = PillarScores(
            internal_consistency=1.0,
            evidence_support=1.0,
            trace_alignment=1.0,
            causal_integrity=1.0,
        )
        assert ps.internal_consistency == 1.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PillarScores(
                internal_consistency=-0.1,
                evidence_support=0.5,
                trace_alignment=0.5,
                causal_integrity=0.5,
            )

    def test_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            PillarScores(
                internal_consistency=0.5,
                evidence_support=1.1,
                trace_alignment=0.5,
                causal_integrity=0.5,
            )


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_all_false(self):
        d = Diagnostics(
            contradictions_detected=False,
            unsupported_claims_detected=False,
            conclusion_drift_detected=False,
            causal_overreach_detected=False,
        )
        assert not d.contradictions_detected

    def test_all_true(self):
        d = Diagnostics(
            contradictions_detected=True,
            unsupported_claims_detected=True,
            conclusion_drift_detected=True,
            causal_overreach_detected=True,
        )
        assert d.causal_overreach_detected

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            Diagnostics(
                contradictions_detected=True,
                # missing other fields
            )


# ---------------------------------------------------------------------------
# validate_raw_response tests
# ---------------------------------------------------------------------------

class TestValidateRawResponse:
    def test_valid_input_parses(self):
        raw = _make_raw(ic=0.8, es=0.7, ta=0.9, ci=0.6)
        result = validate_raw_response(raw)
        assert isinstance(result, CritResult)
        assert result.pillar_scores.internal_consistency == 0.8

    def test_rho_bar_computed_as_mean(self):
        raw = _make_raw(ic=0.8, es=0.6, ta=1.0, ci=0.6)
        result = validate_raw_response(raw)
        expected = (0.8 + 0.6 + 1.0 + 0.6) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_perfect_scores_rho_bar_one(self):
        raw = _make_raw(ic=1.0, es=1.0, ta=1.0, ci=1.0)
        result = validate_raw_response(raw)
        assert result.rho_bar == 1.0

    def test_zero_scores_rho_bar_zero(self):
        raw = _make_raw(ic=0.0, es=0.0, ta=0.0, ci=0.0)
        result = validate_raw_response(raw)
        assert result.rho_bar == 0.0

    def test_missing_pillar_scores_key_raises(self):
        raw = _make_raw()
        del raw["pillar_scores"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_missing_diagnostics_key_raises(self):
        raw = _make_raw()
        del raw["diagnostics"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_missing_explanations_key_raises(self):
        raw = _make_raw()
        del raw["explanations"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_pillar_out_of_range_raises(self):
        raw = _make_raw(ic=1.5)
        with pytest.raises(ValidationError):
            validate_raw_response(raw)

    def test_extra_fields_ignored(self):
        raw = _make_raw()
        raw["extra_field"] = "should be ignored"
        raw["pillar_scores"]["bonus"] = 999
        result = validate_raw_response(raw)
        assert isinstance(result, CritResult)

    def test_diagnostics_boolean_validation(self):
        raw = _make_raw()
        # Pydantic coerces truthy values to bool
        raw["diagnostics"]["contradictions_detected"] = True
        result = validate_raw_response(raw)
        assert result.diagnostics.contradictions_detected is True
