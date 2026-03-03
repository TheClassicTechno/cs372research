"""Unit tests for CRIT schema validation."""

import pytest
from pydantic import ValidationError

from eval.crit.schema import (
    CritResult,
    Diagnostics,
    Explanations,
    PillarScores,
    RoundCritResult,
    aggregate_agent_scores,
    validate_batch_response,
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


def _make_crit_result(ic=0.8, es=0.7, ta=0.9, ci=0.6) -> CritResult:
    """Build a CritResult directly for aggregation tests."""
    return validate_raw_response(_make_raw(ic=ic, es=es, ta=ta, ci=ci))


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


# ---------------------------------------------------------------------------
# RoundCritResult / aggregate_agent_scores tests
# ---------------------------------------------------------------------------

class TestRoundCritResult:
    def test_aggregate_two_agents(self):
        """ρ̄ = mean of per-agent ρ_i scores."""
        macro = _make_crit_result(ic=0.8, es=0.6, ta=1.0, ci=0.6)  # ρ_i = 0.75
        value = _make_crit_result(ic=0.4, es=0.4, ta=0.4, ci=0.4)  # ρ_i = 0.40
        result = aggregate_agent_scores({"macro": macro, "value": value})
        assert isinstance(result, RoundCritResult)
        expected_rho_bar = (0.75 + 0.40) / 2.0
        assert abs(result.rho_bar - expected_rho_bar) < 1e-9

    def test_aggregate_preserves_per_agent_scores(self):
        """Per-agent CritResults are accessible by role name."""
        macro = _make_crit_result(ic=1.0, es=1.0, ta=1.0, ci=1.0)
        value = _make_crit_result(ic=0.0, es=0.0, ta=0.0, ci=0.0)
        result = aggregate_agent_scores({"macro": macro, "value": value})
        assert result.agent_scores["macro"].rho_bar == 1.0
        assert result.agent_scores["value"].rho_bar == 0.0
        assert result.rho_bar == 0.5

    def test_aggregate_single_agent(self):
        """With one agent, ρ̄ = ρ_i."""
        solo = _make_crit_result(ic=0.7, es=0.7, ta=0.7, ci=0.7)
        result = aggregate_agent_scores({"solo": solo})
        assert abs(result.rho_bar - 0.7) < 1e-9

    def test_aggregate_four_agents(self):
        """Standard 4-agent debate: ρ̄ = 1/4 Σ ρ_i."""
        agents = {
            "macro": _make_crit_result(ic=0.9, es=0.9, ta=0.9, ci=0.9),    # 0.9
            "value": _make_crit_result(ic=0.7, es=0.7, ta=0.7, ci=0.7),    # 0.7
            "risk": _make_crit_result(ic=0.5, es=0.5, ta=0.5, ci=0.5),     # 0.5
            "technical": _make_crit_result(ic=0.3, es=0.3, ta=0.3, ci=0.3),# 0.3
        }
        result = aggregate_agent_scores(agents)
        expected = (0.9 + 0.7 + 0.5 + 0.3) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_aggregate_empty_raises(self):
        """Empty agent_scores is invalid."""
        with pytest.raises(ValueError, match="must not be empty"):
            aggregate_agent_scores({})

    def test_round_crit_result_model_dump(self):
        """RoundCritResult serializes correctly for PID event logging."""
        macro = _make_crit_result(ic=0.8, es=0.8, ta=0.8, ci=0.8)
        result = aggregate_agent_scores({"macro": macro})
        dumped = result.model_dump()
        assert "agent_scores" in dumped
        assert "macro" in dumped["agent_scores"]
        assert "rho_bar" in dumped


# ---------------------------------------------------------------------------
# validate_batch_response tests
# ---------------------------------------------------------------------------

class TestValidateBatchResponse:
    def test_valid_two_agents(self):
        """Two-agent batch response parses correctly."""
        raw = {
            "macro": _make_raw(ic=0.9, es=0.8, ta=0.7, ci=0.6),
            "value": _make_raw(ic=0.5, es=0.5, ta=0.5, ci=0.5),
        }
        result = validate_batch_response(raw, {"macro", "value"})
        assert isinstance(result, dict)
        assert set(result.keys()) == {"macro", "value"}
        assert isinstance(result["macro"], CritResult)
        assert isinstance(result["value"], CritResult)
        assert result["macro"].rho_bar == (0.9 + 0.8 + 0.7 + 0.6) / 4.0
        assert result["value"].rho_bar == 0.5

    def test_valid_four_agents(self):
        """Four-agent batch response parses correctly."""
        raw = {
            "macro": _make_raw(ic=0.9, es=0.9, ta=0.9, ci=0.9),
            "value": _make_raw(ic=0.7, es=0.7, ta=0.7, ci=0.7),
            "risk": _make_raw(ic=0.5, es=0.5, ta=0.5, ci=0.5),
            "technical": _make_raw(ic=0.3, es=0.3, ta=0.3, ci=0.3),
        }
        roles = {"macro", "value", "risk", "technical"}
        result = validate_batch_response(raw, roles)
        assert set(result.keys()) == roles

    def test_missing_role_raises(self):
        """Missing role in batch response raises ValueError."""
        raw = {
            "macro": _make_raw(),
            # "value" is missing
        }
        with pytest.raises(ValueError, match="missing roles"):
            validate_batch_response(raw, {"macro", "value"})

    def test_extra_roles_ignored(self):
        """Extra roles in response beyond expected_roles are ignored."""
        raw = {
            "macro": _make_raw(),
            "value": _make_raw(),
            "extra_agent": _make_raw(),
        }
        result = validate_batch_response(raw, {"macro", "value"})
        assert set(result.keys()) == {"macro", "value"}

    def test_invalid_sub_dict_raises(self):
        """Invalid pillar scores in one agent propagate as validation error."""
        raw = {
            "macro": _make_raw(ic=1.5),  # out of range
            "value": _make_raw(),
        }
        with pytest.raises(ValidationError):
            validate_batch_response(raw, {"macro", "value"})

    def test_single_agent_batch(self):
        """Single-agent batch response works."""
        raw = {"solo": _make_raw(ic=0.6, es=0.6, ta=0.6, ci=0.6)}
        result = validate_batch_response(raw, {"solo"})
        assert len(result) == 1
        assert abs(result["solo"].rho_bar - 0.6) < 1e-9

    def test_empty_expected_roles_raises(self):
        """Empty expected_roles with non-empty response still returns empty dict."""
        raw = {"macro": _make_raw()}
        result = validate_batch_response(raw, set())
        assert result == {}
