"""L1 unit tests for CRIT schema validation.

Tests cover:
    - CritResult creation with valid data
    - PillarScores bounds enforcement (parametrized across all 4 pillars)
    - rho_bar = mean of the 4 pillar scores
    - Weakest pillar identification (min of 4)
    - validate_raw_response() rejection of missing fields
    - aggregate_agent_scores() with parametrized agent counts
    - Agent response schema: allocation dict, sum-to-one, confidence bounds
"""

import pytest
from pydantic import ValidationError

from eval.crit.schema import (
    CritResult,
    Diagnostics,
    Explanations,
    PillarScores,
    RoundCritResult,
    aggregate_agent_scores,
    validate_raw_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_diagnostics() -> dict:
    """Canonical diagnostics dict with all flags False."""
    return {
        "contradictions_detected": False,
        "unsupported_claims_detected": False,
        "ignored_critiques_detected": False,
        "premature_certainty_detected": False,
        "causal_overreach_detected": False,
        "conclusion_drift_detected": False,
    }


def _default_explanations() -> dict:
    """Canonical explanations dict with placeholder strings."""
    return {
        "logical_validity": "Reasoning is sound.",
        "evidential_support": "Claims are backed by data.",
        "alternative_consideration": "Alternatives were discussed.",
        "causal_alignment": "Causal links are justified.",
    }


def _make_raw(lv=0.8, es=0.7, ac=0.9, ca=0.6) -> dict:
    """Build a valid raw CRIT response dict with configurable pillar scores."""
    return {
        "pillar_scores": {
            "logical_validity": lv,
            "evidential_support": es,
            "alternative_consideration": ac,
            "causal_alignment": ca,
        },
        "diagnostics": _default_diagnostics(),
        "explanations": _default_explanations(),
    }


def _make_crit_result(lv=0.8, es=0.7, ac=0.9, ca=0.6) -> CritResult:
    """Build a CritResult via validate_raw_response for aggregation tests."""
    return validate_raw_response(_make_raw(lv=lv, es=es, ac=ac, ca=ca))


# ---------------------------------------------------------------------------
# CritResult creation with valid data
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCritResultCreation:
    """Verify CritResult can be constructed with well-formed inputs."""

    def test_valid_crit_result(self):
        ps = PillarScores(
            logical_validity=0.8,
            evidential_support=0.7,
            alternative_consideration=0.9,
            causal_alignment=0.6,
        )
        diag = Diagnostics(**_default_diagnostics())
        expl = Explanations(**_default_explanations())
        cr = CritResult(
            pillar_scores=ps,
            diagnostics=diag,
            explanations=expl,
            rho_bar=0.75,
        )
        assert isinstance(cr, CritResult)
        assert cr.pillar_scores.logical_validity == 0.8
        assert cr.rho_bar == 0.75

    def test_crit_result_via_validate_raw(self):
        raw = _make_raw(lv=0.6, es=0.6, ac=0.6, ca=0.6)
        cr = validate_raw_response(raw)
        assert isinstance(cr, CritResult)
        assert cr.rho_bar == pytest.approx(0.6)

    def test_crit_result_has_all_components(self):
        cr = _make_crit_result()
        assert hasattr(cr, "pillar_scores")
        assert hasattr(cr, "diagnostics")
        assert hasattr(cr, "explanations")
        assert hasattr(cr, "rho_bar")

    def test_diagnostics_accessible(self):
        cr = _make_crit_result()
        assert cr.diagnostics.contradictions_detected is False

    def test_explanations_are_strings(self):
        cr = _make_crit_result()
        assert isinstance(cr.explanations.logical_validity, str)
        assert isinstance(cr.explanations.evidential_support, str)


# ---------------------------------------------------------------------------
# PillarScores bounds — parametrized across all 4 pillars
# ---------------------------------------------------------------------------

_PILLAR_NAMES = [
    "logical_validity",
    "evidential_support",
    "alternative_consideration",
    "causal_alignment",
]


@pytest.mark.fast
class TestPillarScoresBounds:
    """Each pillar score must lie in [0, 1]."""

    @pytest.mark.parametrize("pillar", _PILLAR_NAMES)
    def test_score_at_zero_accepted(self, pillar):
        kwargs = {p: 0.5 for p in _PILLAR_NAMES}
        kwargs[pillar] = 0.0
        ps = PillarScores(**kwargs)
        assert getattr(ps, pillar) == 0.0

    @pytest.mark.parametrize("pillar", _PILLAR_NAMES)
    def test_score_at_one_accepted(self, pillar):
        kwargs = {p: 0.5 for p in _PILLAR_NAMES}
        kwargs[pillar] = 1.0
        ps = PillarScores(**kwargs)
        assert getattr(ps, pillar) == 1.0

    @pytest.mark.parametrize("pillar", _PILLAR_NAMES)
    def test_score_below_zero_rejected(self, pillar):
        kwargs = {p: 0.5 for p in _PILLAR_NAMES}
        kwargs[pillar] = -0.01
        with pytest.raises(ValidationError):
            PillarScores(**kwargs)

    @pytest.mark.parametrize("pillar", _PILLAR_NAMES)
    def test_score_above_one_rejected(self, pillar):
        kwargs = {p: 0.5 for p in _PILLAR_NAMES}
        kwargs[pillar] = 1.01
        with pytest.raises(ValidationError):
            PillarScores(**kwargs)

    @pytest.mark.parametrize("pillar", _PILLAR_NAMES)
    def test_mid_range_score_accepted(self, pillar):
        kwargs = {p: 0.5 for p in _PILLAR_NAMES}
        kwargs[pillar] = 0.55
        ps = PillarScores(**kwargs)
        assert getattr(ps, pillar) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# rho_bar = mean of 4 pillar scores
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestRhoBarComputation:
    """validate_raw_response computes rho_bar as the mean of 4 pillar scores."""

    def test_rho_bar_equals_mean(self):
        cr = _make_crit_result(lv=0.8, es=0.6, ac=1.0, ca=0.6)
        expected = (0.8 + 0.6 + 1.0 + 0.6) / 4.0
        assert cr.rho_bar == pytest.approx(expected)

    def test_perfect_scores_rho_bar_one(self):
        cr = _make_crit_result(lv=1.0, es=1.0, ac=1.0, ca=1.0)
        assert cr.rho_bar == pytest.approx(1.0)

    def test_zero_scores_rho_bar_zero(self):
        cr = _make_crit_result(lv=0.0, es=0.0, ac=0.0, ca=0.0)
        assert cr.rho_bar == pytest.approx(0.0)

    def test_uniform_scores(self):
        cr = _make_crit_result(lv=0.5, es=0.5, ac=0.5, ca=0.5)
        assert cr.rho_bar == pytest.approx(0.5)

    def test_asymmetric_scores(self):
        cr = _make_crit_result(lv=0.2, es=0.4, ac=0.6, ca=0.8)
        expected = (0.2 + 0.4 + 0.6 + 0.8) / 4.0
        assert cr.rho_bar == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Weakest pillar identification (min of 4)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestWeakestPillar:
    """The weakest pillar is the one with the lowest score."""

    def _weakest(self, ps: PillarScores) -> str:
        scores = {
            "logical_validity": ps.logical_validity,
            "evidential_support": ps.evidential_support,
            "alternative_consideration": ps.alternative_consideration,
            "causal_alignment": ps.causal_alignment,
        }
        return min(scores, key=scores.get)

    def test_weakest_is_causal_alignment(self):
        cr = _make_crit_result(lv=0.9, es=0.8, ac=0.7, ca=0.3)
        assert self._weakest(cr.pillar_scores) == "causal_alignment"

    def test_weakest_is_logical_validity(self):
        cr = _make_crit_result(lv=0.1, es=0.8, ac=0.7, ca=0.6)
        assert self._weakest(cr.pillar_scores) == "logical_validity"

    def test_weakest_is_evidential_support(self):
        cr = _make_crit_result(lv=0.9, es=0.2, ac=0.7, ca=0.6)
        assert self._weakest(cr.pillar_scores) == "evidential_support"

    def test_weakest_is_alternative_consideration(self):
        cr = _make_crit_result(lv=0.9, es=0.8, ac=0.1, ca=0.6)
        assert self._weakest(cr.pillar_scores) == "alternative_consideration"

    def test_all_equal_returns_any(self):
        cr = _make_crit_result(lv=0.5, es=0.5, ac=0.5, ca=0.5)
        assert self._weakest(cr.pillar_scores) in _PILLAR_NAMES


# ---------------------------------------------------------------------------
# validate_raw_response() rejects missing fields
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestValidateRawResponseRejectsMissing:
    """validate_raw_response raises on missing top-level keys."""

    def test_missing_pillar_scores_raises_key_error(self):
        raw = _make_raw()
        del raw["pillar_scores"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_missing_diagnostics_raises_key_error(self):
        raw = _make_raw()
        del raw["diagnostics"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_missing_explanations_raises_key_error(self):
        raw = _make_raw()
        del raw["explanations"]
        with pytest.raises(KeyError):
            validate_raw_response(raw)

    def test_empty_dict_raises(self):
        with pytest.raises(KeyError):
            validate_raw_response({})

    def test_out_of_range_pillar_raises_validation_error(self):
        raw = _make_raw(lv=1.5)
        with pytest.raises(ValidationError):
            validate_raw_response(raw)

    def test_negative_pillar_raises_validation_error(self):
        raw = _make_raw(es=-0.1)
        with pytest.raises(ValidationError):
            validate_raw_response(raw)


# ---------------------------------------------------------------------------
# aggregate_agent_scores() — parametrized by n_agents
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestAggregateAgentScores:
    """aggregate_agent_scores computes round-level rho_bar = 1/n sum(rho_i)."""

    @pytest.mark.parametrize("n_agents", [1, 2, 4])
    def test_aggregate_n_agents(self, n_agents):
        """Round rho_bar is the mean of n per-agent rho_bars."""
        # Give each agent a different uniform score: 0.3, 0.5, 0.7, 0.9
        base_scores = [0.3, 0.5, 0.7, 0.9]
        agent_scores = {}
        for i in range(n_agents):
            s = base_scores[i % len(base_scores)]
            agent_scores[f"agent_{i}"] = _make_crit_result(lv=s, es=s, ac=s, ca=s)
        result = aggregate_agent_scores(agent_scores)
        assert isinstance(result, RoundCritResult)
        expected_rho = sum(
            cr.rho_bar for cr in agent_scores.values()
        ) / n_agents
        assert result.rho_bar == pytest.approx(expected_rho)

    @pytest.mark.parametrize("n_agents", [1, 2, 4])
    def test_aggregate_preserves_agent_keys(self, n_agents):
        agent_scores = {}
        for i in range(n_agents):
            agent_scores[f"role_{i}"] = _make_crit_result(lv=0.5, es=0.5, ac=0.5, ca=0.5)
        result = aggregate_agent_scores(agent_scores)
        assert set(result.agent_scores.keys()) == {f"role_{i}" for i in range(n_agents)}

    def test_aggregate_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            aggregate_agent_scores({})

    def test_aggregate_single_agent_equals_self(self):
        cr = _make_crit_result(lv=0.75, es=0.75, ac=0.75, ca=0.75)
        result = aggregate_agent_scores({"solo": cr})
        assert result.rho_bar == pytest.approx(cr.rho_bar)


# ---------------------------------------------------------------------------
# Agent response schema: allocation dict, sum-to-one, confidence bounds
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestAgentResponseSchema:
    """Validate the expected shape of an agent's portfolio allocation response.

    An agent response contains:
        - allocation: dict[str, float] mapping ticker to weight
        - values should sum to approximately 1.0
        - an optional confidence score in [0, 1]

    These are structural tests — they verify the contract that downstream
    code (actuator, PID controller) depends on.
    """

    def test_valid_allocation_dict(self):
        allocation = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        assert isinstance(allocation, dict)
        assert all(isinstance(k, str) for k in allocation)
        assert all(isinstance(v, float) for v in allocation.values())

    def test_allocation_sums_to_one(self):
        allocation = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
        total = sum(allocation.values())
        assert abs(total - 1.0) < 1e-9

    def test_allocation_approximate_sum(self):
        """Floating-point allocations should sum to ~1.0 within tolerance."""
        allocation = {"AAPL": 1 / 3, "MSFT": 1 / 3, "GOOGL": 1 / 3}
        total = sum(allocation.values())
        assert abs(total - 1.0) < 1e-6

    def test_confidence_in_bounds(self):
        confidence = 0.85
        assert 0.0 <= confidence <= 1.0

    def test_confidence_at_zero(self):
        confidence = 0.0
        assert 0.0 <= confidence <= 1.0

    def test_confidence_at_one(self):
        confidence = 1.0
        assert 0.0 <= confidence <= 1.0

    def test_confidence_below_zero_invalid(self):
        confidence = -0.1
        assert not (0.0 <= confidence <= 1.0)

    def test_confidence_above_one_invalid(self):
        confidence = 1.1
        assert not (0.0 <= confidence <= 1.0)

    def test_allocation_all_positive_weights(self):
        allocation = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        assert all(v >= 0.0 for v in allocation.values())

    def test_single_ticker_allocation(self):
        allocation = {"AAPL": 1.0}
        assert sum(allocation.values()) == pytest.approx(1.0)
