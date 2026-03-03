"""Unit tests for CRIT batch scoring logic (CritScorer + rho_bar computation)."""

import json

import pytest

from eval.crit.schema import RoundCritResult, validate_raw_response
from eval.crit.scorer import CritScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_agent_response(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Build a valid CRIT response dict for a single agent (not JSON-encoded)."""
    return {
        "pillar_scores": {
            "internal_consistency": ic,
            "evidence_support": es,
            "trace_alignment": ta,
            "causal_integrity": ci,
        },
        "diagnostics": {
            "contradictions_detected": ic < 0.5,
            "unsupported_claims_detected": es < 0.5,
            "conclusion_drift_detected": ta < 0.5,
            "causal_overreach_detected": ci < 0.5,
        },
        "explanations": {
            "internal_consistency": "Test explanation for IC.",
            "evidence_support": "Test explanation for ES.",
            "trace_alignment": "Test explanation for TA.",
            "causal_integrity": "Test explanation for CI.",
        },
    }


def _make_batch_response(roles: list[str], **kwargs) -> str:
    """Build a batch CRIT JSON response string with the same scores for all roles."""
    agent_data = _make_single_agent_response(**kwargs)
    return json.dumps({role: agent_data for role in roles})


def _make_scorer_uniform(roles: list[str], **kwargs) -> CritScorer:
    """Create a CritScorer with a mock LLM that returns same scores for all roles."""
    response_text = _make_batch_response(roles, **kwargs)
    return CritScorer(llm_fn=lambda sys, usr: response_text)


def _make_scorer_per_role(role_scores: dict[str, dict]) -> CritScorer:
    """Create a CritScorer that returns different scores per agent role.

    Args:
        role_scores: Mapping of role → kwargs for _make_single_agent_response.
    """
    batch = {
        role: _make_single_agent_response(**scores)
        for role, scores in role_scores.items()
    }
    response_text = json.dumps(batch)
    return CritScorer(llm_fn=lambda sys, usr: response_text)


# Standard multi-agent traces for testing
MULTI_AGENT_TRACES = [
    {"role": "macro", "type": "proposal", "content": {"justification": "Growth looks strong."}},
    {"role": "value", "type": "proposal", "content": {"justification": "Valuation is fair."}},
]

MULTI_AGENT_DECISIONS = [
    {"role": "macro", "action_dict": {"orders": [{"ticker": "NVDA", "side": "buy"}], "confidence": 0.8}},
    {"role": "value", "action_dict": {"orders": [{"ticker": "NVDA", "side": "buy"}], "confidence": 0.6}},
]


# ---------------------------------------------------------------------------
# Per-agent scoring tests
# ---------------------------------------------------------------------------

class TestPerAgentScoring:
    def test_returns_round_crit_result(self):
        """score() returns RoundCritResult, not CritResult."""
        scorer = _make_scorer_uniform(["macro", "value"], ic=0.8, es=0.7, ta=0.9, ci=0.6)
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert isinstance(result, RoundCritResult)

    def test_scores_each_agent_individually(self):
        """Each agent gets its own CritResult in agent_scores."""
        scorer = _make_scorer_uniform(["macro", "value"], ic=0.8, es=0.7, ta=0.9, ci=0.6)
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert "macro" in result.agent_scores
        assert "value" in result.agent_scores
        assert len(result.agent_scores) == 2

    def test_different_scores_per_agent(self):
        """Different agents can receive different CRIT scores."""
        scorer = _make_scorer_per_role({
            "macro": {"ic": 1.0, "es": 1.0, "ta": 1.0, "ci": 1.0},
            "value": {"ic": 0.0, "es": 0.0, "ta": 0.0, "ci": 0.0},
        })
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert result.agent_scores["macro"].rho_bar == 1.0
        assert result.agent_scores["value"].rho_bar == 0.0
        assert result.rho_bar == 0.5  # mean of 1.0 and 0.0

    def test_rho_bar_is_mean_of_per_agent_rho_bars(self):
        """ρ̄ = 1/n Σ_i ρ_i per RAudit Algorithm 1 line 8."""
        scorer = _make_scorer_per_role({
            "macro": {"ic": 0.8, "es": 0.8, "ta": 0.8, "ci": 0.8},
            "value": {"ic": 0.4, "es": 0.4, "ta": 0.4, "ci": 0.4},
        })
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        expected = (0.8 + 0.4) / 2.0
        assert abs(result.rho_bar - expected) < 1e-9


# ---------------------------------------------------------------------------
# rho_bar computation tests (uniform scores)
# ---------------------------------------------------------------------------

class TestRhoBarComputation:
    def test_perfect_scores(self):
        scorer = _make_scorer_uniform(["macro"], ic=1.0, es=1.0, ta=1.0, ci=1.0)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 1.0

    def test_zero_scores(self):
        scorer = _make_scorer_uniform(["macro"], ic=0.0, es=0.0, ta=0.0, ci=0.0)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.0

    def test_mixed_scores_correct_mean(self):
        scorer = _make_scorer_uniform(["macro"], ic=0.8, es=0.6, ta=1.0, ci=0.4)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        expected = (0.8 + 0.6 + 1.0 + 0.4) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_boundary_values(self):
        scorer = _make_scorer_uniform(["macro"], ic=0.0, es=1.0, ta=0.0, ci=1.0)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.5


# ---------------------------------------------------------------------------
# CritScorer error handling tests
# ---------------------------------------------------------------------------

class TestCritScorerErrors:
    def test_invalid_json_raises(self):
        scorer = CritScorer(llm_fn=lambda sys, usr: "not valid json at all")
        with pytest.raises(json.JSONDecodeError):
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_partial_response_missing_pillar_raises(self):
        incomplete = json.dumps({
            "macro": {
                "pillar_scores": {
                    "internal_consistency": 0.8,
                    # missing other pillars
                },
                "diagnostics": {
                    "contradictions_detected": False,
                    "unsupported_claims_detected": False,
                    "conclusion_drift_detected": False,
                    "causal_overreach_detected": False,
                },
                "explanations": {
                    "internal_consistency": "ok",
                    "evidence_support": "ok",
                    "trace_alignment": "ok",
                    "causal_integrity": "ok",
                },
            }
        })
        scorer = CritScorer(llm_fn=lambda sys, usr: incomplete)
        with pytest.raises(Exception):  # ValidationError from pydantic
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_empty_json_object_raises(self):
        scorer = CritScorer(llm_fn=lambda sys, usr: "{}")
        with pytest.raises(ValueError, match="missing roles"):
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_markdown_code_fence_stripped(self):
        """CritScorer should handle LLM output wrapped in markdown code fences."""
        inner = _make_batch_response(["macro"], ic=0.7, es=0.7, ta=0.7, ci=0.7)
        wrapped = f"```json\n{inner}\n```"
        scorer = CritScorer(llm_fn=lambda sys, usr: wrapped)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert abs(result.rho_bar - 0.7) < 1e-9

    def test_no_agents_raises(self):
        """Empty traces and decisions should raise ValueError."""
        scorer = _make_scorer_uniform(["macro"])
        with pytest.raises(ValueError, match="No agent roles"):
            scorer.score("case", [], [])

    def test_missing_role_in_batch_raises(self):
        """Batch response missing a role that traces contain raises ValueError."""
        # Response only has "macro" but traces include "value" too
        response = json.dumps({"macro": _make_single_agent_response()})
        scorer = CritScorer(llm_fn=lambda sys, usr: response)
        with pytest.raises(ValueError, match="missing roles"):
            scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)


# ---------------------------------------------------------------------------
# CritScorer determinism tests
# ---------------------------------------------------------------------------

class TestCritScorerDeterminism:
    def test_same_inputs_same_outputs(self):
        """Given the same mock LLM, CRIT produces identical results."""
        scorer = _make_scorer_uniform(["macro"], ic=0.75, es=0.85, ta=0.65, ci=0.55)
        traces = [{"role": "macro", "content": {"justification": "test"}}]
        decisions = [{"role": "macro", "action_dict": {"orders": []}}]

        r1 = scorer.score("same case", traces, decisions)
        r2 = scorer.score("same case", traces, decisions)

        assert r1.rho_bar == r2.rho_bar
        assert r1.agent_scores["macro"].pillar_scores == r2.agent_scores["macro"].pillar_scores
        assert r1.agent_scores["macro"].diagnostics == r2.agent_scores["macro"].diagnostics
