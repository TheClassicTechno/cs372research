"""Unit tests for CRIT per-agent scoring logic (CritScorer + rho_bar computation)."""

import json

import pytest

from eval.crit.schema import RoundCritResult, validate_raw_response
from eval.crit.scorer import CritScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Build a valid CRIT JSON response string for a single agent."""
    return json.dumps({
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
    })


def _make_scorer_uniform(response_text: str) -> CritScorer:
    """Create a CritScorer with a mock LLM that returns the same text for all agents."""
    return CritScorer(llm_fn=lambda sys, usr: response_text)


def _make_scorer_per_role(role_responses: dict[str, str]) -> CritScorer:
    """Create a CritScorer that returns different responses per agent role.

    The mock LLM inspects the user prompt for 'Agent Under Evaluation: ROLE'
    to determine which response to return.
    """
    def _mock_llm(system_prompt: str, user_prompt: str) -> str:
        for role, response in role_responses.items():
            if role.upper() in user_prompt:
                return response
        # Fallback
        return list(role_responses.values())[0]

    return CritScorer(llm_fn=_mock_llm)


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
        scorer = _make_scorer_uniform(_make_mock_response(0.8, 0.7, 0.9, 0.6))
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert isinstance(result, RoundCritResult)

    def test_scores_each_agent_individually(self):
        """Each agent gets its own CritResult in agent_scores."""
        scorer = _make_scorer_uniform(_make_mock_response(0.8, 0.7, 0.9, 0.6))
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert "macro" in result.agent_scores
        assert "value" in result.agent_scores
        assert len(result.agent_scores) == 2

    def test_different_scores_per_agent(self):
        """Different agents can receive different CRIT scores."""
        scorer = _make_scorer_per_role({
            "macro": _make_mock_response(1.0, 1.0, 1.0, 1.0),  # ρ_i = 1.0
            "value": _make_mock_response(0.0, 0.0, 0.0, 0.0),  # ρ_i = 0.0
        })
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        assert result.agent_scores["macro"].rho_bar == 1.0
        assert result.agent_scores["value"].rho_bar == 0.0
        assert result.rho_bar == 0.5  # mean of 1.0 and 0.0

    def test_rho_bar_is_mean_of_per_agent_rho_bars(self):
        """ρ̄ = 1/n Σ_i ρ_i per RAudit Algorithm 1 line 8."""
        scorer = _make_scorer_per_role({
            "macro": _make_mock_response(0.8, 0.8, 0.8, 0.8),  # ρ_i = 0.8
            "value": _make_mock_response(0.4, 0.4, 0.4, 0.4),  # ρ_i = 0.4
        })
        result = scorer.score("case", MULTI_AGENT_TRACES, MULTI_AGENT_DECISIONS)
        expected = (0.8 + 0.4) / 2.0
        assert abs(result.rho_bar - expected) < 1e-9


# ---------------------------------------------------------------------------
# rho_bar computation tests (uniform scores)
# ---------------------------------------------------------------------------

class TestRhoBarComputation:
    def test_perfect_scores(self):
        scorer = _make_scorer_uniform(_make_mock_response(1.0, 1.0, 1.0, 1.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 1.0

    def test_zero_scores(self):
        scorer = _make_scorer_uniform(_make_mock_response(0.0, 0.0, 0.0, 0.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.0

    def test_mixed_scores_correct_mean(self):
        scorer = _make_scorer_uniform(_make_mock_response(0.8, 0.6, 1.0, 0.4))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        expected = (0.8 + 0.6 + 1.0 + 0.4) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_boundary_values(self):
        scorer = _make_scorer_uniform(_make_mock_response(0.0, 1.0, 0.0, 1.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.5


# ---------------------------------------------------------------------------
# CritScorer error handling tests
# ---------------------------------------------------------------------------

class TestCritScorerErrors:
    def test_invalid_json_raises(self):
        scorer = _make_scorer_uniform("not valid json at all")
        with pytest.raises(json.JSONDecodeError):
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_partial_response_missing_pillar_raises(self):
        incomplete = json.dumps({
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
        })
        scorer = _make_scorer_uniform(incomplete)
        with pytest.raises(Exception):  # ValidationError from pydantic
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_empty_json_object_raises(self):
        scorer = _make_scorer_uniform("{}")
        with pytest.raises(KeyError):
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_markdown_code_fence_stripped(self):
        """CritScorer should handle LLM output wrapped in markdown code fences."""
        inner = _make_mock_response(0.7, 0.7, 0.7, 0.7)
        wrapped = f"```json\n{inner}\n```"
        scorer = _make_scorer_uniform(wrapped)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert abs(result.rho_bar - 0.7) < 1e-9

    def test_no_agents_raises(self):
        """Empty traces and decisions should raise ValueError."""
        scorer = _make_scorer_uniform(_make_mock_response())
        with pytest.raises(ValueError, match="No agent roles"):
            scorer.score("case", [], [])


# ---------------------------------------------------------------------------
# CritScorer determinism tests
# ---------------------------------------------------------------------------

class TestCritScorerDeterminism:
    def test_same_inputs_same_outputs(self):
        """Given the same mock LLM, CRIT produces identical results."""
        response = _make_mock_response(0.75, 0.85, 0.65, 0.55)
        scorer = _make_scorer_uniform(response)
        traces = [{"role": "macro", "content": {"justification": "test"}}]
        decisions = [{"role": "macro", "action_dict": {"orders": []}}]

        r1 = scorer.score("same case", traces, decisions)
        r2 = scorer.score("same case", traces, decisions)

        assert r1.rho_bar == r2.rho_bar
        assert r1.agent_scores["macro"].pillar_scores == r2.agent_scores["macro"].pillar_scores
        assert r1.agent_scores["macro"].diagnostics == r2.agent_scores["macro"].diagnostics
