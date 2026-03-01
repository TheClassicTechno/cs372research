"""Unit tests for CRIT scoring logic (CritScorer + rho_bar computation)."""

import json

import pytest

from eval.crit.schema import validate_raw_response
from eval.crit.scorer import CritScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Build a valid CRIT JSON response string."""
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


def _make_scorer(response_text: str) -> CritScorer:
    """Create a CritScorer with a mock LLM that returns the given text."""
    return CritScorer(llm_fn=lambda sys, usr: response_text)


# ---------------------------------------------------------------------------
# rho_bar computation tests
# ---------------------------------------------------------------------------

class TestRhoBarComputation:
    def test_perfect_scores(self):
        scorer = _make_scorer(_make_mock_response(1.0, 1.0, 1.0, 1.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 1.0

    def test_zero_scores(self):
        scorer = _make_scorer(_make_mock_response(0.0, 0.0, 0.0, 0.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.0

    def test_mixed_scores_correct_mean(self):
        scorer = _make_scorer(_make_mock_response(0.8, 0.6, 1.0, 0.4))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        expected = (0.8 + 0.6 + 1.0 + 0.4) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_boundary_values(self):
        scorer = _make_scorer(_make_mock_response(0.0, 1.0, 0.0, 1.0))
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert result.rho_bar == 0.5


# ---------------------------------------------------------------------------
# CritScorer error handling tests
# ---------------------------------------------------------------------------

class TestCritScorerErrors:
    def test_invalid_json_raises(self):
        scorer = _make_scorer("not valid json at all")
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
        scorer = _make_scorer(incomplete)
        with pytest.raises(Exception):  # ValidationError from pydantic
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_empty_json_object_raises(self):
        scorer = _make_scorer("{}")
        with pytest.raises(KeyError):
            scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])

    def test_markdown_code_fence_stripped(self):
        """CritScorer should handle LLM output wrapped in markdown code fences."""
        inner = _make_mock_response(0.7, 0.7, 0.7, 0.7)
        wrapped = f"```json\n{inner}\n```"
        scorer = _make_scorer(wrapped)
        result = scorer.score("case", [{"role": "macro"}], [{"role": "macro"}])
        assert abs(result.rho_bar - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# CritScorer determinism tests
# ---------------------------------------------------------------------------

class TestCritScorerDeterminism:
    def test_same_inputs_same_outputs(self):
        """Given the same mock LLM, CRIT produces identical results."""
        response = _make_mock_response(0.75, 0.85, 0.65, 0.55)
        scorer = _make_scorer(response)
        traces = [{"role": "macro", "content": {"justification": "test"}}]
        decisions = [{"role": "macro", "action_dict": {"orders": []}}]

        r1 = scorer.score("same case", traces, decisions)
        r2 = scorer.score("same case", traces, decisions)

        assert r1.rho_bar == r2.rho_bar
        assert r1.pillar_scores == r2.pillar_scores
        assert r1.diagnostics == r2.diagnostics
