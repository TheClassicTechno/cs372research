"""Integration tests for CRIT scorer in the debate loop.

All tests use mock LLM functions — no real API calls.
"""

import json

import pytest

from eval.crit import CritScorer, CritResult


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

def _mock_crit_llm(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Return a mock LLM function that produces a valid CRIT response."""
    response = json.dumps({
        "pillar_scores": {
            "internal_consistency": ic,
            "evidence_support": es,
            "trace_alignment": ta,
            "causal_integrity": ci,
        },
        "diagnostics": {
            "contradictions_detected": False,
            "unsupported_claims_detected": False,
            "conclusion_drift_detected": False,
            "causal_overreach_detected": False,
        },
        "explanations": {
            "internal_consistency": "No issues found.",
            "evidence_support": "Claims are well supported.",
            "trace_alignment": "Decision follows reasoning.",
            "causal_integrity": "Causal claims are sound.",
        },
    })
    return lambda sys, usr: response


# ---------------------------------------------------------------------------
# Mock debate data
# ---------------------------------------------------------------------------

MOCK_CASE_DATA = "NVDA reported strong Q3 earnings with revenue up 20% YoY."

MOCK_AGENT_TRACES = [
    {
        "role": "macro",
        "type": "proposal",
        "content": {
            "justification": "Strong macro tailwinds support growth.",
            "confidence": 0.8,
        },
    },
    {
        "role": "value",
        "type": "proposal",
        "content": {
            "justification": "Valuation stretched but justified by growth.",
            "confidence": 0.6,
        },
    },
]

MOCK_DECISIONS = [
    {
        "role": "macro",
        "action_dict": {
            "orders": [{"ticker": "NVDA", "side": "buy", "size": 100}],
            "justification": "Buy on strength.",
            "confidence": 0.8,
        },
    },
    {
        "role": "value",
        "action_dict": {
            "orders": [{"ticker": "NVDA", "side": "buy", "size": 50}],
            "justification": "Moderate buy.",
            "confidence": 0.6,
        },
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCritInLoop:
    def test_crit_returns_crit_result(self):
        """CRIT invoked after mock debate round returns CritResult."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert isinstance(result, CritResult)

    def test_rho_bar_computed_correctly(self):
        """rho_bar is mean of four pillar scores."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.8, 0.7, 0.9, 0.6))
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        expected = (0.8 + 0.7 + 0.9 + 0.6) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_no_ground_truth_required(self):
        """CRIT operates without any outcome information in case_data."""
        # case_data has no ground truth, outcomes, or impact scores
        case_data = "Company reported earnings. No forward guidance available."
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(case_data, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert isinstance(result, CritResult)

    def test_no_broker_interaction(self):
        """CRIT does not import or interact with the broker."""
        import eval.crit.scorer as scorer_module
        source = open(scorer_module.__file__).read()
        assert "broker" not in source.lower()

    def test_crit_can_run_without_pid(self):
        """CRIT scorer works standalone without PID controller."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        # No PID involved — just pure CRIT scoring
        assert 0.0 <= result.rho_bar <= 1.0

    def test_crit_output_deterministic(self):
        """Given the same mock LLM, CRIT produces identical results."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.75, 0.85, 0.65, 0.55))
        r1 = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        r2 = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert r1.rho_bar == r2.rho_bar
        assert r1.pillar_scores == r2.pillar_scores
