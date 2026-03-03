"""Integration tests for CRIT scorer in the debate loop.

All tests use mock LLM functions — no real API calls.
Validates per-agent scoring per RAudit paper Section 3.3.
"""

import json

import pytest

from eval.crit import CritScorer, CritResult, RoundCritResult


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

ALL_ROLES = ["macro", "value", "risk", "technical"]


def _make_crit_entry(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Build a single-agent CRIT response dict."""
    return {
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
    }


def _mock_crit_llm(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Return a mock LLM function that produces a valid batch CRIT response."""
    entry = _make_crit_entry(ic, es, ta, ci)
    response = json.dumps({role: entry for role in ALL_ROLES})
    return lambda sys, usr: response


def _mock_crit_llm_per_role(role_scores: dict[str, tuple]):
    """Return a mock LLM that returns different scores per agent role (batch format).

    Args:
        role_scores: Mapping of role name → (ic, es, ta, ci) tuple.
    """
    batch = {}
    for role, (ic, es, ta, ci) in role_scores.items():
        batch[role] = _make_crit_entry(ic, es, ta, ci)
    response = json.dumps(batch)
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
    def test_crit_returns_round_crit_result(self):
        """CRIT invoked after mock debate round returns RoundCritResult."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert isinstance(result, RoundCritResult)

    def test_per_agent_scores_present(self):
        """Each agent has its own CritResult in the RoundCritResult."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert "macro" in result.agent_scores
        assert "value" in result.agent_scores
        assert isinstance(result.agent_scores["macro"], CritResult)
        assert isinstance(result.agent_scores["value"], CritResult)

    def test_rho_bar_is_mean_of_per_agent(self):
        """ρ̄ = 1/n Σ_i ρ_i per RAudit Algorithm 1 line 8."""
        scorer = CritScorer(llm_fn=_mock_crit_llm_per_role({
            "macro": (0.9, 0.9, 0.9, 0.9),  # ρ_i = 0.9
            "value": (0.5, 0.5, 0.5, 0.5),  # ρ_i = 0.5
        }))
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        expected = (0.9 + 0.5) / 2.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_rho_bar_computed_correctly_uniform(self):
        """With uniform mock LLM, all agents get same score, ρ̄ = ρ_i."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.8, 0.7, 0.9, 0.6))
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        expected = (0.8 + 0.7 + 0.9 + 0.6) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_no_ground_truth_required(self):
        """CRIT operates without any outcome information in case_data."""
        case_data = "Company reported earnings. No forward guidance available."
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(case_data, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert isinstance(result, RoundCritResult)

    def test_no_broker_interaction(self):
        """CRIT does not import or interact with the broker."""
        import eval.crit.scorer as scorer_module
        source = open(scorer_module.__file__).read()
        assert "broker" not in source.lower()

    def test_crit_can_run_without_pid(self):
        """CRIT scorer works standalone without PID controller."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert 0.0 <= result.rho_bar <= 1.0

    def test_crit_output_deterministic(self):
        """Given the same mock LLM, CRIT produces identical results."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.75, 0.85, 0.65, 0.55))
        r1 = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        r2 = scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert r1.rho_bar == r2.rho_bar
        assert r1.agent_scores.keys() == r2.agent_scores.keys()

    def test_llm_called_once_for_batch(self):
        """The LLM is invoked once for all agents (batch mode)."""
        call_count = 0
        entry = _make_crit_entry()

        def counting_llm(sys, usr):
            nonlocal call_count
            call_count += 1
            return json.dumps({role: entry for role in ALL_ROLES})

        scorer = CritScorer(llm_fn=counting_llm)
        scorer.score(MOCK_CASE_DATA, MOCK_AGENT_TRACES, MOCK_DECISIONS)
        assert call_count == 1  # single batch call for all agents

    def test_four_agent_scoring(self):
        """Standard 4-agent debate produces 4 per-agent scores."""
        traces = [
            {"role": "macro", "type": "proposal", "content": "macro reasoning"},
            {"role": "value", "type": "proposal", "content": "value reasoning"},
            {"role": "risk", "type": "proposal", "content": "risk reasoning"},
            {"role": "technical", "type": "proposal", "content": "tech reasoning"},
        ]
        decisions = [
            {"role": "macro", "action_dict": {"confidence": 0.8}},
            {"role": "value", "action_dict": {"confidence": 0.6}},
            {"role": "risk", "action_dict": {"confidence": 0.5}},
            {"role": "technical", "action_dict": {"confidence": 0.7}},
        ]
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_CASE_DATA, traces, decisions)
        assert len(result.agent_scores) == 4
        assert set(result.agent_scores.keys()) == {"macro", "value", "risk", "technical"}
