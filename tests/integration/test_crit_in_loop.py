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
            "logical_validity": ic,
            "evidential_support": es,
            "alternative_consideration": ta,
            "causal_alignment": ci,
        },
        "diagnostics": {
            "contradictions_detected": False,
            "unsupported_claims_detected": False,
            "ignored_critiques_detected": False,
            "premature_certainty_detected": False,
            "causal_overreach_detected": False,
            "conclusion_drift_detected": False,
        },
        "explanations": {
            "logical_validity": "No issues found.",
            "evidential_support": "Claims are well supported.",
            "alternative_consideration": "Decision follows reasoning.",
            "causal_alignment": "Causal claims are sound.",
        },
    }


def _mock_crit_llm(ic=0.8, es=0.7, ta=0.9, ci=0.6):
    """Return a mock LLM function that produces a valid single-agent CRIT response."""
    entry = _make_crit_entry(ic, es, ta, ci)
    response = json.dumps(entry)
    return lambda sys, usr, **kw: response


def _mock_crit_llm_per_role(role_scores: dict[str, tuple]):
    """Return a mock LLM that returns different scores based on agent_role in prompt.

    Args:
        role_scores: Mapping of role name → (ic, es, ta, ci) tuple.
    """
    responses = {}
    for role, (ic, es, ta, ci) in role_scores.items():
        responses[role] = json.dumps(_make_crit_entry(ic, es, ta, ci))

    def _llm(sys_prompt: str, usr_prompt: str, **kw) -> str:
        # Match the agent role from the "## Agent Under Evaluation" section.
        # The template renders {{ agent_role | upper }} on its own line.
        for role in role_scores:
            if f"\n{role.upper()}\n" in usr_prompt:
                return responses[role]
        # Fallback: return first entry
        return next(iter(responses.values()))

    return _llm


# ---------------------------------------------------------------------------
# Mock reasoning bundles
# ---------------------------------------------------------------------------

def _make_bundle(role: str) -> dict:
    """Build a minimal reasoning bundle for one agent."""
    return {
        "round": 1,
        "agent_role": role,
        "proposal": {
            "thesis": f"{role} thesis",
            "portfolio_allocation": {"NVDA": 0.5},
            "reasoning": {
                "claims": [],
                "position_rationale": [],
                "thesis": f"{role} thesis",
                "confidence": 0.8,
                "risks_or_falsifiers": [],
            },
            "raw_response": f"{role} reasoning for buy.",
            "evidence_citations": [],
        },
        "critiques_received": [],
        "revised_argument": {
            "thesis": f"{role} revised thesis",
            "portfolio_allocation": {"NVDA": 0.5},
            "reasoning": {
                "claims": [],
                "position_rationale": [],
                "thesis": f"{role} revised thesis",
                "confidence": 0.85,
                "risks_or_falsifiers": [],
            },
            "raw_response": f"{role} revised reasoning.",
            "evidence_citations": [],
        },
    }


MOCK_BUNDLES_2 = {
    "macro": _make_bundle("macro"),
    "value": _make_bundle("value"),
}

MOCK_BUNDLES_4 = {role: _make_bundle(role) for role in ALL_ROLES}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCritInLoop:
    def test_crit_returns_round_crit_result(self):
        """CRIT invoked after mock debate round returns RoundCritResult."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_BUNDLES_2)
        assert isinstance(result, RoundCritResult)

    def test_per_agent_scores_present(self):
        """Each agent has its own CritResult in the RoundCritResult."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_BUNDLES_2)
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
        result = scorer.score(MOCK_BUNDLES_2)
        expected = (0.9 + 0.5) / 2.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_rho_bar_computed_correctly_uniform(self):
        """With uniform mock LLM, all agents get same score, ρ̄ = ρ_i."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.8, 0.7, 0.9, 0.6))
        result = scorer.score(MOCK_BUNDLES_2)
        expected = (0.8 + 0.7 + 0.9 + 0.6) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_no_ground_truth_required(self):
        """CRIT operates without any outcome information in bundles."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_BUNDLES_2)
        assert isinstance(result, RoundCritResult)

    def test_no_broker_interaction(self):
        """CRIT does not import or interact with the broker."""
        import eval.crit.scorer as scorer_module
        source = open(scorer_module.__file__).read()
        assert "broker" not in source.lower()

    def test_crit_can_run_without_pid(self):
        """CRIT scorer works standalone without PID controller."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_BUNDLES_2)
        assert 0.0 <= result.rho_bar <= 1.0

    def test_crit_output_deterministic(self):
        """Given the same mock LLM, CRIT produces identical results."""
        scorer = CritScorer(llm_fn=_mock_crit_llm(0.75, 0.85, 0.65, 0.55))
        r1 = scorer.score(MOCK_BUNDLES_2)
        r2 = scorer.score(MOCK_BUNDLES_2)
        assert r1.rho_bar == r2.rho_bar
        assert r1.agent_scores.keys() == r2.agent_scores.keys()

    def test_llm_called_once_per_agent(self):
        """The LLM is invoked once per agent (parallel per-agent calls)."""
        call_count = 0
        entry = _make_crit_entry()

        def counting_llm(sys, usr, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps(entry)

        scorer = CritScorer(llm_fn=counting_llm)
        scorer.score(MOCK_BUNDLES_4)
        assert call_count == 4  # one call per agent

    def test_four_agent_scoring(self):
        """Standard 4-agent debate produces 4 per-agent scores."""
        scorer = CritScorer(llm_fn=_mock_crit_llm())
        result = scorer.score(MOCK_BUNDLES_4)
        assert len(result.agent_scores) == 4
        assert set(result.agent_scores.keys()) == {"macro", "value", "risk", "technical"}
