"""Unit tests for CRIT per-agent scoring logic (CritScorer + rho_bar computation)."""

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
            "logical_validity": ic,
            "evidential_support": es,
            "alternative_consideration": ta,
            "causal_alignment": ci,
        },
        "diagnostics": {
            "contradictions_detected": ic < 0.5,
            "unsupported_claims_detected": es < 0.5,
            "ignored_critiques_detected": False,
            "premature_certainty_detected": False,
            "causal_overreach_detected": ci < 0.5,
            "conclusion_drift_detected": ta < 0.5,
        },
        "explanations": {
            "logical_validity": "Test explanation for IC.",
            "evidential_support": "Test explanation for ES.",
            "alternative_consideration": "Test explanation for TA.",
            "causal_alignment": "Test explanation for CI.",
        },
    }


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
            "raw_response": f"{role} reasoning.",
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


def _make_scorer_uniform(**kwargs) -> CritScorer:
    """Create a CritScorer with a mock LLM that returns same scores for all agents."""
    response_text = json.dumps(_make_single_agent_response(**kwargs))
    return CritScorer(llm_fn=lambda sys, usr, **kw: response_text)


def _make_scorer_per_role(role_scores: dict[str, dict]) -> CritScorer:
    """Create a CritScorer that returns different scores per agent role.

    Args:
        role_scores: Mapping of role → kwargs for _make_single_agent_response.
    """
    responses = {
        role: json.dumps(_make_single_agent_response(**scores))
        for role, scores in role_scores.items()
    }

    def _llm(sys_prompt: str, usr_prompt: str, **kw) -> str:
        # Match the agent role from the "## Agent Under Evaluation" section.
        # The template renders {{ agent_role | upper }} on its own line.
        for role in role_scores:
            if f"\n{role.upper()}\n" in usr_prompt:
                return responses[role]
        return next(iter(responses.values()))

    return CritScorer(llm_fn=_llm)


BUNDLES_2 = {
    "macro": _make_bundle("macro"),
    "value": _make_bundle("value"),
}

BUNDLES_1 = {"macro": _make_bundle("macro")}


# ---------------------------------------------------------------------------
# Per-agent scoring tests
# ---------------------------------------------------------------------------

class TestPerAgentScoring:
    def test_returns_round_crit_result(self):
        """score() returns RoundCritResult, not CritResult."""
        scorer = _make_scorer_uniform(ic=0.8, es=0.7, ta=0.9, ci=0.6)
        result = scorer.score(BUNDLES_2)
        assert isinstance(result, RoundCritResult)

    def test_scores_each_agent_individually(self):
        """Each agent gets its own CritResult in agent_scores."""
        scorer = _make_scorer_uniform(ic=0.8, es=0.7, ta=0.9, ci=0.6)
        result = scorer.score(BUNDLES_2)
        assert "macro" in result.agent_scores
        assert "value" in result.agent_scores
        assert len(result.agent_scores) == 2

    def test_different_scores_per_agent(self):
        """Different agents can receive different CRIT scores."""
        scorer = _make_scorer_per_role({
            "macro": {"ic": 1.0, "es": 1.0, "ta": 1.0, "ci": 1.0},
            "value": {"ic": 0.0, "es": 0.0, "ta": 0.0, "ci": 0.0},
        })
        result = scorer.score(BUNDLES_2)
        assert result.agent_scores["macro"].rho_bar == 1.0
        assert result.agent_scores["value"].rho_bar == 0.0
        assert result.rho_bar == 0.5  # mean of 1.0 and 0.0

    def test_rho_bar_is_mean_of_per_agent_rho_bars(self):
        """ρ̄ = 1/n Σ_i ρ_i per RAudit Algorithm 1 line 8."""
        scorer = _make_scorer_per_role({
            "macro": {"ic": 0.8, "es": 0.8, "ta": 0.8, "ci": 0.8},
            "value": {"ic": 0.4, "es": 0.4, "ta": 0.4, "ci": 0.4},
        })
        result = scorer.score(BUNDLES_2)
        expected = (0.8 + 0.4) / 2.0
        assert abs(result.rho_bar - expected) < 1e-9


# ---------------------------------------------------------------------------
# rho_bar computation tests (uniform scores)
# ---------------------------------------------------------------------------

class TestRhoBarComputation:
    def test_perfect_scores(self):
        scorer = _make_scorer_uniform(ic=1.0, es=1.0, ta=1.0, ci=1.0)
        result = scorer.score(BUNDLES_1)
        assert result.rho_bar == 1.0

    def test_zero_scores(self):
        scorer = _make_scorer_uniform(ic=0.0, es=0.0, ta=0.0, ci=0.0)
        result = scorer.score(BUNDLES_1)
        assert result.rho_bar == 0.0

    def test_mixed_scores_correct_mean(self):
        scorer = _make_scorer_uniform(ic=0.8, es=0.6, ta=1.0, ci=0.4)
        result = scorer.score(BUNDLES_1)
        expected = (0.8 + 0.6 + 1.0 + 0.4) / 4.0
        assert abs(result.rho_bar - expected) < 1e-9

    def test_boundary_values(self):
        scorer = _make_scorer_uniform(ic=0.0, es=1.0, ta=0.0, ci=1.0)
        result = scorer.score(BUNDLES_1)
        assert result.rho_bar == 0.5


# ---------------------------------------------------------------------------
# CritScorer error handling tests
# ---------------------------------------------------------------------------

class TestCritScorerErrors:
    def test_invalid_json_crashes(self):
        """Invalid JSON exhausts retries and crashes the debate."""
        scorer = CritScorer(llm_fn=lambda sys, usr, **kw: "not valid json at all")
        with pytest.raises(RuntimeError, match="CRIT scoring failed"):
            scorer.score(BUNDLES_1)

    def test_partial_response_missing_pillar_crashes(self):
        """Missing pillar fields exhaust retries and crash the debate."""
        incomplete = json.dumps({
            "pillar_scores": {
                "logical_validity": 0.8,
                # missing other pillars
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
                "logical_validity": "ok",
                "evidential_support": "ok",
                "alternative_consideration": "ok",
                "causal_alignment": "ok",
            },
        })
        scorer = CritScorer(llm_fn=lambda sys, usr, **kw: incomplete)
        with pytest.raises(RuntimeError, match="CRIT scoring failed"):
            scorer.score(BUNDLES_1)

    def test_empty_bundles_raises(self):
        scorer = _make_scorer_uniform()
        with pytest.raises(ValueError, match="must not be empty"):
            scorer.score({})

    def test_markdown_code_fence_stripped(self):
        """CritScorer should handle LLM output wrapped in markdown code fences."""
        inner = json.dumps(_make_single_agent_response(ic=0.7, es=0.7, ta=0.7, ci=0.7))
        wrapped = f"```json\n{inner}\n```"
        scorer = CritScorer(llm_fn=lambda sys, usr, **kw: wrapped)
        result = scorer.score(BUNDLES_1)
        assert abs(result.rho_bar - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# CritScorer determinism tests
# ---------------------------------------------------------------------------

class TestCritScorerDeterminism:
    def test_same_inputs_same_outputs(self):
        """Given the same mock LLM, CRIT produces identical results."""
        scorer = _make_scorer_uniform(ic=0.75, es=0.85, ta=0.65, ci=0.55)

        r1 = scorer.score(BUNDLES_1)
        r2 = scorer.score(BUNDLES_1)

        assert r1.rho_bar == r2.rho_bar
        assert r1.agent_scores["macro"].pillar_scores == r2.agent_scores["macro"].pillar_scores
        assert r1.agent_scores["macro"].diagnostics == r2.agent_scores["macro"].diagnostics
