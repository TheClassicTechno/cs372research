"""Tests for the RAudit-style CRIT scorer (four-pillar blind evaluation).

Tests cover:
1. Logical contradiction → low logical_validity
2. Unsupported claim → low evidential_support
3. No alternative considered → low alternative_consideration
4. Rung collapse → low causal_alignment
5. Clean structured reasoning → high gamma
6. Score normalisation
7. Response parsing
8. Threshold configurability
9. Artifact builder + schema validation
10. Full pipeline with transcript parser
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.crit_raudit import (
    RAuditCRITScorer,
    RAuditCRITResult,
    PillarScore,
    PillarScores,
    _normalize_score,
    build_raudit_eval_artifact,
    DEFAULT_GAMMA_THRESHOLD,
    DEFAULT_THETA_THRESHOLD,
)
from eval.crit.transcript_parser import TranscriptParser

FIXTURES = Path(__file__).parent.parent / "crit" / "tests" / "fixtures"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_mock_response(
    lv: int = 7,
    es: int = 7,
    ac: int = 7,
    ca: int = 7,
    theta: int = 7,
    consistent: bool = True,
    notes: str = "Reasonable argument.",
) -> dict:
    """Create a mock LLM response matching the RAudit CRIT output schema."""
    gamma = (lv + es + ac + ca) / 4
    return {
        "pillars": {
            "logical_validity": {"score": lv, "justification": f"LV score {lv}."},
            "evidential_support": {"score": es, "justification": f"ES score {es}."},
            "alternative_consideration": {"score": ac, "justification": f"AC score {ac}."},
            "causal_alignment": {"score": ca, "justification": f"CA score {ca}."},
        },
        "trace_output_consistent": consistent,
        "gamma": gamma,
        "theta": theta,
        "notes": notes,
    }


def _make_mock_scorer(responses: list[dict] | None = None) -> RAuditCRITScorer:
    """Create a RAuditCRITScorer with a mocked LLM client."""
    import jinja2

    scorer = RAuditCRITScorer.__new__(RAuditCRITScorer)
    scorer._model = "gpt-4o-mini"
    scorer._temperature = 0.0
    scorer._gamma_threshold = DEFAULT_GAMMA_THRESHOLD
    scorer._theta_threshold = DEFAULT_THETA_THRESHOLD
    scorer._jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            str(Path(__file__).parent.parent / "prompts")
        ),
        undefined=jinja2.StrictUndefined,
    )

    if responses is None:
        responses = [_make_mock_response()]

    call_count = {"n": 0}

    async def mock_generate(request):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        resp = MagicMock()
        resp.content = responses[idx]
        return resp

    scorer._client = MagicMock()
    scorer._client.generate = mock_generate
    return scorer


def _run_async(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------

class TestNormalizeScore:
    def test_normalizes_1_to_10(self):
        assert _normalize_score(1) == 0.1
        assert _normalize_score(5) == 0.5
        assert _normalize_score(10) == 1.0

    def test_clamps_low(self):
        assert _normalize_score(0) == 0.0
        assert _normalize_score(-5) == 0.0

    def test_clamps_high(self):
        assert _normalize_score(15) == 1.0

    def test_handles_float(self):
        assert _normalize_score(7.5) == 0.75

    def test_handles_none(self):
        assert _normalize_score(None) == 0.0

    def test_handles_string(self):
        assert _normalize_score("invalid") == 0.0


# ---------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------

class TestParseResponse:
    def test_parses_valid_response(self):
        data = _make_mock_response(lv=8, es=6, ac=7, ca=5, theta=6)
        result = RAuditCRITScorer._parse_response(data)
        assert result.pillars.logical_validity.score == 0.8
        assert result.pillars.evidential_support.score == 0.6
        assert result.pillars.alternative_consideration.score == 0.7
        assert result.pillars.causal_alignment.score == 0.5
        # gamma = mean of pillars = (0.8 + 0.6 + 0.7 + 0.5) / 4 = 0.65
        assert result.gamma == pytest.approx(0.65, abs=1e-4)
        assert result.theta == 0.6
        assert result.notes == "Reasonable argument."

    def test_parses_json_string(self):
        data = json.dumps(_make_mock_response(lv=7, es=7, ac=7, ca=7, theta=7))
        result = RAuditCRITScorer._parse_response(data)
        assert result.gamma == 0.7

    def test_handles_missing_pillars(self):
        result = RAuditCRITScorer._parse_response({"gamma": 5, "theta": 5, "notes": "ok"})
        assert result.pillars.logical_validity.score == 0.0
        # gamma is mean of pillars (all 0), not from raw gamma
        assert result.gamma == 0.0

    def test_handles_empty_response(self):
        result = RAuditCRITScorer._parse_response({})
        assert result.gamma == 0.0
        assert result.theta == 0.0
        assert result.notes == ""

    def test_trace_output_consistency_flag(self):
        data = _make_mock_response(consistent=False)
        result = RAuditCRITScorer._parse_response(data)
        assert result.trace_output_consistent is False


# ---------------------------------------------------------------
# Pathology-specific tests
# ---------------------------------------------------------------

class TestLogicalContradiction:
    """Pillar 1: Conclusion contradicts reasoning → low logical_validity."""

    def test_contradiction_low_logical_validity(self):
        response = _make_mock_response(
            lv=2, es=6, ac=5, ca=6, theta=4,
            consistent=False,
            notes="Conclusion contradicts the reasoning trace.",
        )

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="NVDA is significantly overvalued and facing headwinds.",
                reasons="High P/E ratio, export controls, slowing growth.",
                counterarguments="None identified.",
                assumptions="Growth will decelerate.",
                final_decision="BUY 10% position with high conviction.",
            )

        result = _run_async(_run())
        assert result.pillars.logical_validity.score == 0.2
        assert result.trace_output_consistent is False
        # gamma should be low due to low LV
        assert result.gamma < 0.6


class TestUnsupportedClaim:
    """Pillar 2: Claims made without evidence → low evidential_support."""

    def test_unsupported_claim_low_evidential_support(self):
        response = _make_mock_response(
            lv=6, es=2, ac=5, ca=5, theta=3,
            notes="Claims lack supporting evidence.",
        )

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="AAPL will triple in price next quarter.",
                reasons="It just feels like it will go up.",
                counterarguments="None identified.",
                assumptions="None stated.",
                final_decision="BUY 50% position.",
            )

        result = _run_async(_run())
        assert result.pillars.evidential_support.score == 0.2
        assert result.gamma < 0.6


class TestNoAlternativeConsidered:
    """Pillar 3: No competing hypotheses explored → low alternative_consideration."""

    def test_no_alternatives_low_score(self):
        response = _make_mock_response(
            lv=7, es=6, ac=2, ca=6, theta=5,
            notes="One-sided reasoning with no alternatives considered.",
        )

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="NVDA will continue growing indefinitely.",
                reasons="Strong earnings, CUDA moat, data center demand.",
                counterarguments="None identified.",
                assumptions="Nothing can stop NVDA growth.",
                final_decision="BUY 100% position.",
            )

        result = _run_async(_run())
        assert result.pillars.alternative_consideration.score == 0.2


class TestRungCollapse:
    """Pillar 4: Using L1 correlation evidence for L2 causal claim → low causal_alignment."""

    def test_rung_collapse_low_causal_alignment(self):
        response = _make_mock_response(
            lv=6, es=5, ac=5, ca=2, theta=4,
            notes="Rung collapse: correlation cited as causation.",
        )

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="Increasing GPU sales will cause revenue to grow 50%.",
                reasons="Historical correlation shows GPU sales and revenue move together.",
                counterarguments="None identified.",
                assumptions="Correlation equals causation.",
                final_decision="BUY based on causal relationship.",
            )

        result = _run_async(_run())
        assert result.pillars.causal_alignment.score == 0.2
        assert result.gamma < 0.6


class TestCleanStructuredReasoning:
    """All pillars strong → high gamma, passes threshold."""

    def test_clean_reasoning_high_gamma(self):
        response = _make_mock_response(
            lv=9, es=8, ac=8, ca=9, theta=8,
            notes="Well-structured reasoning with strong evidence.",
        )

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="NVDA is a strong buy based on data center demand acceleration.",
                reasons="40% YoY earnings growth, Blackwell GPU cycle, CUDA ecosystem moat. Data center CapEx growing 30% YoY per hyperscaler reports.",
                counterarguments="Export controls could reduce TAM by 15-20%. 35x forward P/E assumes sustained growth.",
                assumptions="Hyperscaler spending remains elevated. CUDA moat prevents competitive erosion.",
                final_decision="BUY 5% position (confidence: 0.8).",
                context="NVDA at $890, data center revenue accelerating.",
            )

        result = _run_async(_run())
        # gamma = mean of (0.9, 0.8, 0.8, 0.9) = 0.85
        assert result.gamma == pytest.approx(0.85, abs=1e-4)
        assert result.gamma >= DEFAULT_GAMMA_THRESHOLD


# ---------------------------------------------------------------
# Threshold configurability
# ---------------------------------------------------------------

class TestThresholds:
    def test_default_thresholds(self):
        assert DEFAULT_GAMMA_THRESHOLD == 0.8
        assert DEFAULT_THETA_THRESHOLD == 0.5

    def test_custom_threshold_via_artifact(self):
        result = RAuditCRITResult(gamma=0.75, theta=0.6)
        # With default threshold (0.8), this should fail
        results = [("t1", result)]
        artifact = build_raudit_eval_artifact("d1", "r1", results)
        assert artifact["run_summary"]["crit_summary"]["threshold_pass"] is False

        # With custom threshold (0.7), this should pass
        artifact = build_raudit_eval_artifact(
            "d1", "r1", results, gamma_threshold=0.7
        )
        assert artifact["run_summary"]["crit_summary"]["threshold_pass"] is True


# ---------------------------------------------------------------
# Mocked LLM integration
# ---------------------------------------------------------------

class TestScorerWithMockedLLM:
    def test_single_call_per_turn(self):
        """RAudit CRIT makes exactly 1 LLM call per turn (not per-reason)."""
        call_count = {"n": 0}

        async def mock_generate(request):
            call_count["n"] += 1
            resp = MagicMock()
            resp.content = _make_mock_response()
            return resp

        import jinja2
        scorer = RAuditCRITScorer.__new__(RAuditCRITScorer)
        scorer._client = MagicMock()
        scorer._client.generate = mock_generate
        scorer._model = "gpt-4o-mini"
        scorer._temperature = 0.0
        scorer._gamma_threshold = DEFAULT_GAMMA_THRESHOLD
        scorer._theta_threshold = DEFAULT_THETA_THRESHOLD
        scorer._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                str(Path(__file__).parent.parent / "prompts")
            ),
            undefined=jinja2.StrictUndefined,
        )

        _run_async(scorer.score_async(
            claim="Test claim",
            reasons="Test reasons",
            counterarguments="Test counters",
        ))

        assert call_count["n"] == 1

    def test_gamma_is_mean_of_pillars(self):
        """gamma should be computed as mean of 4 pillar scores."""
        response = _make_mock_response(lv=8, es=6, ac=7, ca=5, theta=6)

        async def _run():
            scorer = _make_mock_scorer([response])
            return await scorer.score_async(
                claim="Test", reasons="Reasons",
            )

        result = _run_async(_run())
        expected_gamma = (0.8 + 0.6 + 0.7 + 0.5) / 4
        assert result.gamma == pytest.approx(expected_gamma, abs=1e-4)


# ---------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------

class TestArtifactBuilder:
    def _make_result(self, gamma: float, theta: float) -> RAuditCRITResult:
        return RAuditCRITResult(
            gamma=gamma, theta=theta,
            pillars=PillarScores(), notes="test",
        )

    def test_single_turn_pass(self):
        results = [("turn1", self._make_result(0.85, 0.7))]
        artifact = build_raudit_eval_artifact("debate1", "run1", results)
        assert artifact["schema_version"] == "1.2.0"
        assert artifact["run_summary"]["overall_verdict"] == "pass"
        assert artifact["run_summary"]["crit_summary"]["threshold_pass"] is True

    def test_single_turn_fail(self):
        results = [("turn1", self._make_result(0.5, 0.3))]
        artifact = build_raudit_eval_artifact("debate1", "run1", results)
        assert artifact["run_summary"]["overall_verdict"] == "fail"
        assert artifact["run_summary"]["crit_summary"]["threshold_pass"] is False

    def test_multi_turn_mixed(self):
        results = [
            ("turn1", self._make_result(0.85, 0.7)),
            ("turn2", self._make_result(0.5, 0.3)),
        ]
        artifact = build_raudit_eval_artifact("debate1", "run1", results)
        assert artifact["run_summary"]["overall_verdict"] == "mixed"

    def test_empty_turns(self):
        artifact = build_raudit_eval_artifact("debate1", "run1", [])
        assert artifact["run_summary"]["overall_verdict"] == "fail"
        assert artifact["run_summary"]["crit_summary"]["gamma_mean"] is None

    def test_raudit_version_in_metadata(self):
        results = [("t1", self._make_result(0.85, 0.7))]
        artifact = build_raudit_eval_artifact("d1", "r1", results)
        assert artifact["eval_metadata"]["raudit_version"] is not None
        assert "raudit" in artifact["eval_metadata"]["raudit_version"]

    def test_experiment_config(self):
        results = [("t1", self._make_result(0.85, 0.7))]
        artifact = build_raudit_eval_artifact(
            "d1", "r1", results, experiment_label="raudit_baseline"
        )
        assert artifact["experiment_config"]["label"] == "raudit_baseline"


# ---------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------

class TestSchemaValidation:
    @pytest.fixture
    def eval_schema(self):
        schema_path = Path(__file__).parent.parent / "schemas" / "eval.schema.json"
        if not schema_path.exists():
            pytest.skip("eval.schema.json not found")
        with open(schema_path) as f:
            return json.load(f)

    def test_artifact_validates_against_schema(self, eval_schema):
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        result = RAuditCRITResult(gamma=0.85, theta=0.7, notes="test")
        artifact = build_raudit_eval_artifact("debate1", "run1", [("t1", result)])
        jsonschema.validate(artifact, eval_schema)

    def test_empty_artifact_validates(self, eval_schema):
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        artifact = build_raudit_eval_artifact("debate1", "run1", [])
        jsonschema.validate(artifact, eval_schema)


# ---------------------------------------------------------------
# Full pipeline with transcript parser
# ---------------------------------------------------------------

class TestFullPipelineIntegration:
    @pytest.fixture
    def minimal_trace(self):
        with open(FIXTURES / "sample_minimal_trace.json") as f:
            return json.load(f)

    def test_parse_and_score(self, minimal_trace):
        """End-to-end: parse transcript → score with RAudit CRIT → build artifact."""

        async def _run():
            parser = TranscriptParser(minimal_trace)
            traces = parser.extract_traces()
            assert len(traces) == 2  # 1 proposal + 1 judge

            responses = [
                _make_mock_response(lv=8, es=7, ac=6, ca=7, theta=7),
                _make_mock_response(lv=9, es=8, ac=7, ca=8, theta=8),
            ]
            scorer = _make_mock_scorer(responses)

            turn_results = []
            for trace in traces:
                result = await scorer.score_async(
                    claim=trace.claim,
                    reasons=trace.reasons,
                    counterarguments=trace.counterarguments,
                    assumptions=trace.assumptions,
                    final_decision=trace.final_decision,
                    context=trace.context,
                )
                turn_results.append((trace.turn_id, result))

            return turn_results

        turn_results = _run_async(_run())

        artifact = build_raudit_eval_artifact(
            debate_id="nvda_minimal",
            run_id="test-raudit-001",
            turn_results=turn_results,
        )

        assert artifact["schema_version"] == "1.2.0"
        assert artifact["eval_metadata"]["raudit_version"] is not None
        assert len(artifact["turn_evaluations"]) == 2

        # Both turns should have reasonable scores
        for te in artifact["turn_evaluations"]:
            assert te["crit"]["gamma_mean"] is not None
            assert te["crit"]["gamma_mean"] > 0

    @pytest.fixture
    def full_trace(self):
        with open(FIXTURES / "sample_multi_agent_trace.json") as f:
            return json.load(f)

    def test_full_multi_agent_pipeline(self, full_trace):
        """7-turn multi-agent trace through RAudit CRIT."""

        async def _run():
            parser = TranscriptParser(full_trace)
            traces = parser.extract_traces()
            assert len(traces) == 7

            responses = [
                _make_mock_response(lv=8, es=7, ac=7, ca=7, theta=7)
            ] * 7
            scorer = _make_mock_scorer(responses)

            turn_results = []
            for trace in traces:
                result = await scorer.score_async(
                    claim=trace.claim,
                    reasons=trace.reasons,
                    counterarguments=trace.counterarguments,
                    assumptions=trace.assumptions,
                    final_decision=trace.final_decision,
                    context=trace.context,
                )
                turn_results.append((trace.turn_id, result))

            return turn_results

        turn_results = _run_async(_run())

        artifact = build_raudit_eval_artifact(
            debate_id="multi_agent_test",
            run_id="test-raudit-002",
            turn_results=turn_results,
            experiment_label="raudit_multi_agent",
        )

        assert len(artifact["turn_evaluations"]) == 7
        assert artifact["run_summary"]["overall_verdict"] == "fail"  # 0.725 < 0.8 threshold


# ---------------------------------------------------------------
# PillarScores helper
# ---------------------------------------------------------------

class TestPillarScoresHelper:
    def test_as_list(self):
        ps = PillarScores(
            logical_validity=PillarScore(score=0.8),
            evidential_support=PillarScore(score=0.6),
            alternative_consideration=PillarScore(score=0.7),
            causal_alignment=PillarScore(score=0.9),
        )
        assert ps.as_list() == [0.8, 0.6, 0.7, 0.9]

    def test_default_scores(self):
        ps = PillarScores()
        assert ps.as_list() == [0.0, 0.0, 0.0, 0.0]
