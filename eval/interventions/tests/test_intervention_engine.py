"""Unit tests for the intervention engine, rules, and types."""

from __future__ import annotations

import pytest

from eval.interventions.types import (
    InterventionContext,
    InterventionResult,
    InterventionRule,
)
from eval.interventions.engine import InterventionEngine
from eval.interventions.rules import (
    build_intervention_engine,
    make_js_collapse_rule,
    make_reasoning_quality_rule,
    RULE_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    *,
    stage: str = "post_revision",
    retry_count: int = 0,
    js_proposal: float | None = 0.42,
    js_revision: float | None = 0.08,
    round_num: int = 1,
    intervention_history: list | None = None,
    rho_bar: float | None = None,
    agent_crit_scores: dict | None = None,
) -> InterventionContext:
    return InterventionContext(
        round_num=round_num,
        stage=stage,
        retry_count=retry_count,
        state={"config": {}},
        js_proposal=js_proposal,
        js_revision=js_revision,
        rho_bar=rho_bar,
        agent_crit_scores=agent_crit_scores,
        intervention_history=intervention_history or [],
    )


def _make_mock_crit_scores(
    agent_rhos: dict[str, float],
    pillar_overrides: dict[str, dict[str, float]] | None = None,
    explanation_overrides: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Build mock agent_crit_scores dict (role → mock CritResult-like object)."""
    from types import SimpleNamespace

    pillar_overrides = pillar_overrides or {}
    explanation_overrides = explanation_overrides or {}
    scores = {}
    for role, rho in agent_rhos.items():
        pillars = pillar_overrides.get(role, {})
        ps = SimpleNamespace(
            logical_validity=pillars.get("logical_validity", 0.7),
            evidential_support=pillars.get("evidential_support", 0.7),
            alternative_consideration=pillars.get("alternative_consideration", 0.7),
            causal_alignment=pillars.get("causal_alignment", 0.7),
        )
        expls = explanation_overrides.get(role, {})
        ex = SimpleNamespace(
            logical_validity=expls.get("logical_validity", ""),
            evidential_support=expls.get("evidential_support", ""),
            alternative_consideration=expls.get("alternative_consideration", ""),
            causal_alignment=expls.get("causal_alignment", ""),
        )
        scores[role] = SimpleNamespace(rho_bar=rho, pillar_scores=ps, explanations=ex)
    return scores


# ---------------------------------------------------------------------------
# InterventionContext tests
# ---------------------------------------------------------------------------

class TestInterventionContext:
    def test_default_fields(self):
        ctx = _make_ctx()
        assert ctx.ov_revision is None
        assert ctx.rho_bar is None
        assert ctx.agent_crit_scores is None
        assert ctx.pid_result is None
        assert ctx.intervention_history == []

    def test_history_mutable(self):
        ctx = _make_ctx(intervention_history=[{"round": 1}])
        assert len(ctx.intervention_history) == 1


# ---------------------------------------------------------------------------
# InterventionResult tests
# ---------------------------------------------------------------------------

class TestInterventionResult:
    def test_construction(self):
        r = InterventionResult(
            rule_name="test",
            action="retry_revision",
            nudge_text="Reconsider your position.",
            metrics={"ratio": 0.2},
            severity="warning",
        )
        assert r.rule_name == "test"
        assert r.action == "retry_revision"
        assert r.severity == "warning"
        assert r.metrics["ratio"] == 0.2
        assert r.target_roles is None  # default: broadcast

    def test_target_roles(self):
        r = InterventionResult(
            rule_name="test",
            action="retry_revision",
            nudge_text="Fix it.",
            metrics={},
            severity="warning",
            target_roles=["macro", "value"],
        )
        assert r.target_roles == ["macro", "value"]


# ---------------------------------------------------------------------------
# JS Collapse Rule tests
# ---------------------------------------------------------------------------

class TestJSCollapseRule:
    def test_fires_on_collapse(self):
        """JS drops from 0.42 to 0.08 → collapse_ratio=0.19 < 0.4 → fires."""
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.08)
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_name == "js_collapse"
        assert result.action == "retry_revision"
        assert result.severity == "warning"
        assert result.metrics["collapse_ratio"] == pytest.approx(0.1905, abs=0.001)

    def test_no_fire_above_threshold(self):
        """JS drops modestly → collapse_ratio > threshold → no fire."""
        rule = make_js_collapse_rule(threshold=0.4)
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.30)
        result = rule.evaluate(ctx)
        assert result is None

    def test_min_js_proposal_guard(self):
        """When JS_proposal < min_js_proposal, skip (noise, not herding)."""
        rule = make_js_collapse_rule(min_js_proposal=0.10)
        ctx = _make_ctx(js_proposal=0.03, js_revision=0.01)
        result = rule.evaluate(ctx)
        assert result is None

    def test_moderate_diversity_collapse(self):
        """JS_proposal=0.15 > min, collapse_ratio=0.27 < 0.4 → fires."""
        rule = make_js_collapse_rule(threshold=0.4, min_js_proposal=0.10)
        ctx = _make_ctx(js_proposal=0.15, js_revision=0.04)
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.metrics["collapse_ratio"] == pytest.approx(0.2667, abs=0.001)

    def test_none_js_proposal(self):
        """Missing JS_proposal → skip."""
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=None, js_revision=0.08)
        result = rule.evaluate(ctx)
        assert result is None

    def test_none_js_revision(self):
        """Missing JS_revision → skip."""
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=0.42, js_revision=None)
        result = rule.evaluate(ctx)
        assert result is None

    def test_js_revision_zero(self):
        """JS_revision=0 → collapse_ratio=0 → fires (if proposal had diversity)."""
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.0)
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.metrics["collapse_ratio"] == 0.0

    def test_custom_threshold(self):
        """Custom threshold=0.6 catches milder collapses."""
        rule = make_js_collapse_rule(threshold=0.6)
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.20)
        result = rule.evaluate(ctx)
        assert result is not None  # 0.20/0.42 = 0.476 < 0.6

    def test_nudge_text_present(self):
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.08)
        result = rule.evaluate(ctx)
        assert "converged unusually quickly" in result.nudge_text
        assert "independent analysis" in result.nudge_text

    def test_broadcast_nudge_no_target_roles(self):
        """JS collapse is broadcast — target_roles should be None."""
        rule = make_js_collapse_rule()
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.08)
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.target_roles is None

    def test_rule_metadata(self):
        rule = make_js_collapse_rule(threshold=0.5, max_retries=3)
        assert rule.name == "js_collapse"
        assert rule.stage == "post_revision"
        assert rule.max_retries == 3


# ---------------------------------------------------------------------------
# Reasoning Quality Rule tests
# ---------------------------------------------------------------------------

class TestReasoningQualityRule:
    """All pillar threshold tests use agent_pillar_thresholds (the single
    source of truth).  Pillars not listed for an agent are not checked."""

    # Helper: standard 3-agent thresholds matching the ablation_3 config
    ALL_PILLARS_04 = {
        "macro": {
            "logical_validity": 0.40, "evidential_support": 0.40,
            "alternative_consideration": 0.40, "causal_alignment": 0.50,
        },
        "technical": {
            "logical_validity": 0.50, "evidential_support": 0.40,
            "alternative_consideration": 0.40, "causal_alignment": 0.40,
        },
        "value": {
            "logical_validity": 0.40, "evidential_support": 0.45,
            "alternative_consideration": 0.40, "causal_alignment": 0.40,
        },
    }

    def test_fires_on_low_rho(self):
        """Agent rho below threshold triggers the rule."""
        rule = make_reasoning_quality_rule(rho_threshold=0.5)
        scores = _make_mock_crit_scores({"macro": 0.4, "value": 0.7, "technical": 0.8})
        ctx = _make_ctx(
            stage="post_crit",
            agent_crit_scores=scores,
        )
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.rule_name == "reasoning_quality"
        assert result.action == "retry_revision"
        assert len(result.metrics["weak_agents"]) == 1
        assert result.metrics["weak_agents"][0]["role"] == "macro"
        assert result.target_roles == ["macro"]

    def test_fires_on_low_pillar(self):
        """Agent with one pillar below its threshold triggers the rule."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"macro": {"causal_alignment": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.6, "value": 0.6},
            pillar_overrides={"macro": {"causal_alignment": 0.25}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        assert len(result.metrics["weak_agents"]) == 1
        wa = result.metrics["weak_agents"][0]
        assert wa["role"] == "macro"
        assert "causal_alignment" in wa["weak_pillars"]

    def test_no_fire_above_thresholds(self):
        """All agents above both rho and pillar thresholds → no fire."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.5,
            agent_pillar_thresholds=self.ALL_PILLARS_04,
        )
        scores = _make_mock_crit_scores({"macro": 0.8, "value": 0.7, "technical": 0.8})
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is None

    def test_no_thresholds_means_no_pillar_checks(self):
        """Without agent_pillar_thresholds, only rho is checked."""
        rule = make_reasoning_quality_rule(rho_threshold=0.3)
        scores = _make_mock_crit_scores(
            {"macro": 0.6},
            pillar_overrides={"macro": {"causal_alignment": 0.01}},  # extremely low
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is None  # rho 0.6 > 0.3, no pillar thresholds configured

    def test_none_scores_skip(self):
        """Missing agent_crit_scores → skip."""
        rule = make_reasoning_quality_rule()
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=None)
        result = rule.evaluate(ctx)
        assert result is None

    def test_empty_scores_skip(self):
        """Empty agent_crit_scores dict → skip."""
        rule = make_reasoning_quality_rule()
        ctx = _make_ctx(stage="post_crit", agent_crit_scores={})
        result = rule.evaluate(ctx)
        assert result is None

    def test_multiple_weak_agents(self):
        """Multiple agents with low rho are all reported."""
        rule = make_reasoning_quality_rule(rho_threshold=0.5)
        scores = _make_mock_crit_scores({"macro": 0.3, "value": 0.4, "technical": 0.8})
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        roles = {wa["role"] for wa in result.metrics["weak_agents"]}
        assert roles == {"macro", "value"}
        assert set(result.target_roles) == {"macro", "value"}

    def test_nudge_text_mentions_agent(self):
        """Nudge text identifies the weak agent by name."""
        rule = make_reasoning_quality_rule(rho_threshold=0.5)
        scores = _make_mock_crit_scores({"macro": 0.3, "value": 0.8})
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert isinstance(result.nudge_text, dict)
        assert "macro" in result.nudge_text
        nudge = result.nudge_text["macro"]
        assert "MACRO" in nudge
        assert "rho = 0.30" in nudge

    def test_nudge_pillar_specific_guidance_causal(self):
        """Low causal_alignment triggers Pearl's hierarchy guidance."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"macro": {"causal_alignment": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.6},
            pillar_overrides={"macro": {"causal_alignment": 0.2}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        nudge = result.nudge_text["macro"]
        assert "Pearl" in nudge
        assert "transmission" in nudge.lower()
        assert "rung collapse" in nudge.lower()
        assert "causal_alignment: 0.20" in nudge

    def test_nudge_pillar_specific_guidance_evidential(self):
        """Low evidential_support triggers evidence citation guidance."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"value": {"evidential_support": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"value": 0.6},
            pillar_overrides={"value": {"evidential_support": 0.15}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        nudge = result.nudge_text["value"]
        assert "memo evidence ID" in nudge
        assert "bracketed notation" in nudge

    def test_nudge_pillar_specific_guidance_alternative(self):
        """Low alternative_consideration triggers critique handling guidance."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"technical": {"alternative_consideration": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"technical": 0.6},
            pillar_overrides={"technical": {"alternative_consideration": 0.2}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        nudge = result.nudge_text["technical"]
        assert "critique" in nudge.lower()
        assert "ACCEPT" in nudge
        assert "REJECT" in nudge

    def test_nudge_pillar_specific_guidance_logical(self):
        """Low logical_validity triggers contradiction/reasoning step guidance."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"macro": {"logical_validity": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.6},
            pillar_overrides={"macro": {"logical_validity": 0.25}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        nudge = result.nudge_text["macro"]
        assert "Contradictions" in nudge
        assert "portfolio-reasoning mismatch" in nudge.lower()

    def test_nudge_includes_crit_explanation(self):
        """CRIT auditor explanation is passed through in the nudge."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.3,
            agent_pillar_thresholds={"macro": {"causal_alignment": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.6},
            pillar_overrides={"macro": {"causal_alignment": 0.2}},
            explanation_overrides={"macro": {
                "causal_alignment": "Claim C2 asserts rate impact on tech but provides no mechanism.",
            }},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        nudge = result.nudge_text["macro"]
        assert "Claim C2 asserts rate impact" in nudge
        assert "Auditor finding" in nudge

    def test_rule_metadata(self):
        rule = make_reasoning_quality_rule(rho_threshold=0.6, max_retries=2)
        assert rule.name == "reasoning_quality"
        assert rule.stage == "post_crit"
        assert rule.max_retries == 2

    def test_combined_rho_and_pillar(self):
        """Agent has both low rho and a weak pillar."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.5,
            agent_pillar_thresholds={"macro": {"evidential_support": 0.3}},
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.3},
            pillar_overrides={"macro": {"evidential_support": 0.2}},
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        wa = result.metrics["weak_agents"][0]
        assert wa["rho_i"] == 0.3
        assert "evidential_support" in wa["weak_pillars"]

    # --- Per-agent pillar threshold tests ---

    def test_different_thresholds_per_agent(self):
        """Each agent has its own pillar thresholds; only violations fire."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.2,
            agent_pillar_thresholds={
                "macro": {"causal_alignment": 0.6},
                "value": {"evidential_support": 0.6},
                "technical": {"logical_validity": 0.6},
            },
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.8, "value": 0.8, "technical": 0.8},
            pillar_overrides={
                "macro": {"causal_alignment": 0.5},       # 0.5 < 0.6 → fires
                "value": {"evidential_support": 0.7},      # 0.7 >= 0.6 → ok
                "technical": {"logical_validity": 0.55},   # 0.55 < 0.6 → fires
            },
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        roles = {wa["role"] for wa in result.metrics["weak_agents"]}
        assert roles == {"macro", "technical"}
        assert "value" not in roles

    def test_unlisted_pillar_not_checked(self):
        """Pillars not in agent_pillar_thresholds are never checked."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.2,
            agent_pillar_thresholds={
                "macro": {"causal_alignment": 0.5},  # only this pillar checked
            },
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.8},
            pillar_overrides={
                "macro": {
                    "logical_validity": 0.01,            # very low, but not listed
                    "evidential_support": 0.01,          # very low, but not listed
                    "alternative_consideration": 0.01,   # very low, but not listed
                    "causal_alignment": 0.6,             # above threshold → ok
                },
            },
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is None  # only causal checked, and it's above 0.5

    def test_unlisted_agent_not_checked_for_pillars(self):
        """Agents not in agent_pillar_thresholds have no pillar checks."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.2,
            agent_pillar_thresholds={
                "macro": {"causal_alignment": 0.5},
            },
        )
        scores = _make_mock_crit_scores(
            {"macro": 0.8, "technical": 0.8},
            pillar_overrides={
                "macro": {"causal_alignment": 0.6},       # above → ok
                "technical": {"logical_validity": 0.01},   # very low but agent not listed
            },
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is None

    def test_full_config_all_agents_all_pillars(self):
        """Production-style config: all 3 agents × 4 pillars fully specified."""
        rule = make_reasoning_quality_rule(
            rho_threshold=0.5,
            agent_pillar_thresholds=self.ALL_PILLARS_04,
        )
        # macro: causal_alignment=0.45 < 0.50 → fires
        # value: evidential_support=0.40 < 0.45 → fires
        # technical: all above thresholds → ok
        scores = _make_mock_crit_scores(
            {"macro": 0.8, "value": 0.8, "technical": 0.8},
            pillar_overrides={
                "macro": {"causal_alignment": 0.45},
                "value": {"evidential_support": 0.40},
                "technical": {"logical_validity": 0.55},
            },
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        weak_map = {wa["role"]: wa["weak_pillars"] for wa in result.metrics["weak_agents"]}
        assert "causal_alignment" in weak_map["macro"]
        assert "evidential_support" in weak_map["value"]
        assert "technical" not in weak_map


# ---------------------------------------------------------------------------
# InterventionEngine tests
# ---------------------------------------------------------------------------

class TestInterventionEngine:
    def test_stage_filtering(self):
        """Only rules matching the stage are evaluated."""
        rule_post_rev = make_js_collapse_rule()
        # Create a dummy post_crit rule that always fires
        always_fire = InterventionRule(
            name="always",
            stage="post_crit",
            max_retries=1,
            evaluate=lambda ctx: InterventionResult(
                rule_name="always", action="retry_revision",
                nudge_text="test", metrics={}, severity="warning",
            ),
        )
        engine = InterventionEngine([rule_post_rev, always_fire])

        # post_revision stage — only js_collapse should be evaluated
        ctx = _make_ctx(stage="post_revision", js_proposal=0.42, js_revision=0.08)
        results = engine.evaluate(ctx)
        assert len(results) == 1
        assert results[0].rule_name == "js_collapse"

        # post_crit stage — only "always" should fire
        ctx_crit = _make_ctx(stage="post_crit", js_proposal=0.42, js_revision=0.08)
        results_crit = engine.evaluate(ctx_crit)
        assert len(results_crit) == 1
        assert results_crit[0].rule_name == "always"

    def test_retry_exhaustion(self):
        """Rules with exhausted retries are skipped."""
        rule = make_js_collapse_rule(max_retries=2)
        engine = InterventionEngine([rule])

        # retry_count=0 → fires
        ctx0 = _make_ctx(retry_count=0, js_proposal=0.42, js_revision=0.08)
        assert len(engine.evaluate(ctx0)) == 1

        # retry_count=1 → fires (max_retries=2, so 0 and 1 are both valid)
        ctx1 = _make_ctx(retry_count=1, js_proposal=0.42, js_revision=0.08)
        assert len(engine.evaluate(ctx1)) == 1

        # retry_count=2 → exhausted, skip
        ctx2 = _make_ctx(retry_count=2, js_proposal=0.42, js_revision=0.08)
        assert len(engine.evaluate(ctx2)) == 0

    def test_no_rules_returns_empty(self):
        engine = InterventionEngine([])
        ctx = _make_ctx()
        assert engine.evaluate(ctx) == []

    def test_all_rules_evaluated(self):
        """All matching rules fire (no short-circuit)."""
        rule_a = InterventionRule(
            name="rule_a", stage="post_revision", max_retries=1,
            evaluate=lambda ctx: InterventionResult(
                rule_name="rule_a", action="retry_revision",
                nudge_text="A", metrics={}, severity="warning",
            ),
        )
        rule_b = InterventionRule(
            name="rule_b", stage="post_revision", max_retries=1,
            evaluate=lambda ctx: InterventionResult(
                rule_name="rule_b", action="retry_revision",
                nudge_text="B", metrics={}, severity="critical",
            ),
        )
        engine = InterventionEngine([rule_a, rule_b])
        ctx = _make_ctx()
        results = engine.evaluate(ctx)
        assert len(results) == 2
        names = {r.rule_name for r in results}
        assert names == {"rule_a", "rule_b"}

    def test_rule_returns_none_skipped(self):
        """Rules that return None are not included in results."""
        no_fire = InterventionRule(
            name="no_fire", stage="post_revision", max_retries=1,
            evaluate=lambda ctx: None,
        )
        yes_fire = make_js_collapse_rule()
        engine = InterventionEngine([no_fire, yes_fire])
        ctx = _make_ctx(js_proposal=0.42, js_revision=0.08)
        results = engine.evaluate(ctx)
        assert len(results) == 1
        assert results[0].rule_name == "js_collapse"


# ---------------------------------------------------------------------------
# build_intervention_engine tests
# ---------------------------------------------------------------------------

class TestBuildInterventionEngine:
    def test_none_config(self):
        assert build_intervention_engine(None) is None

    def test_disabled_config(self):
        assert build_intervention_engine({"enabled": False}) is None

    def test_empty_rules(self):
        assert build_intervention_engine({"enabled": True, "rules": {}}) is None

    def test_unknown_rule_skipped(self):
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {"nonexistent_rule": {"threshold": 0.5}},
        })
        assert engine is None

    def test_js_collapse_config(self):
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {
                "js_collapse": {
                    "threshold": 0.5,
                    "min_js_proposal": 0.15,
                    "max_retries": 3,
                },
            },
        })
        assert engine is not None
        assert len(engine.rules) == 1
        rule = engine.rules[0]
        assert rule.name == "js_collapse"
        assert rule.max_retries == 3

    def test_stage_in_config_ignored(self):
        """'stage' in config params shouldn't break the factory."""
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {
                "js_collapse": {"stage": "post_revision", "threshold": 0.4},
            },
        })
        assert engine is not None

    def test_rule_registry_has_js_collapse(self):
        assert "js_collapse" in RULE_REGISTRY

    def test_rule_registry_has_reasoning_quality(self):
        assert "reasoning_quality" in RULE_REGISTRY

    def test_reasoning_quality_config(self):
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {
                "reasoning_quality": {
                    "rho_threshold": 0.6,
                    "max_retries": 2,
                },
            },
        })
        assert engine is not None
        assert len(engine.rules) == 1
        rule = engine.rules[0]
        assert rule.name == "reasoning_quality"
        assert rule.stage == "post_crit"
        assert rule.max_retries == 2

    def test_combined_js_and_reasoning_config(self):
        """Both js_collapse and reasoning_quality can be registered together."""
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {
                "js_collapse": {"threshold": 0.4, "max_retries": 2},
                "reasoning_quality": {"rho_threshold": 0.5, "max_retries": 1},
            },
        })
        assert engine is not None
        assert len(engine.rules) == 2
        names = {r.name for r in engine.rules}
        assert names == {"js_collapse", "reasoning_quality"}
        stages = {r.stage for r in engine.rules}
        assert stages == {"post_revision", "post_crit"}

    def test_reasoning_quality_with_agent_pillar_thresholds_config(self):
        """agent_pillar_thresholds pass through from config as single source of truth."""
        engine = build_intervention_engine({
            "enabled": True,
            "rules": {
                "reasoning_quality": {
                    "rho_threshold": 0.5,
                    "agent_pillar_thresholds": {
                        "macro": {
                            "logical_validity": 0.40,
                            "evidential_support": 0.40,
                            "alternative_consideration": 0.40,
                            "causal_alignment": 0.50,
                        },
                        "value": {
                            "logical_validity": 0.40,
                            "evidential_support": 0.45,
                            "alternative_consideration": 0.40,
                            "causal_alignment": 0.40,
                        },
                        "technical": {
                            "logical_validity": 0.50,
                            "evidential_support": 0.40,
                            "alternative_consideration": 0.40,
                            "causal_alignment": 0.40,
                        },
                    },
                    "max_retries": 1,
                },
            },
        })
        assert engine is not None
        rule = engine.rules[0]
        assert rule.name == "reasoning_quality"

        # macro: causal_alignment=0.48 < 0.50 → fires
        # value: evidential_support=0.42 < 0.45 → fires
        # technical: all above thresholds → ok
        scores = _make_mock_crit_scores(
            {"macro": 0.8, "value": 0.8, "technical": 0.8},
            pillar_overrides={
                "macro": {"causal_alignment": 0.48},
                "value": {"evidential_support": 0.42},
                "technical": {"logical_validity": 0.55},
            },
        )
        ctx = _make_ctx(stage="post_crit", agent_crit_scores=scores)
        result = rule.evaluate(ctx)
        assert result is not None
        roles = {wa["role"] for wa in result.metrics["weak_agents"]}
        assert roles == {"macro", "value"}
