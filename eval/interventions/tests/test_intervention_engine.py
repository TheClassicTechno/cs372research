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
) -> InterventionContext:
    return InterventionContext(
        round_num=round_num,
        stage=stage,
        retry_count=retry_count,
        state={"config": {}},
        js_proposal=js_proposal,
        js_revision=js_revision,
        intervention_history=intervention_history or [],
    )


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

    def test_rule_metadata(self):
        rule = make_js_collapse_rule(threshold=0.5, max_retries=3)
        assert rule.name == "js_collapse"
        assert rule.stage == "post_revision"
        assert rule.max_retries == 3


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
