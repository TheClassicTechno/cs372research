"""Built-in intervention rule implementations.

Each rule is exposed as a factory function that returns an InterventionRule.
New rules are added here and registered in RULE_REGISTRY.
"""

from __future__ import annotations

from .engine import InterventionEngine
from .types import InterventionContext, InterventionResult, InterventionRule


# ---------------------------------------------------------------------------
# JS Collapse Detection (post_revision)
# ---------------------------------------------------------------------------

def make_js_collapse_rule(
    threshold: float = 0.4,
    min_js_proposal: float = 0.10,
    max_retries: int = 2,
    **_extra,
) -> InterventionRule:
    """Detect premature convergence by comparing pre- and post-revision JS.

    Fires when JS_revision / JS_proposal < threshold, indicating agents
    collapsed toward identical allocations during revision.

    Args:
        threshold: collapse_ratio below this triggers the rule.
        min_js_proposal: skip when proposals already near-consensus
            (JS_proposal below this value means there wasn't meaningful
            diversity to protect).
        max_retries: maximum retry attempts for this rule.
    """

    def _evaluate(ctx: InterventionContext) -> InterventionResult | None:
        js_p = ctx.js_proposal
        js_r = ctx.js_revision
        if js_p is None or js_r is None:
            return None
        # Guard: proposals must have meaningful diversity to protect
        if js_p < min_js_proposal:
            return None
        collapse_ratio = js_r / js_p
        if collapse_ratio >= threshold:
            return None  # no collapse detected
        return InterventionResult(
            rule_name="js_collapse",
            action="retry_revision",
            nudge_text=(
                "The debate appears to have converged unusually quickly.\n\n"
                "Before revising your allocation, reconsider whether your position "
                "reflects your independent analysis or whether it may have been "
                "influenced by other agents' arguments.\n\n"
                "Evaluate the evidence again and ensure your allocation reflects "
                "your own analytical perspective. Maintain diversity of reasoning "
                "where your evidence supports it."
            ),
            metrics={
                "js_proposal": js_p,
                "js_revision": js_r,
                "collapse_ratio": round(collapse_ratio, 4),
                "threshold": threshold,
                "min_js_proposal": min_js_proposal,
            },
            severity="warning",
        )

    return InterventionRule(
        name="js_collapse",
        stage="post_revision",
        max_retries=max_retries,
        evaluate=_evaluate,
    )


# ---------------------------------------------------------------------------
# Rule Registry
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, callable] = {
    "js_collapse": make_js_collapse_rule,
}


def build_intervention_engine(config: dict | None) -> InterventionEngine | None:
    """Build an InterventionEngine from a config dict.

    Config format::

        {
            "enabled": true,
            "rules": {
                "js_collapse": {
                    "threshold": 0.4,
                    "min_js_proposal": 0.10,
                    "max_retries": 2,
                },
            },
        }

    Returns None if config is None, disabled, or has no rules.
    """
    if not config or not config.get("enabled"):
        return None
    rules: list[InterventionRule] = []
    for name, params in config.get("rules", {}).items():
        factory = RULE_REGISTRY.get(name)
        if factory is None:
            continue
        # Pass params as kwargs; factories accept **_extra for forward-compat
        rule_params = dict(params) if params else {}
        # Remove 'stage' from params if present — it's set by the factory
        rule_params.pop("stage", None)
        rules.append(factory(**rule_params))
    if not rules:
        return None
    return InterventionEngine(rules)
