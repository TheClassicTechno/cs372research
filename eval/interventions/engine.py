"""InterventionEngine — stateless evaluator for debate intervention rules.

The engine accepts the current debate state + computed metrics, evaluates
all registered rules, and returns those that fire.  It does NOT own the
retry loop — the runner does.
"""

from __future__ import annotations

from .types import InterventionContext, InterventionResult, InterventionRule


class InterventionEngine:
    """Evaluate intervention rules against debate state."""

    def __init__(self, rules: list[InterventionRule]) -> None:
        self._rules = list(rules)

    @property
    def rules(self) -> list[InterventionRule]:
        return list(self._rules)

    def evaluate(self, ctx: InterventionContext) -> list[InterventionResult]:
        """Evaluate all rules for the given stage. Returns fired results.

        Rules are evaluated in registration order.  All matching rules
        fire (no short-circuit) so the runner can prioritize.
        """
        results: list[InterventionResult] = []
        for rule in self._rules:
            if rule.stage != ctx.stage:
                continue
            if ctx.retry_count >= rule.max_retries:
                continue  # exhausted retries for this rule
            result = rule.evaluate(ctx)
            if result is not None:
                results.append(result)
        return results
