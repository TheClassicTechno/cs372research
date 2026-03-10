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

_DEFAULT_JS_NUDGE = (
    "The debate appears to have converged unusually quickly.\n\n"
    "Before revising your allocation, reconsider whether your position "
    "reflects your independent analysis or whether it may have been "
    "influenced by other agents' arguments.\n\n"
    "Evaluate the evidence again and ensure your allocation reflects "
    "your own analytical perspective. Maintain diversity of reasoning "
    "where your evidence supports it."
)


def make_js_collapse_rule(
    threshold: float = 0.4,
    min_js_proposal: float = 0.10,
    max_retries: int = 2,
    nudge_prompts: dict[str, str] | None = None,
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
        nudge_prompts: per-agent nudge text ``{role: text}``.  When
            provided, each listed agent receives its own targeted nudge
            and only those agents retry.  When absent, all agents receive
            the default broadcast nudge.
    """
    _nudge_prompts = nudge_prompts or {}

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

        metrics = {
            "js_proposal": js_p,
            "js_revision": js_r,
            "collapse_ratio": round(collapse_ratio, 4),
            "threshold": threshold,
            "min_js_proposal": min_js_proposal,
        }

        if _nudge_prompts:
            # Per-agent targeted nudge — each agent gets its own text
            return InterventionResult(
                rule_name="js_collapse",
                action="retry_revision",
                nudge_text=_nudge_prompts,  # dict, handled by checkpoint
                metrics=metrics,
                severity="warning",
                target_roles=list(_nudge_prompts.keys()),
            )
        else:
            # Broadcast: all agents get the same default nudge
            return InterventionResult(
                rule_name="js_collapse",
                action="retry_revision",
                nudge_text=_DEFAULT_JS_NUDGE,
                metrics=metrics,
                severity="warning",
            )

    return InterventionRule(
        name="js_collapse",
        stage="post_revision",
        max_retries=max_retries,
        evaluate=_evaluate,
    )


# ---------------------------------------------------------------------------
# Reasoning Quality (post_crit, per-agent, pillar-aware)
# ---------------------------------------------------------------------------

_PILLAR_NAMES = [
    "logical_validity",
    "evidential_support",
    "alternative_consideration",
    "causal_alignment",
]

# Pillar-specific remediation guidance, derived from the CRIT scoring rubric.
# Each entry maps a weak pillar to concrete, actionable instructions that
# reference the same concepts the CRIT auditor evaluates.
_PILLAR_GUIDANCE: dict[str, str] = {
    "logical_validity": (
        "LOGICAL VALIDITY — Your argument contains logical reasoning failures. "
        "Check for and fix the following:\n"
        "  - Contradictions: ensure no two claims imply incompatible outcomes "
        "(e.g., one claim implies recession while another implies cyclical expansion).\n"
        "  - Missing reasoning steps: every conclusion must follow from explicit "
        "premises — do not skip intermediate reasoning between evidence and conclusion.\n"
        "  - Portfolio-reasoning mismatch: verify that your revised allocation "
        "logically follows from your thesis. Every position with weight >10%% must "
        "have a supporting claim that justifies it.\n"
        "  - Circular reasoning: do not use your conclusion as evidence for your premises."
    ),
    "evidential_support": (
        "EVIDENTIAL SUPPORT — Your claims lack adequate evidence backing. "
        "Fix the following:\n"
        "  - Unsupported claims: every factual claim MUST cite at least one "
        "memo evidence ID using exact bracketed notation (e.g., [AAPL-RET60], "
        "[L1-VIX]). Do NOT invent evidence labels.\n"
        "  - Evidence-claim mismatch: verify that each cited evidence ID actually "
        "supports the specific claim it is attached to. Remove citations that "
        "are irrelevant or tangential.\n"
        "  - Overclaiming: do not assert conclusions stronger than your evidence "
        "allows. If evidence is ambiguous, qualify your claim with appropriate "
        "uncertainty.\n"
        "  - Cherry-picking: if contradictory evidence exists in the memo, "
        "acknowledge it rather than citing only supportive evidence."
    ),
    "alternative_consideration": (
        "ALTERNATIVE CONSIDERATION — You are not adequately engaging with "
        "competing explanations and critiques. Fix the following:\n"
        "  - Ignored critiques: you MUST explicitly address every critique you "
        "received — either ACCEPT it (and update your reasoning and allocation "
        "accordingly) or REJECT it (with a specific evidence-based rebuttal). "
        "Superficial acknowledgement without reasoning does not count.\n"
        "  - Premature certainty: if your evidence is limited or ambiguous, "
        "do not express high confidence. Hedge appropriately and acknowledge "
        "what you do not know.\n"
        "  - Missing alternatives: consider at least one competing explanation "
        "for the data you cite. What alternative interpretation could challenge "
        "your thesis? Why do you favor your interpretation over it?\n"
        "  - Use your revision_notes to clearly document which critiques you "
        "accepted, which you rejected, and the reasoning for each decision."
    ),
    "causal_alignment": (
        "CAUSAL ALIGNMENT — Your causal reasoning is not justified by your "
        "evidence. Fix the following using Pearl's causal hierarchy:\n"
        "  - Rung collapse: if you claim a causal mechanism (L2), your claim "
        "text must describe the transmission pathway explicitly (e.g., "
        "'Fed raises rates → higher discount rates → lower equity valuations'). "
        "Citing a correlation without describing the mechanism is L1 evidence "
        "supporting an L2 claim — this is causal overreach.\n"
        "  - Counterfactual claims (L3): if you reason about what would have "
        "happened under different conditions, you must explicitly state the "
        "counterfactual scenario. Do not assert counterfactual conclusions "
        "without counterfactual reasoning structure.\n"
        "  - Macro-to-asset mapping: do not directly map macro relationships "
        "to asset outcomes without explaining the transmission mechanism "
        "through which macro conditions affect the specific assets in your "
        "allocation.\n"
        "  - Reasoning type consistency: ensure your claims' reasoning_type "
        "labels (causal/observational/risk_assessment/pattern) match the "
        "actual reasoning structure in the claim text."
    ),
}


def make_reasoning_quality_rule(
    rho_threshold: float = 0.5,
    agent_pillar_thresholds: dict[str, dict[str, float]] | None = None,
    nudge_prompts: dict[str, str] | None = None,
    max_retries: int = 1,
    **_extra,
) -> InterventionRule:
    """Detect weak per-agent reasoning quality after CRIT scoring.

    Fires when any individual agent's rho_i is below *rho_threshold*, OR
    any agent has a pillar score below its configured threshold.  The nudge
    is targeted: only the weak agent(s) receive it.  The nudge includes
    pillar-specific remediation guidance derived from the CRIT rubric, plus
    the CRIT auditor's own explanation of the weakness.

    Pillar thresholds are specified entirely via *agent_pillar_thresholds*.
    Every agent and every pillar MUST be listed — there is no fallback
    default.  Pillars not listed for an agent are not checked.

    Args:
        rho_threshold: agent rho below this triggers the rule.
        agent_pillar_thresholds: ``{agent: {pillar: threshold}}`` — the
            single source of truth for pillar-level intervention triggers.
        nudge_prompts: per-agent custom nudge preamble ``{role: text}``.
            When provided, the custom text is prepended before the
            auto-generated pillar guidance for that agent.
        max_retries: post-crit retries are expensive (re-run revise + CRIT).
    """
    _agent_pillar_thresholds = agent_pillar_thresholds or {}
    _nudge_prompts = nudge_prompts or {}

    def _evaluate(ctx: InterventionContext) -> InterventionResult | None:
        scores = ctx.agent_crit_scores
        if not scores:
            return None

        weak_agents: list[dict] = []

        for role, crit_result in scores.items():
            rho_i = getattr(crit_result, "rho_bar", None)
            pillar_scores = getattr(crit_result, "pillar_scores", None)
            if rho_i is None:
                continue

            weak_pillars: dict[str, float] = {}
            agent_thresholds = _agent_pillar_thresholds.get(role, {})
            if pillar_scores is not None:
                for pname in _PILLAR_NAMES:
                    thresh = agent_thresholds.get(pname)
                    if thresh is None:
                        continue  # no threshold configured → not checked
                    val = getattr(pillar_scores, pname, None)
                    if val is None:
                        continue
                    if val < thresh:
                        weak_pillars[pname] = round(val, 3)

            if rho_i < rho_threshold or weak_pillars:
                # Extract CRIT explanations if available
                explanations = getattr(crit_result, "explanations", None)
                pillar_explanations: dict[str, str] = {}
                if explanations is not None:
                    for pname in weak_pillars:
                        expl = getattr(explanations, pname, None)
                        if expl:
                            pillar_explanations[pname] = expl

                weak_agents.append({
                    "role": role,
                    "rho_i": round(rho_i, 3),
                    "weak_pillars": weak_pillars,
                    "pillar_explanations": pillar_explanations,
                })

        if not weak_agents:
            return None

        # Build per-agent nudge text: custom preamble + pillar guidance
        # When nudge_prompts are configured, each agent gets its own nudge
        # dict. Otherwise, all weak agents share one combined nudge string.
        per_agent_nudges: dict[str, str] = {}

        for wa in weak_agents:
            role = wa["role"]
            lines: list[str] = []

            # Custom preamble from config (if provided for this agent)
            custom = _nudge_prompts.get(role)
            if custom:
                lines.append(custom.strip())
                lines.append("")

            lines.append(
                "A reasoning quality audit identified specific weaknesses in your "
                "revised argument. Your overall reasoning score (rho) and the "
                "specific pillar deficiencies are listed below.\n"
                "\n"
                "You MUST address these weaknesses in your revised allocation. "
                "Focus on the specific remediation steps for each flagged pillar.\n"
            )
            lines.append(f"Agent: {wa['role'].upper()} | rho = {wa['rho_i']:.2f}")
            lines.append("")

            if wa["weak_pillars"]:
                for pname, pval in wa["weak_pillars"].items():
                    # Pillar score + CRIT's own explanation
                    lines.append(f"  [{pname}: {pval:.2f}]")
                    crit_expl = wa.get("pillar_explanations", {}).get(pname)
                    if crit_expl:
                        lines.append(f"  Auditor finding: {crit_expl}")

                    # Pillar-specific remediation guidance
                    guidance = _PILLAR_GUIDANCE.get(pname)
                    if guidance:
                        lines.append(f"  {guidance}")
                    lines.append("")
            else:
                # Triggered by low rho but no individual pillar below threshold
                lines.append(
                    "  Your overall reasoning score is below the quality "
                    "threshold. Review your argument for the issues described "
                    "below and strengthen your weakest areas.\n"
                )
                # Include guidance for all pillars when rho is low globally
                for pname in _PILLAR_NAMES:
                    pillar_scores = scores[wa["role"]].pillar_scores
                    val = getattr(pillar_scores, pname, None)
                    if val is not None:
                        lines.append(f"  [{pname}: {val:.2f}]")
                        crit_expl = wa.get("pillar_explanations", {}).get(pname)
                        if crit_expl:
                            lines.append(f"  Auditor finding: {crit_expl}")
                        # Only include remediation for pillars that are
                        # actually below rho_star (not perfect)
                        if val < 0.7:
                            guidance = _PILLAR_GUIDANCE.get(pname)
                            if guidance:
                                lines.append(f"  {guidance}")
                        lines.append("")

            per_agent_nudges[role] = "\n".join(lines)

        return InterventionResult(
            rule_name="reasoning_quality",
            action="retry_revision",
            nudge_text=per_agent_nudges,
            metrics={
                "weak_agents": weak_agents,
                "rho_threshold": rho_threshold,
            },
            severity="warning",
            target_roles=[wa["role"] for wa in weak_agents],
        )

    return InterventionRule(
        name="reasoning_quality",
        stage="post_crit",
        max_retries=max_retries,
        evaluate=_evaluate,
    )


# ---------------------------------------------------------------------------
# Rule Registry
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, callable] = {
    "js_collapse": make_js_collapse_rule,
    "reasoning_quality": make_reasoning_quality_rule,
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
        # Skip disabled rules
        if not rule_params.pop("enabled", True):
            continue
        # Remove 'stage' from params if present — it's set by the factory
        rule_params.pop("stage", None)
        rules.append(factory(**rule_params))
    if not rules:
        return None
    return InterventionEngine(rules)
