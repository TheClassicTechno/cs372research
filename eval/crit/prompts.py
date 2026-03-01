"""
Prompt templates for the CRIT blind reasoning auditor.

CRIT evaluates ONLY the logical integrity of agent arguments.
It never sees ground truth, market outcomes, or impact scores.
"""

from __future__ import annotations

import json

CRIT_SYSTEM_PROMPT = """\
You are a reasoning quality auditor (CRIT). Your job is to evaluate the \
logical integrity of arguments produced by a single trading agent during \
a structured debate.

You will be given one agent's reasoning trace and trading decision, along \
with the case context that the agent saw. Evaluate ONLY this agent's \
reasoning — ignore any other agents mentioned in the context.

## What you evaluate

You score FOUR pillars of reasoning quality:

1. **Internal Consistency** — Are the agent's claims logically compatible \
with each other? Does the agent contradict itself within its argument?

2. **Evidence Support** — Are factual claims backed by cited evidence from \
the case context? Are there unsupported assertions?

3. **Trace Alignment** — Does the final trading decision (buy/sell/hold, \
sizing, confidence) logically follow from the reasoning presented? Or does \
the conclusion drift from the argument?

4. **Causal Integrity** — Are causal claims properly scoped? Does the agent \
distinguish between correlation (L1), intervention (L2), and counterfactual \
(L3) reasoning? Are causal leaps flagged?

## What you do NOT evaluate

- You do NOT evaluate correctness of predictions or trading outcomes.
- You do NOT evaluate profitability or market performance.
- You do NOT reward agreement between agents or penalize disagreement.
- You do NOT have access to ground truth or actual market outcomes.

## Scoring rubric

Each pillar is scored on a continuous scale [0.0, 1.0]:
- 0.0 = Severe failure (contradictions, unsupported claims, conclusion drift, causal overreach)
- 0.5 = Mixed / partial (some issues but also some sound reasoning)
- 1.0 = Rigorous (internally consistent, well-supported, aligned, causally sound)

## Output format

You MUST respond with a single JSON object (no markdown, no explanation outside the JSON):

```json
{
  "pillar_scores": {
    "internal_consistency": <float 0.0-1.0>,
    "evidence_support": <float 0.0-1.0>,
    "trace_alignment": <float 0.0-1.0>,
    "causal_integrity": <float 0.0-1.0>
  },
  "diagnostics": {
    "contradictions_detected": <bool>,
    "unsupported_claims_detected": <bool>,
    "conclusion_drift_detected": <bool>,
    "causal_overreach_detected": <bool>
  },
  "explanations": {
    "internal_consistency": "<1-2 sentence explanation>",
    "evidence_support": "<1-2 sentence explanation>",
    "trace_alignment": "<1-2 sentence explanation>",
    "causal_integrity": "<1-2 sentence explanation>"
  }
}
```
"""


def build_crit_user_prompt(
    case_data: str,
    agent_arguments: list[dict],
    decisions: list[dict],
) -> str:
    """Build the CRIT user prompt from case data, agent arguments, and decisions.

    NOTE: This is the legacy multi-agent prompt.  The per-agent scorer uses
    build_crit_single_agent_prompt() instead.

    Args:
        case_data: The enriched context that agents saw (no ground truth).
        agent_arguments: List of agent argument dicts from the debate round.
            Each dict should have at least 'role' and reasoning content.
        decisions: List of agent decision dicts (proposals or revisions).
            Each dict should have at least 'role' and 'action_dict'.

    Returns:
        Formatted user prompt string for the CRIT scorer.
    """
    sections = []

    sections.append("## Case Context\n")
    sections.append(case_data)
    sections.append("")

    sections.append("## Agent Arguments\n")
    for i, arg in enumerate(agent_arguments, 1):
        role = arg.get("role", f"agent_{i}")
        content = arg.get("content", arg)
        sections.append(f"### Agent: {role}")
        if isinstance(content, dict):
            sections.append(json.dumps(content, indent=2))
        else:
            sections.append(str(content))
        sections.append("")

    sections.append("## Agent Decisions\n")
    for i, dec in enumerate(decisions, 1):
        role = dec.get("role", f"agent_{i}")
        action = dec.get("action_dict", dec)
        sections.append(f"### Decision by {role}")
        if isinstance(action, dict):
            sections.append(json.dumps(action, indent=2))
        else:
            sections.append(str(action))
        sections.append("")

    sections.append(
        "## Instructions\n"
        "Evaluate the reasoning quality of the above agent arguments and "
        "decisions across the four pillars: internal_consistency, "
        "evidence_support, trace_alignment, causal_integrity.\n"
        "Respond with the JSON object described in your system prompt."
    )

    return "\n".join(sections)


def build_crit_single_agent_prompt(
    case_data: str,
    role: str,
    agent_traces: list[dict],
    decision: dict | None,
) -> str:
    """Build a CRIT user prompt for a single agent (per-agent ρ_i scoring).

    Per the RAudit paper (Section 3.3), CRIT evaluates each agent's
    reasoning quality individually.  This prompt presents only one
    agent's traces and decision against the shared case context.

    Args:
        case_data: The enriched context that agents saw (no ground truth).
        role: The agent's role name (e.g. "macro", "value").
        agent_traces: List of debate turn dicts for this agent only.
            Each dict should have 'type' (proposal/critique/revision)
            and 'content' with the agent's reasoning.
        decision: The agent's final decision dict (proposal or revision),
            or None if the agent has no decision this round.

    Returns:
        Formatted user prompt string for scoring this single agent.
    """
    sections = []

    sections.append("## Case Context\n")
    sections.append(case_data)
    sections.append("")

    sections.append(f"## Agent Under Evaluation: {role.upper()}\n")

    sections.append("### Reasoning Trace\n")
    if agent_traces:
        for trace in agent_traces:
            trace_type = trace.get("type", "unknown")
            content = trace.get("content", trace)
            sections.append(f"**{trace_type.capitalize()}:**")
            if isinstance(content, dict):
                sections.append(json.dumps(content, indent=2))
            else:
                sections.append(str(content))
            sections.append("")
    else:
        sections.append("(No reasoning traces available for this agent.)")
        sections.append("")

    sections.append("### Trading Decision\n")
    if decision:
        action = decision.get("action_dict", decision)
        if isinstance(action, dict):
            sections.append(json.dumps(action, indent=2))
        else:
            sections.append(str(action))
    else:
        sections.append("(No decision available for this agent.)")
    sections.append("")

    sections.append(
        "## Instructions\n"
        f"Evaluate the reasoning quality of the {role.upper()} agent's "
        "arguments and decision across the four pillars: "
        "internal_consistency, evidence_support, trace_alignment, "
        "causal_integrity.\n"
        "Respond with the JSON object described in your system prompt."
    )

    return "\n".join(sections)
