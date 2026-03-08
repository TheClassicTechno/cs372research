"""
Prompt templates for the CRIT blind reasoning auditor.

CRIT evaluates ONLY the logical integrity of agent arguments.
It never sees ground truth, market outcomes, or impact scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import jinja2

# Module-level Jinja environment singleton — templates loaded once and cached.
# Templates live alongside this __init__.py in the prompts/ package directory.
_TEMPLATE_DIR = Path(__file__).parent
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    undefined=jinja2.StrictUndefined,
)


def render_crit_prompts(
    bundle: dict,
    system_template: str = "crit_system_enumerated.jinja",
    user_template: str = "crit_user_master.jinja",
) -> tuple[str, str]:
    """Render CRIT system + user prompts from a reasoning bundle.

    Args:
        bundle: Dict with keys: round, agent_role, proposal,
                critiques_received, revised_argument.
        system_template: Filename for the CRIT system prompt template.
        user_template: Filename for the CRIT user prompt template.

    Returns:
        (system_prompt, user_prompt) tuple of rendered strings.
    """
    system_tmpl = _JINJA_ENV.get_template(system_template)
    user_tmpl = _JINJA_ENV.get_template(user_template)
    system_prompt = system_tmpl.render(agent_role=bundle["agent_role"])

    # Strip raw_response from bundle sections before rendering — it duplicates
    # the structured reasoning field and wastes tokens in the CRIT prompt.
    def _strip_raw(d):
        if isinstance(d, dict):
            return {k: v for k, v in d.items() if k != "raw_response"}
        return d

    user_prompt = user_tmpl.render(
        round=bundle["round"],
        agent_role=bundle["agent_role"],
        proposal=_strip_raw(bundle["proposal"]),
        critiques_received=bundle["critiques_received"],
        revised_argument=_strip_raw(bundle["revised_argument"]),
    )
    return system_prompt, user_prompt

# DEPRECATED: Use render_crit_prompts() with Jinja templates instead.
CRIT_SYSTEM_PROMPT = """\
You are a reasoning quality auditor (CRIT). Your job is to evaluate the \
logical integrity of arguments produced by a single trading agent during \
a structured debate.

You will be given one agent's reasoning trace and trading decision, along \
with the case context that the agent saw. Evaluate ONLY this agent's \
reasoning — ignore any other agents mentioned in the context.

## What you evaluate

You score FOUR pillars of reasoning quality:

1. **Logical Validity** — Does the reasoning logically support the conclusion? \
Are the agent's claims logically compatible with each other?

2. **Evidential Support** — Are factual claims backed by cited evidence from \
the case context? Are there unsupported assertions?

3. **Alternative Consideration** — Does the agent consider competing \
explanations? Are alternative hypotheses acknowledged or ignored?

4. **Causal Alignment** — Are causal claims justified by the evidence? Does \
the agent distinguish between correlation (L1), intervention (L2), and \
counterfactual (L3) reasoning?

## What you do NOT evaluate

- You do NOT evaluate correctness of predictions or trading outcomes.
- You do NOT evaluate profitability or market performance.
- You do NOT reward agreement between agents or penalize disagreement.
- You do NOT have access to ground truth or actual market outcomes.

## Scoring rubric

Each pillar is scored on a continuous scale [0.0, 1.0]:
- 0.0 = Severe failure (contradictions, unsupported claims, ignored alternatives, causal overreach)
- 0.5 = Mixed / partial (some issues but also some sound reasoning)
- 1.0 = Rigorous (logically valid, well-supported, considers alternatives, causally sound)

## Output format

You MUST respond with a single JSON object (no markdown, no explanation outside the JSON):

```json
{
  "pillar_scores": {
    "logical_validity": <float 0.0-1.0>,
    "evidential_support": <float 0.0-1.0>,
    "alternative_consideration": <float 0.0-1.0>,
    "causal_alignment": <float 0.0-1.0>
  },
  "diagnostics": {
    "contradictions_detected": <bool>,
    "unsupported_claims_detected": <bool>,
    "ignored_critiques_detected": <bool>,
    "premature_certainty_detected": <bool>,
    "causal_overreach_detected": <bool>,
    "conclusion_drift_detected": <bool>
  },
  "explanations": {
    "logical_validity": "<1-2 sentence explanation>",
    "evidential_support": "<1-2 sentence explanation>",
    "alternative_consideration": "<1-2 sentence explanation>",
    "causal_alignment": "<1-2 sentence explanation>"
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

    .. deprecated:: Use render_crit_prompts() with Jinja templates instead.

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
        role = arg["role"]
        content = arg.get("content") or arg
        sections.append(f"### Agent: {role}")
        if isinstance(content, dict):
            sections.append(json.dumps(content, indent=2))
        else:
            sections.append(str(content))
        sections.append("")

    sections.append("## Agent Decisions\n")
    for i, dec in enumerate(decisions, 1):
        role = dec["role"]
        action = dec.get("action_dict") or dec
        sections.append(f"### Decision by {role}")
        if isinstance(action, dict):
            sections.append(json.dumps(action, indent=2))
        else:
            sections.append(str(action))
        sections.append("")

    sections.append(
        "## Instructions\n"
        "Evaluate the reasoning quality of the above agent arguments and "
        "decisions across the four pillars: logical_validity, "
        "evidential_support, alternative_consideration, causal_alignment.\n"
        "Respond with the JSON object described in your system prompt."
    )

    return "\n".join(sections)


# DEPRECATED: Use render_crit_prompts() with Jinja templates instead.
CRIT_BATCH_SYSTEM_PROMPT = """\
You are a reasoning quality auditor (CRIT). Your job is to evaluate the \
logical integrity of arguments produced by multiple trading agents during \
a structured debate.

You will be given each agent's reasoning trace and trading decision, along \
with the case context that the agents saw. Evaluate each agent independently \
— one agent's weak reasoning must not inflate or deflate another's score.

## What you evaluate

You score FOUR pillars of reasoning quality for EACH agent:

1. **Logical Validity** — Does the reasoning logically support the conclusion? \
Are the agent's claims logically compatible with each other?

2. **Evidential Support** — Are factual claims backed by cited evidence from \
the case context? Are there unsupported assertions?

3. **Alternative Consideration** — Does the agent consider competing \
explanations? Are alternative hypotheses acknowledged or ignored?

4. **Causal Alignment** — Are causal claims justified by the evidence? Does \
the agent distinguish between correlation (L1), intervention (L2), and \
counterfactual (L3) reasoning?

## What you do NOT evaluate

- You do NOT evaluate correctness of predictions or trading outcomes.
- You do NOT evaluate profitability or market performance.
- You do NOT reward agreement between agents or penalize disagreement.
- You do NOT have access to ground truth or actual market outcomes.

## Scoring rubric

Each pillar is scored on a continuous scale [0.0, 1.0]:
- 0.0 = Severe failure (contradictions, unsupported claims, ignored alternatives, causal overreach)
- 0.5 = Mixed / partial (some issues but also some sound reasoning)
- 1.0 = Rigorous (logically valid, well-supported, considers alternatives, causally sound)

## Output format

You MUST respond with a single JSON object (no markdown, no explanation outside the JSON). \
The object is keyed by agent role name. Each value contains pillar_scores, diagnostics, \
and explanations:

```json
{
  "<role_name>": {
    "pillar_scores": {
      "logical_validity": <float 0.0-1.0>,
      "evidential_support": <float 0.0-1.0>,
      "alternative_consideration": <float 0.0-1.0>,
      "causal_alignment": <float 0.0-1.0>
    },
    "diagnostics": {
      "contradictions_detected": <bool>,
      "unsupported_claims_detected": <bool>,
      "ignored_critiques_detected": <bool>,
      "premature_certainty_detected": <bool>,
      "causal_overreach_detected": <bool>,
      "conclusion_drift_detected": <bool>
    },
    "explanations": {
      "logical_validity": "<1-2 sentence explanation>",
      "evidential_support": "<1-2 sentence explanation>",
      "alternative_consideration": "<1-2 sentence explanation>",
      "causal_alignment": "<1-2 sentence explanation>"
    }
  }
}
```

Return one such entry for EVERY agent presented. Do not omit any agent.
"""


def build_crit_batch_prompt(
    case_data: str,
    traces_by_role: dict[str, list[dict]],
    decisions_by_role: dict[str, dict | None],
) -> str:
    """Build a CRIT user prompt that evaluates all agents in one call.

    .. deprecated:: Use render_crit_prompts() with Jinja templates instead.

    Args:
        case_data: The enriched context that agents saw (no ground truth).
        traces_by_role: Mapping of role name → list of debate turn dicts.
        decisions_by_role: Mapping of role name → decision dict (or None).

    Returns:
        Formatted user prompt string for batch CRIT scoring.
    """
    sections = []

    sections.append("## Case Context\n")
    sections.append(case_data)
    sections.append("")

    all_roles = sorted(set(traces_by_role.keys()) | set(decisions_by_role.keys()))

    for role in all_roles:
        sections.append(f"## Agent: {role.upper()}\n")

        sections.append("### Reasoning Trace\n")
        role_traces = traces_by_role.get(role) or []
        if role_traces:
            for trace in role_traces:
                trace_type = trace["type"]
                content = trace.get("content") or trace
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
        decision = decisions_by_role.get(role)
        if decision:
            action = decision.get("action_dict") or decision
            if isinstance(action, dict):
                sections.append(json.dumps(action, indent=2))
            else:
                sections.append(str(action))
        else:
            sections.append("(No decision available for this agent.)")
        sections.append("")

    sections.append(
        "## Instructions\n"
        "Evaluate the reasoning quality of EACH agent listed above "
        "across the four pillars: logical_validity, evidential_support, "
        "alternative_consideration, causal_alignment.\n"
        "Respond with the JSON object described in your system prompt, "
        "with one entry per agent keyed by role name."
    )

    return "\n".join(sections)


def build_crit_single_agent_prompt(
    case_data: str,
    role: str,
    agent_traces: list[dict],
    decision: dict | None,
) -> str:
    """Build a CRIT user prompt for a single agent (per-agent ρ_i scoring).

    .. deprecated:: Use render_crit_prompts() with Jinja templates instead.

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
            trace_type = trace["type"]
            content = trace.get("content") or trace
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
        action = decision.get("action_dict") or decision
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
        "logical_validity, evidential_support, alternative_consideration, "
        "causal_alignment.\n"
        "Respond with the JSON object described in your system prompt."
    )

    return "\n".join(sections)
