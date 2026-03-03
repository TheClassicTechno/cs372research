"""
CritScorer — Blind reasoning quality auditor for multi-agent debate.

BLINDNESS: This scorer never sees ground truth, market outcomes, or impact
scores. It evaluates ONLY the logical structure of the reasoning presented
in agent traces.

BATCH SCORING: CRIT evaluates all agents in a single LLM call per phase.
The LLM returns a JSON object keyed by role name, each containing the
standard pillar/diagnostic/explanation structure. This reduces LLM calls
from N (one per agent) to 1 per phase — a significant cost/latency win.

The batch response is validated per-agent using the same validate_raw_response()
logic, then aggregated into ρ̄ = 1/n Σ_i ρ_i for the PID controller.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Callable

from eval.crit.prompts import (
    CRIT_BATCH_SYSTEM_PROMPT,
    build_crit_batch_prompt,
)
from eval.crit.schema import (
    RoundCritResult,
    aggregate_agent_scores,
    validate_batch_response,
)

class CritScorer:
    """Blind reasoning quality auditor for multi-agent debate.

    Scores all agents in one LLM call, then aggregates into ρ̄.

    Usage:
        scorer = CritScorer(llm_fn=my_llm_caller)
        result = scorer.score(case_data, agent_traces, decisions)
        # result.rho_bar → feed to PID controller
        # result.agent_scores["macro"].rho_bar → per-agent ρ_i
    """

    def __init__(self, llm_fn: Callable[[str, str], str]) -> None:
        """
        Args:
            llm_fn: Function that takes (system_prompt, user_prompt)
                     and returns raw LLM response string.
                     Dependency injection for testability.
        """
        self._llm_fn = llm_fn

    def score(
        self,
        case_data: str,
        agent_traces: list[dict],
        decisions: list[dict],
    ) -> RoundCritResult:
        """Run CRIT audit on one debate round, scoring all agents in one call.

        Makes a single LLM call with all agents' traces and decisions,
        then validates and aggregates the batch response.

        Args:
            case_data: Rendered case context (what agents saw).
            agent_traces: List of agent trace dicts from the round.
                Each dict must have a 'role' field.
            decisions: List of agent decision dicts (proposals or revisions).
                Each dict must have a 'role' field.

        Returns:
            RoundCritResult with per-agent scores and aggregated rho_bar.

        Raises:
            ValueError: If no agents found or LLM response is malformed.
            json.JSONDecodeError: If LLM response is not valid JSON.
        """
        # Group traces and decisions by agent role
        traces_by_role: dict[str, list[dict]] = defaultdict(list)
        for trace in agent_traces:
            role = trace.get("role", "unknown")
            traces_by_role[role].append(trace)

        decisions_by_role: dict[str, dict | None] = {}
        for dec in decisions:
            role = dec.get("role", "unknown")
            decisions_by_role[role] = dec  # latest decision per role

        # Determine all agent roles (union of traces and decisions)
        all_roles = set(traces_by_role.keys()) | set(decisions_by_role.keys())
        if not all_roles:
            raise ValueError("No agent roles found in traces or decisions")

        # Build one batch prompt with all agents
        system_prompt = CRIT_BATCH_SYSTEM_PROMPT
        user_prompt = build_crit_batch_prompt(
            case_data, dict(traces_by_role), decisions_by_role
        )

        # Single LLM call for all agents
        raw_text = self._llm_fn(system_prompt, user_prompt)

        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        raw_dict = json.loads(cleaned)

        # Validate and parse per-agent results
        agent_scores = validate_batch_response(raw_dict, all_roles)

        return aggregate_agent_scores(agent_scores)
