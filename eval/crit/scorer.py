"""
CritScorer — Blind reasoning quality auditor for multi-agent debate.

BLINDNESS: This scorer never sees ground truth, market outcomes, or impact
scores. It evaluates ONLY the logical structure of the reasoning presented
in agent traces.

PER-AGENT SCORING: Per the RAudit paper (Section 3.3, Algorithm 1 lines
7-8), CRIT scores each agent individually (ρ_i), then averages into
ρ̄ = 1/n Σ_i ρ_i.  Each agent gets its own LLM call so the evaluation
is independent — one agent's weak reasoning cannot inflate another's score.

WHY THIS FEEDS PID: The PID controller needs a quality signal (rho_bar) to
determine whether to push agents harder or ease off. CRIT provides this
signal by measuring reasoning integrity, not outcome correctness.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Callable

from eval.crit.prompts import (
    CRIT_SYSTEM_PROMPT,
    build_crit_single_agent_prompt,
)
from eval.crit.schema import (
    CritResult,
    RoundCritResult,
    aggregate_agent_scores,
    validate_raw_response,
)


class CritScorer:
    """Blind reasoning quality auditor for multi-agent debate.

    Scores each agent individually (ρ_i), then aggregates into ρ̄.

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
        """Run CRIT audit on one debate round, scoring each agent individually.

        Per the RAudit paper (Algorithm 1):
            1. For each agent i, evaluate ρ_i from their traces + decision
            2. Compute ρ̄ = 1/n Σ_i ρ_i

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

        decisions_by_role: dict[str, dict] = {}
        for dec in decisions:
            role = dec.get("role", "unknown")
            decisions_by_role[role] = dec  # latest decision per role

        # Determine all agent roles (union of traces and decisions)
        all_roles = set(traces_by_role.keys()) | set(decisions_by_role.keys())
        if not all_roles:
            raise ValueError("No agent roles found in traces or decisions")

        # Score each agent individually
        agent_scores: dict[str, CritResult] = {}
        for role in sorted(all_roles):
            role_traces = traces_by_role.get(role, [])
            role_decision = decisions_by_role.get(role)
            agent_scores[role] = self._score_single_agent(
                case_data, role, role_traces, role_decision
            )

        return aggregate_agent_scores(agent_scores)

    def _score_single_agent(
        self,
        case_data: str,
        role: str,
        agent_traces: list[dict],
        decision: dict | None,
    ) -> CritResult:
        """Run CRIT audit on a single agent's reasoning.

        Args:
            case_data: Rendered case context.
            role: Agent role name.
            agent_traces: This agent's debate turn dicts.
            decision: This agent's decision dict, or None.

        Returns:
            CritResult (ρ_i) for this agent.
        """
        system_prompt = CRIT_SYSTEM_PROMPT
        user_prompt = build_crit_single_agent_prompt(
            case_data, role, agent_traces, decision
        )
        raw_text = self._llm_fn(system_prompt, user_prompt)

        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        raw_dict = json.loads(cleaned)
        return validate_raw_response(raw_dict)
