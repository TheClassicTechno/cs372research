"""
CritScorer — Blind reasoning quality auditor for multi-agent debate.

BLINDNESS: This scorer never sees ground truth, market outcomes, or impact
scores. It evaluates ONLY the logical structure of the reasoning presented
in agent traces.

WHY THIS FEEDS PID: The PID controller needs a quality signal (rho_bar) to
determine whether to push agents harder or ease off. CRIT provides this
signal by measuring reasoning integrity, not outcome correctness.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from eval.crit.prompts import CRIT_SYSTEM_PROMPT, build_crit_user_prompt
from eval.crit.schema import CritResult, validate_raw_response


class CritScorer:
    """Blind reasoning quality auditor for multi-agent debate.

    Usage:
        scorer = CritScorer(llm_fn=my_llm_caller)
        result = scorer.score(case_data, agent_traces, decisions)
        # result.rho_bar → feed to PID controller
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
    ) -> CritResult:
        """Run CRIT audit on one debate round.

        Args:
            case_data: Rendered case context (what agents saw).
            agent_traces: List of agent argument dicts from the round.
            decisions: List of agent decision dicts from the round.

        Returns:
            CritResult with pillar_scores, rho_bar, diagnostics, explanations.

        Raises:
            ValueError: If LLM response is malformed or fails validation.
            json.JSONDecodeError: If LLM response is not valid JSON.
        """
        system_prompt = CRIT_SYSTEM_PROMPT
        user_prompt = build_crit_user_prompt(case_data, agent_traces, decisions)
        raw_text = self._llm_fn(system_prompt, user_prompt)

        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        raw_dict = json.loads(cleaned)
        return validate_raw_response(raw_dict)
