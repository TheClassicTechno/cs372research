"""
CritScorer — Blind reasoning quality auditor for multi-agent debate.

BLINDNESS: This scorer never sees ground truth, market outcomes, or impact
scores. It evaluates ONLY the logical structure of the reasoning presented
in agent traces.

PER-AGENT SCORING: CRIT evaluates each agent independently via parallel
LLM calls (one per agent). Each call receives only that agent's reasoning
bundle (proposal → critiques received → revised argument) with embedded
evidence citations. No cross-agent contamination.

The per-agent results are aggregated into ρ̄ = 1/n Σ_i ρ_i for the PID
controller.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)

from eval.crit.prompts import render_crit_prompts
from eval.crit.schema import (
    CritResult,
    RoundCritResult,
    aggregate_agent_scores,
    validate_raw_response,
)


class CritScorer:
    """Blind reasoning quality auditor for multi-agent debate.

    Scores each agent independently via parallel LLM calls, then
    aggregates into ρ̄.

    Usage:
        scorer = CritScorer(llm_fn=my_llm_caller)
        bundles = {"macro": {...}, "risk": {...}, ...}
        result = scorer.score(bundles)
        # result.rho_bar → feed to PID controller
        # result.agent_scores["macro"].rho_bar → per-agent ρ_i
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        capture_fn: Callable[[str, str, str, str], None] | None = None,
        crit_system_template: str = "crit_system.jinja",
        crit_user_template: str = "crit_user.jinja",
    ) -> None:
        """
        Args:
            llm_fn: Function that takes (system_prompt, user_prompt)
                     and returns raw LLM response string.
                     Dependency injection for testability.
            capture_fn: Optional callback (role, system_prompt, user_prompt,
                        raw_response) called after each LLM scoring call.
                        Used to capture CRIT prompts/responses for diagnostics.
            crit_system_template: CRIT system prompt template filename.
            crit_user_template: CRIT user prompt template filename.
        """
        self._llm_fn = llm_fn
        self._capture_fn = capture_fn
        self._crit_system_template = crit_system_template
        self._crit_user_template = crit_user_template

    def _score_single_agent(self, role: str, bundle: dict) -> tuple[str, CritResult]:
        """Score a single agent's reasoning bundle via one LLM call.

        Args:
            role: Agent role name (e.g. "macro").
            bundle: Reasoning bundle dict with keys: round, agent_role,
                    proposal, critiques_received, revised_argument.

        Returns:
            (role, CritResult) tuple.

        Raises:
            ValueError: If LLM response is malformed.
        """
        system_prompt, user_prompt = render_crit_prompts(
            bundle,
            system_template=self._crit_system_template,
            user_template=self._crit_user_template,
        )

        raw_text = self._llm_fn(system_prompt, user_prompt)

        if self._capture_fn:
            self._capture_fn(role, system_prompt, user_prompt, raw_text)

        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            raw_dict = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                "CRIT JSON parse failed for %s: %s\n"
                "  prompt size: system=%d chars, user=%d chars\n"
                "  raw LLM response (%d chars): %.500s",
                role, e, len(system_prompt), len(user_prompt),
                len(raw_text), raw_text,
            )
            raise

        try:
            result = validate_raw_response(raw_dict)
        except (ValueError, KeyError) as e:
            logger.error(
                "CRIT validation failed for %s: %s\n"
                "  prompt size: system=%d chars, user=%d chars\n"
                "  raw LLM response (%d chars): %.500s",
                role, e, len(system_prompt), len(user_prompt),
                len(raw_text), raw_text,
            )
            raise

        return role, result

    def score(self, reasoning_bundles: dict[str, dict]) -> RoundCritResult:
        """Run CRIT audit on one debate round, scoring each agent in parallel.

        Makes one LLM call per agent via ThreadPoolExecutor, then validates
        and aggregates the results.

        Args:
            reasoning_bundles: Dict mapping role name → reasoning bundle dict.
                Each bundle has keys: round, agent_role, proposal,
                critiques_received, revised_argument.

        Returns:
            RoundCritResult with per-agent scores and aggregated rho_bar.

        Raises:
            ValueError: If no agents or if any LLM response is malformed.
        """
        if not reasoning_bundles:
            raise ValueError("reasoning_bundles must not be empty")

        agent_scores: dict[str, CritResult] = {}

        with ThreadPoolExecutor(max_workers=len(reasoning_bundles)) as executor:
            futures = {
                executor.submit(self._score_single_agent, role, bundle): role
                for role, bundle in reasoning_bundles.items()
            }
            for future in futures:
                role = futures[future]
                try:
                    scored_role, result = future.result()
                    agent_scores[scored_role] = result
                except Exception as e:
                    logger.error(
                        "CRIT scoring failed for agent '%s': %s", role, e,
                    )
                    raise

        return aggregate_agent_scores(agent_scores)
