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
    Diagnostics,
    Explanations,
    PillarScores,
    RoundCritResult,
    aggregate_agent_scores,
    validate_raw_response,
)

_CRIT_MAX_RETRIES = 2


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
        crit_system_template: str = "crit_system_enumerated.jinja",
        crit_user_template: str = "crit_user_master.jinja",
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

        Retries up to _CRIT_MAX_RETRIES times on parse/validation failure,
        then falls back to a low-quality default CritResult so the simulation
        can continue.

        Args:
            role: Agent role name (e.g. "macro").
            bundle: Reasoning bundle dict with keys: round, agent_role,
                    proposal, critiques_received, revised_argument.

        Returns:
            (role, CritResult) tuple.
        """
        system_prompt, user_prompt = render_crit_prompts(
            bundle,
            system_template=self._crit_system_template,
            user_template=self._crit_user_template,
        )

        last_error: Exception | None = None

        for attempt in range(_CRIT_MAX_RETRIES + 1):
            try:
                raw_text = self._llm_fn(
                    system_prompt, user_prompt,
                    role=role, round_num=bundle.get("round", 0),
                )

                if self._capture_fn:
                    self._capture_fn(role, system_prompt, user_prompt, raw_text)

                # Strip markdown code fences if present
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

                raw_dict = json.loads(cleaned)
                result = validate_raw_response(raw_dict)
                return role, result

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                if attempt < _CRIT_MAX_RETRIES:
                    logger.warning(
                        "CRIT scoring attempt %d/%d failed for %s: %s — retrying",
                        attempt + 1, _CRIT_MAX_RETRIES + 1, role, e,
                    )

        # All retries exhausted — crash the debate
        raise RuntimeError(
            f"CRIT scoring failed for '{role}' after {_CRIT_MAX_RETRIES + 1} attempts: {last_error}\n"
            f"  raw LLM response ({len(raw_text)} chars): {raw_text[:500]}"
        ) from last_error

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
                scored_role, result = future.result()
                agent_scores[scored_role] = result

        return aggregate_agent_scores(agent_scores)
