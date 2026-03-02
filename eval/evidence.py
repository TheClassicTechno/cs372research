"""
Evidence span extraction and overlap computation for debate agents.

Evidence overlap is the average pairwise Jaccard similarity of the
normalized causal variables cited by debate agents.  Each agent's
proposal/revision contains structured ``claims``, each with a
``variables`` list naming the causal variables in their reasoning.

Variables are normalized (lowercased, underscores/hyphens stripped) to
handle minor naming differences across agents that see the same case.

Pipeline context:
    state["revisions"] or state["proposals"]  (list of agent decision dicts)
        │
        ▼
    extract_agent_evidence_spans()  →  {role: set of normalized variables}
        │
        ▼
    compute_mean_overlap()          →  float (mean pairwise Jaccard)
        │
        ▼
    PIDController.step(rho_bar, js, ov)   ← ov is this value
"""

from __future__ import annotations

import re
from itertools import combinations

from eval.PID.sycophancy import evidence_overlap


def normalize_variable(var: str) -> str:
    """Normalize a causal variable name for Jaccard comparison.

    Lowercases, strips underscores, hyphens, and extra whitespace so
    that e.g. ``"NVDA_revenue"``, ``"nvda-revenue"``, and
    ``"nvda revenue"`` all map to ``"nvdarevenue"``.
    """
    s = var.lower()
    s = re.sub(r"[_\-\s]+", "", s)
    return s


def extract_evidence_spans(decision: dict) -> set[str]:
    """Extract normalized causal variables from one agent's decision.

    Collects all ``variables`` entries from the ``claims`` list in the
    decision dict, normalizes each, and returns the union as a set.

    If no claims or no variables are present, falls back to splitting
    ``claim_text`` values into whitespace-delimited tokens (normalized).
    """
    claims = decision.get("claims", [])
    if isinstance(decision.get("action_dict"), dict):
        claims = decision["action_dict"].get("claims", claims)

    spans: set[str] = set()
    fallback_texts: list[str] = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        variables = claim.get("variables", [])
        if variables:
            for v in variables:
                normed = normalize_variable(str(v))
                if normed:
                    spans.add(normed)
        claim_text = claim.get("claim_text", "")
        if claim_text:
            fallback_texts.append(claim_text)

    if not spans and fallback_texts:
        for text in fallback_texts:
            for token in text.split():
                normed = normalize_variable(token)
                if len(normed) > 2:
                    spans.add(normed)

    return spans


def extract_agent_evidence_spans(
    decisions: list[dict],
) -> dict[str, set[str]]:
    """Extract per-agent normalized evidence sets from decision dicts.

    Each dict in *decisions* must have a ``role`` key identifying the
    agent.  Returns a mapping of role -> set of normalized variable names.
    """
    result: dict[str, set[str]] = {}
    for dec in decisions:
        role = dec.get("role", "unknown")
        spans = extract_evidence_spans(dec)
        if role in result:
            result[role] |= spans
        else:
            result[role] = spans
    return result


def compute_mean_overlap(evidence_sets: dict[str, set[str]]) -> float:
    """Average pairwise Jaccard similarity across all agent pairs.

    Uses ``eval.PID.sycophancy.evidence_overlap()`` for each pair.
    Returns 0.0 if fewer than 2 agents have non-empty evidence sets.
    """
    roles = [r for r, s in evidence_sets.items() if s]
    if len(roles) < 2:
        return 0.0

    total = 0.0
    n_pairs = 0
    for a, b in combinations(roles, 2):
        total += evidence_overlap(evidence_sets[a], evidence_sets[b])
        n_pairs += 1

    return total / n_pairs if n_pairs > 0 else 0.0
