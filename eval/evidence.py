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

# Regex to match memo evidence IDs: [L1-FF], [AAPL-RET60], [NVDA-F3], etc.
_EVIDENCE_ID_RE = re.compile(r"\[([A-Z0-9]+-[A-Z0-9_]+|L1-[A-Z0-9]+)\]")


def extract_evidence_ids(text: str) -> set[str]:
    """Extract memo evidence IDs from agent text output.

    Matches patterns like [AAPL-RET60], [L1-VIX], [NVDA-F3].
    Returns the set of matched IDs (without brackets).
    """
    return set(_EVIDENCE_ID_RE.findall(text))


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
    """Extract evidence identifiers from one agent's decision.

    First attempts to extract memo evidence IDs (e.g. [AAPL-RET60], [L1-VIX])
    from justification text, claim_text values, and raw_response. These are
    canonical, unambiguous references used in memo-based allocation mode.

    Falls back to the legacy path: extracting normalized causal variables from
    ``claims[].variables``, then whitespace-tokenizing ``claim_text`` values.
    """
    # --- Try memo evidence IDs first (allocation mode) ---
    action_dict = decision.get("action_dict", {})
    if not isinstance(action_dict, dict):
        action_dict = {}

    texts_to_scan: list[str] = []

    justification = action_dict.get("justification", "")
    if justification:
        texts_to_scan.append(justification)

    for claim in action_dict.get("claims", []):
        if isinstance(claim, dict):
            ct = claim.get("claim_text", "")
            if ct:
                texts_to_scan.append(ct)

    raw = decision.get("raw_response", "")
    if raw:
        texts_to_scan.append(raw)

    ids: set[str] = set()
    for text in texts_to_scan:
        ids |= extract_evidence_ids(text)

    if ids:
        return ids

    # --- Fallback: legacy claims[].variables path ---
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


def compute_mean_overlap(evidence_sets: dict[str, set[str]]) -> float | None:
    """Average pairwise Jaccard similarity across all agent pairs.

    Uses ``eval.PID.sycophancy.evidence_overlap()`` for each pair.
    Returns None if fewer than 2 agents.
    Returns 0.0 if no agents have non-empty evidence sets.
    """
    roles = evidence_sets.keys()
    if len(roles) < 2:
        return None

    total = 0.0
    n_pairs = 0
    for a, b in combinations(roles, 2):
        total += evidence_overlap(evidence_sets[a], evidence_sets[b])
        n_pairs += 1

    return total / n_pairs if n_pairs > 0 else None
