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

import logging
import re
from itertools import combinations

from eval.PID.sycophancy import evidence_overlap

logger = logging.getLogger(__name__)

# Regex to match memo evidence IDs: [L1-FF], [AAPL-RET60], [NVDA-F3], etc.
_EVIDENCE_ID_RE = re.compile(r"\[([A-Z0-9]+-[A-Z0-9_]+|L1-[A-Z0-9]+)\]")


def extract_evidence_ids(text: str) -> set[str]:
    """Extract memo evidence IDs from agent text output.

    Matches patterns like [AAPL-RET60], [L1-VIX], [NVDA-F3].
    Returns the set of matched IDs (without brackets).
    """
    return set(_EVIDENCE_ID_RE.findall(text))


# Regex for memo evidence lines: "[ID]  text..." or "[ID] text..."
_MEMO_LINE_RE = re.compile(r"^\s*\[([A-Z0-9]+-[A-Z0-9_]+|L1-[A-Z0-9]+)\]\s+(.+)$", re.MULTILINE)


def parse_memo_evidence(enriched_context: str) -> dict[str, str]:
    """Parse memo text into a flat {evidence_id: evidence_text} lookup dict.

    Single-pass regex over lines: extract bracketed ID and rest of line.
    Works with macros ([L1-VIX]  VIX: 17.35), ticker metrics
    ([AAPL-RET60]  60D Return: +11.9%), and filing summaries
    ([NVDA-F1] Operations: ...).

    Args:
        enriched_context: The full enriched context string containing the memo.

    Returns:
        Dict mapping evidence_id (without brackets) to evidence_text (trimmed).
    """
    lookup: dict[str, str] = {}
    for match in _MEMO_LINE_RE.finditer(enriched_context):
        eid = match.group(1)
        text = match.group(2).strip()
        lookup[eid] = text
    return lookup


def enrich_evidence_citations(
    citations: list[dict], lookup: dict[str, str],
) -> list[dict]:
    """Expand evidence citations with text from the memo lookup.

    Each citation dict must have an ``evidence_id`` key. This function
    fills in ``evidence_text`` from the lookup. If an ID is missing from
    the lookup, logs ERROR and sets evidence_text = "MISSING_EVIDENCE".

    Enriches existing dicts in place and also returns the list.

    Args:
        citations: List of citation dicts with at least ``evidence_id``.
        lookup: Memo evidence lookup from ``parse_memo_evidence()``.

    Returns:
        The same list, with ``evidence_text`` filled in on each dict.
    """
    for cite in citations:
        eid = cite.get("evidence_id", "")
        if eid in lookup:
            cite["evidence_text"] = lookup[eid]
        else:
            logger.error(
                "Evidence ID '%s' not found in memo lookup — setting MISSING_EVIDENCE",
                eid,
            )
            cite["evidence_text"] = "MISSING_EVIDENCE"
    return citations


def expand_evidence_ids_inline(text: str, lookup: dict[str, str]) -> str:
    """Replace [ID] with [ID: evidence_text] inline.

    Only expands IDs found in the lookup. Unknown IDs are left unchanged.
    """
    def _replacer(match):
        eid = match.group(1)
        if eid in lookup:
            return f"[{eid}: {lookup[eid]}]"
        return match.group(0)  # leave unknown IDs as-is
    return _EVIDENCE_ID_RE.sub(_replacer, text)


def normalize_variable(var: str) -> str:
    """Normalize a causal variable name for Jaccard comparison.

    Lowercases, strips underscores, hyphens, and extra whitespace so
    that e.g. ``"NVDA_revenue"``, ``"nvda-revenue"``, and
    ``"nvda revenue"`` all map to ``"nvdarevenue"``.
    """
    s = var.lower()
    s = re.sub(r"[_\-\s]+", "", s)
    return s


def extract_evidence_spans(decision: dict, allocation_mode: bool = False) -> set[str]:
    """Extract evidence identifiers from one agent's decision.

    First attempts to extract memo evidence IDs (e.g. [AAPL-RET60], [L1-VIX])
    from justification text, claim_text values, and raw_response. These are
    canonical, unambiguous references used in memo-based allocation mode.

    Falls back to the legacy path: extracting normalized causal variables from
    ``claims[].variables``, then whitespace-tokenizing ``claim_text`` values.
    The fallback is skipped in allocation mode — agents must use canonical
    bracketed IDs; free-text slugs would mask citation failures.
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

    # In allocation mode, don't fall back to normalized slugs —
    # agents must cite canonical bracketed IDs from the memo.
    if allocation_mode:
        return set()

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
    allocation_mode: bool = False,
) -> dict[str, set[str]]:
    """Extract per-agent normalized evidence sets from decision dicts.

    Each dict in *decisions* must have a ``role`` key identifying the
    agent.  Returns a mapping of role -> set of normalized variable names.
    """
    result: dict[str, set[str]] = {}
    for dec in decisions:
        role = dec.get("role", "unknown")
        spans = extract_evidence_spans(dec, allocation_mode=allocation_mode)
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
