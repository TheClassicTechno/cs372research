"""Render a previous proposal action_dict into structured natural-language text.

Used by critique and revision prompts so the LLM sees a human-readable
representation of the prior round's output rather than raw JSON.

The JSON output schema is NOT modified — this is a read-only rendering layer.
"""

from __future__ import annotations


def render_previous_proposal(action_dict: dict) -> str:
    """Convert an action_dict into structured text for prompt injection.

    Handles missing fields gracefully.  Claims are numbered using
    ``claim_id`` if present, otherwise deterministic C1/C2/… IDs are
    generated from list order.

    Args:
        action_dict: The parsed JSON output from a propose or revise phase.

    Returns:
        Multi-line structured text suitable for ``{{ my_proposal_v2 }}``.
    """
    if not action_dict:
        return "(No previous proposal available.)"

    parts: list[str] = []

    # --- Portfolio allocation ---
    allocation = action_dict.get("allocation")
    if allocation and isinstance(allocation, dict):
        lines = [f"{ticker}: {weight}" for ticker, weight in sorted(allocation.items())]
        parts.append("## Previous Portfolio Allocation\n" + "\n".join(lines))

    # --- Thesis / justification ---
    justification = action_dict.get("justification")
    if justification:
        parts.append(f"## Previous Thesis\n{justification}")

    # --- Claims ---
    claims = action_dict.get("claims")
    if claims and isinstance(claims, list):
        claim_lines: list[str] = []
        for i, claim in enumerate(claims):
            cid = claim.get("claim_id") or f"C{i + 1}"
            text = claim.get("claim_text", "")
            pearl = claim.get("pearl_level", "")
            variables = claim.get("variables", [])
            assumptions = claim.get("assumptions", [])
            confidence = claim.get("confidence")

            entry = f"{cid}: {text}"
            if pearl:
                entry += f"\n  Pearl Level: {pearl}"
            if variables:
                entry += f"\n  Variables: {', '.join(str(v) for v in variables)}"
            if assumptions:
                entry += f"\n  Assumptions: {', '.join(str(a) for a in assumptions)}"
            if confidence is not None:
                entry += f"\n  Confidence: {confidence}"
            claim_lines.append(entry)

        parts.append("## Previous Claims\n" + "\n\n".join(claim_lines))

    # --- Falsifiers ---
    falsifiers = action_dict.get("risks_or_falsifiers")
    if falsifiers:
        parts.append(f"## Previous Falsifiers\n{falsifiers}")

    # --- Confidence ---
    confidence = action_dict.get("confidence")
    if confidence is not None:
        parts.append(f"## Previous Confidence\n{confidence}")

    return "\n\n".join(parts) if parts else "(No previous proposal available.)"


def render_others_proposals(source: list[dict], exclude_role: str) -> str:
    """Render all other agents' proposals as structured text.

    Args:
        source: List of proposal entries (each with 'role' and 'action_dict').
        exclude_role: The role to exclude (the agent's own role).

    Returns:
        Multi-line structured text suitable for ``{{ others_text_v2 }}``.
    """
    parts: list[str] = []
    for entry in source:
        if entry.get("role") == exclude_role:
            continue
        role = entry.get("role", "unknown")
        action_dict = entry.get("action_dict", {})
        rendered = render_previous_proposal(action_dict)
        parts.append(f"### {role.upper()} agent proposed:\n{rendered}")
    return "\n\n".join(parts) if parts else "(No other proposals available.)"


def render_critiques_received(critiques: list[dict]) -> str:
    """Render critique data as structured text for the revision prompt.

    Handles both the new extended format (with target_claim, counter_evidence,
    etc.) and the old 3-field format (from_role, objection, falsifier).

    Args:
        critiques: List of critique dicts with at least 'from_role' and 'objection'.

    Returns:
        Multi-line structured text suitable for ``{{ critiques_text_v2 }}``.
    """
    if not critiques:
        return "(No critiques targeted at you this round.)"

    parts: list[str] = []
    for i, c in enumerate(critiques, 1):
        from_role = c.get("from_role", "unknown").upper()
        lines = [f"### Critique {i} (from {from_role})"]

        target_claim = c.get("target_claim")
        if target_claim:
            lines.append(f"- Target claim: {target_claim}")

        objection = c.get("objection", "")
        lines.append(f"- Objection: {objection}")

        counter_evidence = c.get("counter_evidence")
        if counter_evidence:
            if isinstance(counter_evidence, list):
                lines.append(f"- Counter-evidence: {', '.join(str(e) for e in counter_evidence)}")
            else:
                lines.append(f"- Counter-evidence: {counter_evidence}")

        portfolio_implication = c.get("portfolio_implication")
        if portfolio_implication:
            lines.append(f"- Portfolio implication: {portfolio_implication}")

        suggested_adjustment = c.get("suggested_adjustment")
        if suggested_adjustment:
            lines.append(f"- Suggested adjustment: {suggested_adjustment}")

        falsifier = c.get("falsifier")
        if falsifier:
            lines.append(f"- Falsifier: {falsifier}")

        objection_confidence = c.get("objection_confidence")
        if objection_confidence is not None:
            lines.append(f"- Objection confidence: {objection_confidence}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)
