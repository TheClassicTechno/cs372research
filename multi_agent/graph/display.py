"""Verbose display helpers for debate progress output."""

from __future__ import annotations

_ROLE_LABELS = {
    "macro": "MACRO STRATEGIST",
    "value": "VALUE ANALYST",
    "risk": "RISK MANAGER",
    "technical": "TECHNICAL ANALYST",
    "sentiment": "SENTIMENT ANALYST",
    "devils_advocate": "DEVIL'S ADVOCATE",
}


def _print_allocation(role: str, action_dict: dict, phase: str = "proposes") -> None:
    """Print a compact allocation map for a single agent."""
    alloc = action_dict.get("allocation")
    if not alloc:
        return
    conf = action_dict.get("confidence", 0.5)
    # Sort by weight descending, format as "TICKER:XX%"
    sorted_alloc = sorted(alloc.items(), key=lambda x: -x[1])
    parts = [f"{t}:{w:.0%}" for t, w in sorted_alloc if w > 1e-4]
    zeros = [t for t, w in sorted_alloc if w <= 1e-4]
    line = "  ".join(parts)
    zero_str = f"  (zero: {', '.join(zeros)})" if zeros else ""
    print(f"    [{role.upper()} {phase}] conf={conf:.0%}  {line}{zero_str}", flush=True)


def _print_critique_summary(role: str, result: dict) -> None:
    """Print a compact critique summary for a single agent."""
    critiques = result.get("critiques", [])
    if not critiques:
        return
    parts = []
    for c in critiques[:4]:
        target = c.get("target_role", "?").upper()
        obj = c.get("objection", "")[:80]
        parts.append(f"{target}: {obj}")
    print(f"    [{role.upper()} critiques] {' | '.join(parts)}", flush=True)


def _verbose_proposal(role: str, result: dict) -> None:
    """Print a formatted proposal for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    hyp = result.get("hypothesis", result.get("justification", ""))[:200]
    claims = result.get("claims", [])

    print(f"\n    ┌─── {label} proposes ───", flush=True)
    print(f"    │ Orders: {orders_str}", flush=True)
    print(f"    │ Confidence: {conf:.0%}", flush=True)
    print(f"    │ Thesis: {hyp}", flush=True)
    for c in claims[:2]:
        lvl = c.get("pearl_level", "?")
        txt = c.get("claim_text", "")[:120]
        print(f"    │ Claim [{lvl}]: {txt}", flush=True)
    falsifiers = result.get("risks_or_falsifiers", "")
    if falsifiers:
        falsifiers_str = str(falsifiers)
        print(f"    │ Falsifier: {falsifiers_str[:120]}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_critique(role: str, result: dict) -> None:
    """Print a formatted critique for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    critiques = result.get("critiques", [])
    self_crit = result.get("self_critique", "")

    print(f"\n    ┌─── {label} critiques ───", flush=True)
    for c in critiques[:3]:
        target = _ROLE_LABELS.get(c.get("target_role", ""), c.get("target_role", "?"))
        obj = c.get("objection", "")[:150]
        print(f"    │ → {target}: {obj}", flush=True)
        falsifier = c.get("falsifier", "")
        if falsifier:
            print(f"    │   Falsifier: {falsifier[:100]}", flush=True)
    if self_crit:
        print(f"    │ Self-critique: {self_crit[:120]}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_revision(role: str, result: dict) -> None:
    """Print a formatted revision for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    notes = result.get("revision_notes", result.get("justification", ""))[:200]

    print(f"\n    ┌─── {label} revises ───", flush=True)
    print(f"    │ Orders: {orders_str}", flush=True)
    print(f"    │ Confidence: {conf:.0%}", flush=True)
    if notes:
        print(f"    │ Revision: {notes}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_judge(result: dict) -> None:
    """Print a formatted judge decision for verbose mode."""
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    memo = result.get("audited_memo", result.get("justification", ""))[:300]
    objection = result.get("strongest_objection", "")[:200]

    print(f"\n    ╔{'═' * 54}", flush=True)
    print(f"    ║  JUDGE FINAL DECISION", flush=True)
    print(f"    ╠{'═' * 54}", flush=True)
    print(f"    ║  Orders: {orders_str}", flush=True)
    print(f"    ║  Confidence: {conf:.0%}", flush=True)
    if memo:
        # Wrap long memo text
        for line in [memo[i:i+70] for i in range(0, len(memo), 70)]:
            print(f"    ║  {line}", flush=True)
    if objection:
        print(f"    ║", flush=True)
        print(f"    ║  Strongest objection preserved:", flush=True)
        print(f"    ║  {objection}", flush=True)
    print(f"    ╚{'═' * 54}", flush=True)


def _print_comparison_table(agents: list[dict], phase: str = "Allocations") -> None:
    """Print a side-by-side comparison table of allocations across agents.

    Args:
        agents: List of dicts with ``role`` and ``action_dict`` keys.
        phase: Label for the table header (e.g. "Proposals", "Revisions").
    """
    if not agents:
        return

    # Deduplicate by role (keep last entry — matches runner dedup behaviour)
    seen: dict[str, dict] = {}
    for a in agents:
        seen[a.get("role", "?")] = a
    agents = list(seen.values())

    roles: list[str] = []
    allocations: list[dict[str, float]] = []
    confidences: list[float] = []
    for a in agents:
        roles.append(a.get("role", "?").upper())
        ad = a.get("action_dict", {})
        allocations.append(ad.get("allocation", {}))
        confidences.append(ad.get("confidence", 0.5))

    # Collect tickers, sort by average weight descending
    all_tickers: set[str] = set()
    for alloc in allocations:
        all_tickers.update(alloc.keys())

    def _avg(t: str) -> float:
        return sum(al.get(t, 0.0) for al in allocations) / len(allocations)

    ticker_order = sorted(all_tickers, key=_avg, reverse=True)
    # Only show tickers with non-trivial weight in at least one agent
    ticker_order = [t for t in ticker_order if any(al.get(t, 0) > 0.005 for al in allocations)]

    if not ticker_order:
        return

    # Column widths
    col_w = max(max(len(r) for r in roles), 4) + 1
    ticker_w = max(max(len(t) for t in ticker_order), 6)
    total_w = ticker_w + 2 + len(roles) * (col_w + 2)

    # Header
    header_cols = "  ".join(f"{r:>{col_w}}" for r in roles)
    sep = "─" * total_w

    print(f"\n    ── {phase} {sep[len(phase) + 4:]}", flush=True)
    print(f"    {'':>{ticker_w}}  {header_cols}", flush=True)
    print(f"    {sep}", flush=True)

    # Ticker rows
    for t in ticker_order:
        cells = []
        for alloc in allocations:
            w = alloc.get(t, 0.0)
            if w > 0.005:
                cells.append(f"{w:>{col_w}.0%}")
            else:
                cells.append(f"{'·':>{col_w}}")
        print(f"    {t:<{ticker_w}}  {'  '.join(cells)}", flush=True)

    # Confidence row
    print(f"    {sep}", flush=True)
    conf_cells = [f"{c:>{col_w}.0%}" for c in confidences]
    print(f"    {'conf':<{ticker_w}}  {'  '.join(conf_cells)}", flush=True)
    print(f"    {sep}", flush=True)
