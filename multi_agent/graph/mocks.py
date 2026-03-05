"""Deterministic mock response generators for testing without API keys."""

from __future__ import annotations


def _mock_proposal(role: str, obs_dict: dict, config: dict | None = None) -> dict:
    """Generate a deterministic mock proposal (allocation mode) for a given role."""
    tickers = obs_dict.get("universe", ["AAPL"])
    if not tickers:
        tickers = ["AAPL"]

    eq = round(1.0 / len(tickers), 4)
    return {
        "allocation": {t: eq for t in tickers},
        "justification": f"[{role} mock] Equal-weight allocation across {len(tickers)} tickers",
        "confidence": 0.5,
        "risks_or_falsifiers": f"A reversal in {role} signals would change this view",
        "claims": [
            {
                "claim_text": f"[{role}] Equal-weight is optimal given balanced signals [{tickers[0]}-RET60] [L1-VIX]",
                "pearl_level": "L2",
                "variables": tickers[:2],
                "confidence": 0.55,
            }
        ],
    }


def _mock_critique(role: str, proposals: list) -> dict:
    """Generate a deterministic mock critique."""
    others = [p for p in proposals if p.get("role") != role][:2]
    return {
        "critiques": [
            {
                "target_role": o.get("role", "unknown"),
                "objection": (
                    f"[mock] {role} challenges {o.get('role')}'s core assumption; "
                    f"possible confounder: regime change could invalidate their signal"
                ),
                "alternative_explanation": f"Market noise rather than genuine {o.get('role')} signal",
                "falsifier": "Unexpected macro shock or data revision",
                "objection_confidence": 0.6,
            }
            for o in others
        ],
        "self_critique": f"[mock] {role} may be overweighting recent data; sample size is small",
    }


def _mock_revision(role: str, original_action: dict, obs_dict: dict, config: dict | None = None) -> dict:
    """Generate a deterministic mock revision."""
    proposal = _mock_proposal(role, obs_dict, config)
    proposal["confidence"] = max(0.25, proposal["confidence"] - 0.1)
    proposal["revision_notes"] = f"[mock] {role} reduced confidence after considering critiques"
    return proposal


def _mock_judge(revisions: list, config: dict | None = None) -> dict:
    """Generate a deterministic mock judge decision (allocation mode)."""
    all_allocs: list[dict[str, float]] = []
    for r in revisions:
        alloc = r.get("action_dict", {}).get("allocation", {})
        if alloc:
            all_allocs.append(alloc)
    if all_allocs:
        tickers = sorted(set().union(*all_allocs))
        avg = {}
        for t in tickers:
            avg[t] = sum(a.get(t, 0.0) for a in all_allocs) / len(all_allocs)
        # Normalize
        total = sum(avg.values())
        if total > 0:
            avg = {t: w / total for t, w in avg.items()}
    else:
        avg = {}
    return {
        "allocation": avg,
        "audited_memo": f"[Judge mock] Averaged {len(all_allocs)} agent allocations.",
        "strongest_objection": "Risk agent raised concentration concerns",
        "confidence": 0.55,
        "risks_or_falsifiers": "Unexpected correlation breakdown or macro shock",
        "claims": [
            {
                "claim_text": "Averaged allocation reduces individual agent bias [L1-VIX]",
                "pearl_level": "L2",
                "variables": ["consensus", "allocation"],
                "confidence": 0.6,
            }
        ],
    }
