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
                "reasoning_type": "causal",
                "confidence": 0.55,
                "evidence": [f"[{tickers[0]}-RET60]", "[L1-VIX]"],
                "assumptions": [f"{role} signals remain stable"],
                "falsifiers": ["Regime change invalidates signal"],
                "impacts_positions": [tickers[0]],
            }
        ],
        "position_rationale": [
            {
                "ticker": t,
                "weight": eq,
                "supported_by_claims": [f"[{role}] Equal-weight is optimal given balanced signals [{tickers[0]}-RET60] [L1-VIX]"],
                "explanation": f"Equal-weight allocation to {t} based on {role} analysis",
            }
            for t in tickers
        ],
    }


def _mock_critique(role: str, proposals: list) -> dict:
    """Generate a deterministic mock critique."""
    others = [p for p in proposals if p.get("role") != role][:2]
    return {
        "critiques": [
            {
                "target_role": o.get("role", "unknown"),
                "target_claim": f"[{o.get('role', 'unknown')}] Equal-weight is optimal",
                "objection": (
                    f"[mock] {role} challenges {o.get('role')}'s core assumption; "
                    f"possible confounder: regime change could invalidate their signal"
                ),
                "counter_evidence": [f"[L1-VIX]"],
                "alternative_explanation": f"Market noise rather than genuine {o.get('role')} signal",
                "portfolio_implication": f"Over-concentration risk if {o.get('role')} signal is noise",
                "suggested_adjustment": f"Reduce weight on {o.get('role')} signal by 10-20%",
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
    proposal["critique_responses"] = [
        {
            "target_role": role,
            "response": f"[mock] {role} acknowledges critique and adjusts confidence",
        }
    ]
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
        tickers = ["AAPL"]
        avg = {"AAPL": 1.0}
    eq = round(1.0 / len(tickers), 4) if tickers else 1.0
    return {
        "allocation": avg,
        "audited_memo": f"[Judge mock] Averaged {len(all_allocs)} agent allocations.",
        "strongest_objection": "Risk agent raised concentration concerns",
        "confidence": 0.55,
        "risks_or_falsifiers": "Unexpected correlation breakdown or macro shock",
        "claims": [
            {
                "claim_text": "Averaged allocation reduces individual agent bias [L1-VIX]",
                "reasoning_type": "causal",
                "confidence": 0.6,
                "evidence": ["[L1-VIX]"],
                "assumptions": ["Agent allocations are independent"],
                "falsifiers": ["Correlated agent biases"],
                "impacts_positions": list(avg.keys())[:1] if avg else [],
            }
        ],
        "position_rationale": [
            {
                "ticker": t,
                "weight": avg.get(t, eq),
                "supported_by_claims": ["Averaged allocation reduces individual agent bias [L1-VIX]"],
                "explanation": f"Averaged weight for {t} across agent revisions",
            }
            for t in tickers
        ],
        "orders": [
            {
                "ticker": t,
                "side": "buy",
                "size": int(avg.get(t, 0) * 1000),
                "type": "market",
            }
            for t in tickers
            if avg.get(t, 0) > 0
        ],
    }
