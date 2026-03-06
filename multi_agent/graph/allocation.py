"""Allocation normalization and weight enforcement."""

from __future__ import annotations

import logging

logger = logging.getLogger("multi_agent.graph.allocation")


def normalize_allocation(
    raw: dict[str, float],
    universe: list[str],
    max_weight: float,
    min_holdings: int,
) -> dict[str, float]:
    """Validate, constrain, and normalize allocation weights.

    Steps:
    1. Drop tickers not in universe, warn
    2. Set missing universe tickers to 0.0
    3. Clamp negative weights to 0.0, warn
    4. If all weights zero -> equal-weight fallback
    5. Normalize to sum to 1.0
    6. Enforce max_weight cap (redistribute excess proportionally)
    7. Enforce min_holdings (force minimum weight on zero tickers)
    8. Re-normalize to sum to 1.0
    """
    # Guard: empty universe
    if not universe:
        logger.warning("Empty universe — returning empty allocation")
        return {}

    # Steps 1-3: clean up
    alloc = {}
    for t in universe:
        w = raw.get(t, 0.0)
        if w < 0:
            logger.warning("Negative weight for %s (%.4f), clamping to 0", t, w)
            w = 0.0
        alloc[t] = w

    unknown = set(raw.keys()) - set(universe)
    if unknown:
        logger.warning("Allocation contains unknown tickers: %s — dropped", unknown)

    # Step 4: all-zero fallback
    total = sum(alloc.values())
    if total < 1e-8:
        logger.warning("All weights zero, defaulting to equal-weight")
        eq = 1.0 / len(universe)
        return {t: eq for t in universe}

    # Step 5: normalize
    alloc = {t: w / total for t, w in alloc.items()}
    naive_alloc = dict(alloc)  # Keep for comparison

    # Step 6: enforce max_weight (iterative cap-and-redistribute)
    alloc = _enforce_max_weight(alloc, max_weight)

    # Step 7: enforce min_holdings
    non_zero = sum(1 for w in alloc.values() if w > 1e-8)
    if non_zero < min_holdings and len(universe) >= min_holdings:
        logger.warning("Min holdings constraint violated (%d < %d). Forcing weights on zero-tickers.", non_zero, min_holdings)
        sorted_tickers = sorted(alloc.items(), key=lambda x: x[1])
        need = min_holdings - non_zero
        min_w = 0.01  # 1% floor for forced holdings
        for t, w in sorted_tickers:
            if w < 1e-8 and need > 0:
                alloc[t] = min_w
                need -= 1

    # Step 8: re-normalize then re-enforce max_weight (min_holdings may
    # have pushed weights around)
    final_total = sum(alloc.values())
    if final_total > 0:
        alloc = {t: w / final_total for t, w in alloc.items()}
    alloc = _enforce_max_weight(alloc, max_weight)

    # Step 9: post-renorm max_weight re-check.
    # With few tickers, Step 8 re-normalization can push weights above
    # max_weight (e.g. 2 tickers both at max_weight=0.40, Step 7 adds a
    # 1% floor → renorm pushes them above 0.40). Re-cap and re-normalize.
    if any(w > max_weight + 1e-8 for w in alloc.values()):
        alloc = {t: min(w, max_weight) for t, w in alloc.items()}
        recapped_total = sum(alloc.values())
        if recapped_total > 0:
            alloc = {t: w / recapped_total for t, w in alloc.items()}

    # Final comparison: warn if constraints changed weights significantly
    for t in universe:
        diff = alloc[t] - naive_alloc[t]
        if abs(diff) > 1e-4:
            logger.warning(
                "Constraint enforcement changed %s allocation: %.4f -> %.4f (diff: %+.4f)",
                t, naive_alloc[t], alloc[t], diff
            )

    return alloc


def _enforce_max_weight(alloc: dict[str, float], max_weight: float) -> dict[str, float]:
    """Iteratively cap weights at max_weight and redistribute excess.

    If n * max_weight < 1.0 the constraint is infeasible (can't sum to 1.0
    with all weights <= max_weight). In that case the effective cap is raised
    to 1/n so equal-weight is the tightest feasible solution.
    """
    n = len(alloc)
    if n == 0:
        return alloc
    effective_max = max(max_weight, 1.0 / n)

    for _ in range(10):
        capped = {t: min(w, effective_max) for t, w in alloc.items()}
        excess = sum(alloc.values()) - sum(capped.values())
        if excess < 1e-8:
            return capped
        # Redistribute excess proportionally to uncapped tickers
        uncapped = {t: w for t, w in capped.items() if w < effective_max - 1e-8}
        uncapped_total = sum(uncapped.values())
        if uncapped_total < 1e-8:
            # All at cap — normalize to sum to 1.0
            total = sum(capped.values())
            if total > 0:
                return {t: w / total for t, w in capped.items()}
            return capped
        alloc = dict(capped)
        for t in uncapped:
            alloc[t] += excess * (uncapped[t] / uncapped_total)
    return alloc
