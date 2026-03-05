"""Sector-level allocation constraints.

Pure functions for:
  1. Building a ticker→sector lookup from the sector definition
  2. Enforcing per-sector min/max exposure limits on a final allocation
  3. Filtering an agent's allocation by role-specific sector permissions
"""

from __future__ import annotations

import logging

logger = logging.getLogger("multi_agent.graph.sector_constraints")


def build_sector_constraint_text(
    sector_config: dict | None,
    role: str,
    *,
    include_permissions: bool = True,
) -> str:
    """Build human-readable constraint text for an agent's prompt.

    Composes up to three blocks (permissions, sector limits, max sector weight)
    into a single string.  Returns "" if no constraints apply.
    """
    if not sector_config:
        return ""

    parts: list[str] = []

    # 1) Agent sector permissions (role-specific)
    if include_permissions:
        perms = sector_config.get("agent_sector_permissions") or {}
        if role in perms:
            allowed = perms[role]
            if isinstance(allowed, list) and "*" not in allowed:
                parts.append(
                    f"MANDATORY SECTOR CONSTRAINT: As the {role.upper()} agent, you may ONLY "
                    f"allocate to these sectors: {', '.join(sorted(allowed))}.\n"
                    f"You MUST assign 0.0 weight to all tickers outside your allowed sectors."
                )

    # 2) Sector limits
    limits = sector_config.get("sector_limits") or {}
    if limits:
        lines: list[str] = []
        for sector, bounds in sorted(limits.items()):
            lo = bounds.get("min", 0.0)
            hi = bounds.get("max", 1.0)
            if lo > 0 and hi < 1.0:
                lines.append(f"  - {sector}: min {lo:.0%}, max {hi:.0%}")
            elif lo > 0:
                lines.append(f"  - {sector}: min {lo:.0%}")
            elif hi < 1.0:
                lines.append(f"  - {sector}: max {hi:.0%}")
        if lines:
            parts.append(
                "MANDATORY SECTOR LIMITS (applied to final portfolio):\n" + "\n".join(lines)
            )

    # 3) Max sector weight
    max_sw = sector_config.get("max_sector_weight")
    if max_sw is not None:
        parts.append(
            f"MANDATORY MAX SECTOR WEIGHT: No single sector may exceed {max_sw:.0%} "
            f"of the portfolio."
        )

    return "\n\n".join(parts)


def build_sector_map(sectors: dict[str, list[str]]) -> dict[str, str]:
    """Invert sector definition into a ticker→sector lookup.

    >>> build_sector_map({"tech": ["AAPL", "NVDA"], "energy": ["XOM"]})
    {'AAPL': 'tech', 'NVDA': 'tech', 'XOM': 'energy'}
    """
    ticker_to_sector: dict[str, str] = {}
    for sector, tickers in sectors.items():
        for t in tickers:
            ticker_to_sector[t] = sector
    return ticker_to_sector


def enforce_sector_limits(
    alloc: dict[str, float],
    sector_map: dict[str, str],
    sector_limits: dict[str, dict],
) -> dict[str, float]:
    """Enforce per-sector min/max exposure on a normalized allocation.

    Parameters
    ----------
    alloc:
        Normalized allocation {ticker: weight} summing to ~1.0.
    sector_map:
        {ticker: sector_name} lookup.
    sector_limits:
        {sector_name: {"min": float, "max": float}}.

    Returns
    -------
    Adjusted allocation respecting sector bounds, still summing to ~1.0.
    Intra-sector ticker ratios are preserved (proportional scaling).

    Algorithm (two-pass, iterative):
        1. Clamp each constrained sector to [min, max] by scaling its tickers.
        2. Compute budget used by clamped sectors.
        3. Distribute remaining budget (1.0 - clamped) among free sectors
           proportionally to their current weights.
        4. Iterate until stable (clamping can cascade).
    """
    if not sector_limits or not alloc:
        return alloc

    result = dict(alloc)

    for _ in range(10):
        sector_totals = _compute_sector_totals(result, sector_map)
        limited_sectors = set(sector_limits.keys())

        # Pass 1: clamp constrained sectors to [min, max]
        clamped_budget = 0.0
        clamped_sectors: set[str] = set()

        for sector, limits in sector_limits.items():
            total = sector_totals.get(sector, 0.0)
            min_w = limits.get("min", 0.0)
            max_w = limits.get("max", 1.0)

            if total > max_w + 1e-8:
                # Over max → scale down
                _scale_sector(result, sector_map, sector, max_w / total)
                clamped_budget += max_w
                clamped_sectors.add(sector)
            elif total < min_w - 1e-8:
                # Under min → scale up (or distribute equally if zero)
                if total > 1e-8:
                    _scale_sector(result, sector_map, sector, min_w / total)
                else:
                    sector_tickers = [t for t in result if sector_map.get(t) == sector]
                    if sector_tickers:
                        for t in sector_tickers:
                            result[t] = min_w / len(sector_tickers)
                clamped_budget += min_w
                clamped_sectors.add(sector)
            else:
                # Within bounds — leave alone for now
                pass

        # Pass 2: distribute remaining budget among free sectors
        free_budget = 1.0 - clamped_budget
        if free_budget < -1e-8:
            # Infeasible — min constraints exceed 1.0.  Do best effort.
            logger.warning("Sector limits infeasible (clamped budget %.4f > 1.0)", clamped_budget)
            break

        # Collect tickers in non-clamped sectors
        free_tickers = [
            t for t in result
            if sector_map.get(t) not in clamped_sectors
        ]
        free_total = sum(result[t] for t in free_tickers)

        if free_tickers and free_total > 1e-8:
            scale = free_budget / free_total
            for t in free_tickers:
                result[t] *= scale
        elif free_tickers and free_budget > 1e-8:
            # All free tickers are zero — split equally
            per_t = free_budget / len(free_tickers)
            for t in free_tickers:
                result[t] = per_t

        # Check convergence — are all constraints satisfied?
        violations = 0
        new_totals = _compute_sector_totals(result, sector_map)
        for sector, limits in sector_limits.items():
            total = new_totals.get(sector, 0.0)
            if total > limits.get("max", 1.0) + 1e-8:
                violations += 1
            if total < limits.get("min", 0.0) - 1e-8:
                violations += 1
        if violations == 0:
            break

    # Final normalization to exactly 1.0 and clamp negatives
    total = sum(result.values())
    if total > 1e-8:
        result = {t: w / total for t, w in result.items()}
    result = {t: max(w, 0.0) for t, w in result.items()}

    return result


def filter_allocation_by_permissions(
    alloc: dict[str, float],
    role: str,
    sector_map: dict[str, str],
    permissions: dict[str, list[str]],
) -> dict[str, float]:
    """Zero out tickers in sectors the role is not allowed to invest in.

    Parameters
    ----------
    alloc:
        Raw allocation {ticker: weight} (not yet normalized).
    role:
        Agent role string (e.g. "macro", "value").
    sector_map:
        {ticker: sector_name} lookup.
    permissions:
        {role: [allowed_sectors]} or {role: ["*"]} for all.

    Returns
    -------
    Filtered allocation with disallowed tickers zeroed out.
    Caller should pass result through normalize_allocation().
    """
    if role not in permissions:
        return alloc

    allowed = permissions[role]
    # ["*"] means all sectors allowed
    if isinstance(allowed, list) and "*" in allowed:
        return alloc

    allowed_set = set(allowed) if isinstance(allowed, list) else set()
    filtered = {}
    zeroed = []
    for t, w in alloc.items():
        sector = sector_map.get(t)
        if sector is not None and sector not in allowed_set:
            filtered[t] = 0.0
            if w > 1e-8:
                zeroed.append(t)
        else:
            filtered[t] = w

    if zeroed:
        logger.info(
            "Role '%s': zeroed allocation for %s (sector not in allowed: %s)",
            role, zeroed, sorted(allowed_set),
        )

    return filtered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_sector(
    alloc: dict[str, float],
    sector_map: dict[str, str],
    sector: str,
    scale: float,
) -> None:
    """Scale all tickers in a sector by a factor (in-place)."""
    for t in alloc:
        if sector_map.get(t) == sector:
            alloc[t] *= scale


def _compute_sector_totals(
    alloc: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, float]:
    """Sum allocation weights by sector."""
    totals: dict[str, float] = {}
    for t, w in alloc.items():
        sector = sector_map.get(t)
        if sector:
            totals[sector] = totals.get(sector, 0.0) + w
    return totals


def _sector_total(
    alloc: dict[str, float],
    sector_map: dict[str, str],
    sector: str,
) -> float:
    """Sum weights for a single sector."""
    return sum(w for t, w in alloc.items() if sector_map.get(t) == sector)
