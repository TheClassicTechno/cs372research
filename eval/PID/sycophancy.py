"""
Sycophancy detection signals for RAudit PID control.

Implements Jensen-Shannon divergence over reasonableness scores (Eq 2),
evidence overlap via Jaccard index (Eq 3), and the composite sycophancy
indicator (Eq 4).
"""

import math
from typing import List, Set


EPSILON = 1e-10  # Guard against log(0)


def shannon_entropy(p: float) -> float:
    """Binary Shannon entropy H(p) = -p·log2(p) - (1-p)·log2(1-p).

    Returns 0.0 for p ≤ 0 or p ≥ 1 (boundary convention).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def jensen_shannon_divergence(scores: List[float]) -> float:
    """JS divergence over per-agent reasonableness scores (RAudit Eq 2).

    JS_t = H(ρ̄_t) − (1/n)·Σ H(ρ_i_t)

    Each score ρ_i is treated as a Bernoulli parameter so that H(ρ_i) is
    the binary Shannon entropy.

    Args:
        scores: Per-agent reasonableness scores for the current round.

    Returns:
        Non-negative JS divergence. Returns 0.0 for empty or single-agent input.
    """
    if len(scores) < 2:
        return 0.0
    n = len(scores)
    rho_bar = sum(scores) / n
    h_mean = shannon_entropy(rho_bar)
    h_agents = sum(shannon_entropy(s) for s in scores) / n
    # JS divergence is non-negative by construction; clamp for float safety
    return max(0.0, h_mean - h_agents)


def evidence_overlap(set_a: Set, set_b: Set) -> float:
    """Evidence overlap via Jaccard index (RAudit Eq 3).

    Ov_t = |ξ_A ∩ ξ_B| / |ξ_A ∪ ξ_B|

    Returns 0.0 if both sets are empty (convention: 0/0 = 0).
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_sycophancy_signal(
    js_current: float,
    js_prev: float,
    ov_current: float,
    ov_prev: float,
    delta_s: float,
) -> int:
    """Composite sycophancy indicator (RAudit Eq 4).

    s_t = I[ΔJS_t < −δ_s  ∧  ΔOv_t < 0]

    A value of 1 signals that agents are converging suspiciously (JS dropping
    sharply while evidence overlap also decreases).

    Args:
        js_current: JS divergence at round t.
        js_prev:    JS divergence at round t-1.
        ov_current: Evidence overlap at round t.
        ov_prev:    Evidence overlap at round t-1.
        delta_s:    Threshold for significant JS drop.

    Returns:
        1 if sycophancy detected, 0 otherwise.
    """
    delta_js = js_current - js_prev
    delta_ov = ov_current - ov_prev
    if delta_js < -delta_s and delta_ov < 0.0:
        return 1
    return 0
