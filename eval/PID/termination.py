"""
Logarithmic termination bound for RAudit PID control (Proposition 2).

Implements the expected contraction inequality (Eq 23), the minimum-rounds
formula T* (Eq 26), convergence detection, and a contraction simulator.
"""

import math
from typing import List


def termination_bound(d0: float, epsilon: float, kappa: float) -> float:
    """Minimum rounds T* for convergence (RAudit Eq 26).

    T* >= log(D_0 / ε) / log(1 / κ)

    Args:
        d0:      Initial divergence D_0 (must be > 0).
        epsilon: Convergence tolerance ε (must be > 0).
        kappa:   Contraction rate κ ∈ (0, 1).

    Returns:
        T* as a float (caller may ceil for integer rounds).

    Raises:
        ValueError: If inputs are out of valid range.
    """
    if d0 <= 0.0:
        raise ValueError(f"d0 must be > 0, got {d0}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if kappa <= 0.0 or kappa >= 1.0:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    if d0 <= epsilon:
        return 0.0  # Already converged
    return math.log(d0 / epsilon) / math.log(1.0 / kappa)


def check_convergence(js_current: float, epsilon: float) -> bool:
    """Check whether the current JS divergence is below the convergence threshold.

    Returns True if js_current <= epsilon.
    """
    return js_current <= epsilon


def expected_divergence(d0: float, kappa: float, t: int) -> float:
    """Expected divergence upper bound at round t (RAudit Eq 23).

    E[D_t] <= κ^t · D_0

    Args:
        d0:    Initial divergence.
        kappa: Contraction rate κ ∈ [0, 1].
        t:     Round index.

    Returns:
        Upper bound on expected divergence at round t.
    """
    return (kappa ** t) * d0


def simulate_contraction(
    d0: float,
    kappa: float,
    epsilon: float,
    max_steps: int = 1000,
) -> List[float]:
    """Simulate deterministic contraction until convergence or max_steps.

    Returns a list of divergence values [D_0, D_1, ..., D_T] where T is the
    first step at which D_t <= epsilon, or max_steps if convergence is not reached.

    Args:
        d0:        Initial divergence.
        kappa:     Contraction rate κ ∈ [0, 1].
        epsilon:   Convergence tolerance.
        max_steps: Safety cap on iteration count.

    Returns:
        List of divergence values from D_0 through the convergence step.
    """
    trajectory = [d0]
    d_t = d0
    for _ in range(max_steps):
        d_t = d_t * kappa
        trajectory.append(d_t)
        if d_t <= epsilon:
            break
    return trajectory
