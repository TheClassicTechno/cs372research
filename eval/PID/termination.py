"""
Logarithmic termination bound for RAudit PID control (Proposition 2).

=================================================================================
PURPOSE
=================================================================================

After how many rounds can we stop the debate?  This file answers that
question in two ways:

    1. THEORETICAL BOUND — Given a contraction rate (how fast agents are
       converging), compute the minimum number of rounds T* needed for the
       divergence to drop below a tolerance epsilon.  This gives a
       GUARANTEED upper bound before the debate starts.

    2. RUNTIME CHECK — Each round, check whether the current divergence is
       already below epsilon.  If so, stop early.

The key insight (Proposition 2 in RAudit) is that if the PID controller is
working correctly, the disagreement between agents shrinks geometrically:

    D_t  ≤  κ^t * D_0

where:
    D_0   = initial divergence (how much agents disagree at the start)
    κ     = contraction rate (0 < κ < 1; smaller = faster convergence)
    D_t   = divergence at round t

This geometric shrinkage gives a logarithmic bound on the number of rounds:

    T*  ≥  log(D_0 / ε) / log(1 / κ)

So even if D_0 is large, the number of rounds needed only grows as the LOG
of the initial divergence — debates don't drag on forever.

=================================================================================
WHERE THIS IS USED IN THE PIPELINE
=================================================================================

    Before the debate:
        T_star = termination_bound(d0, epsilon, kappa)
        → Sets the maximum number of rounds to run.
        → Can be compared against T_max in PIDConfig.

    During the debate (each round):
        if check_convergence(js_current, epsilon):
            → Stop the debate early — agents have converged enough.

    For analysis/visualization:
        trajectory = simulate_contraction(d0, kappa, epsilon)
        → Produces the full expected divergence curve [D_0, D_1, ..., D_T]
        → Useful for plotting convergence behavior.

    Note: kappa (the contraction rate) is not something the PID controller
    computes directly — it's a property of the debate dynamics that you
    estimate empirically or derive from the stability analysis.  The PID
    controller's job is to ensure that kappa < 1 (i.e., that agents ARE
    converging), and then this module tells you how long that convergence
    will take.
"""

import math
from typing import List


def termination_bound(d0: float, epsilon: float, kappa: float) -> float:
    """Minimum rounds needed for the debate to converge. (RAudit Eq 26)

    Formula:  T*  =  log(D_0 / ε) / log(1 / κ)

    In plain English: "If divergence shrinks by a factor of κ each round,
    how many rounds until it drops from D_0 to below ε?"

    The answer is logarithmic in D_0/ε — so doubling the initial divergence
    only adds one extra round (at the same contraction rate).

    Examples:
        d0=1.0, epsilon=0.01, kappa=0.5  →  T* ≈ 6.6 rounds
        d0=1.0, epsilon=0.01, kappa=0.9  →  T* ≈ 43.7 rounds
        d0=1.0, epsilon=0.01, kappa=0.1  →  T* ≈ 2.0 rounds

    Pipeline context:
        Called before the debate to estimate how many rounds will be needed.
        The result can be compared to PIDConfig.T_max to check whether the
        allocated round budget is sufficient.

        Inputs come from:
            d0      ← initial JS divergence (from round 0 scoring)
            epsilon ← PIDConfig.epsilon (convergence tolerance)
            kappa   ← estimated or derived contraction rate

    Args:
        d0:      Initial divergence D_0.  Must be positive.
        epsilon: Convergence tolerance.  Must be positive.  The debate stops
                 when divergence drops below this.
        kappa:   Contraction rate.  Must be in (0, 1).  Smaller κ = faster
                 convergence.  κ = 0.5 means divergence halves each round.

    Returns:
        T* as a float.  The caller should ceil() this to get integer rounds.
        Returns 0.0 if d0 ≤ epsilon (already converged, no rounds needed).

    Raises:
        ValueError: If d0 ≤ 0, epsilon ≤ 0, or kappa is outside (0, 1).
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
    """Check whether the debate has converged enough to stop.

    Returns True when the current JS divergence (agent disagreement) is at
    or below the convergence tolerance.

    Pipeline context:
        Called each round after computing JS divergence.  When this returns
        True, the debate orchestrator can stop the debate early instead of
        running to T_max rounds.

        Input: js_current from sycophancy.jensen_shannon_divergence() or
               whatever external JS computation is being used.

    Args:
        js_current: Current round's JS divergence.
        epsilon:    Convergence tolerance (from PIDConfig.epsilon).

    Returns:
        True if js_current ≤ epsilon (debate has converged).
    """
    return js_current <= epsilon


def expected_divergence(d0: float, kappa: float, t: int) -> float:
    """Predicted upper bound on divergence at round t. (RAudit Eq 23)

    Formula:  E[D_t]  ≤  κ^t * D_0

    In plain English: "If divergence shrinks by factor κ each round, what's
    the worst it could be at round t?"

    This is the geometric contraction model that underlies the termination
    bound.  It assumes the PID controller is working correctly and the
    debate dynamics are stable.

    Useful for: plotting the expected convergence curve, comparing actual
    divergence against the theoretical prediction, detecting anomalies
    (if actual divergence exceeds this bound, something is wrong).

    Not used in the main pipeline — this is for analysis and debugging.

    Args:
        d0:    Initial divergence.
        kappa: Contraction rate κ ∈ [0, 1].
        t:     Round index (0 = start, 1 = after one round, etc.).

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
    """Simulate the deterministic contraction of divergence over rounds.

    Produces a trajectory [D_0, D_1, D_2, ...] where each D_{t+1} = κ * D_t,
    stopping when D_t drops below epsilon or max_steps is reached.

    This is a discrete simulation of the geometric contraction model.  It's
    useful for:
        - Visualizing how fast the debate should converge
        - Sanity-checking the termination_bound formula
        - Running what-if scenarios with different κ values

    Not used in the main pipeline — this is for analysis and visualization.

    Args:
        d0:        Initial divergence.
        kappa:     Contraction rate κ ∈ [0, 1].
        epsilon:   Convergence tolerance — stop when D_t ≤ epsilon.
        max_steps: Safety cap to prevent infinite loops if κ is very
                   close to 1 and convergence is extremely slow.

    Returns:
        List of divergence values from D_0 through the step where
        convergence was reached (or max_steps if not reached).
    """
    trajectory = [d0]
    d_t = d0
    for _ in range(max_steps):
        d_t = d_t * kappa
        trajectory.append(d_t)
        if d_t <= epsilon:
            break
    return trajectory
