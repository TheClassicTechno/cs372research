"""
Beta (behavior dial) dynamics for RAudit PID control.

Implements the unclipped update (Eq 15), clipped update (Eq 14),
and steady-state formula (Eq 16).
"""


def update_beta_unclipped(beta: float, gamma_beta: float, u_t: float) -> float:
    """Unclipped beta update (RAudit Eq 15).

    β̄_{t+1} = β_t · γ_β + u_t
    """
    return beta * gamma_beta + u_t


def update_beta_clipped(beta: float, gamma_beta: float, u_t: float) -> float:
    """Clipped beta update (RAudit Eq 14).

    β_{t+1} = clip(β_t · γ_β + u_t, 0, 1)
    """
    return max(0.0, min(1.0, beta * gamma_beta + u_t))


def steady_state_beta(u: float, gamma_beta: float) -> float:
    """Steady-state beta under constant PID output (RAudit Eq 16).

    β_ss = u / (1 − γ_β)

    Requires gamma_beta < 1 to avoid division by zero.

    Raises:
        ValueError: If gamma_beta >= 1 (no steady state exists).
    """
    if gamma_beta >= 1.0:
        raise ValueError(
            f"gamma_beta must be < 1 for steady state to exist, got {gamma_beta}"
        )
    return u / (1.0 - gamma_beta)
