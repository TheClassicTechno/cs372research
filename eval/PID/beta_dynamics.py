"""
Beta (behavior dial) dynamics for RAudit PID control.

=================================================================================
PURPOSE
=================================================================================

β (beta) is the "behavior dial" — a number between 0 and 1 that tells the
actuator layer how much agents should explore (try new ideas) vs. exploit
(refine existing arguments):

    β close to 0  →  agents should tighten up, stick to what's working
    β close to 1  →  agents should explore, try new approaches
    β = 0.5       →  neutral starting point

This file provides the math for how β changes from one round to the next.
The PID controller computes a correction signal u_t, and these functions
apply it to β.

=================================================================================
THE UPDATE FORMULA
=================================================================================

Each round, β is updated with two ingredients:

    1. MOMENTUM — β carries over partially from the previous round, scaled
       by gamma_beta (γ_β).  This prevents β from jumping around wildly.
       Think of γ_β as "inertia" — higher values mean β changes more slowly.

    2. CORRECTION — The PID output u_t nudges β up or down based on how the
       debate is going.

    Combined (RAudit Eq 15, unclipped):
        β_new = γ_β * β_old + u_t

    With safety clamping (RAudit Eq 14, clipped):
        β_new = clip(γ_β * β_old + u_t,  0,  1)

    The clipped version is what the controller actually uses, because β
    outside [0, 1] has no meaning.

=================================================================================
WHERE THIS IS USED IN THE PIPELINE
=================================================================================

    PIDController.step()
        │
        ├── computes u_t via compute_pid_output()
        │
        └── calls update_beta_clipped(beta_old, gamma_beta, u_t)
                │
                └── returns β_new → stored in PIDState, returned in PIDStepResult
                                       │
                                       ▼
                            Actuator policy reads β_new

    The unclipped version (update_beta_unclipped) is provided for analysis
    and testing — to see what β WOULD be without the safety clamp.

    The steady-state function is for theoretical analysis: "If the PID
    output were constant forever, what would β converge to?"
"""


def update_beta_unclipped(beta: float, gamma_beta: float, u_t: float) -> float:
    """Compute the next β WITHOUT clamping to [0, 1]. (RAudit Eq 15)

    Formula:  β_new = γ_β * β_old + u_t

    This is the "raw" update before safety clamping.  Useful for analysis
    to see whether the system is trying to push β out of bounds.

    Not used in the main pipeline — update_beta_clipped() is used instead.

    Args:
        beta:       Current β value.
        gamma_beta: Momentum factor γ_β ∈ [0, 1).  How much of the old β
                    carries over.  0.9 means "90% of old β + correction."
        u_t:        PID correction signal from compute_pid_output().
                    Positive = push β up (more exploration).
                    Negative = pull β down (less exploration).
    """
    return beta * gamma_beta + u_t


def update_beta_clipped(beta: float, gamma_beta: float, u_t: float) -> float:
    """Compute the next β WITH clamping to [0, 1]. (RAudit Eq 14)

    Formula:  β_new = clip(γ_β * β_old + u_t,  0,  1)

    This is the version actually used by PIDController.step().  The clamp
    ensures β always stays in the valid range regardless of how large u_t is.

    Args:
        beta:       Current β value.
        gamma_beta: Momentum factor γ_β ∈ [0, 1).
        u_t:        PID correction signal from compute_pid_output().
    """
    return max(0.0, min(1.0, beta * gamma_beta + u_t))


def steady_state_beta(u: float, gamma_beta: float) -> float:
    """What β would converge to if the PID output were constant. (RAudit Eq 16)

    Formula:  β_ss = u / (1 - γ_β)

    This is found by setting β_new = β_old = β_ss in the unclipped update:
        β_ss = γ_β * β_ss + u
        β_ss - γ_β * β_ss = u
        β_ss = u / (1 - γ_β)

    Useful for:
        - Verifying that a given PID configuration will settle to a
          reasonable β value in the long run.
        - Understanding the relationship between gain tuning and steady
          behavior: if u is small and γ_β is high, steady-state β can
          still be large (because the momentum amplifies small corrections).

    Not used in the main pipeline — this is an analysis/debugging tool.

    Args:
        u:          Constant PID output.
        gamma_beta: Momentum factor γ_β.  Must be < 1, otherwise β grows
                    without bound (no steady state exists).

    Raises:
        ValueError: If gamma_beta >= 1 (no steady state exists).
    """
    if gamma_beta >= 1.0:
        raise ValueError(
            f"gamma_beta must be < 1 for steady state to exist, got {gamma_beta}"
        )
    return u / (1.0 - gamma_beta)
