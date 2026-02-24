"""
Stability analysis for RAudit PID control gains.

=================================================================================
PURPOSE
=================================================================================

Before starting a debate, we need to verify that the PID gains (Kp, Ki, Kd)
are safe.  "Safe" means:

    1. The correction signal u_t can't grow so large that β flies off to
       infinity (or negative infinity) before the clamp catches it.

    2. β won't oscillate wildly between 0 and 1 from round to round,
       which would make the agents flip between explore and exploit
       erratically.

This file provides the checks.  You run them ONCE during setup (before the
debate starts) to validate your configuration.  If they fail, you need to
lower your gains or increase gamma_beta.

=================================================================================
THE KEY IDEA
=================================================================================

The PID output u_t is a weighted sum of three terms:
    u_t = Kp * e_t  +  Ki * sum(errors)  +  Kd * (e_t - e_prev)

Each error e_t is between -1 and +1 (since scores are in [0,1]).  So the
worst case for |u_t| is when every term is at its maximum:

    |u_t|  ≤  Kp * 1  +  Ki * T_max * 1  +  Kd * 2 * 1
              ─────────   ──────────────    ────────────
              worst P     worst I           worst D
              (max error) (max errors for   (error jumps from
                           T_max rounds)    -1 to +1)

This gives us the output bound (Eq 12):
    |u_t| ≤ Kp + T_max * Ki + 2 * Kd

For β to stay bounded, this worst-case u_t must not overwhelm the
momentum term.  The stability condition (Eq 17) says:

    Kp + T_max * Ki + 2 * Kd  <  1 / γ_β

And the stricter non-oscillation condition (Eq 18) says:

    Kp + T_max * Ki + 2 * Kd  <  1 - γ_β

The non-oscillation check is stricter — it guarantees not just that β is
bounded, but that it converges smoothly without bouncing around.

=================================================================================
WHERE THIS IS USED IN THE PIPELINE
=================================================================================

    Before the debate starts:

        config = PIDConfig(gains=PIDGains(...), ...)
        validate_gains(config.gains, config.T_max, config.gamma_beta)
          │
          ├── If gains are safe:  proceed to create PIDController
          └── If gains are unsafe: raises GainInstabilityError
                                   (caller must fix config)

    These checks are NOT called during the debate loop — they're a one-time
    safety gate at setup time.
"""

from eval.PID.types import PIDGains


class GainInstabilityError(Exception):
    """Raised when PID gains violate the stability condition.

    This means the chosen Kp, Ki, Kd values are too aggressive for the
    given T_max and gamma_beta.  The correction signal could grow large
    enough to make β unstable.

    Fix by: lowering Kp/Ki/Kd, reducing T_max, or increasing gamma_beta.
    """
    pass


def max_pid_output_bound(gains: PIDGains, T_max: int) -> float:
    """Worst-case upper bound on the correction signal |u_t|. (RAudit Eq 12)

    Formula:  |u_t| ≤ Kp + T_max * Ki + 2 * Kd

    This assumes the worst possible error sequence: every round's error is
    at its maximum magnitude (1.0), and the error can swing from -1 to +1
    between consecutive rounds.

    Used by: check_stability() and check_non_oscillation() to verify that
    the gains are safe before starting a debate.

    Args:
        gains: The Kp, Ki, Kd gain triple.
        T_max: Maximum number of debate rounds.  Needed because the I-term
               accumulates errors over all rounds, so more rounds =
               potentially larger integral = larger u_t.
    """
    return gains.Kp + T_max * gains.Ki + 2.0 * gains.Kd


def check_stability(gains: PIDGains, T_max: int, gamma_beta: float) -> bool:
    """Check whether β is guaranteed to stay bounded. (RAudit Eq 17)

    Condition:  Kp + T_max * Ki + 2 * Kd  <  1 / γ_β

    In plain English: "The worst-case correction signal must be smaller
    than the inverse of the momentum factor."  If this fails, there exist
    error sequences where β would grow without bound (before clamping).

    Returns True if the gains are safe.  False if they might cause instability.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.  When gamma_beta ≤ 0
                    there is no momentum to amplify corrections, so the
                    condition is trivially satisfied.
    """
    if gamma_beta <= 0.0:
        return True  # No momentum means no amplification — always safe
    bound = max_pid_output_bound(gains, T_max)
    return bound < 1.0 / gamma_beta


def check_non_oscillation(gains: PIDGains, T_max: int, gamma_beta: float) -> bool:
    """Stricter check: will β converge smoothly without bouncing? (RAudit Eq 18)

    Condition:  Kp + T_max * Ki + 2 * Kd  <  1 - γ_β

    This is harder to satisfy than check_stability().  A configuration can
    pass stability (β won't blow up) but fail non-oscillation (β might
    bounce between high and low values from round to round).

    Returns True if β is guaranteed to converge smoothly.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.
    """
    bound = max_pid_output_bound(gains, T_max)
    return bound < 1.0 - gamma_beta


def validate_gains(gains: PIDGains, T_max: int, gamma_beta: float) -> None:
    """Validate PID gains and raise an error if they're unsafe.

    Call this once when setting up a PIDController to fail fast if the
    configuration would produce unstable behavior.

    Does nothing if gains are safe.  Raises GainInstabilityError with a
    detailed message if not.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.

    Raises:
        GainInstabilityError: If the stability condition (Eq 17) is violated.
    """
    if not check_stability(gains, T_max, gamma_beta):
        bound = max_pid_output_bound(gains, T_max)
        limit = 1.0 / gamma_beta if gamma_beta > 0.0 else float("inf")
        raise GainInstabilityError(
            f"Gains violate stability (Eq 17): "
            f"|u_t| bound = {bound:.4f} >= 1/gamma_beta = {limit:.4f}. "
            f"Gains: Kp={gains.Kp}, Ki={gains.Ki}, Kd={gains.Kd}, "
            f"T_max={T_max}, gamma_beta={gamma_beta}"
        )
