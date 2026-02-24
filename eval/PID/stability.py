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

The error e_t = (rho_star - rho_bar) + mu * s_t can be larger than 1.0
when the sycophancy penalty fires (s_t=1, mu > 0).  The worst-case
magnitude of e_t is:

    e_max = max(rho_star + mu,  1 - rho_star)
            ───────────────     ─────────────
            worst positive      worst negative
            (rho_bar=0, s_t=1)  (rho_bar=1, s_t=0)

The worst case for |u_t| is when every term is at its maximum:

    |u_t|  ≤  e_max * Kp  +  T_max * e_max * Ki  +  2 * e_max * Kd
           =  e_max * (Kp + T_max * Ki + 2 * Kd)

This gives us the output bound (Eq 12, sycophancy-corrected):
    |u_t| ≤ e_max * (Kp + T_max * Ki + 2 * Kd)

For β to stay bounded, this worst-case u_t must not overwhelm the
momentum term.  The stability condition (Eq 17) says:

    e_max * (Kp + T_max * Ki + 2 * Kd)  <  1 / γ_β

And the stricter non-oscillation condition (Eq 18) says:

    e_max * (Kp + T_max * Ki + 2 * Kd)  <  1 - γ_β

The non-oscillation check is stricter — it guarantees not just that β is
bounded, but that it converges smoothly without bouncing around.

=================================================================================
WHERE THIS IS USED IN THE PIPELINE
=================================================================================

    Before the debate starts:

        config = PIDConfig(gains=PIDGains(...), ...)
        validate_gains(config.gains, config.T_max, config.gamma_beta,
                       rho_star=config.rho_star, mu=config.mu)
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


def _max_error(rho_star: float, mu: float) -> float:
    """Worst-case |e_t| given the target score and sycophancy weight.

    The error formula is:  e_t = (rho_star - rho_bar) + mu * s_t

    The most positive error occurs when rho_bar = 0 and s_t = 1:
        e_max_positive = rho_star + mu

    The most negative error occurs when rho_bar = 1 and s_t = 0:
        e_max_negative = rho_star - 1   (absolute value = 1 - rho_star)

    Returns the larger of the two magnitudes.
    """
    return max(rho_star + mu, 1.0 - rho_star)


def max_pid_output_bound(
    gains: PIDGains,
    T_max: int,
    rho_star: float = 1.0,
    mu: float = 0.0,
) -> float:
    """Worst-case upper bound on the correction signal |u_t|. (RAudit Eq 12)

    Formula:  |u_t| ≤ e_max * (Kp + T_max * Ki + 2 * Kd)

    where e_max = max(rho_star + mu, 1 - rho_star) is the worst-case
    error magnitude accounting for the sycophancy penalty.

    When rho_star and mu are not provided, defaults give e_max = 1.0,
    which is the conservative bound assuming |e_t| ≤ 1 (no sycophancy).

    Used by: check_stability() and check_non_oscillation() to verify that
    the gains are safe before starting a debate.

    Args:
        gains:    The Kp, Ki, Kd gain triple.
        T_max:    Maximum number of debate rounds.  Needed because the I-term
                  accumulates errors over all rounds, so more rounds =
                  potentially larger integral = larger u_t.
        rho_star: Target reasonableness score.  Default 1.0 gives the most
                  conservative e_max when mu=0.
        mu:       Sycophancy weight.  Default 0.0 (no sycophancy penalty).
    """
    e_max = _max_error(rho_star, mu)
    return e_max * (gains.Kp + T_max * gains.Ki + 2.0 * gains.Kd)


def check_stability(
    gains: PIDGains,
    T_max: int,
    gamma_beta: float,
    rho_star: float = 1.0,
    mu: float = 0.0,
) -> bool:
    """Check whether β is guaranteed to stay bounded. (RAudit Eq 17)

    Condition:  e_max * (Kp + T_max * Ki + 2 * Kd)  <  1 / γ_β

    In plain English: "The worst-case correction signal (including
    sycophancy-inflated errors) must be smaller than the inverse of the
    momentum factor."  If this fails, there exist error sequences where β
    would grow without bound (before clamping).

    Returns True if the gains are safe.  False if they might cause instability.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.  When gamma_beta ≤ 0
                    there is no momentum to amplify corrections, so the
                    condition is trivially satisfied.
        rho_star:   Target reasonableness score (for error bound calc).
        mu:         Sycophancy weight (for error bound calc).
    """
    if gamma_beta <= 0.0:
        return True  # No momentum means no amplification — always safe
    bound = max_pid_output_bound(gains, T_max, rho_star, mu)
    return bound < 1.0 / gamma_beta


def check_non_oscillation(
    gains: PIDGains,
    T_max: int,
    gamma_beta: float,
    rho_star: float = 1.0,
    mu: float = 0.0,
) -> bool:
    """Stricter check: will β converge smoothly without bouncing? (RAudit Eq 18)

    Condition:  e_max * (Kp + T_max * Ki + 2 * Kd)  <  1 - γ_β

    This is harder to satisfy than check_stability().  A configuration can
    pass stability (β won't blow up) but fail non-oscillation (β might
    bounce between high and low values from round to round).

    Returns True if β is guaranteed to converge smoothly.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.
        rho_star:   Target reasonableness score (for error bound calc).
        mu:         Sycophancy weight (for error bound calc).
    """
    bound = max_pid_output_bound(gains, T_max, rho_star, mu)
    return bound < 1.0 - gamma_beta


def validate_gains(
    gains: PIDGains,
    T_max: int,
    gamma_beta: float,
    rho_star: float = 1.0,
    mu: float = 0.0,
) -> None:
    """Validate PID gains and raise an error if they're unsafe.

    Call this once when setting up a PIDController to fail fast if the
    configuration would produce unstable behavior.

    Does nothing if gains are safe.  Raises GainInstabilityError with a
    detailed message if not.

    Args:
        gains:      The Kp, Ki, Kd gain triple.
        T_max:      Maximum debate rounds.
        gamma_beta: Momentum factor for β updates.
        rho_star:   Target reasonableness score (for error bound calc).
        mu:         Sycophancy weight (for error bound calc).

    Raises:
        GainInstabilityError: If the stability condition (Eq 17) is violated.
    """
    if not check_stability(gains, T_max, gamma_beta, rho_star, mu):
        bound = max_pid_output_bound(gains, T_max, rho_star, mu)
        limit = 1.0 / gamma_beta if gamma_beta > 0.0 else float("inf")
        raise GainInstabilityError(
            f"Gains violate stability (Eq 17): "
            f"|u_t| bound = {bound:.4f} >= 1/gamma_beta = {limit:.4f}. "
            f"Gains: Kp={gains.Kp}, Ki={gains.Ki}, Kd={gains.Kd}, "
            f"T_max={T_max}, gamma_beta={gamma_beta}, "
            f"rho_star={rho_star}, mu={mu}"
        )
