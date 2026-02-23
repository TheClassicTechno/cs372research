"""
Stability analysis for RAudit PID control gains.

Implements the bounded correction condition (Proposition 1):
  - PID output bound (Eq 12)
  - Stability condition (Eq 17)
  - Non-oscillation condition (Eq 18)
"""

from eval.PID.types import PIDGains


class GainInstabilityError(Exception):
    """Raised when PID gains violate the stability condition (Prop 1, Eq 17)."""
    pass


def max_pid_output_bound(gains: PIDGains, T_max: int) -> float:
    """Upper bound on |u_t| assuming |e_t| ≤ 1 (RAudit Eq 12).

    |u_t| ≤ Kp + T_max·Ki + 2·Kd
    """
    return gains.Kp + T_max * gains.Ki + 2.0 * gains.Kd


def check_stability(gains: PIDGains, T_max: int, gamma_beta: float) -> bool:
    """Check whether the stability condition holds (RAudit Eq 17).

    Kp + T_max·Ki + 2·Kd < 1/γ_β

    Returns True if the gains are stable under the given parameters.
    """
    if gamma_beta <= 0.0:
        return True  # No decay constraint to violate
    bound = max_pid_output_bound(gains, T_max)
    return bound < 1.0 / gamma_beta


def check_non_oscillation(gains: PIDGains, T_max: int, gamma_beta: float) -> bool:
    """Stricter non-oscillation condition (RAudit Eq 18).

    Kp + T_max·Ki + 2·Kd < 1 − γ_β

    Returns True if the gains satisfy the non-oscillation bound.
    """
    bound = max_pid_output_bound(gains, T_max)
    return bound < 1.0 - gamma_beta


def validate_gains(gains: PIDGains, T_max: int, gamma_beta: float) -> None:
    """Validate PID gains against the stability condition.

    Raises:
        GainInstabilityError: If Eq 17 is violated.
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
