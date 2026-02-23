"""
PID control law for RAudit debate quality regulation.

Implements the sycophancy-augmented error (Eq 5) and the standard PID
output formula (Eq 6), plus a stateful PIDController wrapper.
"""

from eval.PID.types import PIDGains, PIDConfig, PIDState, PIDStepResult
from eval.PID.beta_dynamics import update_beta_clipped
from eval.PID.sycophancy import compute_sycophancy_signal


def compute_error(rho_star: float, rho_bar: float, mu: float, s_t: int) -> float:
    """Sycophancy-augmented error (RAudit Eq 5).

    e_t = (ρ* − ρ̄_t) + μ · s_t

    Args:
        rho_star: Target reasonableness score.
        rho_bar:  Observed mean reasonableness score for the current round.
        mu:       Sycophancy weighting coefficient.
        s_t:      Binary sycophancy indicator (0 or 1).
    """
    return (rho_star - rho_bar) + mu * s_t


def compute_pid_output(
    gains: PIDGains,
    e_t: float,
    integral: float,
    e_prev: float,
) -> tuple:
    """Standard PID output (RAudit Eq 6).

    u_t = Kp·e_t + Ki·Σe_j + Kd·(e_t − e_{t−1})

    Returns:
        (u_t, p_term, i_term, d_term)
    """
    p_term = gains.Kp * e_t
    i_term = gains.Ki * integral
    d_term = gains.Kd * (e_t - e_prev)
    u_t = p_term + i_term + d_term
    return u_t, p_term, i_term, d_term


class PIDController:
    """Stateful PID controller for multi-round debate regulation.

    Wraps compute_error, compute_pid_output, and beta update into a single
    step() call that advances internal state.
    """

    def __init__(self, config: PIDConfig, initial_beta: float = 0.5) -> None:
        self.config = config
        self.state = PIDState(beta=initial_beta)

    def step(
        self,
        rho_bar: float,
        js_current: float = 0.0,
        ov_current: float = 0.0,
    ) -> PIDStepResult:
        """Advance the controller by one round.

        Args:
            rho_bar:    Mean reasonableness score for this round.
            js_current: JS divergence for this round (for sycophancy detection).
            ov_current: Evidence overlap for this round (for sycophancy detection).

        Returns:
            PIDStepResult with all computed quantities.
        """
        cfg = self.config
        st = self.state

        # Track JS / overlap history
        st.js_history.append(js_current)
        st.ov_history.append(ov_current)

        # Sycophancy signal (need at least 2 rounds of history)
        if len(st.js_history) >= 2:
            s_t = compute_sycophancy_signal(
                js_current=st.js_history[-1],
                js_prev=st.js_history[-2],
                ov_current=st.ov_history[-1],
                ov_prev=st.ov_history[-2],
                delta_s=cfg.delta_s,
            )
        else:
            s_t = 0

        # Error (Eq 5)
        e_t = compute_error(cfg.rho_star, rho_bar, cfg.mu, s_t)

        # Accumulate integral
        st.integral += e_t

        # PID output (Eq 6)
        u_t, p_term, i_term, d_term = compute_pid_output(
            cfg.gains, e_t, st.integral, st.e_prev
        )

        # Beta update (Eq 14)
        beta_new = update_beta_clipped(st.beta, cfg.gamma_beta, u_t)

        # Build result
        result = PIDStepResult(
            e_t=e_t,
            u_t=u_t,
            beta_new=beta_new,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            s_t=s_t,
        )

        # Advance state
        st.e_prev = e_t
        st.e_t = e_t
        st.beta = beta_new
        st.t += 1

        return result

    @property
    def beta(self) -> float:
        """Current behavior dial value."""
        return self.state.beta

    @property
    def t(self) -> int:
        """Current round index."""
        return self.state.t
