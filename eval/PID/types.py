"""
Data structures for the RAudit PID control loop.

Defines gains, configuration, mutable state, and per-step result containers
used across controller, beta_dynamics, stability, sycophancy, and termination.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PIDGains:
    """Proportional-Integral-Derivative gain triple (RAudit Eq 6)."""
    Kp: float
    Ki: float
    Kd: float


@dataclass
class PIDConfig:
    """Full configuration for one PID control loop instance.

    Attributes:
        gains:      PID gain triple (Kp, Ki, Kd).
        rho_star:   Target reasonableness score (ρ*).
        gamma_beta: Decay/momentum factor for beta update (γ_β ∈ [0, 1)).
        mu:         Sycophancy weighting coefficient (μ ≥ 0).
        delta_s:    JS-divergence drop threshold for sycophancy detection (δ_s > 0).
        T_max:      Maximum allowed debate rounds.
        epsilon:    Convergence tolerance for termination check.
    """
    gains: PIDGains
    rho_star: float = 0.8
    gamma_beta: float = 0.9
    mu: float = 1.0
    delta_s: float = 0.05
    T_max: int = 20
    epsilon: float = 0.01


@dataclass
class PIDState:
    """Mutable state carried across PID steps.

    Attributes:
        t:          Current round index (0-based).
        e_t:        Most recent error value.
        integral:   Running sum of errors (for the I-term).
        e_prev:     Error from the previous step (for the D-term).
        beta:       Current behavior dial value β ∈ [0, 1].
        js_history: Per-round JS divergence values for sycophancy tracking.
        ov_history: Per-round evidence-overlap values for sycophancy tracking.
    """
    t: int = 0
    e_t: float = 0.0
    integral: float = 0.0
    e_prev: float = 0.0
    beta: float = 0.5
    js_history: List[float] = field(default_factory=list)
    ov_history: List[float] = field(default_factory=list)


@dataclass
class PIDStepResult:
    """Output of a single PID controller step.

    Attributes:
        e_t:      Sycophancy-augmented error (RAudit Eq 5).
        u_t:      Raw PID output signal (RAudit Eq 6).
        beta_new: Updated behavior dial after clipping (RAudit Eq 14).
        p_term:   Proportional contribution Kp * e_t.
        i_term:   Integral contribution Ki * Σe_j.
        d_term:   Derivative contribution Kd * (e_t - e_{t-1}).
        s_t:      Sycophancy indicator (0 or 1) for this step.
    """
    e_t: float
    u_t: float
    beta_new: float
    p_term: float
    i_term: float
    d_term: float
    s_t: int
