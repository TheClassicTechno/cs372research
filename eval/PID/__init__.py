"""
RAudit PID — Control-theoretic math core for multi-agent debate quality regulation.

Implements the PID control loop from Chang & Geng (2026) for adjusting agent
behavior via a sycophancy-augmented error signal and logarithmic termination bound.

Modules:
    types          — Dataclasses for PID state, config, and step results
    controller     — PID control law (Eq 5-6), error computation
    beta_dynamics  — Beta update with clipping (Eq 13-15), steady-state
    stability      — Bounded correction condition (Prop 1, Eq 12/17)
    sycophancy     — Sycophancy signal s_t (Eq 4), JS divergence (Eq 2), overlap (Eq 3)
    termination    — Logarithmic termination bound T* (Prop 2, Eq 25-27)
"""

from eval.PID.types import PIDGains, PIDConfig, PIDState, PIDStepResult
from eval.PID.controller import compute_error, compute_pid_output, PIDController
from eval.PID.beta_dynamics import update_beta_unclipped, update_beta_clipped, steady_state_beta
from eval.PID.stability import (
    max_pid_output_bound,
    check_stability,
    check_non_oscillation,
    validate_gains,
    GainInstabilityError,
)
from eval.PID.sycophancy import (
    shannon_entropy,
    jensen_shannon_divergence,
    evidence_overlap,
    compute_sycophancy_signal,
)
from eval.PID.termination import (
    termination_bound,
    check_convergence,
    expected_divergence,
    simulate_contraction,
)

__all__ = [
    # types
    "PIDGains",
    "PIDConfig",
    "PIDState",
    "PIDStepResult",
    # controller
    "compute_error",
    "compute_pid_output",
    "PIDController",
    # beta_dynamics
    "update_beta_unclipped",
    "update_beta_clipped",
    "steady_state_beta",
    # stability
    "max_pid_output_bound",
    "check_stability",
    "check_non_oscillation",
    "validate_gains",
    "GainInstabilityError",
    # sycophancy
    "shannon_entropy",
    "jensen_shannon_divergence",
    "evidence_overlap",
    "compute_sycophancy_signal",
    # termination
    "termination_bound",
    "check_convergence",
    "expected_divergence",
    "simulate_contraction",
]
