"""
RAudit PID — Control-theoretic math core for multi-agent debate quality regulation.

=================================================================================
WHAT THIS MODULE IS
=================================================================================

This package contains the pure math behind the PID feedback loop that
regulates multi-agent debates in the RAudit framework (Chang & Geng, 2026).

Think of it like a thermostat for debate quality:
  - The CRIT scorer (external, not in this package) measures how "reasonable"
    the agents' arguments are each round, producing a score ρ̄_t.
  - This PID module compares that score to a target, computes a correction
    signal, and updates a "behavior dial" (β) that tells the actuator layer
    how aggressively agents should explore vs. exploit.
  - The actuator layer (eval/actuator/) then translates β into a concrete
    strategy (EXPLORE / REFINE / CONSOLIDATE / NO_OP).

=================================================================================
WHERE THIS SITS IN THE PIPELINE
=================================================================================

    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
    │  Agents run  │────▶│  CRIT scores  │────▶│  PID math    │────▶│ Actuator │
    │  a debate    │     │  each round   │     │  (this pkg)  │     │  policy  │
    │  round       │     │  ρ̄_t, JS, Ov  │     │  → u_t, β    │     │  → mode  │
    └─────────────┘     └──────────────┘     └──────────────┘     └──────────┘
          ▲                                                            │
          └────────────────────────────────────────────────────────────┘
                          β feeds back into the next round

This package implements ONLY the middle box.  No LLM calls, no prompts, no
schema mutations.  Pure functions and dataclasses.

=================================================================================
MODULE MAP
=================================================================================

    types.py        — Data containers: gains, config, state, step results.
                      Every other module in this package uses these.

    controller.py   — The core PID loop: compute error, compute correction
                      signal u_t, and the stateful PIDController that ties
                      everything together round-by-round.

    beta_dynamics.py — How the behavior dial β gets updated each round
                       (with and without clamping to [0,1]).

    stability.py    — Safety checks: do the PID gains guarantee that β
                      won't blow up or oscillate?  Run these before starting
                      a debate to validate your configuration.

    sycophancy.py   — Detecting "fake agreement": when agents converge on
                      the same answer not because they reasoned well, but
                      because they're copying each other.  Uses JS divergence
                      over scores and Jaccard overlap over cited evidence.

    termination.py  — "How many rounds do we need?" Logarithmic bound on
                      when the debate should stop because agents have
                      converged enough (or will never converge).

=================================================================================
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
