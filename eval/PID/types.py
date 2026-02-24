"""
Data structures for the RAudit PID control loop.

=================================================================================
PURPOSE
=================================================================================

This file defines the four containers that flow through the PID pipeline:

    PIDGains       — The three tuning knobs (Kp, Ki, Kd) that control how
                     aggressively the system reacts to errors.

    PIDConfig      — Everything you need to set up a PID controller for a
                     debate: gains, target score, decay rate, thresholds, etc.

    PIDState       — The memory the controller carries between rounds: what
                     the error was last time, accumulated error history, the
                     current value of β, etc.

    PIDStepResult  — What one round of the PID loop produces: the error,
                     the correction signal, the new β, and a breakdown of
                     which term (P, I, or D) contributed how much.

=================================================================================
WHERE THESE ARE USED
=================================================================================

    PIDConfig  ──▶  PIDController.__init__()   (one-time setup before debate)
    PIDState   ──▶  PIDController.state        (mutated each round by step())
    PIDGains   ──▶  compute_pid_output()       (used in the P/I/D formula)
    PIDStepResult ◀── PIDController.step()     (returned each round to the caller)

The caller (eventually the debate orchestrator) reads PIDStepResult.beta_new
and PIDStepResult.u_t to decide what happens next round.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PIDGains:
    """The three tuning knobs of a PID controller.

    These control how the correction signal u_t responds to the error:

        Kp (Proportional gain) — How strongly to react to the CURRENT error.
            Large Kp = aggressive immediate corrections.
            Think: "The score is 0.2 below target right now, so push hard."

        Ki (Integral gain) — How strongly to react to ACCUMULATED past errors.
            Large Ki = the system "remembers" that it has been off-target for
            many rounds and pushes harder to compensate.
            Think: "We've been below target for 5 rounds straight, ramp up."

        Kd (Derivative gain) — How strongly to react to the RATE OF CHANGE
            of the error.  Large Kd = dampen sudden swings.
            Think: "The error jumped a lot this round, slow down to avoid
            overshooting."

    These appear directly in the PID output formula (RAudit Eq 6):
        u_t = Kp * e_t  +  Ki * sum(e_j)  +  Kd * (e_t - e_{t-1})

    Used by: compute_pid_output() in controller.py, and by the stability
    checks in stability.py to verify the gains won't cause β to blow up.
    """
    Kp: float
    Ki: float
    Kd: float


@dataclass
class PIDConfig:
    """Complete configuration for one PID-controlled debate.

    You create one of these before the debate starts and pass it to
    PIDController().  It bundles together all the knobs and thresholds.

    Attributes:
        gains:      The Kp/Ki/Kd tuning triple (see PIDGains above).

        rho_star:   The target "reasonableness" score that we want the debate
                    to reach.  CRIT (external scorer) produces a per-round
                    mean score rho_bar; the PID controller tries to drive
                    rho_bar toward rho_star.  Higher = stricter quality bar.
                    Default 0.8 means "we want 80% reasonableness."

        gamma_beta: Momentum/decay factor for the behavior dial β.  Each
                    round, β is partially carried over from the previous
                    round: β_{t+1} = γ_β * β_t + u_t.  Values close to 1.0
                    mean β changes slowly (high momentum); values close to
                    0.0 mean β resets quickly.  Must be < 1 for the system
                    to be stable.  Default 0.9.

        mu:         How much weight the sycophancy alarm adds to the error.
                    When the sycophancy detector fires (s_t = 1), the error
                    increases by μ, making the controller push harder.
                    Default 1.0.

        delta_s:    Sensitivity threshold for sycophancy detection.  The
                    sycophancy signal fires only if JS divergence drops by
                    more than delta_s in one round.  Smaller = more sensitive.
                    Default 0.05.

        T_max:      Maximum number of debate rounds allowed.  Used by the
                    stability checker to verify that accumulated integral
                    error won't blow up the correction signal.  Default 20.

        epsilon:    Convergence tolerance.  When JS divergence (the measure
                    of how much agents disagree) drops below epsilon, the
                    debate is considered converged and can stop.  Default 0.01.
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
    """The controller's memory between rounds.

    PIDController creates one of these internally and mutates it each time
    step() is called.  You normally don't construct this directly — the
    controller manages it.

    Attributes:
        t:          Which round we're on (0 = first round, increments by 1).

        e_t:        The error computed during the most recent round.  This is
                    "how far off the debate quality is from target, plus a
                    sycophancy penalty if detected."

        integral:   Running total of all past errors.  This is the sum that
                    the I-term (Ki * integral) uses.  If the debate has been
                    below target for many rounds, this number grows, which
                    makes the controller push harder over time.

        e_prev:     The error from the PREVIOUS round.  The D-term uses the
                    difference (e_t - e_prev) to detect whether the error is
                    getting better or worse.

        beta:       The current value of the "behavior dial" β ∈ [0, 1].
                    β = 0 means "fully exploit" (agents should tighten up).
                    β = 1 means "fully explore" (agents should try new ideas).
                    The actuator layer reads this to decide strategy.

        js_history: List of JS divergence values, one per round.  JS measures
                    how much the agents disagree with each other.  We keep
                    the history so the sycophancy detector can compare
                    round-over-round changes.

        ov_history: List of evidence-overlap values, one per round.  Overlap
                    measures how much the agents are citing the same evidence.
                    Also used by the sycophancy detector for round-over-round
                    comparison.
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
    """What one round of the PID controller produces.

    Returned by PIDController.step() after processing one debate round.
    This is the main output that the rest of the pipeline reads.

    In the pipeline:
        CRIT scorer → rho_bar, js, ov → PIDController.step() → PIDStepResult
                                                                     │
        Actuator policy reads result.u_t and result.beta_new ◀───────┘

    Attributes:
        e_t:      The error for this round.  Positive means "quality is below
                  target" (agents need to do better).  Negative means "quality
                  exceeds target."  Includes the sycophancy penalty if s_t=1.
                  (RAudit Eq 5)

        u_t:      The raw correction signal — the single number that
                  summarizes "how much should we adjust?"  Positive u_t means
                  "push β up (more exploration)."  Negative means "pull β
                  down (less exploration)."  This is the weighted sum of the
                  P, I, and D terms.  (RAudit Eq 6)

        beta_new: The updated behavior dial after applying u_t and clamping
                  to [0, 1].  This is what the actuator layer will use.
                  (RAudit Eq 14)

        p_term:   Proportional contribution: Kp * e_t.
                  "How much of u_t came from reacting to the current error?"

        i_term:   Integral contribution: Ki * sum(all past errors).
                  "How much of u_t came from reacting to accumulated history?"

        d_term:   Derivative contribution: Kd * (e_t - e_{t-1}).
                  "How much of u_t came from reacting to the error trend?"

        s_t:      Sycophancy indicator: 1 if the detector flagged this round
                  as suspicious (agents converging without good reason),
                  0 otherwise.  When s_t=1, the error e_t gets a bonus
                  penalty of +μ, which makes the controller push harder.
    """
    e_t: float
    u_t: float
    beta_new: float
    p_term: float
    i_term: float
    d_term: float
    s_t: int
