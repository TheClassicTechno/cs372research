"""
PID control law for RAudit debate quality regulation.

=================================================================================
PURPOSE
=================================================================================

This is the central file of the PID package.  It answers the question:

    "Given how well the debate is going this round, how much should we
     adjust the agents' behavior for the next round?"

It does this in three steps each round:

    1. COMPUTE ERROR  — How far is the current quality score from the target?
       If the sycophancy detector flagged this round, add a penalty.
       (compute_error, Eq 5)

    2. COMPUTE PID OUTPUT — Combine the current error (P-term), the history
       of all past errors (I-term), and the rate of change (D-term) into
       a single correction signal u_t.
       (compute_pid_output, Eq 6)

    3. UPDATE BETA — Apply u_t to the behavior dial β (with momentum and
       clamping to [0,1]) so the actuator knows what strategy to use next.
       (done via beta_dynamics.update_beta_clipped, Eq 14)

=================================================================================
WHERE THIS SITS IN THE PIPELINE
=================================================================================

    Per-round flow:

        CRIT scorer produces:  rho_bar (quality score), js (disagreement), ov (overlap)
              │
              ▼
        PIDController.step(rho_bar, js, ov)
              │
              ├── sycophancy.compute_sycophancy_signal(js, ov, history)  → s_t
              ├── compute_error(rho_star, rho_bar, mu, s_t)              → e_t
              ├── compute_pid_output(gains, e_t, integral, e_prev)       → u_t
              ├── beta_dynamics.update_beta_clipped(beta, gamma, u_t)    → β_new
              │
              ▼
        PIDStepResult(e_t, u_t, beta_new, p_term, i_term, d_term, s_t)
              │
              ▼
        Actuator policy reads u_t and beta_new to decide next-round strategy

=================================================================================
"""

from eval.PID.types import PIDGains, PIDConfig, PIDState, PIDStepResult, Quadrant
from eval.PID.beta_dynamics import update_beta_clipped
from eval.PID.sycophancy import compute_sycophancy_signal


def classify_quadrant(
    js: float, rho_bar: float, delta_js: float, rho_star: float
) -> Quadrant:
    """Classify debate state into one of 4 quadrants for actuator routing.

    RAudit paper references:
        - Eq 7 (p.4): div(t) = 1[JS(t) >= delta_JS], qual(t) = 1[rho_bar(t) >= rho*]
        - Table 1 (p.4): Quadrant-based control table
        - Algorithm 1, line 13 (p.19, Appendix E): div/qual signal computation

    Args:
        js: Current Jensen-Shannon divergence between agent beliefs.
        rho_bar: Current mean CRIT reasoning quality score.
        delta_js: Diversity threshold (delta_JS from Eq 7, p.4).
                  NOT the same as delta_s (sycophancy threshold, Eq 4).
        rho_star: Target quality threshold (rho* from Eq 5, p.4).

    Returns:
        Quadrant enum value: STUCK, CHAOTIC, CONVERGED, or HEALTHY.
    """
    div = js >= delta_js
    qual = rho_bar >= rho_star
    if not div and not qual:
        return Quadrant.STUCK
    if div and not qual:
        return Quadrant.CHAOTIC
    if not div and qual:
        return Quadrant.CONVERGED
    return Quadrant.HEALTHY


def compute_error(rho_star: float, rho_bar: float, mu: float, s_t: int) -> float:
    """Compute how far the debate quality is from the target. (RAudit Eq 5)

    This is the "error signal" — the gap between where we want quality to be
    (rho_star) and where it actually is (rho_bar), plus an optional penalty
    if the sycophancy detector fired this round.

    The formula:
        e_t = (rho_star - rho_bar) + mu * s_t

    Plain English:
        error = (target score - actual score) + sycophancy penalty

    Sign conventions:
        e_t > 0  →  quality is BELOW target (need to push agents harder)
        e_t < 0  →  quality EXCEEDS target (can ease off)
        e_t = 0  →  quality is exactly on target

    When s_t = 1 (sycophancy detected), the error gets a bonus of +mu,
    which forces the controller to increase β even if rho_bar looks fine.
    This prevents the system from being fooled by agents who agree with
    each other without actually reasoning well.

    Pipeline context:
        Inputs come from:
            rho_star  ← PIDConfig (set once before debate)
            rho_bar   ← CRIT scorer's mean reasonableness score for this round
            mu        ← PIDConfig (set once before debate)
            s_t       ← sycophancy.compute_sycophancy_signal() (this round)

        Output feeds into:
            compute_pid_output() as the current error value e_t

    Args:
        rho_star: Target reasonableness score (e.g. 0.8 = "80% reasonable").
        rho_bar:  Observed mean reasonableness score from CRIT for this round.
        mu:       Sycophancy weighting coefficient.  How much extra error to
                  inject when sycophancy is detected.  mu=0 disables this.
        s_t:      Binary sycophancy indicator — 1 if detected, 0 if not.
    """
    return (rho_star - rho_bar) + mu * s_t


def compute_pid_output(
    gains: PIDGains,
    e_t: float,
    integral: float,
    e_prev: float,
) -> tuple:
    """Combine the three PID terms into a single correction signal. (RAudit Eq 6)

    This is the classic PID formula adapted for debate control.  It takes the
    current error and produces a number u_t that says "adjust β by this much."

    The formula:
        u_t = Kp * e_t  +  Ki * integral  +  Kd * (e_t - e_prev)
              ─────────    ──────────────    ─────────────────────
              P-term       I-term            D-term

    What each term does:

        P-term (Proportional):  Reacts to the error RIGHT NOW.
            "Quality is 0.2 below target, so push proportionally."
            Fast response, but can't eliminate steady-state offset alone.

        I-term (Integral):  Reacts to the SUM of all past errors.
            "We've been below target for 5 rounds, clearly we need more push."
            Eliminates persistent offset, but can overshoot if gains are high.

        D-term (Derivative):  Reacts to how FAST the error is changing.
            "Error jumped up suddenly — add extra push" or
            "Error is already improving — ease off to avoid overshoot."
            Provides damping against oscillation.

    Pipeline context:
        Inputs come from:
            gains     ← PIDConfig.gains (set once before debate)
            e_t       ← compute_error() (this round)
            integral  ← PIDState.integral (running sum, updated by PIDController)
            e_prev    ← PIDState.e_prev (error from the previous round)

        Output feeds into:
            beta_dynamics.update_beta_clipped(beta, gamma_beta, u_t)

    Args:
        gains:    The Kp, Ki, Kd gain triple.
        e_t:      Current round's error (from compute_error).
        integral: Sum of all errors from round 0 through the current round.
        e_prev:   Error from the previous round (0.0 on the first round).

    Returns:
        Tuple of (u_t, p_term, i_term, d_term) so the caller can inspect
        how much each component contributed.
    """
    p_term = gains.Kp * e_t
    i_term = gains.Ki * integral
    d_term = gains.Kd * (e_t - e_prev)
    u_t = p_term + i_term + d_term
    return u_t, p_term, i_term, d_term


class PIDController:
    """Stateful PID controller for multi-round debate regulation.

    This is the main entry point for the PID package.  You create one of
    these before the debate starts, then call step() after each round with
    the latest scores.  It returns a PIDStepResult telling you the new β
    and correction signal.

    Usage:
        config = PIDConfig(gains=PIDGains(Kp=0.3, Ki=0.05, Kd=0.1))
        ctrl = PIDController(config, initial_beta=0.5)

        for each debate round:
            rho_bar = crit_scorer.score(round)   # external
            js = compute_js(round)               # from sycophancy.py or external
            ov = compute_overlap(round)           # from sycophancy.py or external

            result = ctrl.step(rho_bar, js, ov)
            # result.beta_new  → send to actuator
            # result.u_t       → send to actuator
            # result.s_t       → log whether sycophancy was detected

    Internal flow of step():
        1. Append js/ov to history lists (for sycophancy round-over-round comparison)
        2. Check sycophancy: did JS drop sharply while overlap also dropped?
        3. Compute error: (target - actual) + sycophancy penalty
        4. Accumulate error into the integral (for the I-term)
        5. Compute PID output u_t from the P, I, D terms
        6. Update β with momentum and clamp to [0, 1]
        7. Advance state (save current error as e_prev for next round)
        8. Return PIDStepResult

    Pipeline context:
        Created by: debate orchestrator (before debate starts)
        Called by:   debate orchestrator (once per round, after CRIT scoring)
        Outputs to:  actuator policy layer (eval/actuator/)
    """

    def __init__(self, config: PIDConfig, initial_beta: float = 0.5) -> None:
        """Set up the controller with a config and starting β value.

        Args:
            config:       Full PID configuration (gains, target, thresholds).
            initial_beta: Starting value of the behavior dial.  0.5 is a
                          neutral midpoint between explore and exploit.
        """
        self.config = config
        self.state = PIDState(beta=initial_beta)

    def step(
        self,
        rho_bar: float,
        js_current: float = 0.0,
        ov_current: float = 0.0,
    ) -> PIDStepResult:
        """Run one PID control step with quadrant-based actuator routing.

        RAudit paper references:
            - Algorithm 1 (p.19, Appendix E): Full control loop pseudocode
            - Table 1 (p.4): Quadrant-based control table
            - Section 3.5 (p.4): Regulated Search Architecture
            - Eq 5-6 (p.4): Error and PID output formulas

        Args:
            rho_bar:    Mean reasonableness score from the CRIT scorer for
                        this round.  Range [0, 1].  This is the "measured
                        temperature" that the PID thermostat compares to
                        the target (rho_star).

            js_current: Jensen-Shannon divergence across agent scores for
                        this round.  Measures how much agents disagree.
                        High JS = lots of disagreement.  Low JS = agents
                        are converging.  Used by the sycophancy detector
                        and quadrant classification.
                        Default 0.0.

            ov_current: Evidence overlap (Jaccard index) for this round.
                        Measures how much the agents cited the same evidence.
                        Used by the sycophancy detector alongside JS.
                        Default 0.0.

        Returns:
            PIDStepResult containing:
                e_t        — the error for this round
                u_t        — the correction signal
                beta_new   — the new behavior dial value
                p/i/d_term — breakdown of the correction signal
                s_t        — whether sycophancy was detected (0 or 1)
                quadrant   — which of the 4 quadrants ("stuck"/"chaotic"/"converged"/"healthy")
                div_signal — True if JS >= δ_JS
                qual_signal — True if ρ̄ >= ρ*
        """
        cfg = self.config
        st = self.state

        # ---------------------------------------------------------------
        # Step 1: Record this round's JS and overlap values.
        # We keep the full history so the sycophancy detector can compare
        # this round to last round.
        # ---------------------------------------------------------------
        st.js_history.append(js_current)
        st.ov_history.append(ov_current)

        # ---------------------------------------------------------------
        # Step 2: Sycophancy detection.
        # On the first round there's no "previous round" to compare to,
        # so we skip detection and assume no sycophancy (s_t = 0).
        # From round 2 onward, compare JS and overlap trends.
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # Step 3: Compute this round's error (Eq 5).
        # Positive error = quality below target, negative = above target.
        # ---------------------------------------------------------------
        e_t = compute_error(cfg.rho_star, rho_bar, cfg.mu, s_t)

        # ---------------------------------------------------------------
        # Step 4: Accumulate error into the integral.
        # The integral is the running sum of all errors from round 0 to now.
        # The I-term (Ki * integral) will use this to correct persistent
        # offsets — if quality has been below target for many rounds, the
        # integral grows and the controller pushes harder.
        # ---------------------------------------------------------------
        st.integral += e_t

        # ---------------------------------------------------------------
        # Step 5: Compute the PID output u_t (Eq 6).
        # This combines the P, I, and D terms into one correction number.
        # ---------------------------------------------------------------
        u_t, p_term, i_term, d_term = compute_pid_output(
            cfg.gains, e_t, st.integral, st.e_prev
        )

        # ---------------------------------------------------------------
        # Step 6: Quadrant classification.
        # RAudit Eq 7 (p.4): div(t) = 1[JS >= δ_JS], qual(t) = 1[ρ̄ >= ρ*]
        # Algorithm 1, line 13 (p.19): div/qual signal computation
        # Uses δ_JS (diversity threshold), NOT δ_s (sycophancy threshold)
        # ---------------------------------------------------------------
        quadrant = classify_quadrant(
            js_current, rho_bar, cfg.delta_js, cfg.rho_star
        )

        # ---------------------------------------------------------------
        # Step 7: Quadrant-routed β update.
        # RAudit Algorithm 1 lines 19-29 (p.19, Appendix E)
        # Table 1 (p.4): Stuck=β↑, Chaotic=Hold β, Converged=β↓, Healthy=Natural decay
        # ---------------------------------------------------------------
        if quadrant == Quadrant.STUCK:
            # EXPLORE: force β up by Δβ increment (not PID-driven).
            # Algorithm 1 L22 (p.19): β ← min(β + Δβ, 1)
            # Table 1 (p.4): "β↑ + EXPLORE"
            # NOTE: Δβ increment is FINAL — no natural decay applied after.
            # Algorithm 1 L29 applies decay unconditionally in pseudocode,
            # but we treat the Stuck increment as final to match Table 1's
            # clear "β↑" directive. Applying decay after would partially
            # negate the forced exploration intent.
            beta_new = min(st.beta + cfg.delta_beta, 1.0)

        elif quadrant == Quadrant.CHAOTIC:
            # REFINE: PID-directed β adjustment without γ_β decay.
            # Algorithm 1 L24 (p.19): "Apply REFINE" — no explicit β formula
            # Table 1 (p.4): "Hold β + REFINE"
            # We interpret REFINE as PID-directed: β ← clip(β + u_t, 0, 1)
            # This makes each quadrant behaviorally distinct — Chaotic gets
            # direct PID control without the γ_β momentum that
            # Converged/Healthy use.
            beta_new = update_beta_clipped(st.beta, 1.0, u_t)  # γ_β=1.0 → just β + u_t

        elif quadrant == Quadrant.CONVERGED:
            # CONSOLIDATE: gentle decay.
            # Algorithm 1 L29 (p.19): β ← clip(β · γ_β, 0, 1)
            # Table 1 (p.4): "β↓ + CONSOLIDATE"
            beta_new = max(0.0, min(1.0, st.beta * cfg.gamma_beta))

        else:  # HEALTHY
            # Natural decay.
            # Algorithm 1 L29 (p.19): β ← clip(β · γ_β, 0, 1)
            # Table 1 (p.4): "Natural decay"
            beta_new = max(0.0, min(1.0, st.beta * cfg.gamma_beta))

        # ---------------------------------------------------------------
        # Package up the results.
        # ---------------------------------------------------------------
        result = PIDStepResult(
            e_t=e_t,
            u_t=u_t,
            beta_new=beta_new,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            s_t=s_t,
            quadrant=quadrant.value,
            div_signal=js_current >= cfg.delta_js,
            qual_signal=rho_bar >= cfg.rho_star,
        )

        # ---------------------------------------------------------------
        # Step 8: Advance state for the next round.
        # Save current error as "previous error" (for the D-term next round).
        # Update β and round counter.
        # ---------------------------------------------------------------
        st.e_prev = e_t
        st.e_t = e_t
        st.beta = beta_new
        st.t += 1

        return result

    @property
    def beta(self) -> float:
        """Current behavior dial value (shortcut for self.state.beta)."""
        return self.state.beta

    @property
    def t(self) -> int:
        """Current round index (shortcut for self.state.t)."""
        return self.state.t
