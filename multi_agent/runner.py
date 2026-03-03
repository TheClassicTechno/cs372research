"""
High-level runner for the multi-agent debate system.

WHAT CHANGED AND WHY
---------------------
Previously, the runner compiled a single monolithic LangGraph and called
graph.invoke() once.  The debate loop (propose → critique → revise →
repeat → judge) was managed entirely inside the graph via a conditional
edge (should_continue).  This made it impossible for external code to
inspect or modify state between rounds — there was no seam to hook into.

This module now decomposes execution into three phases, each backed by
its own LangGraph sub-graph:

    Phase 1 — Pipeline:      news → data → build_context
    Phase 2 — Debate rounds:  for t in range(max_rounds):
                                  propose → critique → revise
    Phase 3 — Finalize:       judge → build_trace

The runner owns the iteration loop (Phase 2), calling single_round_graph
once per round.  Between rounds, external controllers can observe the
state and decide whether to continue, adjust agreeableness, etc.

When PID is enabled, Phase 2 decomposes each round into three separate
sub-graph invocations (propose, critique, revise).  CRIT scores and PID
updates beta after EACH phase, so the updated beta feeds into the next
phase within the same round:

    for t in range(max_rounds):
        state = self._run_round_with_pid(state, t + 1)
        # Inside _run_round_with_pid:
        #   propose(β₀) → CRIT → PID.step → β₁
        #   critique(β₁) → CRIT → PID.step → β₂
        #   revise(β₂)  → CRIT → PID.step → β₃
        if self.should_terminate(state):
            break

HOW INVARIANTS ARE PRESERVED
-----------------------------
1. All three phases use graph.invoke() — never manual state.update().
   This ensures LangGraph's reducers (Annotated[list, operator.add]
   for debate_turns) are applied correctly at every phase boundary.

2. Between sub-graph invocations the runner passes the full state dict.
   LangGraph treats the input dict as starting state and applies each
   node's returned dict through the reducer.  So debate_turns accumulates
   across pipeline → rounds → finalize seamlessly.

3. propose_node has an idempotency guard: if proposals already exist,
   it returns {} (no-op).  This means the single_round_graph can include
   propose in every round, but proposals are only generated once (round 1).

4. current_round is set by the runner to t+1 before each round AND is
   returned by the node functions (for monolithic graph compatibility).
   The two sources agree: runner sets t+1, propose returns 1 (round 1 only),
   revise returns current_round+1 (same as the next iteration's t+1).

5. proposals (plain list, no reducer) — set once, preserved by guard.
   critiques/revisions (plain list, no reducer) — replaced each round.
   debate_turns (Annotated[list, operator.add]) — appended each round.
   These behaviors match the monolithic graph exactly.

6. run_returning_state() duplicates run()'s logic but returns raw state
   instead of (Action, AgentTrace), without disk I/O.  This enables
   equivalence testing against the old monolithic path.

Usage:
    from multi_agent import MultiAgentRunner, DebateConfig, Observation, AgentRole

    config = DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL],
        max_rounds=2,
        agreeableness=0.3,
        enable_adversarial=True,
        mock=True,
    )
    runner = MultiAgentRunner(config)
    action, trace = runner.run(observation)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Dedicated PID loggers.  pid.metrics logs at INFO so PID output is
# visible by default (pid_log_metrics defaults to True).  Instance flags
# gate whether info() calls are made, so multiple runners with different
# pid_log_* settings don't conflict on the singleton.
pid_metrics_logger = logging.getLogger("pid.metrics")
pid_metrics_logger.setLevel(logging.INFO)
pid_llm_logger = logging.getLogger("pid.llm")


def _round_floats(obj, ndigits=4):
    """Recursively round floats in nested dicts/lists for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


from .config import AgentRole, DebateConfig
from .prompts.registry import beta_to_bucket, reset_registry_cache
from .graph import (
    compile_finalize_graph,
    compile_parallel_single_round_graph,
    compile_pipeline_graph,
    compile_single_round_graph,
    # Per-phase graphs (for PID intervention)
    compile_propose_graph,
    compile_critique_graph,
    compile_revise_graph,
    compile_parallel_propose_graph,
    compile_parallel_critique_graph,
    compile_parallel_revise_graph,
    _call_llm,
)
from .models import (
    Action,
    AgentTrace,
    Claim,
    ControllerOutput,
    DebateTurn,
    Observation,
    Order,
    PIDEvent,
    RoundMetrics,
)


class MultiAgentRunner:
    """
    Orchestrates multi-agent debate for trading decisions.

    Wraps three LangGraph sub-graphs (pipeline, single-round, finalize)
    with a clean interface matching Deveen's SingleAgentRunner /
    MajorityVoteRunner / DebateRunner pattern.

    The iteration loop lives in the runner so that external controllers
    (e.g. PID) can intervene between rounds.
    """

    def __init__(self, config: DebateConfig | None = None):
        self.config = config or DebateConfig()

        # Auto-inject devil's advocate if adversarial mode is enabled.
        # Copy roles to avoid mutating a shared config object.
        if self.config.enable_adversarial:
            if AgentRole.DEVILS_ADVOCATE not in self.config.roles:
                self.config.roles = list(self.config.roles) + [AgentRole.DEVILS_ADVOCATE]

        self.pipeline_graph = compile_pipeline_graph(self.config)
        if self.config.parallel_agents:
            self.single_round_graph = compile_parallel_single_round_graph(self.config)
        else:
            self.single_round_graph = compile_single_round_graph(self.config)
        self.finalize_graph = compile_finalize_graph(self.config)

        # --- Per-phase sub-graphs (compiled only when PID enabled) ---
        self._propose_graph = None
        self._critique_graph = None
        self._revise_graph = None

        if self.config.pid_enabled:
            if self.config.parallel_agents:
                self._propose_graph = compile_parallel_propose_graph(self.config)
                self._critique_graph = compile_parallel_critique_graph(self.config)
                self._revise_graph = compile_parallel_revise_graph(self.config)
            else:
                self._propose_graph = compile_propose_graph(self.config)
                self._critique_graph = compile_critique_graph(self.config)
                self._revise_graph = compile_revise_graph(self.config)

        # --- PID logging flags (instance-level, not mutating global logger) ---
        self._log_metrics = self.config.pid_log_metrics
        self._log_llm = self.config.pid_log_llm_calls

        # --- CRIT scorer (uses same LLM config as debate agents) ---
        self._crit_scorer = None
        if self.config.pid_enabled:
            from eval.crit import CritScorer

            base_llm_fn = lambda sys, usr: _call_llm(self.config.to_dict(), sys, usr)

            def _logging_llm_fn(system_prompt: str, user_prompt: str) -> str:
                if self._log_llm:
                    pid_llm_logger.debug(
                        "[CRIT LLM REQUEST]\n"
                        "===== SYSTEM PROMPT =====\n%s\n"
                        "===== USER PROMPT =====\n%s",
                        system_prompt, user_prompt,
                    )
                response = base_llm_fn(system_prompt, user_prompt)
                if self._log_llm:
                    pid_llm_logger.debug(
                        "[CRIT LLM RESPONSE]\n%s", response,
                    )
                return response

            self._crit_scorer = CritScorer(llm_fn=_logging_llm_fn)

        # --- PID controller ---
        self._pid_controller = None
        self._pid_events: list[PIDEvent] = []
        self._debate_id: str = str(uuid.uuid4())
        self._pid_phase_data: list[dict] = []
        self._stable_rounds: int = 0
        self._prev_rho_bar: float | None = None
        self._original_agreeableness = self.config.agreeableness

        if self.config.pid_config:
            from eval.PID.controller import PIDController
            from eval.PID.stability import validate_gains

            pid_cfg = self.config.pid_config
            validate_gains(
                pid_cfg.gains,
                pid_cfg.T_max,
                pid_cfg.gamma_beta,
                rho_star=pid_cfg.rho_star,
                mu=pid_cfg.mu,
            )
            self._pid_controller = PIDController(pid_cfg, self.config.initial_beta)

    def _reset_per_invocation_state(self) -> None:
        """Reset mutable state that must not leak between run() calls.

        A single DebateAgentSystem (and thus MultiAgentRunner) is reused
        across all decision points within an episode.  PID events and
        controller state must be fresh for each decision point.
        """
        self._pid_events = []
        self._debate_id = str(uuid.uuid4())
        self._pid_phase_data = []
        self._stable_rounds = 0
        self._prev_rho_bar = None
        reset_registry_cache()
        if self.config.pid_config:
            from eval.PID.controller import PIDController
            self._pid_controller = PIDController(
                self.config.pid_config, self.config.initial_beta
            )

    def run(self, observation: Observation) -> tuple[Action, AgentTrace]:
        """
        Run the full debate pipeline on an observation.

        Returns:
            (action, trace) -- the final Action and the full AgentTrace
        """
        self._reset_per_invocation_state()
        state = self._run_pipeline(observation)

        # Parse final action
        final_dict = state.get("final_action", {})
        action = self._parse_action(final_dict)

        # Parse trace
        trace_dict = state.get("trace", {})
        trace = self._parse_trace(trace_dict, observation, action)

        # Save to disk
        self._save_trace(trace, state)

        return action, trace

    def _run_pipeline(self, observation: Observation) -> dict:
        """Execute all three phases and return the raw state dict.

        Shared implementation for run() and run_returning_state() so
        the pipeline/loop/finalize logic is defined in exactly one place.
        """
        state = self._initialize_state(observation)

        # Phase 1: Pipeline (news digest, data analysis, context building)
        state = self.pipeline_graph.invoke(state)

        # Phase 2: Debate rounds
        terminated_early = False
        for t in range(self.config.max_rounds):
            state["current_round"] = t + 1
            # Reset critiques so operator.add (parallel graph) doesn't
            # accumulate across rounds.  Harmless for the sequential graph
            # (batch nodes replace lists entirely).
            # NOTE: revisions are NOT reset here because critique nodes need
            # to see the previous round's revisions as their source.
            if t > 0:
                state["critiques"] = []

            if self.config.pid_enabled:
                state = self._run_round_with_pid(state, t + 1)
            else:
                state = self.run_single_round(state)

            if self.config.parallel_agents:
                # With operator.add, revisions accumulate (previous round's
                # entries + this round's entries).  Keep only the latest
                # entry per role so the next round and judge see clean state.
                seen = set()
                deduped = []
                for rev in reversed(state.get("revisions", [])):
                    if rev["role"] not in seen:
                        seen.add(rev["role"])
                        deduped.append(rev)
                state["revisions"] = sorted(reversed(deduped), key=lambda r: r["role"])

                # Sort proposals and critiques by role for deterministic
                # ordering — parallel fan-out merges via operator.add in
                # thread-completion order, which is non-deterministic.
                state["proposals"] = sorted(state.get("proposals", []), key=lambda p: p["role"])
                state["critiques"] = sorted(state.get("critiques", []), key=lambda c: c["role"])

            terminated_early = self.should_terminate(state)
            if terminated_early:
                break

        # Emit end-of-debate PID summary
        if self._pid_controller and self._pid_phase_data and self._log_metrics:
            summary = self._build_pid_summary(terminated_early=terminated_early)
            pid_metrics_logger.info(json.dumps(summary, indent=2))

        # Phase 3: Judge + trace
        state = self.finalize_graph.invoke(state)

        return state

    def _run_round_with_pid(self, state: dict, round_num: int) -> dict:
        """Run one debate round with per-phase CRIT scoring and PID control.

        Each phase (propose, critique, revise) is invoked as a separate
        sub-graph.  After each phase, CRIT scores the current state and
        PID updates beta — so the updated beta feeds into the next phase.

        Also injects ``_current_beta`` into config so the modular prompt
        registry can map beta to tone bucket without seeing the numeric value.
        """
        beta = self._pid_controller.beta if self._pid_controller else self._original_agreeableness

        # --- Propose phase (no tone injection) ---
        propose_a = beta if self.config.pid_propose else self._original_agreeableness
        state["config"]["agreeableness"] = propose_a
        state["config"]["_current_beta"] = None
        state = self._propose_graph.invoke(state)

        if self._pid_controller and self._crit_scorer:
            try:
                self._pid_phase_step(state, round_num, "propose", beta_in=beta)
                beta = self._pid_controller.beta
            except Exception as exc:
                logger.warning(
                    "CRIT/PID failed in round %d propose: %s — keeping beta %.3f",
                    round_num, exc, beta,
                )

        # --- Critique phase ---
        critique_a = beta if self.config.pid_critique else self._original_agreeableness
        state["config"]["agreeableness"] = critique_a
        state["config"]["_current_beta"] = beta if self.config.pid_critique else None
        state = self._critique_graph.invoke(state)

        if self._pid_controller and self._crit_scorer:
            try:
                self._pid_phase_step(state, round_num, "critique", beta_in=beta)
                beta = self._pid_controller.beta
            except Exception as exc:
                logger.warning(
                    "CRIT/PID failed in round %d critique: %s — keeping beta %.3f",
                    round_num, exc, beta,
                )

        # --- Revise phase ---
        revise_a = beta if self.config.pid_revise else self._original_agreeableness
        state["config"]["agreeableness"] = revise_a
        state["config"]["_current_beta"] = beta if self.config.pid_revise else None
        state = self._revise_graph.invoke(state)

        if self._pid_controller and self._crit_scorer:
            try:
                self._pid_phase_step(state, round_num, "revise", beta_in=beta)
            except Exception as exc:
                logger.warning(
                    "CRIT/PID failed in round %d revise: %s — keeping beta %.3f",
                    round_num, exc, self._pid_controller.beta,
                )

        return state

    # Turn types allowed in CRIT traces for each phase.
    _PHASE_ALLOWED_TYPES: dict[str, set[str]] = {
        "propose": {"proposal"},
        "critique": {"proposal", "critique"},
        "revise": {"proposal", "critique", "revision"},
    }

    def _pid_phase_step(
        self, state: dict, round_num: int, phase: str, *, beta_in: float,
    ) -> None:
        """Run CRIT scorer + PID controller step after one debate phase.

        Called after each phase (propose, critique, revise) within a round.
        Selects appropriate decisions and trace context for the phase:
          - propose/critique: decisions = proposals
          - revise: decisions = revisions
        Traces are filtered to only include turns up to the current phase.
        """
        from eval.PID.sycophancy import jensen_shannon_divergence
        from eval.evidence import extract_agent_evidence_spans, compute_mean_overlap

        # --- Select decisions and traces for this phase ---
        if phase == "revise":
            decisions = state.get("revisions", state.get("proposals", []))
        else:
            decisions = state.get("proposals", [])

        allowed_types = self._PHASE_ALLOWED_TYPES.get(phase, {"proposal", "critique", "revision"})
        filtered_traces = [
            t for t in state.get("debate_turns", [])
            if t.get("type", "") in allowed_types
        ]

        # --- CRIT audit (per-agent scoring → RoundCritResult) ---
        round_crit = self._crit_scorer.score(
            case_data=state.get("enriched_context", ""),
            agent_traces=filtered_traces,
            decisions=decisions,
        )

        # --- JS divergence from agent confidences ---
        confidences = [
            d.get("action_dict", {}).get("confidence", 0.5)
            for d in decisions
        ]
        js = jensen_shannon_divergence(confidences) if len(confidences) >= 2 else 0.0

        # --- Evidence overlap (average pairwise Jaccard) ---
        evidence_sets = extract_agent_evidence_spans(
            decisions, allocation_mode=True,
        )
        ov = compute_mean_overlap(evidence_sets)
        if ov is None:
            ov = 0.0

        # Guard rho_bar against None/NaN
        rho_bar = round_crit.rho_bar
        if rho_bar is None or math.isnan(rho_bar):
            logger.warning("rho_bar is %s (round %d, %s) — defaulting to 0.0",
                           rho_bar, round_num, phase)
            rho_bar = 0.0

        # --- PID controller step ---
        old_beta = self._pid_controller.beta
        pid_result = self._pid_controller.step(rho_bar, js, ov)

        # Record PIDEvent
        ctrl_output = ControllerOutput(new_agreeableness=pid_result.beta_new)
        event = PIDEvent(
            round_index=round_num,
            metrics=RoundMetrics(
                round_index=round_num,
                rho_bar=rho_bar,
                js_divergence=js,
                ov_overlap=ov,
            ),
            crit_result=round_crit.model_dump(),
            pid_step={
                "e_t": pid_result.e_t,
                "u_t": pid_result.u_t,
                "beta_new": pid_result.beta_new,
                "p_term": pid_result.p_term,
                "i_term": pid_result.i_term,
                "d_term": pid_result.d_term,
                "s_t": pid_result.s_t,
                "quadrant": pid_result.quadrant,
                "div_signal": pid_result.div_signal,
                "qual_signal": pid_result.qual_signal,
            },
            controller_output=ctrl_output,
        )
        self._pid_events.append(event)

        # --- Structured JSON logging ---
        if self._log_metrics:
            agent_confidences = {}
            for d in decisions:
                role = d.get("role", "unknown")
                agent_confidences[role] = d.get("action_dict", {}).get("confidence", 0.5)

            agent_evidence = {
                role: sorted(spans) for role, spans in evidence_sets.items()
            }

            phase_data = {
                "type": "pid_phase",
                "debate_id": self._debate_id,
                "phase_id": str(uuid.uuid4()),
                "round": round_num,
                "phase": phase,
                "beta_in": beta_in,
                "tone_bucket": beta_to_bucket(pid_result.beta_new),
                "crit": {
                    "rho_bar": round_crit.rho_bar,
                    "agents": {
                        role: {
                            "rho_i": cr.rho_bar,
                            "pillars": {
                                "IC": cr.pillar_scores.internal_consistency,
                                "ES": cr.pillar_scores.evidence_support,
                                "TA": cr.pillar_scores.trace_alignment,
                                "CI": cr.pillar_scores.causal_integrity,
                            },
                            "diagnostics": {
                                "contradictions": cr.diagnostics.contradictions_detected,
                                "unsupported_claims": cr.diagnostics.unsupported_claims_detected,
                                "conclusion_drift": cr.diagnostics.conclusion_drift_detected,
                                "causal_overreach": cr.diagnostics.causal_overreach_detected,
                            },
                        }
                        for role, cr in round_crit.agent_scores.items()
                    },
                },
                "pid": {
                    "e_t": pid_result.e_t,
                    "integral": self._pid_controller.state.integral,
                    "e_prev": self._pid_controller.state.e_prev,
                    "p_term": pid_result.p_term,
                    "i_term": pid_result.i_term,
                    "d_term": pid_result.d_term,
                    "u_t": pid_result.u_t,
                    "beta_old": old_beta,
                    "beta_new": pid_result.beta_new,
                    "quadrant": pid_result.quadrant,
                    "div_signal": pid_result.div_signal,
                    "qual_signal": pid_result.qual_signal,
                    "sycophancy": pid_result.s_t,
                },
                "divergence": {
                    "js": js,
                    "ov": ov,
                    "agent_confidences": agent_confidences,
                    "agent_evidence_ids": agent_evidence,
                },
            }

            # Convergence metrics only meaningful on revise (end-of-round)
            if phase == "revise":
                phase_data["convergence"] = {
                    "stable_rounds": self._stable_rounds,
                    "delta_rho_actual": (
                        abs(rho_bar - self._prev_rho_bar)
                        if self._prev_rho_bar is not None else None
                    ),
                    "delta_rho_threshold": self.config.delta_rho,
                }

            phase_data = _round_floats(phase_data)
            self._pid_phase_data.append(phase_data)
            pid_metrics_logger.info(json.dumps(phase_data, indent=2))

    def _build_pid_summary(self, *, terminated_early: bool) -> dict:
        """Assemble end-of-debate JSON summary from accumulated phase data."""
        pid_cfg = self._pid_controller.config
        last = self._pid_phase_data[-1] if self._pid_phase_data else {}
        total_rounds = len({d["round"] for d in self._pid_phase_data}) if self._pid_phase_data else 0
        return {
            "type": "pid_summary",
            "debate_id": self._debate_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "Kp": pid_cfg.gains.Kp,
                "Ki": pid_cfg.gains.Ki,
                "Kd": pid_cfg.gains.Kd,
                "rho_star": pid_cfg.rho_star,
                "gamma_beta": pid_cfg.gamma_beta,
                "initial_beta": self.config.initial_beta,
                "epsilon": pid_cfg.epsilon,
                "T_max": pid_cfg.T_max,
                "mu": pid_cfg.mu,
                "delta_s": pid_cfg.delta_s,
                "delta_js": pid_cfg.delta_js,
                "delta_beta": pid_cfg.delta_beta,
                "pid_propose": self.config.pid_propose,
                "pid_critique": self.config.pid_critique,
                "pid_revise": self.config.pid_revise,
            },
            "phases": [rd["phase_id"] for rd in self._pid_phase_data],
            "outcome": {
                "total_rounds": total_rounds,
                "total_phase_steps": len(self._pid_phase_data),
                "terminated_early": terminated_early,
                "termination_reason": "stable_convergence" if terminated_early else "max_rounds",
                "final_beta": last.get("pid", {}).get("beta_new"),
                "final_rho_bar": last.get("crit", {}).get("rho_bar"),
                "final_js": last.get("divergence", {}).get("js"),
            },
        }

    def _initialize_state(self, observation: Observation) -> dict:
        """Build the initial state dict for the debate.

        This dict matches the DebateState TypedDict schema exactly.
        Every field is initialized to its zero-value so that nodes can
        safely read from state without KeyError, and so that LangGraph's
        reducers have a valid starting point (e.g. debate_turns starts
        as [] so operator.add can append to it).
        """
        return {
            "observation": observation.model_dump(),
            "config": self.config.to_dict(),
            "news_digest": "",
            "data_analysis": "",
            "enriched_context": "",
            "proposals": [],
            "critiques": [],
            "revisions": [],
            "current_round": 0,
            "debate_turns": [],
            "final_action": {},
            "strongest_objection": "",
            "audited_memo": "",
            "trace": {},
        }

    def run_single_round(self, state: dict) -> dict:
        """Execute one debate round via LangGraph: propose → critique → revise.

        This is the primitive that external controllers will call.  The
        caller sets state["current_round"] before invoking, then inspects
        the returned state to decide whether to continue.

        Invariants:
          - propose is idempotent (no-op if proposals exist)
          - debate_turns accumulates via LangGraph's operator.add reducer
          - critiques/revisions are replaced (plain list, no reducer)
        """
        return self.single_round_graph.invoke(state)

    def should_terminate(self, state: dict) -> bool:
        """Check if debate should end early via stable convergence.

        Requires ALL of the following for `convergence_window` consecutive
        rounds:
          1. quadrant == "converged" (high quality + low diversity)
          2. JS < epsilon (agents agree)
          3. |rho_bar(t) - rho_bar(t-1)| < delta_rho (quality plateau)

        Quality alone is insufficient — we require stability over time.
        When PID is disabled, always returns False.
        """
        if not (self._pid_controller and self._pid_events):
            return False

        last = self._pid_events[-1]
        js = last.metrics.js_divergence
        rho_bar = last.metrics.rho_bar
        quadrant = last.pid_step.get("quadrant", "")

        js_ok = js < self._pid_controller.config.epsilon
        quad_ok = quadrant == "converged"
        rho_stable = (
            self._prev_rho_bar is not None
            and abs(rho_bar - self._prev_rho_bar) < self.config.delta_rho
        )

        if js_ok and quad_ok and rho_stable:
            self._stable_rounds += 1
        else:
            self._stable_rounds = 0

        self._prev_rho_bar = rho_bar

        if self._stable_rounds >= self.config.convergence_window:
            logger.info(
                "Terminating debate: stable convergence for %d consecutive rounds "
                "(JS=%.5f, rho_bar=%.3f, quadrant=%s)",
                self._stable_rounds, js, rho_bar, quadrant,
            )
            return True
        return False

    def run_returning_state(self, observation: Observation) -> dict:
        """Run full pipeline and return raw LangGraph state dict.

        Same three-phase logic as run() but returns the raw state dict
        instead of (Action, AgentTrace).  No disk I/O — does NOT call
        _save_trace.  Used by equivalence tests to compare against the
        old monolithic compile_debate_graph().invoke() path.
        """
        self._reset_per_invocation_state()
        return self._run_pipeline(observation)

    def _parse_action(self, d: dict) -> Action:
        """Convert a raw dict into a validated Action model."""
        orders = []
        for o in d.get("orders", []):
            orders.append(
                Order(
                    ticker=o.get("ticker", ""),
                    side=o.get("side", "buy"),
                    size=o.get("size", 0),
                    type=o.get("type", "market"),
                )
            )

        claims = []
        for c in d.get("claims", []):
            claims.append(
                Claim(
                    claim_text=c.get("claim_text", ""),
                    pearl_level=c.get("pearl_level", "L1"),
                    variables=c.get("variables", []),
                    assumptions=c.get("assumptions"),
                    confidence=c.get("confidence", 0.5),
                )
            )

        return Action(
            orders=orders,
            allocation=d.get("allocation"),
            justification=d.get("justification", ""),
            confidence=d.get("confidence", 0.5),
            claims=claims,
        )

    def _parse_trace(
        self,
        trace_dict: dict,
        observation: Observation,
        action: Action,
    ) -> AgentTrace:
        """Convert a raw trace dict into a validated AgentTrace model."""
        return AgentTrace(
            observation_timestamp=trace_dict.get(
                "observation_timestamp", observation.timestamp
            ),
            architecture="debate",
            what_i_saw=trace_dict.get("what_i_saw", ""),
            hypothesis=trace_dict.get("hypothesis", ""),
            decision=trace_dict.get("decision", "Hold"),
            risks_or_falsifiers=trace_dict.get("risks_or_falsifiers"),
            strongest_objection=trace_dict.get("strongest_objection"),
            debate_turns=None,  # Stored in full_state for auditability
            action=action,
            logged_at=datetime.now(timezone.utc).isoformat(),
            initial_market_state=observation.market_state,
            initial_portfolio_state=observation.portfolio_state,
            pid_events=self._pid_events if self._pid_events else None,
        )

    def _save_trace(self, trace: AgentTrace, full_state: dict) -> None:
        """Save the trace + full debate history to disk."""
        trace_dir = Path(self.config.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        now = datetime.now()
        safe_ts = now.strftime("%Y-%m-%d_%I-%M-%S%p").lower()
        filename = f"debate_langgraph_{safe_ts}.json"
        filepath = trace_dir / filename

        output = {
            "trace": trace.model_dump(),
            "debate_turns": full_state.get("debate_turns", []),
            "config": full_state.get("config", {}),
        }

        filepath.write_text(json.dumps(output, indent=2, default=str))
        print(f"[MultiAgent] Trace written to {filepath}")
