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
sub-graph invocations (propose, critique, revise) so that agreeableness
can be adjusted per-phase based on the PID controller's output:

    for t in range(max_rounds):
        state = self._run_round_with_pid(state, t + 1)
        crit_result = crit_scorer.score(...)
        pid_result = controller.step(crit_result.rho_bar, js, ov)
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
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from .config import AgentRole, DebateConfig
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

        # Auto-inject devil's advocate if adversarial mode is enabled
        if self.config.enable_adversarial:
            if AgentRole.DEVILS_ADVOCATE not in self.config.roles:
                self.config.roles.append(AgentRole.DEVILS_ADVOCATE)

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

        # --- CRIT scorer (uses same LLM config as debate agents) ---
        self._crit_scorer = None
        if self.config.pid_enabled:
            from eval.crit import CritScorer
            self._crit_scorer = CritScorer(
                llm_fn=lambda sys, usr: _call_llm(self.config.to_dict(), sys, usr)
            )

        # --- PID controller ---
        self._pid_controller = None
        self._pid_events: list[PIDEvent] = []
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
                state["revisions"] = list(reversed(deduped))

            # CRIT + PID step (after full round completes)
            if self._pid_controller and self._crit_scorer:
                try:
                    self._pid_step(state, t + 1)
                except Exception as exc:
                    logger.warning(
                        "CRIT/PID step failed in round %d: %s — "
                        "continuing with current beta (%.3f)",
                        t + 1,
                        exc,
                        self._pid_controller.beta if self._pid_controller else 0.0,
                    )

            if self.should_terminate(state):
                break

        # Phase 3: Judge + trace
        state = self.finalize_graph.invoke(state)

        return state

    def _run_round_with_pid(self, state: dict, round_num: int) -> dict:
        """Run one debate round with per-phase agreeableness control.

        When PID is active, each phase (propose, critique, revise) is
        invoked as a separate sub-graph so agreeableness can be adjusted
        between them based on per-phase toggle configuration.
        """
        beta = self._pid_controller.beta if self._pid_controller else self._original_agreeableness

        # Propose phase
        if self.config.pid_propose:
            state["config"]["agreeableness"] = beta
        else:
            state["config"]["agreeableness"] = self._original_agreeableness
        state = self._propose_graph.invoke(state)

        # Critique phase
        if self.config.pid_critique:
            state["config"]["agreeableness"] = beta
        else:
            state["config"]["agreeableness"] = self._original_agreeableness
        state = self._critique_graph.invoke(state)

        # Revise phase
        if self.config.pid_revise:
            state["config"]["agreeableness"] = beta
        else:
            state["config"]["agreeableness"] = self._original_agreeableness
        state = self._revise_graph.invoke(state)

        return state

    def _pid_step(self, state: dict, round_num: int) -> None:
        """Run CRIT scorer + PID controller step after a debate round.

        Computes rho_bar from CRIT, feeds it into the PID controller,
        and records a PIDEvent for auditing.
        """
        from eval.PID.sycophancy import jensen_shannon_divergence

        # Run CRIT audit
        crit_result = self._crit_scorer.score(
            case_data=state.get("enriched_context", ""),
            agent_traces=state.get("debate_turns", []),
            decisions=state.get("revisions", state.get("proposals", [])),
        )

        # Compute JS divergence from agent confidences
        decisions = state.get("revisions", state.get("proposals", []))
        confidences = [
            d.get("action_dict", {}).get("confidence", 0.5)
            for d in decisions
        ]
        js = jensen_shannon_divergence(confidences) if len(confidences) >= 2 else 0.0

        # Evidence overlap: average Jaccard across agent pairs
        # For now, use 0.0 as default (evidence extraction is not yet implemented)
        ov = 0.0

        # PID step
        pid_result = self._pid_controller.step(crit_result.rho_bar, js, ov)

        # Record event
        ctrl_output = ControllerOutput(new_agreeableness=pid_result.beta_new)
        event = PIDEvent(
            round_index=round_num,
            metrics=RoundMetrics(
                round_index=round_num,
                rho_bar=crit_result.rho_bar,
                js_divergence=js,
                ov_overlap=ov,
            ),
            crit_result=crit_result.model_dump(),
            pid_step={
                "e_t": pid_result.e_t,
                "u_t": pid_result.u_t,
                "beta_new": pid_result.beta_new,
                "p_term": pid_result.p_term,
                "i_term": pid_result.i_term,
                "d_term": pid_result.d_term,
                "s_t": pid_result.s_t,
            },
            controller_output=ctrl_output,
        )
        self._pid_events.append(event)

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
        """Check if debate should end early.

        When PID is enabled, checks convergence via JS divergence.
        When PID is disabled, always returns False (debates run for max_rounds).
        """
        if self._pid_controller and self._pid_events:
            from eval.PID.termination import check_convergence
            last_event = self._pid_events[-1]
            js = last_event.metrics.js_divergence
            return check_convergence(js, self._pid_controller.config.epsilon)
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
