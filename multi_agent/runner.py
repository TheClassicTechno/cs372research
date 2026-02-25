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
state and decide whether to continue, adjust agreeableness, etc:

    for t in range(max_rounds):
        state = self.run_single_round(state)
        # <-- FUTURE: controller_step(state) goes here
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
from datetime import datetime, timezone
from pathlib import Path

from .config import AgentRole, DebateConfig
from .graph import (
    compile_finalize_graph,
    compile_pipeline_graph,
    compile_single_round_graph,
)
from .models import Action, AgentTrace, Claim, DebateTurn, Observation, Order


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
        self.single_round_graph = compile_single_round_graph(self.config)
        self.finalize_graph = compile_finalize_graph(self.config)

    def run(self, observation: Observation) -> tuple[Action, AgentTrace]:
        """
        Run the full debate pipeline on an observation.

        Returns:
            (action, trace) -- the final Action and the full AgentTrace
        """
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
            state = self.run_single_round(state)
            if self.should_terminate(state):
                break

        # Phase 3: Judge + trace
        state = self.finalize_graph.invoke(state)

        return state

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

        Currently always returns False (debates run for max_rounds).
        Future PRs will implement dynamic termination based on PID
        controller output, convergence detection, etc.
        """
        return False

    def run_returning_state(self, observation: Observation) -> dict:
        """Run full pipeline and return raw LangGraph state dict.

        Same three-phase logic as run() but returns the raw state dict
        instead of (Action, AgentTrace).  No disk I/O — does NOT call
        _save_trace.  Used by equivalence tests to compare against the
        old monolithic compile_debate_graph().invoke() path.
        """
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
