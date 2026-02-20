"""
High-level runner for the multi-agent debate system.

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
from .graph import compile_debate_graph
from .models import Action, AgentTrace, Claim, DebateTurn, Observation, Order


class MultiAgentRunner:
    """
    Orchestrates multi-agent debate for trading decisions.

    Wraps the LangGraph debate graph with a clean interface matching
    Deveen's SingleAgentRunner / MajorityVoteRunner / DebateRunner pattern.
    """

    def __init__(self, config: DebateConfig | None = None):
        self.config = config or DebateConfig()

        # Auto-inject devil's advocate if adversarial mode is enabled
        if self.config.enable_adversarial:
            if AgentRole.DEVILS_ADVOCATE not in self.config.roles:
                self.config.roles.append(AgentRole.DEVILS_ADVOCATE)

        self.graph = compile_debate_graph(self.config)

    def run(self, observation: Observation) -> tuple[Action, AgentTrace]:
        """
        Run the full debate pipeline on an observation.

        Returns:
            (action, trace) -- the final Action and the full AgentTrace
        """
        initial_state = {
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

        # Run the LangGraph
        result = self.graph.invoke(initial_state)

        # Parse final action
        final_dict = result.get("final_action", {})
        action = self._parse_action(final_dict)

        # Parse trace
        trace_dict = result.get("trace", {})
        trace = self._parse_trace(trace_dict, observation, action)

        # Save to disk
        self._save_trace(trace, result)

        return action, trace

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
