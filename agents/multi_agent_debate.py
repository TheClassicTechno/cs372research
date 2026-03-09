"""Adapter: plugs the multi-agent debate system into the simulation runner.

The simulation runner speaks ``AgentSystem`` (bind_tools / invoke) with
``Case`` / ``Decision`` models.  The debate system speaks
``MultiAgentRunner.run(Observation) → (Action, AgentTrace)``.

This module bridges the two by:
    1. Converting ``Case`` → ``Observation``  (via feature_engineering)
    2. Invoking ``MultiAgentRunner.run`` (sync, so we wrap in a thread)
    3. Converting ``Action`` → ``Decision``
    4. Returning the Decision — the **runner** calls the broker, not us.

Layer boundary: this adapter must NOT call the broker, compute eval
metrics, or write eval artifacts.  See documentation/integration_plan.md §2.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from agents.base import AgentSystem
from agents.registry import register
from models.agents import AgentInvocation, AgentInvocationResult
from models.config import AgentConfig
from models.decision import Decision
from models.decision import Order as SimOrder
from multi_agent.config import DebateConfig
from multi_agent.models import Action
from multi_agent.runner import MultiAgentRunner
from simulation.feature_engineering import build_observation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Action → Decision conversion
# ------------------------------------------------------------------


def allocation_to_decision(
    allocation: dict[str, float],
    prices: dict[str, float],
    cash: float,
) -> Decision:
    """Convert allocation weights to buy orders.

    Each weight * cash = dollar amount for that ticker.
    Dollar amount / price = shares (integer, floor).
    Remaining cash (from rounding) is unallocated.
    """
    orders: list[SimOrder] = []
    for ticker, weight in allocation.items():
        if ticker == "_CASH_":
            continue
        if weight <= 0 or ticker not in prices:
            continue
        price = prices[ticker]
        if price <= 0:
            continue
        dollar_amount = weight * cash
        shares = int(dollar_amount / price)
        if shares > 0:
            orders.append(SimOrder(ticker=ticker, side="buy", quantity=shares))
    return Decision(orders=orders)


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


@register("multi_agent_debate")
class DebateAgentSystem(AgentSystem):
    """Adapter that wraps ``MultiAgentRunner`` as an ``AgentSystem``.

    Architecture (from integration_plan.md §8 — Debate Flow):
        1. Agent returns ``Decision``
        2. Runner calls broker directly
        3. No tool used

    The adapter does NOT call the broker or the submit_decision tool.
    It converts Case → Observation, runs the debate, converts Action →
    Decision, and returns the result.  The simulation runner is
    responsible for executing the decision through the broker.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

        # Enable mock mode via system_prompt_override containing "mock".
        use_mock = (
            config.system_prompt_override is not None
            and "mock" in config.system_prompt_override.lower()
        )

        # --- New agent profile system ---
        # If config.agents is set (dict of role->profile_name), use new system.
        loaded_agent_profiles: dict = {}
        loaded_judge_profile: dict = {}
        if config.agents:
            from multi_agent.prompts.profile_loader import get_agent_profiles
            profiles = get_agent_profiles(
                config.agents,
                judge_profile_name=config.judge_profile,
            )
            # Separate judge from agent profiles
            loaded_judge_profile = profiles.pop("judge", {})
            loaded_agent_profiles = profiles

        # Build the debate roster from YAML config, or fall back to defaults.
        # Any string is a valid role — no enum validation needed.
        roles = None
        if config.agents:
            roles = [r.lower() for r in config.agents.keys()]
        elif config.debate_roles:
            roles = [r.lower() for r in config.debate_roles]

        # Build DebateConfig with all simulation-layer fields.
        # PID object construction is handled inside DebateConfig.__post_init__
        # so this adapter stays decoupled and has no direct dependency on PID.
        debate_kwargs: dict[str, Any] = dict(
            llm_provider=config.llm_provider,
            model_name=config.llm_model,
            role_llms=config.role_llms or {},
            phase_llms=config.phase_llms or {},
            temperature=config.temperature,
            mock=use_mock,
            max_rounds=config.max_rounds,
            propose_only=config.propose_only,
            round_robin_mode=config.round_robin_mode,
            judge_type=config.judge_type,
            enable_adversarial=config.enable_adversarial,
            _pid_enabled_flag=config.pid_enabled,
            pid_kp=config.pid_kp,
            pid_ki=config.pid_ki,
            pid_kd=config.pid_kd,
            pid_rho_star=config.pid_rho_star,
            pid_epsilon=config.pid_epsilon,
            convergence_window=config.convergence_window,
            delta_rho=config.delta_rho,
            initial_beta=config.pid_initial_beta,
            pid_log_metrics=config.pid_log_metrics,
            pid_log_llm_calls=config.pid_log_llm_calls,
            log_tokens=config.log_tokens,
            log_rendered_prompts=config.log_rendered_prompts,
            log_prompt_manifest=config.log_prompt_manifest,
            prompt_logging=config.prompt_logging,
            logging_mode=config.logging_mode,
            experiment_name=config.experiment_name,
            parallel_agents=config.parallel_agents,
            no_rate_limit=config.no_rate_limit,
            llm_stagger_ms=config.llm_stagger_ms,
            max_concurrent_llm=config.max_concurrent_llm,
            console_display=config.console_display,
            prompt_file_overrides=config.prompt_file_overrides or {},
            prompt_profile=config.prompt_profile,
            role_overrides=config.role_overrides or {},
            crit_model_name=config.crit_llm_model,
            crit_system_template=config.crit_system_template,
            crit_user_template=config.crit_user_template,
            run_command=config.run_command,
            config_paths=config.config_paths,
            sector_config=config.sector_config,
            allocation_constraints=config.allocation_constraints,
            agent_profiles=loaded_agent_profiles,
            agent_profile_names=config.agents or {},
            judge_profile=loaded_judge_profile,
        )
        if roles is not None:
            debate_kwargs["roles"] = roles

        self._debate_cfg = DebateConfig(**debate_kwargs)
        self._debate_runner = MultiAgentRunner(self._debate_cfg)

    def bind_tools(self, submit_decision_fn: Callable[..., Any]) -> None:
        """Accept the tool for interface compliance.

        The debate adapter does not use the submit_decision tool — the
        runner calls the broker directly after receiving our Decision.
        We store it only to satisfy the AgentSystem contract.
        """
        self._submit_decision_tool = submit_decision_fn

    async def invoke(self, invocation: AgentInvocation) -> AgentInvocationResult:
        """Run the debate system for one decision point.

        Steps:
            1. Convert ``Case`` → ``Observation`` (via feature_engineering)
            2. Run ``MultiAgentRunner.run(observation)`` in a thread
            3. Convert ``Action`` → ``Decision``
            4. Return ``AgentInvocationResult`` — runner handles broker execution
        """
        case = invocation.case

        # 1. Convert Case → Observation (feature engineering layer)
        observation = build_observation(case)
        logger.info(
            "Debate agent: converted case %s → observation (universe=%s, cash=%.2f)",
            case.case_id,
            observation.universe,
            observation.portfolio_state.cash,
        )

        # 2. Run debate (sync call → offload to thread pool)
        action, trace = await asyncio.to_thread(self._debate_runner.run, observation)
        logger.info(
            "Debate agent: received action with %d order(s), confidence=%.2f",
            len(action.orders),
            action.confidence,
        )

        # 3. Convert allocation → Decision (buy orders from weights)
        prices = {t: sd.current_price for t, sd in case.stock_data.items()}
        cash = case.portfolio.cash
        allocation = action.allocation or {}
        decision = allocation_to_decision(allocation, prices, cash)
        logger.info(
            "Debate agent: %d ticker(s) allocated, %d buy order(s) for case %s",
            sum(1 for w in allocation.values() if w > 0),
            len(decision.orders),
            case.case_id,
        )

        # 4. Build raw output for logging (includes full debate trace)
        raw_output = {
            "debate_action": action.model_dump(),
            "debate_trace": trace.model_dump(),
            "debate_justification": action.justification,
            "debate_confidence": action.confidence,
        }

        return AgentInvocationResult(decision=decision, raw_output=raw_output)
