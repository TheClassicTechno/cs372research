"""Async simulation runner: the main orchestration loop.

Lifecycle:
    1. Load config and case templates.
    2. For each episode:
        a. Initialise broker and agent system.
        b. For each decision point (case template):
            - Build Case with current portfolio.
            - Create fresh submit_decision tool (bound to broker + case).
            - Invoke the agent asynchronously.
            - Log the decision point.
        c. Record episode result.
    3. Finalise and write summary.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.registry import create_agent_system
from agents.tools import make_submit_decision_tool
from models.agents import AgentInvocation
from models.case import Case
from models.config import SimulationConfig
from models.decision import Decision
from models.log import DecisionPointLog, EpisodeLog
from simulation.broker import Broker
from simulation.case_loader import build_case, load_case_templates
from simulation.sim_logging import SimulationLogger, run_name_from_config_path

logger = logging.getLogger(__name__)


class AsyncSimulationRunner:
    """Drives the simulation loop across episodes and decision points."""

    def __init__(
        self,
        config: SimulationConfig,
        config_yaml_path: str,
        output_dir: str = "results",
    ) -> None:
        self._config = config
        self._config_yaml_path = config_yaml_path
        self._run_name = run_name_from_config_path(config_yaml_path)
        self._sim_logger = SimulationLogger(output_dir, config, self._run_name)

    async def run(self) -> None:
        """Execute the full simulation."""
        self._sim_logger.init_run(self._config_yaml_path)

        # Load case templates once (shared across episodes).
        templates = load_case_templates(self._config.dataset_path)
        num_cases = len(templates)
        logger.info(
            "Starting simulation '%s': %d episode(s), %d case(s) each.",
            self._run_name,
            self._config.num_episodes,
            num_cases,
        )

        for ep_idx in range(self._config.num_episodes):
            episode_id = f"ep_{ep_idx:03d}"
            agent_id = f"{self._config.agent.agent_system}_{ep_idx:03d}"

            try:
                episode_log = await self._run_episode(
                    episode_id=episode_id,
                    agent_id=agent_id,
                    templates=templates,
                )
                self._sim_logger.write_episode(episode_log)
            except Exception as exc:
                msg = f"Episode '{episode_id}' failed: {exc}"
                logger.exception(msg)
                self._sim_logger.record_error(msg)

        # Finalize with a lightweight summary.
        summary = self._build_summary()
        self._sim_logger.finalize(summary)
        logger.info("Simulation '%s' complete. Output: %s", self._run_name, self._sim_logger.run_dir)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    async def _run_episode(
        self,
        episode_id: str,
        agent_id: str,
        templates: list[Case],
    ) -> EpisodeLog:
        """Run a single episode: iterate decision points, return the log."""
        broker = Broker(self._config.broker, self._config.tickers)
        agent = create_agent_system(self._config.agent)

        decision_point_logs: list[DecisionPointLog] = []
        num_cases = len(templates)

        logger.info("Episode '%s' starting with %d decision points.", episode_id, num_cases)

        for dp_idx, template in enumerate(templates):
            case_id = f"{episode_id}:{dp_idx}"
            steps_remaining = num_cases - dp_idx - 1

            # Snapshot portfolio *before* the agent acts.
            portfolio_before = broker.get_portfolio()

            # Build case with live portfolio.
            case = build_case(
                template,
                portfolio_before,
                case_id=case_id,
                decision_point_idx=dp_idx,
            )

            # Create a fresh tool bound to the current broker state and case.
            tool = make_submit_decision_tool(broker, case, agent_id)
            agent.bind_tools(tool)

            invocation = AgentInvocation(
                case=case,
                episode_id=episode_id,
                agent_id=agent_id,
                steps_remaining=steps_remaining,
            )

            # Invoke the agent.
            t0 = time.monotonic()
            try:
                result = await agent.invoke(invocation)
                decision = result.decision
                agent_output: dict | str | None = result.raw_output
            except Exception as exc:
                logger.warning(
                    "Agent error on case %s: %s — defaulting to hold.",
                    case_id,
                    exc,
                )
                decision = Decision(orders=[])
                agent_output = f"ERROR: {exc}"

            elapsed = time.monotonic() - t0
            logger.info(
                "Case %s: %d order(s), %.1fs elapsed.",
                case_id,
                len(decision.orders),
                elapsed,
            )

            # Retrieve the execution result from the tool (may be None if
            # the agent never called submit_decision).
            execution_result = getattr(
                getattr(tool, "func", None), "_last_result", None
            )

            # Snapshot portfolio *after* the decision has settled.
            portfolio_after = broker.get_portfolio()

            dp_log = DecisionPointLog(
                case_id=case_id,
                decision_point_idx=dp_idx,
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                extracted_decision=decision,
                execution_result=execution_result,
                agent_output=agent_output,
                elapsed_seconds=elapsed,
            )
            decision_point_logs.append(dp_log)

        # Build the flattened episode log.
        final_portfolio = broker.get_portfolio()
        final_prices = broker.get_last_prices()
        episode_log = EpisodeLog(
            episode_id=episode_id,
            agent_id=agent_id,
            decision_point_logs=decision_point_logs,
            trades=broker.get_trade_history(),
            final_portfolio=final_portfolio,
            final_prices=final_prices,
        )

        logger.info(
            "Episode '%s' complete. Final cash: $%.2f, positions: %s, book value: $%.2f",
            episode_id,
            final_portfolio.cash,
            final_portfolio.positions,
            episode_log.book_value or 0.0,
        )
        return episode_log

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict[str, Any]:
        """Build a lightweight summary dict for the run."""
        episodes = self._sim_logger.simulation_log.episode_logs
        initial_cash = self._config.broker.initial_cash
        summaries = []
        for ep in episodes:
            if ep.final_portfolio is None:
                continue

            # Per-position market values.
            position_values = {
                ticker: qty * ep.final_prices.get(ticker, 0.0)
                for ticker, qty in ep.final_portfolio.positions.items()
            }
            book_value = ep.book_value or 0.0

            summaries.append(
                {
                    "episode_id": ep.episode_id,
                    "initial_cash": initial_cash,
                    "final_cash": ep.final_portfolio.cash,
                    "final_positions": ep.final_portfolio.positions,
                    "final_prices": ep.final_prices,
                    "position_values": position_values,
                    "book_value": book_value,
                    "return_pct": ((book_value - initial_cash) / initial_cash) * 100,
                    "total_trades": len(ep.trades),
                }
            )
        return {
            "run_name": self._run_name,
            "num_episodes": len(episodes),
            "episode_summaries": summaries,
        }
