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
from eval.financial import compute_financial_metrics
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
        # All filtering is controlled via YAML config fields:
        #   tickers  — which tickers to include
        #   quarters — which quarters to include (optional)
        #   top_n_news — cap news items per case (optional)
        templates = load_case_templates(
            self._config.dataset_path,
            top_n_news=self._config.top_n_news,
            ticker_filter=self._config.tickers,
            quarters=self._config.quarters,
            merge_tickers=self._config.merge_tickers,
        )
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

        # Post-hoc evaluation on all completed episodes.
        self._run_posthoc_eval()

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

        from datetime import datetime, timezone
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

            # If the agent returned orders but did not call the tool
            # (e.g. the multi-agent debate adapter), the runner executes
            # the decision through the broker directly.
            if execution_result is None and decision.orders:
                execution_result = broker.execute_decision(decision, case, agent_id)
                logger.info(
                    "Runner executed decision for case %s directly (agent did not call tool).",
                    case_id,
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
                timestamp=datetime.now(timezone.utc).isoformat(),
                case_prices={
                    t: sd.current_price for t, sd in case.stock_data.items()
                },
            )
            decision_point_logs.append(dp_log)

        # Build the flattened episode log.
        final_portfolio = broker.get_portfolio()
        final_prices = broker.get_last_prices()
        from datetime import datetime, timezone
        episode_log = EpisodeLog(
            episode_id=episode_id,
            agent_id=agent_id,
            decision_point_logs=decision_point_logs,
            trades=broker.get_trade_history(),
            final_portfolio=final_portfolio,
            final_prices=final_prices,
            timestamp=datetime.now(timezone.utc).isoformat(),
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
    # Post-hoc evaluation
    # ------------------------------------------------------------------

    def _run_posthoc_eval(self) -> None:
        """Run the post-hoc eval pipeline on all completed episodes."""
        import json

        from eval.pipeline import EvalPipeline
        from multi_agent.graph import _call_llm

        episodes = self._sim_logger.simulation_log.episode_logs
        if not episodes:
            return

        llm_config = {
            "model_name": self._config.agent.llm_model,
            "temperature": self._config.agent.temperature,
        }
        llm_fn = lambda sys_prompt, usr_prompt: _call_llm(llm_config, sys_prompt, usr_prompt)

        pipeline = EvalPipeline(
            llm_fn=llm_fn,
            run_crit=True,
            run_consistency=True,
            run_divergence=True,
            run_financials=True,
        )
        initial_cash = self._config.broker.initial_cash

        for episode_log in episodes:
            eval_artifacts = []
            for dp_log in episode_log.decision_point_logs:
                trace_data = dp_log.agent_output
                if not isinstance(trace_data, dict):
                    continue

                artifact = pipeline.evaluate_trace(
                    trace_data=trace_data,
                    episode_log=episode_log,
                    initial_cash=initial_cash,
                    debate_id=dp_log.case_id,
                    run_id=self._run_name,
                )
                eval_artifacts.append(artifact.model_dump())

            if eval_artifacts:
                ep_dir = self._sim_logger.run_dir / "episodes" / episode_log.episode_id
                ep_dir.mkdir(parents=True, exist_ok=True)
                eval_path = ep_dir / "eval.json"
                eval_path.write_text(
                    json.dumps(eval_artifacts, indent=2, default=str),
                    encoding="utf-8",
                )
                logger.info("Post-hoc eval written to %s", eval_path)

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

            fin = compute_financial_metrics(ep, initial_cash)

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
                    "sharpe_ratio": fin.sharpe_ratio,
                    "sortino_ratio": fin.sortino_ratio,
                    "max_drawdown": fin.max_drawdown,
                    "max_drawdown_pct": fin.max_drawdown_pct,
                    "calmar_ratio": fin.calmar_ratio,
                }
            )
        return {
            "run_name": self._run_name,
            "num_episodes": len(episodes),
            "episode_summaries": summaries,
        }
