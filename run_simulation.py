#!/usr/bin/env python3
"""CLI entrypoint for the market simulation harness.

Overview
--------
This script loads a YAML config, reads case data from disk, and runs an
agent-based trading simulation.  Each "case" is a decision point: the agent
sees market data (prices, news, earnings) and its current portfolio, then
decides whether to buy, sell, or hold.

The pipeline is:

    YAML config
        -> load case JSON files from data/cases/
        -> filter by tickers (from config)
        -> filter by quarters (from config, optional)
        -> filter news items by top_n_news (from config, optional)
        -> for each episode:
            for each case (decision point):
                send case data to LLM via agent system
                agent returns trading decision
                broker validates and executes trades
                portfolio carries forward to next case

What gets sent to the LLM
--------------------------
Each case's full JSON payload (stock prices, daily bars, news items, portfolio
state) is sent to the LLM at every decision point.  Unfiltered cases can be
100-300 KB each (~25K-75K tokens).  Use top_n_news in the YAML config to
reduce this dramatically (e.g. top_n_news: 5 reduces cases to ~5-6 KB each).

Case selection (config-driven)
------------------------------
All case selection is controlled via the YAML config file:

- ``tickers``: Universe of ticker symbols. Only cases containing these
  tickers are loaded, and these are the only tickers the broker can trade.
- ``quarters`` (optional): If set, only cases whose filename matches one of
  the specified quarters are loaded (e.g. ``['Q1', 'Q3']``).
- ``top_n_news`` (optional): Cap news items per case to the top N by
  abs(impact_score).

Example configs::

    # NVDA only, Q1 and Q2, top 5 news:
    tickers: [NVDA]
    quarters: [Q1, Q2]
    top_n_news: 5

    # All tickers, all quarters:
    tickers: [NVDA, AAPL, MSFT]
    # quarters omitted = all quarters loaded

Example commands
----------------
# Run with default config (single LLM agent, NVDA only, 1 episode):
    python run_simulation.py --agents config/example.yaml

# Run the multi-agent debate (4 specialists + judge, real API calls):
    python run_simulation.py --agents config/debate.yaml

# Run the debate in mock mode (no API calls, deterministic):
    python run_simulation.py --agents config/debate_mock.yaml

# List all available tickers in the dataset:
    python run_simulation.py --agents config/example.yaml --list-tickers

# Save results to a custom directory:
    python run_simulation.py --agents config/debate.yaml --output-dir results/experiment_1

# Override tickers/constraints with a scenario file:
    python run_simulation.py --agents config/agents/debate_diverse_agents.yaml \
        --scenario config/scenarios/financials_heavy.yaml

# Quiet mode — only show errors:
    python run_simulation.py --agents config/debate.yaml --log-level ERROR
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from models.config import SimulationConfig
from simulation.runner import AsyncSimulationRunner

# Load .env file (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY) into environment.
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent trading simulation.",
    )
    parser.add_argument(
        "--agents",
        required=True,
        type=str,
        help="Path to the agent YAML configuration file (roles, prompts, PID, logging).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        type=str,
        help="Path to a scenario YAML file (tickers, invest_quarter, "
        "allocation_constraints, etc.). Scenario values override the base config.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help="Directory where simulation results will be written (default: results/).",
    )

    # ------------------------------------------------------------------
    # Utility flags
    # ------------------------------------------------------------------
    parser.add_argument(
        "--list-tickers",
        action="store_true",
        help="List all available tickers in the dataset and exit.",
    )
    parser.add_argument(
        "--dump-prompts",
        action="store_true",
        help="Print the full system + user prompts for one propose round "
        "(all roles) and exit. No LLM calls are made.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--logging-mode",
        default=None,
        choices=["standard", "debug", "off"],
        help="Override debate logging mode from config. "
        "'standard' writes artifacts only, 'debug' adds prompt files, 'off' disables.",
    )
    parser.add_argument(
        "--crit-model",
        default=None,
        metavar="MODEL",
        help="Override LLM model used for CRIT scoring (default: gpt-5).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run debate agents sequentially instead of in parallel. "
        "Easier to debug but slower.",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable LLM call stagger entirely. All parallel calls fire at once.",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=None,
        metavar="MS",
        help="Milliseconds between parallel LLM call starts (default: 200). "
        "Set to 0 to disable stagger.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable Rich-formatted terminal display. "
        "Uses minimal plain-text output instead.",
    )
    return parser.parse_args()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _setup_logging(level: str) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy HTTP-level logs (e.g. "HTTP Request: POST ... 200 OK")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _dump_prompts(config: SimulationConfig) -> None:
    """Print full system + user prompts for one propose round and exit."""
    from multi_agent.config import AgentRole
    from multi_agent.prompts import (
        SYSTEM_CAUSAL_CONTRACT,
        build_proposal_user_prompt,
        get_role_prompts,
    )
    from simulation.feature_engineering import build_observation
    from simulation.memo_loader import load_memo_cases

    # --- Load one case ---
    cases = load_memo_cases(
        config.dataset_path,
        invest_quarter=config.invest_quarter,
        memo_format=config.memo_format,
        tickers=config.tickers,
    )

    if not cases:
        print("ERROR: No cases found for the given config.")
        return

    case = cases[0]
    obs = build_observation(case)

    # --- Build enriched context (allocation mode) ---
    header = (
        f"## Portfolio Allocation Task\n"
        f"- Cash to allocate: ${obs.portfolio_state.cash:,.2f}\n"
        f"- Allocation universe: {', '.join(obs.universe)}\n"
        f"- As-of: {obs.timestamp}\n"
    )
    memo_context = obs.text_context or ""
    context = header + "\n" + memo_context

    # --- Resolve roles ---
    role_strs = config.agent.debate_roles or ["macro", "value", "risk", "technical"]
    roles = []
    for r in role_strs:
        try:
            roles.append(AgentRole(r.lower()))
        except ValueError:
            pass
    if not roles:
        roles = [AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL]

    use_cc = config.agent.use_system_causal_contract

    # --- Build and print prompts ---
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  PROMPT DUMP — {len(roles)} roles, use_system_causal_contract={use_cc}")
    print(f"{sep}\n")

    rp = get_role_prompts(use_cc)
    user_prompt = build_proposal_user_prompt(
        context,
        use_system_causal_contract=use_cc,
    )

    for role in roles:
        role_system = rp.get(role, rp.get(AgentRole.MACRO, ""))
        if use_cc:
            role_system = SYSTEM_CAUSAL_CONTRACT + "\n\n" + role_system

        print(f"\n{'─' * 80}")
        print(f"  ROLE: {role.value.upper()}")
        print(f"{'─' * 80}")

        print(f"\n┌── SYSTEM PROMPT ({len(role_system)} chars) ──┐\n")
        print(role_system)

        print(f"\n┌── USER PROMPT ({len(user_prompt)} chars) ──┐\n")
        print(user_prompt)

    print(f"\n{sep}")
    print(f"  END PROMPT DUMP")
    print(f"{sep}\n")


async def _main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Loading agent config from '%s'...", args.agents)

    # --- Load base config, optionally merge scenario overrides ---
    if args.scenario:
        scenario_path = Path(args.scenario)
        if not scenario_path.exists():
            logger.error("Scenario file not found: %s", scenario_path)
            sys.exit(1)
        with open(args.agents, encoding="utf-8") as fh:
            base_raw = yaml.safe_load(fh) or {}
        with scenario_path.open(encoding="utf-8") as fh:
            scenario_raw = yaml.safe_load(fh) or {}
        merged_raw = _deep_merge(base_raw, scenario_raw)
        logger.info("Merged scenario '%s' into base config", args.scenario)
        config = SimulationConfig(**merged_raw)
    else:
        config = SimulationConfig.from_yaml(args.agents)

    # --logging-mode: override debate logging mode from CLI.
    if args.logging_mode is not None:
        config.agent.logging_mode = args.logging_mode
        logger.info("Logging mode overridden to '%s'", args.logging_mode)

    # --crit-model: override CRIT scoring model.
    if args.crit_model is not None:
        config.agent.crit_llm_model = args.crit_model
        logger.info("CRIT scoring model overridden to '%s'", args.crit_model)

    # --no-parallel: force sequential agent execution.
    if args.no_parallel:
        config.agent.parallel_agents = False
        logger.info("Parallel agents disabled (sequential mode)")

    # --no-rate-limit: disable LLM call stagger entirely.
    if args.no_rate_limit:
        config.agent.no_rate_limit = True
        logger.info("LLM call stagger disabled (all calls fire at once)")

    # --stagger-ms: override stagger interval.
    if args.stagger_ms is not None:
        config.agent.llm_stagger_ms = args.stagger_ms
        logger.info("LLM stagger interval set to %dms", args.stagger_ms)

    # --no-display: disable Rich console display.
    if args.no_display:
        config.agent.console_display = False
        logger.info("Rich console display disabled (plain text mode)")

    # --- Build effective command string with all resolved args ---
    cmd_parts = ["python run_simulation.py"]
    cmd_parts.append(f"--agents {args.agents}")
    if args.scenario:
        cmd_parts.append(f"--scenario {args.scenario}")
    cmd_parts.append(f"--output-dir {args.output_dir}")
    cmd_parts.append(f"--log-level {args.log_level}")
    cmd_parts.append(f"--logging-mode {config.agent.logging_mode}")
    cmd_parts.append(f"--crit-model {config.agent.crit_llm_model}")
    if not config.agent.parallel_agents:
        cmd_parts.append("--no-parallel")
    if config.agent.no_rate_limit:
        cmd_parts.append("--no-rate-limit")
    if args.stagger_ms is not None:
        cmd_parts.append(f"--stagger-ms {args.stagger_ms}")
    if not config.agent.console_display:
        cmd_parts.append("--no-display")
    config.agent.run_command = " ".join(cmd_parts)

    config_paths = [str(Path(args.agents).resolve())]
    if args.scenario:
        config_paths.append(str(Path(args.scenario).resolve()))
    config.agent.config_paths = config_paths

    # --list-tickers: print configured tickers and exit (no simulation).
    if args.list_tickers:
        print("Configured tickers:")
        for t in config.tickers:
            print(f"  {t}")
        return

    # --dump-prompts: print full prompts for one propose round and exit.
    if args.dump_prompts:
        _dump_prompts(config)
        return

    logger.info("Config loaded: agent='%s'", config.agent.agent_system)

    # --- Auto-generate snapshot data if using memo pipeline ---
    if "final_snapshots" in config.dataset_path:
        logger.info("Auto-generating snapshots for %s (%d tickers)...",
                     config.invest_quarter, len(config.tickers))
        subprocess.run(
            [
                sys.executable,
                str(Path("data-pipeline/final_snapshots/snapshot_builder.py")),
                "--tickers", ",".join(config.tickers),
                "--invest-quarter", config.invest_quarter,
            ],
            check=True,
        )

    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=args.agents,
        output_dir=args.output_dir,
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_main())
