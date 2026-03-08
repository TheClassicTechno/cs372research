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
        default=None,
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
        help="Print system + user prompts for all roles × 3 phases "
        "(propose, critique, revise) and exit. No LLM calls are made. "
        "Memo and inter-phase data use {{placeholders}}.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print full system + user prompts for each agent profile, then exit. "
        "Uses the new agent profile system (config.debate_setup.agents).",
    )
    parser.add_argument(
        "--print-prompt-manifest",
        action="store_true",
        help="Print prompt manifest JSON showing block composition for "
        "each agent profile, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: DEBUG).",
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
        help="Override LLM model used for CRIT scoring (default: gpt-5-mini).",
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
    parser.add_argument(
        "--log-tokens",
        action="store_true",
        help="Print per-request token counts (prompt, completion, total) to console.",
    )
    parser.add_argument(
        "--custom-memo",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a user-provided memo file. Skips memo generation entirely.",
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


def _parse_invest_quarter(invest_quarter: str) -> tuple[int, str]:
    """Parse '2025Q1' -> (2025, 'Q1')."""
    year = int(invest_quarter[:4])
    q = invest_quarter[4:]
    return year, q


def _prior_quarter(year: int, q: str) -> tuple[int, str]:
    """Return the prior calendar quarter: (year, 'Qn')."""
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(q)
    if idx == 0:
        return year - 1, "Q4"
    return year, labels[idx - 1]


def _derive_scenario_memo_path(scenario_path: str, invest_quarter: str) -> Path:
    """Derive a scenario-specific memo cache path.

    Example: scenario '2021Q3_inflation_emergence.yaml' with invest_quarter '2021Q3'
    -> scenario_memos/2021Q3_inflation_emergence_memo_2021_Q2.txt
    """
    scenario_name = Path(scenario_path).stem  # strip .yaml
    year, q = _parse_invest_quarter(invest_quarter)
    prior_year, prior_q = _prior_quarter(year, q)
    return (
        Path("data-pipeline/final_snapshots/scenario_memos")
        / f"{scenario_name}_memo_{prior_year}_{prior_q}.txt"
    )


def _dump_prompts(config: SimulationConfig) -> None:
    """Print system + user prompts for the configured agents × 3 phases and exit.

    Uses the real prompt assembly pipeline (profiles or legacy registry)
    but substitutes {{placeholders}} for the memo and inter-phase data
    so the output shows exact prompt structure without needing real data.
    """
    from multi_agent.prompts import (
        build_proposal_user_prompt,
        build_critique_prompt,
        build_revision_prompt,
        resolve_prompt_profile,
    )
    from multi_agent.prompts.registry import (
        beta_to_bucket,
        get_registry,
        resolve_beta,
    )
    from multi_agent.graph.sector_constraints import build_sector_constraint_text

    agent = config.debate_setup
    beta = agent.pid_initial_beta
    tone = beta_to_bucket(beta)

    # --- Build the same config dict that the real debate pipeline uses ---
    use_profiles = bool(agent.agents)

    if use_profiles:
        from multi_agent.prompts.profile_loader import get_agent_profiles
        all_profiles = get_agent_profiles(
            agent.agents,
            judge_profile_name=agent.judge_profile,
        )
        judge_profile_data = all_profiles.pop("judge", {})
        role_strs = list(agent.agents.keys())

        cfg = {
            "agent_profiles": all_profiles,
            "judge_profile": judge_profile_data,
            "sector_config": agent.sector_config,
            "prompt_file_overrides": agent.prompt_file_overrides or {},
            "_current_beta": beta,
        }
    else:
        role_strs = agent.debate_roles or ["macro", "value", "risk", "technical"]
        role_strs = [r.lower() for r in role_strs]
        cfg = {
            "prompt_file_overrides": agent.prompt_file_overrides or {},
            "prompt_profile": agent.prompt_profile,
            "role_overrides": agent.role_overrides or {},
            "sector_config": agent.sector_config,
        }

    # --- Placeholder context ---
    context = (
        "## Portfolio Allocation Task\n"
        f"- Cash to allocate: ${config.broker.initial_cash:,.2f}\n"
        f"- Allocation universe: {', '.join(config.tickers)}\n"
        f"- As-of: {config.invest_quarter}\n"
        "\n{{memo}}\n"
    )

    sep = "=" * 80
    phase_sep = "─" * 80
    profile_label = ", ".join(f"{r}={agent.agents[r]}" for r in role_strs) if use_profiles else (agent.prompt_profile or "default")

    print(f"\n{sep}")
    print(f"  PROMPT DUMP — {len(role_strs)} agents × 3 phases")
    print(f"  {profile_label}")
    print(f"  beta={beta:.3f}  tone={tone}")
    print(f"{sep}")

    registry = get_registry(cfg)

    def _build_system(role, phase, user_prompt):
        if use_profiles:
            profile_data = cfg["agent_profiles"].get(role, {})
            return registry.build_from_profile(
                role=role, phase=phase, profile=profile_data,
                beta=resolve_beta(cfg, phase),
                user_prompt=user_prompt,
            ).system_prompt
        else:
            prof = resolve_prompt_profile(cfg, role, phase)
            return registry.build(
                role=role, phase=phase,
                beta=resolve_beta(cfg, phase),
                user_prompt=user_prompt,
                block_order=prof.get("system_blocks"),
                prompt_file_overrides=cfg.get("prompt_file_overrides"),
            ).system_prompt

    def _get_user_sections(role, phase):
        if use_profiles:
            profile_data = cfg["agent_profiles"].get(role, {})
            return profile_data.get("user_prompts", {}).get(phase, {}).get("sections")
        else:
            return resolve_prompt_profile(cfg, role, phase).get("user_sections")

    for role in role_strs:
        overrides = cfg.get("prompt_file_overrides")
        sector_text = build_sector_constraint_text(cfg.get("sector_config"), role)

        # ── Propose ──
        propose_user = build_proposal_user_prompt(
            context,
            prompt_file_overrides=overrides,
            user_sections=_get_user_sections(role, "propose"),
            sector_constraints=sector_text,
        )
        propose_sys = _build_system(role, "propose", propose_user)

        # ── Critique ──
        all_proposals = [
            {"role": r, "proposal": "{{" + r + "_proposal}}"}
            for r in role_strs
        ]
        my_proposal = "{{" + role + "_proposal}}"
        cfg["_current_beta"] = beta

        critique_user = build_critique_prompt(
            role, context, all_proposals, my_proposal,
            prompt_file_overrides=overrides,
            user_sections=_get_user_sections(role, "critique"),
            sector_constraints=sector_text,
        )
        critique_sys = _build_system(role, "critique", critique_user)

        # ── Revise ──
        critiques_received = [
            {"from_role": r, "objection": "{{" + r + "_critique_of_" + role + "}}"}
            for r in role_strs if r != role
        ]
        revise_user = build_revision_prompt(
            role, context, my_proposal, critiques_received,
            prompt_file_overrides=overrides,
            user_sections=_get_user_sections(role, "revise"),
            sector_constraints=sector_text,
        )
        revise_sys = _build_system(role, "revise", revise_user)

        # ── Print all 3 phases for this role ──
        for phase, sys_prompt, usr_prompt in [
            ("Propose", propose_sys, propose_user),
            ("Critique", critique_sys, critique_user),
            ("Revise", revise_sys, revise_user),
        ]:
            print(f"\n{phase_sep}")
            print(f"  {role.upper()} — {phase}")
            print(f"{phase_sep}")
            print(f"\n┌── SYSTEM PROMPT ({len(sys_prompt):,} chars) ──┐\n")
            print(sys_prompt)
            print(f"\n┌── USER PROMPT ({len(usr_prompt):,} chars) ──┐\n")
            print(usr_prompt)

    print(f"\n{sep}")
    print(f"  END PROMPT DUMP")
    print(f"{sep}\n")


def _print_profile_prompts(config: SimulationConfig) -> None:
    """Print prompts for all agents using the new profile system."""
    from multi_agent.prompts.profile_loader import get_agent_profiles
    from multi_agent.prompts import build_proposal_user_prompt
    from multi_agent.prompts.registry import get_registry, resolve_beta, beta_to_bucket

    agent = config.debate_setup
    if not agent.agents:
        print("Error: --print-prompts requires agent.agents to be set.")
        return

    profiles = get_agent_profiles(
        agent.agents,
        judge_profile_name=agent.judge_profile,
    )

    context = (
        "## Portfolio Allocation Task\n"
        f"- Cash to allocate: ${config.broker.initial_cash:,.2f}\n"
        f"- Allocation universe: {', '.join(config.tickers)}\n"
        f"- As-of: {config.invest_quarter}\n"
        "\n{{memo}}\n"
    )

    beta = agent.pid_initial_beta
    cfg = {"_current_beta": beta}
    registry = get_registry(cfg)
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  PROFILE PROMPT DUMP — {len(agent.agents)} agents")
    print(f"  beta={beta:.3f}  tone={beta_to_bucket(beta)}")
    print(f"{sep}")

    for role, profile in profiles.items():
        if role == "judge":
            continue
        for phase in ["propose", "critique", "revise"]:
            sys_blocks = profile.get("system_prompts", {}).get(phase, [])
            usr_config = profile.get("user_prompts", {}).get(phase, {})

            user_prompt = build_proposal_user_prompt(
                context,
                user_sections=usr_config.get("sections"),
            ) if phase == "propose" else f"{{{{user_prompt_{phase}}}}}"

            build_result = registry.build_from_profile(
                role=role, phase=phase, profile=profile,
                beta=resolve_beta(cfg, phase),
                user_prompt=user_prompt,
            )

            print(f"\n{'─' * 80}")
            print(f"  {role.upper()} — {phase} (blocks: {build_result.blocks_used})")
            print(f"{'─' * 80}")
            print(f"\n┌── SYSTEM PROMPT ({len(build_result.system_prompt):,} chars) ──┐\n")
            print(build_result.system_prompt[:2000])
            if len(build_result.system_prompt) > 2000:
                print(f"\n... ({len(build_result.system_prompt) - 2000} more chars)")

    print(f"\n{sep}\n  END PROFILE PROMPT DUMP\n{sep}\n")


def _print_prompt_manifest(config: SimulationConfig) -> None:
    """Print prompt manifest JSON for all agent profiles."""
    import json as _json
    from multi_agent.prompts.profile_loader import get_agent_profiles

    agent = config.debate_setup
    if not agent.agents:
        print("Error: --print-prompt-manifest requires agent.agents to be set.")
        return

    profiles = get_agent_profiles(
        agent.agents,
        judge_profile_name=agent.judge_profile,
    )

    manifest = {}
    for role, profile in profiles.items():
        manifest[role] = {
            "system_prompts": profile.get("system_prompts", {}),
            "user_prompts": {
                phase: {
                    "template": cfg.get("template"),
                    "sections": cfg.get("sections"),
                }
                for phase, cfg in profile.get("user_prompts", {}).items()
            },
        }

    print(_json.dumps(manifest, indent=2))


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
        config.debate_setup.logging_mode = args.logging_mode
        logger.info("Logging mode overridden to '%s'", args.logging_mode)

    # --crit-model: override CRIT scoring model.
    if args.crit_model is not None:
        config.debate_setup.crit_llm_model = args.crit_model
        logger.info("CRIT scoring model overridden to '%s'", args.crit_model)

    # --no-parallel: force sequential agent execution.
    if args.no_parallel:
        config.debate_setup.parallel_agents = False
        logger.info("Parallel agents disabled (sequential mode)")

    # --no-rate-limit: disable LLM call stagger entirely.
    if args.no_rate_limit:
        config.debate_setup.no_rate_limit = True
        logger.info("LLM call stagger disabled (all calls fire at once)")

    # --stagger-ms: override stagger interval.
    if args.stagger_ms is not None:
        config.debate_setup.llm_stagger_ms = args.stagger_ms
        logger.info("LLM stagger interval set to %dms", args.stagger_ms)

    # --no-display: disable Rich console display.
    if args.no_display:
        config.debate_setup.console_display = False
        logger.info("Rich console display disabled (plain text mode)")

    # --log-tokens: print per-request token counts to console.
    if args.log_tokens:
        config.debate_setup.log_tokens = True
        logger.info("Per-request token logging enabled")

    # --- Build effective command string with all resolved args ---
    cmd_parts = ["python run_simulation.py"]
    cmd_parts.append(f"--agents {args.agents}")
    if args.scenario:
        cmd_parts.append(f"--scenario {args.scenario}")
    cmd_parts.append(f"--output-dir {args.output_dir}")
    cmd_parts.append(f"--log-level {args.log_level}")
    cmd_parts.append(f"--logging-mode {config.debate_setup.logging_mode}")
    cmd_parts.append(f"--crit-model {config.debate_setup.crit_llm_model}")
    if not config.debate_setup.parallel_agents:
        cmd_parts.append("--no-parallel")
    if config.debate_setup.no_rate_limit:
        cmd_parts.append("--no-rate-limit")
    if args.stagger_ms is not None:
        cmd_parts.append(f"--stagger-ms {args.stagger_ms}")
    if not config.debate_setup.console_display:
        cmd_parts.append("--no-display")
    if config.debate_setup.log_tokens:
        cmd_parts.append("--log-tokens")
    if args.custom_memo:
        cmd_parts.append(f"--custom-memo {args.custom_memo}")
    config.debate_setup.run_command = " ".join(cmd_parts)

    config_paths = [str(Path(args.agents).resolve())]
    if args.scenario:
        config_paths.append(str(Path(args.scenario).resolve()))
    config.debate_setup.config_paths = config_paths

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

    # --print-prompts: print prompts using new agent profile system and exit.
    if args.print_prompts:
        _print_profile_prompts(config)
        return

    # --print-prompt-manifest: print prompt manifest JSON and exit.
    if args.print_prompt_manifest:
        _print_prompt_manifest(config)
        return

    # Validate that tickers + invest_quarter are present (supplied by scenario).
    config.validate_ready()

    # Validate agent profiles if using the new system
    if config.debate_setup.agents:
        from multi_agent.prompts.profile_loader import validate_all_profiles
        errors = validate_all_profiles(
            config.debate_setup.agents,
            judge_profile_name=config.debate_setup.judge_profile,
        )
        if errors:
            for err in errors:
                logger.error("Profile validation: %s", err)
            sys.exit(1)
        logger.info("Agent profiles validated: %s", list(config.debate_setup.agents.keys()))

    logger.info("Config loaded: agent='%s'", config.debate_setup.agent_system)

    # --- Scenario memo caching ---
    memo_override_path: str | None = None

    if args.custom_memo:
        # User-provided memo — skip all generation.
        custom = Path(args.custom_memo)
        if not custom.exists():
            logger.error("Custom memo file not found: %s", custom)
            sys.exit(1)
        memo_override_path = str(custom)
        logger.info("Using custom memo: %s", memo_override_path)

    # --- Auto-generate snapshot data if using memo pipeline ---
    if "final_snapshots" in config.dataset_path:
        scenario_memo_path: Path | None = None
        if args.scenario and not args.custom_memo:
            scenario_memo_path = _derive_scenario_memo_path(
                args.scenario, config.invest_quarter,
            )

            # Always regenerate snapshot + memo to avoid stale cache issues.
            logger.info(
                "Auto-generating snapshots for %s (%d tickers), "
                "scenario memo -> %s ...",
                config.invest_quarter, len(config.tickers), scenario_memo_path,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path("data-pipeline/final_snapshots/snapshot_builder.py")),
                    "--tickers", ",".join(config.tickers),
                    "--invest-quarter", config.invest_quarter,
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path("data-pipeline/final_snapshots/generate_scenario_memo.py")),
                    "--scenario", args.scenario,
                ],
                check=True,
            )
            memo_override_path = str(scenario_memo_path)
            logger.info("Generated scenario memo: %s", memo_override_path)
        elif not args.custom_memo:
            # No scenario — generate snapshots as before (no scenario memo).
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

    config.memo_override_path = memo_override_path

    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=args.agents,
        output_dir=args.output_dir,
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_main())
