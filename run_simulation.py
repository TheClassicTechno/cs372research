#!/usr/bin/env python3
"""CLI entrypoint for the market simulation harness.

Usage::

    python run_simulation.py --config config/example.yaml
    python run_simulation.py --config config/example.yaml --output-dir results/

The simulation loads a YAML configuration file, instantiates the agent system
and broker, then runs the async simulation loop.  The run name is derived
automatically from the config file name (e.g. ``example.yaml`` -> ``example``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from models.config import SimulationConfig
from simulation.runner import AsyncSimulationRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent trading simulation.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help="Directory where simulation results will be written (default: results/).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


async def _main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Loading config from '%s'...", args.config)

    config = SimulationConfig.from_yaml(args.config)
    logger.info("Config loaded: agent='%s'", config.agent.agent_system)

    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=args.config,
        output_dir=args.output_dir,
    )
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_main())
