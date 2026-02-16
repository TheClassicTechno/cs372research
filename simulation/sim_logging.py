"""Simulation output logging: persists SimulationLog, EpisodeLogs, and reasoning traces.

The output directory structure is::

    {output_dir}/{run_name}/
    ├── config.yaml
    ├── simulation_log.json
    ├── episodes/
    │   ├── ep_001/
    │   │   ├── episode_log.json
    │   │   ├── trades.json
    │   │   └── reasoning/
    │   │       ├── case_000.txt
    │   │       └── case_001.txt
    │   └── ...
    └── summary.json
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from models.config import SimulationConfig
from models.log import EpisodeLog, SimulationLog

logger = logging.getLogger(__name__)


def run_name_from_config_path(config_path: str | Path) -> str:
    """Derive a run name from the configuration file path (stem without extension)."""
    return Path(config_path).stem


class SimulationLogger:
    """Manages on-disk output for a simulation run.

    Call ``init_run`` once at the start, ``write_episode`` after each episode
    completes, and ``finalize`` at the very end.
    """

    def __init__(
        self,
        output_dir: str,
        config: SimulationConfig,
        run_name: str,
    ) -> None:
        self._run_dir = _unique_run_dir(Path(output_dir), run_name)
        self._episodes_dir = self._run_dir / "episodes"
        self._simulation_log = SimulationLog(
            run_name=self._run_dir.name,
            config=config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_run(self, config_yaml_path: str | None = None) -> None:
        """Create the output directory tree and optionally copy the config."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_dir.mkdir(exist_ok=True)
        if config_yaml_path is not None:
            dest = self._run_dir / "config.yaml"
            shutil.copy2(config_yaml_path, dest)
            logger.info("Copied config to %s", dest)

    def write_episode(self, episode_log: EpisodeLog) -> None:
        """Persist a completed episode's log and trade history to disk."""
        ep_id = episode_log.episode_id
        ep_dir = self._episodes_dir / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Episode log (full audit trail).
        _write_json(ep_dir / "episode_log.json", episode_log.model_dump())

        # Trades (separate convenience file).
        if episode_log.trades:
            _write_json(
                ep_dir / "trades.json",
                [t.model_dump() for t in episode_log.trades],
            )

        # Reasoning traces — one file per decision point.
        reasoning_dir = ep_dir / "reasoning"
        reasoning_dir.mkdir(exist_ok=True)
        for dp_log in episode_log.decision_point_logs:
            if dp_log.agent_output is not None:
                trace_path = reasoning_dir / f"case_{dp_log.decision_point_idx:03d}.txt"
                content = (
                    dp_log.agent_output
                    if isinstance(dp_log.agent_output, str)
                    else json.dumps(dp_log.agent_output, indent=2)
                )
                trace_path.write_text(content, encoding="utf-8")

        # Accumulate in the run-level log.
        self._simulation_log.episode_logs.append(episode_log)
        logger.info("Wrote episode log for '%s' to %s", ep_id, ep_dir)

    def record_error(self, message: str) -> None:
        """Append an error message to the run-level log."""
        self._simulation_log.errors.append(message)
        logger.error("Simulation error: %s", message)

    def finalize(self, summary: dict[str, Any] | None = None) -> None:
        """Write the run-level simulation log and optional summary."""
        _write_json(
            self._run_dir / "simulation_log.json",
            self._simulation_log.model_dump(),
        )
        if summary is not None:
            _write_json(self._run_dir / "summary.json", summary)
        logger.info("Simulation log finalized at %s", self._run_dir)

    @property
    def simulation_log(self) -> SimulationLog:
        """Expose the in-memory simulation log (used by the runner for summaries)."""
        return self._simulation_log

    @property
    def run_dir(self) -> Path:
        return self._run_dir


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _unique_run_dir(output_dir: Path, run_name: str) -> Path:
    """Return a run directory that does not already exist.

    If ``output_dir/run_name`` is free, use it directly (first run keeps
    a clean name).  Otherwise append an incrementing suffix:
    ``run_name_001``, ``run_name_002``, etc.
    """
    candidate = output_dir / run_name
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = output_dir / f"{run_name}_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_json(path: Path, data: Any) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
