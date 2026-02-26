"""Tests for the --num-cases flag.

Verifies that the CLI flag is parsed correctly and that
AsyncSimulationRunner truncates the templates list when num_cases
is provided, while leaving it unchanged when omitted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.config import AgentConfig, BrokerConfig, SimulationConfig
from simulation.runner import AsyncSimulationRunner


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sim_config(tmp_path) -> SimulationConfig:
    """A minimal SimulationConfig pointing at a temp dataset path."""
    return SimulationConfig(
        dataset_path=str(tmp_path),
        agent=AgentConfig(
            agent_system="single_llm",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        ),
        broker=BrokerConfig(initial_cash=100_000.0),
        tickers=["AAPL"],
        num_episodes=1,
    )


def _make_fake_templates(n: int) -> list:
    """Create n distinct sentinel objects to act as case templates."""
    return [MagicMock(name=f"case_{i}") for i in range(n)]


def _run_and_get_templates(runner) -> list:
    """Run the runner with mocked internals and return the templates passed to _run_episode."""
    mock_episode = AsyncMock(return_value=MagicMock())

    with patch.object(
        runner._sim_logger, "init_run"
    ), patch.object(
        runner, "_run_episode", mock_episode
    ), patch.object(
        runner._sim_logger, "write_episode"
    ), patch.object(
        runner, "_build_summary", return_value={}
    ), patch.object(
        runner._sim_logger, "finalize"
    ):
        asyncio.run(runner.run())

    call_kwargs = mock_episode.call_args
    return call_kwargs.kwargs.get(
        "templates", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )


# =============================================================================
# TEST 1: num_cases=None preserves all templates (default behavior)
# =============================================================================


def test_num_cases_none_uses_all_templates(sim_config, tmp_path):
    """When num_cases is not set, all loaded templates are used."""
    templates = _make_fake_templates(10)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
        num_cases=None,
    )

    with patch("simulation.runner.load_case_templates", return_value=templates):
        passed = _run_and_get_templates(runner)

    assert len(passed) == 10


# =============================================================================
# TEST 2: num_cases truncates templates to the specified count
# =============================================================================


def test_num_cases_truncates_templates(sim_config, tmp_path):
    """When num_cases=3, only the first 3 templates are used."""
    templates = _make_fake_templates(10)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
        num_cases=3,
    )

    with patch("simulation.runner.load_case_templates", return_value=templates):
        passed = _run_and_get_templates(runner)

    assert len(passed) == 3
    assert passed == templates[:3]


# =============================================================================
# TEST 3: num_cases larger than dataset uses all templates
# =============================================================================


def test_num_cases_larger_than_dataset(sim_config, tmp_path):
    """When num_cases exceeds the dataset size, all templates are used."""
    templates = _make_fake_templates(5)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
        num_cases=100,
    )

    with patch("simulation.runner.load_case_templates", return_value=templates):
        passed = _run_and_get_templates(runner)

    assert len(passed) == 5


# =============================================================================
# TEST 4: num_cases=1 runs exactly one case
# =============================================================================


def test_num_cases_one(sim_config, tmp_path):
    """num_cases=1 should pass exactly one template to the episode."""
    templates = _make_fake_templates(10)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
        num_cases=1,
    )

    with patch("simulation.runner.load_case_templates", return_value=templates):
        passed = _run_and_get_templates(runner)

    assert len(passed) == 1
    assert passed[0] is templates[0]


# =============================================================================
# TEST 5: CLI argparse parses --num-cases correctly
# =============================================================================


def test_cli_parses_num_cases_flag():
    """The --num-cases flag is parsed as an int."""
    from run_simulation import _parse_args

    with patch("sys.argv", ["run_simulation.py", "--config", "test.yaml", "--num-cases", "5"]):
        args = _parse_args()
    assert args.num_cases == 5


def test_cli_num_cases_defaults_to_none():
    """When --num-cases is omitted, it defaults to None."""
    from run_simulation import _parse_args

    with patch("sys.argv", ["run_simulation.py", "--config", "test.yaml"]):
        args = _parse_args()
    assert args.num_cases is None


# =============================================================================
# TEST 6: Constructor stores num_cases
# =============================================================================


def test_runner_stores_num_cases(sim_config, tmp_path):
    """AsyncSimulationRunner stores num_cases on the instance."""
    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
        num_cases=7,
    )
    assert runner._num_cases == 7


def test_runner_num_cases_defaults_to_none(sim_config, tmp_path):
    """When num_cases is omitted, it defaults to None."""
    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
    )
    assert runner._num_cases is None
