"""Tests for config-driven case filtering.

Verifies that AsyncSimulationRunner passes the config's tickers, quarters,
and top_n_news to load_case_templates, and that the CLI no longer accepts
the removed --ticker / --num-cases flags.
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


def _run_and_capture_load_call(runner, templates):
    """Run the runner with mocked internals and return the kwargs passed to load_case_templates."""
    mock_load = MagicMock(return_value=templates)
    mock_episode = AsyncMock(return_value=MagicMock())

    with patch(
        "simulation.runner.load_case_templates", mock_load
    ), patch.object(
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

    return mock_load.call_args


# =============================================================================
# TEST 1: Runner passes config tickers to load_case_templates
# =============================================================================


def test_runner_passes_config_tickers(sim_config, tmp_path):
    """Runner should pass config.tickers as ticker_filter."""
    templates = _make_fake_templates(5)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
    )

    call_args = _run_and_capture_load_call(runner, templates)
    assert call_args.kwargs["ticker_filter"] == ["AAPL"]


# =============================================================================
# TEST 2: Runner passes config quarters to load_case_templates
# =============================================================================


def test_runner_passes_config_quarters(tmp_path):
    """Runner should pass config.quarters to load_case_templates."""
    config = SimulationConfig(
        dataset_path=str(tmp_path),
        agent=AgentConfig(
            agent_system="single_llm",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        ),
        broker=BrokerConfig(initial_cash=100_000.0),
        tickers=["NVDA"],
        quarters=["Q1", "Q3"],
        num_episodes=1,
    )
    templates = _make_fake_templates(3)

    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=str(tmp_path / "test.yaml"),
    )

    call_args = _run_and_capture_load_call(runner, templates)
    assert call_args.kwargs["quarters"] == ["Q1", "Q3"]


# =============================================================================
# TEST 3: Runner passes None quarters when not set in config
# =============================================================================


def test_runner_passes_none_quarters_when_omitted(sim_config, tmp_path):
    """When quarters is not set in config, None should be passed."""
    templates = _make_fake_templates(5)

    runner = AsyncSimulationRunner(
        sim_config,
        config_yaml_path=str(tmp_path / "test.yaml"),
    )

    call_args = _run_and_capture_load_call(runner, templates)
    assert call_args.kwargs["quarters"] is None


# =============================================================================
# TEST 4: Runner no longer accepts num_cases or ticker_filter params
# =============================================================================


def test_runner_rejects_removed_params(sim_config, tmp_path):
    """AsyncSimulationRunner should not accept num_cases or ticker_filter."""
    with pytest.raises(TypeError):
        AsyncSimulationRunner(
            sim_config,
            config_yaml_path=str(tmp_path / "test.yaml"),
            num_cases=3,
        )

    with pytest.raises(TypeError):
        AsyncSimulationRunner(
            sim_config,
            config_yaml_path=str(tmp_path / "test.yaml"),
            ticker_filter=["NVDA"],
        )


# =============================================================================
# TEST 5: CLI no longer accepts --num-cases or --ticker
# =============================================================================


def test_cli_rejects_num_cases_flag():
    """The --num-cases flag should no longer be accepted."""
    from run_simulation import _parse_args

    with patch("sys.argv", ["run_simulation.py", "--config", "test.yaml", "--num-cases", "5"]):
        with pytest.raises(SystemExit):
            _parse_args()


def test_cli_rejects_ticker_flag():
    """The --ticker flag should no longer be accepted."""
    from run_simulation import _parse_args

    with patch("sys.argv", ["run_simulation.py", "--config", "test.yaml", "--ticker", "NVDA"]):
        with pytest.raises(SystemExit):
            _parse_args()


# =============================================================================
# TEST 6: CLI still accepts --config, --output-dir, --list-tickers, --log-level
# =============================================================================


def test_cli_accepts_remaining_flags():
    """The kept flags should still parse correctly."""
    from run_simulation import _parse_args

    with patch("sys.argv", [
        "run_simulation.py",
        "--config", "test.yaml",
        "--output-dir", "out",
        "--log-level", "ERROR",
    ]):
        args = _parse_args()

    assert args.config == "test.yaml"
    assert args.output_dir == "out"
    assert args.log_level == "ERROR"
    assert args.list_tickers is False
