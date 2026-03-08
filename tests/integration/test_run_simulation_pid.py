"""End-to-end integration test: run_simulation → disk output with PID events.

Tests the complete simulation pipeline — the same path executed by
``python run_simulation.py --agents <yaml>``:

    YAML config on disk
        → SimulationConfig.from_yaml()
        → AsyncSimulationRunner.run()
            → load_case_templates() from disk
            → Broker + DebateAgentSystem creation
            → agent.invoke() (Case → Observation → debate → Action → Decision)
            → broker.execute_decision()
            → DecisionPointLog (with agent_output containing debate_trace)
            → EpisodeLog
        → SimulationLogger writes to disk:
            episode_log.json, reasoning/case_000.txt,
            simulation_log.json, summary.json

Verifies PID events survive the full path and are present in every
output artifact.  All tests use mock=True — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from models.config import SimulationConfig
from simulation.runner import AsyncSimulationRunner


# ---------------------------------------------------------------------------
# Mock CRIT response for _call_llm (debate nodes use their own mock helpers)
# ---------------------------------------------------------------------------

_CRIT_ENTRY = {
    "pillar_scores": {
        "logical_validity": 0.8,
        "evidential_support": 0.7,
        "alternative_consideration": 0.9,
        "causal_alignment": 0.6,
    },
    "diagnostics": {
        "contradictions_detected": False,
        "unsupported_claims_detected": False,
        "ignored_critiques_detected": False,
        "premature_certainty_detected": False,
        "causal_overreach_detected": False,
        "conclusion_drift_detected": False,
    },
    "explanations": {
        "logical_validity": "ok",
        "evidential_support": "ok",
        "alternative_consideration": "ok",
        "causal_alignment": "ok",
    },
}

# Single-agent format: scorer now calls _llm_fn once per agent independently.
MOCK_CRIT_RESPONSE = json.dumps(_CRIT_ENTRY)


def _mock_call_llm(config: dict, system_prompt: str, user_prompt: str, **kw) -> str:
    """Drop-in replacement for ``multi_agent.graph._call_llm``.

    Debate graph nodes never reach _call_llm in mock mode (they short-circuit
    via _mock_proposal / _mock_critique / _mock_revision / _mock_judge).
    The only caller in mock mode is the CRIT scorer, which needs a valid
    CRIT JSON response.
    """
    return MOCK_CRIT_RESPONSE


# ---------------------------------------------------------------------------
# Minimal snapshot JSON (written to disk during test setup)
# invest_quarter=2025Q1 → loader reads prior quarter 2024Q4
# ---------------------------------------------------------------------------

SNAPSHOT_2024_Q4 = {
    "as_of_date": "2024-12-31",
    "ticker_data": {
        "NVDA": {
            "asset_features": {
                "close": 150.0,
            }
        }
    },
}


# ---------------------------------------------------------------------------
# YAML config template (PID enabled + mock mode)
# ---------------------------------------------------------------------------

YAML_PID_ENABLED = """\
dataset_path: "{dataset_path}"
tickers: [NVDA]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
debate_setup:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
  pid_enabled: true
  pid_kp: 0.05
  pid_ki: 0.005
  pid_kd: 0.01
  pid_rho_star: 0.8
  pid_initial_beta: 0.5
"""

YAML_PID_DISABLED = """\
dataset_path: "{dataset_path}"
tickers: [NVDA]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
debate_setup:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simulation_dir():
    """Create a tmpdir with dataset + config + output dirs on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Snapshot JSON directory (memo loader expects json_data/)
        json_dir = root / "dataset" / "json_data"
        json_dir.mkdir(parents=True)
        snapshot_file = json_dir / "snapshot_2024_Q4.json"
        snapshot_file.write_text(json.dumps(SNAPSHOT_2024_Q4), encoding="utf-8")

        # Output directory
        output_dir = root / "results"
        output_dir.mkdir()

        yield root


@pytest.fixture
def pid_config_path(simulation_dir):
    """Write PID-enabled YAML config and return its path."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_PID_ENABLED.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_pid.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


@pytest.fixture
def no_pid_config_path(simulation_dir):
    """Write PID-disabled YAML config and return its path."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_PID_DISABLED.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_no_pid.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


def _run_simulation(config_path: str, output_dir: str) -> Path:
    """Run the full simulation pipeline and return the run output directory.

    This is the same path as ``python run_simulation.py --agents <yaml>``.
    """
    config = SimulationConfig.from_yaml(config_path)
    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=config_path,
        output_dir=output_dir,
    )
    asyncio.run(runner.run())
    # Find the run directory (named after config stem)
    results_root = Path(output_dir)
    run_dirs = [d for d in results_root.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    return run_dirs[0]


# ---------------------------------------------------------------------------
# Tests: PID enabled → disk output contains PID events
# ---------------------------------------------------------------------------

class TestRunSimulationWithPID:
    """Full pipeline test: YAML → AsyncSimulationRunner → disk output."""

    def test_episode_log_contains_pid_events(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """episode_log.json has pid_events in agent_output.debate_trace."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        assert ep_log_path.exists()

        ep_log = json.loads(ep_log_path.read_text())
        dp_logs = ep_log["decision_point_logs"]
        assert len(dp_logs) >= 1

        agent_output = dp_logs[0]["agent_output"]
        assert isinstance(agent_output, dict)
        assert "debate_trace" in agent_output

        trace = agent_output["debate_trace"]
        assert "pid_events" in trace
        assert trace["pid_events"] is not None
        assert len(trace["pid_events"]) >= 1

    def test_pid_event_structure_in_episode_log(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """Each PIDEvent in episode_log.json has the correct fields."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        ep_log = json.loads(ep_log_path.read_text())
        events = ep_log["decision_point_logs"][0]["agent_output"]["debate_trace"]["pid_events"]

        for event in events:
            assert "round_index" in event
            assert "metrics" in event
            assert "crit_result" in event
            assert "pid_step" in event
            assert "controller_output" in event

            # Metrics sub-fields
            metrics = event["metrics"]
            assert "rho_bar" in metrics
            assert "js_divergence" in metrics
            assert "ov_overlap" in metrics

            # PID step sub-fields
            pid_step = event["pid_step"]
            assert "e_t" in pid_step
            assert "u_t" in pid_step
            assert "beta_new" in pid_step

            # Controller output
            ctrl = event["controller_output"]
            assert "new_beta" in ctrl

    def test_reasoning_trace_file_contains_pid_events(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """reasoning/case_000.txt contains PID events as JSON."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        reasoning_path = run_dir / "episodes" / "ep_000" / "reasoning" / "case_000.txt"
        assert reasoning_path.exists()

        content = json.loads(reasoning_path.read_text())
        assert "debate_trace" in content
        assert content["debate_trace"]["pid_events"] is not None
        assert len(content["debate_trace"]["pid_events"]) >= 1

    def test_simulation_log_contains_episode_with_pid(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """simulation_log.json has episode data with PID events."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        sim_log_path = run_dir / "simulation_log.json"
        assert sim_log_path.exists()

        sim_log = json.loads(sim_log_path.read_text())
        assert "episode_logs" in sim_log
        assert len(sim_log["episode_logs"]) == 1

        ep = sim_log["episode_logs"][0]
        assert ep["episode_id"] == "ep_000"
        dp = ep["decision_point_logs"][0]
        trace = dp["agent_output"]["debate_trace"]
        assert trace["pid_events"] is not None

    def test_summary_json_has_valid_structure(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """summary.json contains episode summary with financial metrics."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        summary_path = run_dir / "summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["num_episodes"] == 1
        assert len(summary["episode_summaries"]) == 1

        ep_summary = summary["episode_summaries"][0]
        assert "initial_cash" in ep_summary
        assert "final_cash" in ep_summary
        assert "book_value" in ep_summary
        assert "return_pct" in ep_summary
        assert "total_trades" in ep_summary
        assert ep_summary["initial_cash"] == 100000.0

    def test_decision_was_executed(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """The debate produced a decision and the broker executed it."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        ep_log = json.loads(ep_log_path.read_text())

        dp = ep_log["decision_point_logs"][0]
        decision = dp["extracted_decision"]
        assert "orders" in decision

        # Portfolio final_snapshots exist
        assert dp["portfolio_before"]["cash"] == 100000.0
        assert "portfolio_after" in dp

    def test_crit_rho_bar_in_pid_events(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """rho_bar from CRIT scorer is the mean of 4 pillar scores."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        ep_log = json.loads(ep_log_path.read_text())
        events = ep_log["decision_point_logs"][0]["agent_output"]["debate_trace"]["pid_events"]

        expected_rho = (0.8 + 0.7 + 0.9 + 0.6) / 4.0
        for event in events:
            assert abs(event["metrics"]["rho_bar"] - expected_rho) < 1e-9

    def test_config_yaml_copied_to_output(
        self, simulation_dir, pid_config_path, monkeypatch
    ):
        """The YAML config is copied into the run output directory."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(pid_config_path, output_dir)

        config_copy = run_dir / "config.yaml"
        assert config_copy.exists()
        content = config_copy.read_text()
        assert "pid_enabled: true" in content


# ---------------------------------------------------------------------------
# Tests: PID disabled → no PID events in output
# ---------------------------------------------------------------------------

class TestRunSimulationWithoutPID:
    """Full pipeline test: verify no PID events when PID is disabled."""

    def test_no_pid_events_in_episode_log(
        self, simulation_dir, no_pid_config_path, monkeypatch
    ):
        """Without PID, pid_events is null in episode_log.json."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(no_pid_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        ep_log = json.loads(ep_log_path.read_text())
        trace = ep_log["decision_point_logs"][0]["agent_output"]["debate_trace"]
        assert trace["pid_events"] is None

    def test_no_pid_events_in_reasoning_file(
        self, simulation_dir, no_pid_config_path, monkeypatch
    ):
        """Without PID, reasoning trace file has pid_events=null."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(no_pid_config_path, output_dir)

        reasoning_path = run_dir / "episodes" / "ep_000" / "reasoning" / "case_000.txt"
        content = json.loads(reasoning_path.read_text())
        assert content["debate_trace"]["pid_events"] is None

    def test_simulation_completes_without_pid(
        self, simulation_dir, no_pid_config_path, monkeypatch
    ):
        """Full pipeline completes successfully without PID."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(no_pid_config_path, output_dir)

        sim_log_path = run_dir / "simulation_log.json"
        summary_path = run_dir / "summary.json"
        assert sim_log_path.exists()
        assert summary_path.exists()

        sim_log = json.loads(sim_log_path.read_text())
        assert len(sim_log["errors"]) == 0
        assert len(sim_log["episode_logs"]) == 1
