"""End-to-end integration tests for PID through the full simulation stack.

Tests the complete path:
    YAML config → AgentConfig → DebateAgentSystem → MultiAgentRunner → output

All tests use mock=True — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.multi_agent_debate import DebateAgentSystem
from models.agents import AgentInvocation, AgentInvocationResult
from models.case import Case, CaseData, CaseDataItem, ClosePricePoint, StockData
from models.config import AgentConfig, SimulationConfig
from models.portfolio import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_case() -> Case:
    """A realistic Case from the simulation side."""
    return Case(
        case_data=CaseData(
            items=[
                CaseDataItem(kind="earnings", content="NVDA beat earnings by 10%."),
                CaseDataItem(kind="news", content="AI chip demand surging."),
            ]
        ),
        stock_data={
            "NVDA": StockData(
                ticker="NVDA",
                current_price=150.0,
                daily_bars=[
                    ClosePricePoint(timestamp="2025-01-01", close=145.0),
                    ClosePricePoint(timestamp="2025-01-02", close=148.0),
                    ClosePricePoint(timestamp="2025-01-03", close=150.0),
                ],
            ),
        },
        portfolio=PortfolioSnapshot(
            cash=100000.0,
            positions={"NVDA": 0},
        ),
        case_id="ep_000:0",
        decision_point_idx=0,
        information_cutoff_timestamp="2025-01-03T00:00:00Z",
    )


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """AgentConfig with mock mode and PID disabled."""
    return AgentConfig(
        agent_system="multi_agent_debate",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        temperature=0.3,
        system_prompt_override="mock",
    )


@pytest.fixture
def pid_agent_config() -> AgentConfig:
    """AgentConfig with mock mode and PID enabled via flat YAML fields."""
    return AgentConfig(
        agent_system="multi_agent_debate",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        temperature=0.3,
        system_prompt_override="mock",
        pid_enabled=True,
        pid_kp=0.05,
        pid_ki=0.005,
        pid_kd=0.01,
        pid_rho_star=0.8,
        pid_initial_beta=0.5,
        pid_propose=False,
        pid_critique=True,
        pid_revise=True,
    )


# ---------------------------------------------------------------------------
# 1. YAML → AgentConfig → PID field propagation
# ---------------------------------------------------------------------------

class TestYAMLConfigPropagation:
    """Verify PID config fields propagate from YAML through to DebateConfig."""

    def test_yaml_to_agent_config(self):
        """SimulationConfig.from_yaml produces AgentConfig with PID fields."""
        yaml_content = """\
dataset_path: "data/cases"
tickers: [NVDA]
invest_quarter: "2025Q1"
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  pid_enabled: true
  pid_kp: 0.1
  pid_ki: 0.02
  pid_kd: 0.05
  pid_rho_star: 0.85
  pid_initial_beta: 0.6
  pid_propose: true
  pid_critique: true
  pid_revise: false
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config = SimulationConfig.from_yaml(f.name)

        agent = config.agent
        assert agent.pid_enabled is True
        assert agent.pid_kp == 0.1
        assert agent.pid_ki == 0.02
        assert agent.pid_kd == 0.05
        assert agent.pid_rho_star == 0.85
        assert agent.pid_initial_beta == 0.6
        assert agent.pid_propose is True
        assert agent.pid_critique is True
        assert agent.pid_revise is False

    def test_yaml_pid_disabled_by_default(self):
        """Without pid_enabled in YAML, PID stays off."""
        yaml_content = """\
dataset_path: "data/cases"
tickers: [NVDA]
invest_quarter: "2025Q1"
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config = SimulationConfig.from_yaml(f.name)

        assert config.agent.pid_enabled is False

    def test_agent_config_to_debate_config(self, pid_agent_config: AgentConfig):
        """AgentConfig with PID fields creates DebateAgentSystem with PID enabled."""
        agent = DebateAgentSystem(pid_agent_config)
        debate_cfg = agent._debate_cfg
        assert debate_cfg.pid_enabled is True
        assert debate_cfg.pid_config is not None
        assert debate_cfg.pid_config.gains.Kp == 0.05
        assert debate_cfg.pid_config.gains.Ki == 0.005
        assert debate_cfg.pid_config.gains.Kd == 0.01
        assert debate_cfg.pid_config.rho_star == 0.8
        assert debate_cfg.initial_beta == 0.5
        assert debate_cfg.pid_propose is False
        assert debate_cfg.pid_critique is True
        assert debate_cfg.pid_revise is True


# ---------------------------------------------------------------------------
# 2. DebateAgentSystem.invoke() end-to-end with PID
# ---------------------------------------------------------------------------

class TestAdapterInvokeWithPID:
    """Test full adapter invoke path with PID enabled.

    Path: AgentConfig → DebateAgentSystem → MultiAgentRunner.run() → output
    """

    def test_invoke_with_pid_returns_decision(
        self, pid_agent_config: AgentConfig, sample_case: Case
    ):
        """Full invoke with PID enabled returns a valid AgentInvocationResult."""
        agent = DebateAgentSystem(pid_agent_config)

        # Patch CRIT scorer to use mock LLM
        self._patch_crit_scorer(agent)

        agent.bind_tools(MagicMock())
        invocation = AgentInvocation(
            case=sample_case,
            episode_id="ep_000",
            agent_id="debate_000",
        )
        result = asyncio.run(
            agent.invoke(invocation)
        )

        assert isinstance(result, AgentInvocationResult)
        assert isinstance(result.decision, type(result.decision))  # Decision type
        assert "debate_action" in result.raw_output
        assert "debate_trace" in result.raw_output

    def test_invoke_pid_events_in_raw_output(
        self, pid_agent_config: AgentConfig, sample_case: Case
    ):
        """PID events are included in the raw_output trace."""
        agent = DebateAgentSystem(pid_agent_config)
        self._patch_crit_scorer(agent)

        agent.bind_tools(MagicMock())
        invocation = AgentInvocation(
            case=sample_case,
            episode_id="ep_000",
            agent_id="debate_000",
        )
        result = asyncio.run(
            agent.invoke(invocation)
        )

        trace = result.raw_output["debate_trace"]
        assert trace["pid_events"] is not None
        assert len(trace["pid_events"]) >= 1
        event = trace["pid_events"][0]
        assert "round_index" in event
        assert "metrics" in event
        assert "pid_step" in event
        assert "controller_output" in event

    def test_invoke_without_pid_no_events(
        self, mock_agent_config: AgentConfig, sample_case: Case
    ):
        """Without PID, pid_events is None in the trace."""
        agent = DebateAgentSystem(mock_agent_config)
        agent.bind_tools(MagicMock())
        invocation = AgentInvocation(
            case=sample_case,
            episode_id="ep_000",
            agent_id="debate_000",
        )
        result = asyncio.run(
            agent.invoke(invocation)
        )

        trace = result.raw_output["debate_trace"]
        assert trace["pid_events"] is None

    @staticmethod
    def _patch_crit_scorer(agent: DebateAgentSystem) -> None:
        """Replace CRIT scorer's LLM with a mock that returns valid batch JSON."""
        runner = agent._debate_runner
        if runner._crit_scorer:
            entry = {
                "pillar_scores": {
                    "internal_consistency": 0.8,
                    "evidence_support": 0.7,
                    "trace_alignment": 0.9,
                    "causal_integrity": 0.6,
                },
                "diagnostics": {
                    "contradictions_detected": False,
                    "unsupported_claims_detected": False,
                    "conclusion_drift_detected": False,
                    "causal_overreach_detected": False,
                },
                "explanations": {
                    "internal_consistency": "ok",
                    "evidence_support": "ok",
                    "trace_alignment": "ok",
                    "causal_integrity": "ok",
                },
            }
            role_names = [r.value for r in runner.config.roles]
            mock_response = json.dumps({role: entry for role in role_names})
            runner._crit_scorer._llm_fn = lambda sys, usr: mock_response


# ---------------------------------------------------------------------------
# 3. Trace output on disk contains PID events
# ---------------------------------------------------------------------------

class TestDiskOutputWithPID:
    """Verify PID events are persisted in the trace file written to disk."""

    def test_trace_file_contains_pid_events(
        self, pid_agent_config: AgentConfig, sample_case: Case
    ):
        """The JSON trace file on disk includes pid_events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DebateAgentSystem(pid_agent_config)
            # Point trace output to tmpdir
            agent._debate_runner.config.trace_dir = tmpdir
            TestAdapterInvokeWithPID._patch_crit_scorer(agent)

            agent.bind_tools(MagicMock())
            invocation = AgentInvocation(
                case=sample_case,
                episode_id="ep_000",
                agent_id="debate_000",
            )
            asyncio.run(
                agent.invoke(invocation)
            )

            # Find the trace file
            trace_files = list(Path(tmpdir).glob("debate_langgraph_*.json"))
            assert len(trace_files) == 1

            with open(trace_files[0]) as f:
                data = json.load(f)

            # Trace should contain pid_events
            assert "trace" in data
            assert "pid_events" in data["trace"]
            assert data["trace"]["pid_events"] is not None
            assert len(data["trace"]["pid_events"]) >= 1

    def test_trace_file_no_pid_when_disabled(
        self, mock_agent_config: AgentConfig, sample_case: Case
    ):
        """When PID is disabled, trace file has pid_events=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DebateAgentSystem(mock_agent_config)
            agent._debate_runner.config.trace_dir = tmpdir

            agent.bind_tools(MagicMock())
            invocation = AgentInvocation(
                case=sample_case,
                episode_id="ep_000",
                agent_id="debate_000",
            )
            asyncio.run(
                agent.invoke(invocation)
            )

            trace_files = list(Path(tmpdir).glob("debate_langgraph_*.json"))
            assert len(trace_files) == 1

            with open(trace_files[0]) as f:
                data = json.load(f)

            assert data["trace"]["pid_events"] is None


# ---------------------------------------------------------------------------
# 4. PID config round-trips through all layers
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Verify PID gains survive the full config chain."""

    def test_gains_match_at_controller(self, pid_agent_config: AgentConfig):
        """PID gains in AgentConfig match what PIDController receives."""
        agent = DebateAgentSystem(pid_agent_config)
        runner = agent._debate_runner
        ctrl = runner._pid_controller

        assert ctrl is not None
        assert ctrl.config.gains.Kp == pid_agent_config.pid_kp
        assert ctrl.config.gains.Ki == pid_agent_config.pid_ki
        assert ctrl.config.gains.Kd == pid_agent_config.pid_kd
        assert ctrl.config.rho_star == pid_agent_config.pid_rho_star
        assert ctrl.state.beta == pid_agent_config.pid_initial_beta

    def test_per_phase_toggles_match(self, pid_agent_config: AgentConfig):
        """Per-phase toggles in AgentConfig match DebateConfig."""
        agent = DebateAgentSystem(pid_agent_config)
        cfg = agent._debate_runner.config

        assert cfg.pid_propose == pid_agent_config.pid_propose
        assert cfg.pid_critique == pid_agent_config.pid_critique
        assert cfg.pid_revise == pid_agent_config.pid_revise

    def test_phase_graphs_compiled_when_pid_enabled(
        self, pid_agent_config: AgentConfig
    ):
        """When PID is enabled, per-phase sub-graphs are compiled."""
        agent = DebateAgentSystem(pid_agent_config)
        runner = agent._debate_runner

        assert runner._propose_graph is not None
        assert runner._critique_graph is not None
        assert runner._revise_graph is not None

    def test_phase_graphs_not_compiled_when_pid_disabled(
        self, mock_agent_config: AgentConfig
    ):
        """When PID is disabled, per-phase sub-graphs are None."""
        agent = DebateAgentSystem(mock_agent_config)
        runner = agent._debate_runner

        assert runner._propose_graph is None
        assert runner._critique_graph is None
        assert runner._revise_graph is None
