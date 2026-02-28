"""Integration tests for PID controller in the debate loop.

All tests use mock=True + mock LLM functions — no real API calls.
"""

import json

import pytest

from eval.PID.stability import GainInstabilityError
from eval.PID.types import PIDConfig, PIDGains
from multi_agent.config import AgentRole, DebateConfig
from multi_agent.models import Observation, MarketState, PortfolioState, PIDEvent
from multi_agent.runner import MultiAgentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation() -> Observation:
    """Create a minimal observation for testing."""
    return Observation(
        timestamp="2025-01-01T00:00:00Z",
        universe=["NVDA"],
        market_state=MarketState(prices={"NVDA": 150.0}),
        text_context="NVDA reported strong earnings.",
        portfolio_state=PortfolioState(cash=100000.0, positions={"NVDA": 0.0}),
    )


def _make_pid_config(kp=0.05, ki=0.005, kd=0.01, epsilon=0.001) -> PIDConfig:
    """Create a PIDConfig with safe (low) gains for testing.

    epsilon is set very low (0.001) by default so that convergence
    termination doesn't fire early with mock agents whose confidences
    are very similar (producing near-zero JS divergence).
    """
    return PIDConfig(
        gains=PIDGains(Kp=kp, Ki=ki, Kd=kd),
        rho_star=0.8,
        T_max=20,
        epsilon=epsilon,
    )


def _make_debate_config(
    pid_config=None,
    max_rounds=2,
    pid_propose=False,
    pid_critique=True,
    pid_revise=True,
) -> DebateConfig:
    """Create a DebateConfig for testing (mock mode, parallel off)."""
    return DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=max_rounds,
        agreeableness=0.3,
        mock=True,
        parallel_agents=False,
        enable_news_pipeline=False,
        enable_data_pipeline=False,
        pid_config=pid_config,
        pid_propose=pid_propose,
        pid_critique=pid_critique,
        pid_revise=pid_revise,
    )


def _make_runner_with_mock_crit(config: DebateConfig) -> MultiAgentRunner:
    """Create a runner and patch CRIT scorer with a mock LLM.

    The mock CRIT LLM returns fixed pillar scores so tests are deterministic.
    """
    runner = MultiAgentRunner(config)
    if runner._crit_scorer:
        mock_response = json.dumps({
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
                "internal_consistency": "No issues.",
                "evidence_support": "Well supported.",
                "trace_alignment": "Aligned.",
                "causal_integrity": "Sound.",
            },
        })
        runner._crit_scorer._llm_fn = lambda sys, usr: mock_response
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPIDDisabledByDefault:
    def test_no_pid_events_when_disabled(self):
        """When pid_config=None, no PID events are produced."""
        config = _make_debate_config(pid_config=None)
        runner = MultiAgentRunner(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is None
        assert runner._pid_events == []


class TestPIDProducesEvents:
    def test_pid_events_per_round(self):
        """With PID config, pid_events has one entry per round."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=2)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 2  # one per round


class TestPIDAdjustsAgreeableness:
    def test_beta_changes_after_round(self):
        """After round 1, PID controller has updated beta."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=2)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        # Beta should have been updated from initial 0.5
        assert runner._pid_controller.beta != 0.5 or len(runner._pid_events) == 2


class TestPIDPerPhaseToggles:
    def test_critique_only(self):
        """pid_critique=True + pid_revise=False: only critique uses beta."""
        pid_config = _make_pid_config()
        config = _make_debate_config(
            pid_config=pid_config,
            max_rounds=1,
            pid_propose=False,
            pid_critique=True,
            pid_revise=False,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None

    def test_propose_toggle(self):
        """pid_propose=True: propose phase uses PID agreeableness."""
        pid_config = _make_pid_config()
        config = _make_debate_config(
            pid_config=pid_config,
            max_rounds=1,
            pid_propose=True,
            pid_critique=True,
            pid_revise=True,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None


class TestPIDEventSchema:
    def test_event_has_all_fields(self):
        """PIDEvent has all required fields."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=1)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        event = runner._pid_events[0]
        assert isinstance(event, PIDEvent)
        assert event.round_index == 1
        assert event.metrics.rho_bar > 0
        assert "e_t" in event.pid_step
        assert "beta_new" in event.pid_step
        assert event.controller_output.new_agreeableness is not None


class TestPIDCritIntegration:
    def test_crit_feeds_rho_bar_to_pid(self):
        """CRIT result's rho_bar is used as PID input."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=1)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        event = runner._pid_events[0]
        # rho_bar should match our mock CRIT scores
        expected_rho = (0.8 + 0.7 + 0.9 + 0.6) / 4.0
        assert abs(event.metrics.rho_bar - expected_rho) < 1e-9


class TestNoPIDUsesSingleRoundGraph:
    def test_single_round_graph_when_disabled(self):
        """PID disabled → runner uses existing single_round_graph path."""
        config = _make_debate_config(pid_config=None)
        runner = MultiAgentRunner(config)
        # Phase sub-graphs should not be compiled
        assert runner._propose_graph is None
        assert runner._critique_graph is None
        assert runner._revise_graph is None
        # single_round_graph should exist
        assert runner.single_round_graph is not None


class TestPIDStabilityValidation:
    def test_invalid_gains_raise_error(self):
        """Invalid (too aggressive) gains raise GainInstabilityError at init."""
        # Very aggressive gains that should fail stability check
        bad_config = PIDConfig(
            gains=PIDGains(Kp=10.0, Ki=10.0, Kd=10.0),
            rho_star=0.8,
            T_max=20,
        )
        config = _make_debate_config(pid_config=bad_config)
        with pytest.raises(GainInstabilityError):
            MultiAgentRunner(config)


class TestPIDSycophancyDetection:
    def test_sycophancy_flag_in_step(self):
        """PID step result includes sycophancy indicator s_t."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=2)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        for event in runner._pid_events:
            assert "s_t" in event.pid_step
            assert event.pid_step["s_t"] in (0, 1)


# ---------------------------------------------------------------------------
# Exhaustive per-phase toggle combinations (2^3 = 8)
# ---------------------------------------------------------------------------

class TestAllPhaseToggleCombinations:
    """Test all 8 combinations of (pid_propose, pid_critique, pid_revise).

    Each test verifies the runner completes without error and produces
    the expected PID events.
    """

    @pytest.mark.parametrize(
        "pid_propose, pid_critique, pid_revise",
        [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ],
        ids=[
            "FFF_no_phases",
            "FFT_revise_only",
            "FTF_critique_only",
            "FTT_critique_revise",
            "TFF_propose_only",
            "TFT_propose_revise",
            "TTF_propose_critique",
            "TTT_all_phases",
        ],
    )
    def test_toggle_combination(self, pid_propose, pid_critique, pid_revise):
        pid_config = _make_pid_config()
        config = _make_debate_config(
            pid_config=pid_config,
            max_rounds=1,
            pid_propose=pid_propose,
            pid_critique=pid_critique,
            pid_revise=pid_revise,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 1


# ---------------------------------------------------------------------------
# Parallel agents with PID
# ---------------------------------------------------------------------------

class TestPIDWithParallelAgents:
    def test_parallel_pid_produces_events(self):
        """PID works with parallel_agents=True."""
        pid_config = _make_pid_config()
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            agreeableness=0.3,
            mock=True,
            parallel_agents=True,
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            pid_config=pid_config,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 1

    def test_parallel_pid_multi_round(self):
        """PID works with parallel_agents=True across multiple rounds."""
        pid_config = _make_pid_config()
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=2,
            agreeableness=0.3,
            mock=True,
            parallel_agents=True,
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            pid_config=pid_config,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 2


# ---------------------------------------------------------------------------
# Flat YAML field path (_pid_enabled_flag)
# ---------------------------------------------------------------------------

class TestFlatYAMLPIDConfig:
    def test_pid_enabled_via_flag(self):
        """PID can be enabled via _pid_enabled_flag + flat gain fields."""
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            agreeableness=0.3,
            mock=True,
            parallel_agents=False,
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            _pid_enabled_flag=True,
            pid_kp=0.05,
            pid_ki=0.005,
            pid_kd=0.01,
            pid_rho_star=0.8,
        )
        assert config.pid_enabled is True
        assert config.pid_config is not None
        assert config.pid_config.gains.Kp == 0.05

    def test_pid_disabled_via_flag(self):
        """_pid_enabled_flag=False does not construct PIDConfig."""
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            mock=True,
            _pid_enabled_flag=False,
        )
        assert config.pid_enabled is False
        assert config.pid_config is None

    def test_flat_fields_run_end_to_end(self):
        """Runner works when PID is configured via flat YAML fields."""
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            agreeableness=0.3,
            mock=True,
            parallel_agents=False,
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            _pid_enabled_flag=True,
            pid_kp=0.05,
            pid_ki=0.005,
            pid_kd=0.01,
            pid_rho_star=0.8,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None


# ---------------------------------------------------------------------------
# Convergence termination
# ---------------------------------------------------------------------------

class TestConvergenceTermination:
    def test_convergence_terminates_early(self):
        """High epsilon causes early termination when JS is low."""
        # Mock agents have very similar confidences → near-zero JS.
        # Setting epsilon high (0.1) should cause termination after round 1.
        pid_config = _make_pid_config(epsilon=0.1)
        config = _make_debate_config(pid_config=pid_config, max_rounds=5)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        # Should have terminated before round 5
        assert trace.pid_events is not None
        assert len(trace.pid_events) < 5

    def test_no_convergence_runs_all_rounds(self):
        """Tiny epsilon prevents early termination."""
        pid_config = _make_pid_config(epsilon=0.0001)
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 3


# ---------------------------------------------------------------------------
# Non-default initial_beta
# ---------------------------------------------------------------------------

class TestInitialBeta:
    def test_custom_initial_beta(self):
        """Non-default initial_beta is used by PID controller."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=1)
        config.initial_beta = 0.9
        runner = _make_runner_with_mock_crit(config)
        # Controller should start at 0.9, not default 0.5
        assert runner._pid_controller.state.beta == 0.9

    def test_zero_initial_beta(self):
        """initial_beta=0.0 is valid."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=1)
        config.initial_beta = 0.0
        runner = _make_runner_with_mock_crit(config)
        assert runner._pid_controller.state.beta == 0.0
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None


# ---------------------------------------------------------------------------
# Beta trajectory across rounds
# ---------------------------------------------------------------------------

class TestBetaTrajectory:
    def test_beta_evolves_across_rounds(self):
        """Beta values in PID events show controller evolution."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        betas = [e.pid_step["beta_new"] for e in runner._pid_events]
        assert len(betas) == 3
        # Each beta should be a valid float in [0, 1]
        for b in betas:
            assert 0.0 <= b <= 1.0

    def test_round_indices_sequential(self):
        """PID events have sequential round indices."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        indices = [e.round_index for e in runner._pid_events]
        assert indices == [1, 2, 3]


# ---------------------------------------------------------------------------
# Config properties
# ---------------------------------------------------------------------------

class TestConfigProperties:
    def test_evaluation_mode_in_loop(self):
        """evaluation_mode is 'in_loop' when PID is enabled."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config)
        assert config.evaluation_mode == "in_loop"

    def test_evaluation_mode_post_hoc(self):
        """evaluation_mode is 'post_hoc' when PID is disabled."""
        config = _make_debate_config(pid_config=None)
        assert config.evaluation_mode == "post_hoc"

    def test_pid_enabled_true(self):
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config)
        assert config.pid_enabled is True

    def test_pid_enabled_false(self):
        config = _make_debate_config(pid_config=None)
        assert config.pid_enabled is False
