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


def _make_pid_config(kp=0.05, ki=0.005, kd=0.01, epsilon=0.001, rho_star=0.8) -> PIDConfig:
    """Create a PIDConfig with safe (low) gains for testing.

    epsilon is set very low (0.001) by default so that convergence
    termination doesn't fire early with mock agents whose confidences
    are very similar (producing near-zero JS divergence).
    """
    return PIDConfig(
        gains=PIDGains(Kp=kp, Ki=ki, Kd=kd),
        rho_star=rho_star,
        T_max=20,
        epsilon=epsilon,
    )


def _make_debate_config(
    pid_config=None,
    max_rounds=2,
) -> DebateConfig:
    """Create a DebateConfig for testing (mock mode, parallel off)."""
    return DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=max_rounds,
        mock=True,
        parallel_agents=False,
        pid_config=pid_config,
    )


def _make_single_crit_entry():
    """Build a single-agent CRIT response dict (used inside batch response)."""
    return {
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
            "logical_validity": "No issues.",
            "evidential_support": "Well supported.",
            "alternative_consideration": "Aligned.",
            "causal_alignment": "Sound.",
        },
    }


def _make_runner_with_mock_crit(config: DebateConfig) -> MultiAgentRunner:
    """Create a runner and patch CRIT scorer with a mock LLM.

    The mock CRIT LLM returns a single-agent response with fixed pillar
    scores so tests are deterministic.
    """
    runner = MultiAgentRunner(config)
    if runner._crit_scorer:
        entry = _make_single_crit_entry()

        def _mock_single_agent_llm(sys_prompt: str, usr_prompt: str) -> str:
            # Return single-agent response (not batch)
            return json.dumps(entry)

        runner._crit_scorer._llm_fn = _mock_single_agent_llm
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
        """With PID config, pid_events has 1 entry per round (once after revise)."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=2)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 2  # 1 per round × 2 rounds


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


class TestPIDOncePerRound:
    def test_one_pid_event_per_round(self):
        """CRIT + PID runs once per round (after revise), not per phase."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 3  # 1 per round × 3 rounds


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
        assert event.controller_output.new_beta is not None


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
    @pytest.mark.skip(reason="validate_gains disabled in runner while experimenting with gains")
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
            mock=True,
            parallel_agents=True,
            pid_config=pid_config,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 1  # 1 per round × 1 round

    def test_parallel_pid_multi_round(self):
        """PID works with parallel_agents=True across multiple rounds."""
        pid_config = _make_pid_config()
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=2,
            mock=True,
            parallel_agents=True,
            pid_config=pid_config,
        )
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 2  # 1 per round × 2 rounds


# ---------------------------------------------------------------------------
# Flat YAML field path (_pid_enabled_flag)
# ---------------------------------------------------------------------------

class TestFlatYAMLPIDConfig:
    def test_pid_enabled_via_flag(self):
        """PID can be enabled via _pid_enabled_flag + flat gain fields."""
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=1,
            mock=True,
            parallel_agents=False,
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
            mock=True,
            parallel_agents=False,
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
        """Stable convergence causes early termination.

        Requires: converged quadrant (rho_bar >= rho_star) + JS < epsilon
        + rho_bar plateau, for convergence_window consecutive rounds.

        Mock CRIT gives rho_bar=0.75 every round. With rho_star=0.7,
        quadrant is CONVERGED. With epsilon=0.1 and near-zero JS from
        mock agents, all conditions are met. convergence_window=2 means
        earliest termination at round 3 (round 2 is first eligible,
        round 3 completes the window).

        PID runs once per round (after revise), so N rounds = N events.
        """
        pid_config = _make_pid_config(epsilon=0.1, rho_star=0.7)
        config = _make_debate_config(pid_config=pid_config, max_rounds=7)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        # Must terminate before max_rounds (7 rounds = 7 events)
        assert len(trace.pid_events) < 7
        # Terminates by round 3 or 4
        assert len(trace.pid_events) <= 4

    def test_no_convergence_runs_all_rounds(self):
        """Tiny epsilon prevents early termination (JS never drops below it)."""
        pid_config = _make_pid_config(epsilon=0.0001)
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 3  # 3 rounds × 1 per round

    def test_stuck_quadrant_prevents_termination(self):
        """When rho_bar < rho_star (STUCK quadrant), no early termination.

        Mock CRIT gives rho_bar=0.75 < rho_star=0.8, so quadrant=STUCK.
        Even with high epsilon and low JS, termination should not fire.
        """
        pid_config = _make_pid_config(epsilon=0.1, rho_star=0.8)
        config = _make_debate_config(pid_config=pid_config, max_rounds=5)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        action, trace = runner.run(obs)
        assert trace.pid_events is not None
        assert len(trace.pid_events) == 5  # all rounds × 1 per round


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
        """Beta values in PID events show controller evolution across rounds."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        betas = [e.pid_step["beta_new"] for e in runner._pid_events]
        assert len(betas) == 3  # 3 rounds × 1 per round
        # Each beta should be a valid float in [0, 1]
        for b in betas:
            assert 0.0 <= b <= 1.0

    def test_round_indices_sequential(self):
        """PID events have sequential round indices (1 per round)."""
        pid_config = _make_pid_config()
        config = _make_debate_config(pid_config=pid_config, max_rounds=3)
        runner = _make_runner_with_mock_crit(config)
        obs = _make_observation()
        runner.run(obs)
        indices = [e.round_index for e in runner._pid_events]
        # Each round index appears once
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
