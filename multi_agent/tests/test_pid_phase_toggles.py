"""Tests for once-per-round PID beta routing.

Verifies that _run_round_with_pid() correctly routes β:
    - Propose: _current_beta=None (no tone)
    - Critique: _current_beta=β
    - Revise: _current_beta=β

β stays constant across all three phases within a round. CRIT + PID
update happens once after revise, providing β for the next round.
"""

import pytest
from unittest.mock import MagicMock

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.runner import MultiAgentRunner


def _make_runner(
    initial_beta: float = 0.7,
) -> MultiAgentRunner:
    """Create a mock-mode runner with PID enabled."""
    config = DebateConfig(
        mock=True,
        roles=[AgentRole.MACRO, AgentRole.VALUE],
        max_rounds=2,
        _pid_enabled_flag=True,
        pid_kp=0.15,
        pid_ki=0.0,
        pid_kd=0.0,
        pid_rho_star=0.8,
        initial_beta=initial_beta,
    )
    return MultiAgentRunner(config)


class TestBetaRouting:
    """Verify that each phase gets the correct _current_beta value."""

    def test_propose_has_no_beta(self):
        """Propose always has _current_beta=None (no tone injection)."""
        runner = _make_runner(initial_beta=0.7)

        recorded = {}

        def mock_propose_invoke(state):
            recorded["propose_beta"] = state["config"].get("_current_beta")
            return state

        def mock_critique_invoke(state):
            recorded["critique_beta"] = state["config"].get("_current_beta")
            return state

        def mock_revise_invoke(state):
            recorded["revise_beta"] = state["config"].get("_current_beta")
            return state

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = mock_propose_invoke
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = mock_critique_invoke
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = mock_revise_invoke

        state = {"config": {}}
        runner._run_round_with_pid(state, round_num=1)

        # Propose uses no tone
        assert recorded["propose_beta"] is None

        # Critique + Revise use PID beta (0.7)
        assert recorded["critique_beta"] == pytest.approx(0.7)
        assert recorded["revise_beta"] == pytest.approx(0.7)

    def test_beta_constant_within_round(self):
        """β stays the same across propose/critique/revise in a single round."""
        runner = _make_runner(initial_beta=0.6)

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {}}
        runner._run_round_with_pid(state, round_num=1)

        # Critique and revise should get the same beta
        assert recorded["critique_beta"] == recorded["revise_beta"]


class TestBetaValuePropagation:
    """Verify that the actual PID controller β value is what gets propagated."""

    def test_beta_changes_after_pid_step(self):
        """After a PID step changes β, the new β is used in the next round."""
        runner = _make_runner(initial_beta=0.5)

        # Manually change controller beta to simulate a PID step
        runner._pid_controller.state.beta = 0.85

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {}}
        runner._run_round_with_pid(state, round_num=2)

        # Propose is skipped in round 2+ (uses prior revisions), so no
        # propose_beta is recorded.
        assert "propose_beta" not in recorded
        # Critique + Revise use the updated beta from controller
        assert recorded["critique_beta"] == pytest.approx(0.85)
        assert recorded["revise_beta"] == pytest.approx(0.85)

    def test_beta_at_extremes(self):
        """β at 0.0 and 1.0 extremes are correctly propagated."""
        for beta_val in [0.0, 1.0]:
            runner = _make_runner(initial_beta=0.5)
            runner._pid_controller.state.beta = beta_val

            recorded = {}

            def capture(phase):
                def fn(state):
                    recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                    return state
                return fn

            runner._propose_graph = MagicMock()
            runner._propose_graph.invoke = capture("propose")
            runner._critique_graph = MagicMock()
            runner._critique_graph.invoke = capture("critique")
            runner._revise_graph = MagicMock()
            runner._revise_graph.invoke = capture("revise")

            state = {"config": {}}
            runner._run_round_with_pid(state, round_num=1)

            assert recorded["critique_beta"] == pytest.approx(beta_val)
            assert recorded["revise_beta"] == pytest.approx(beta_val)


class TestToneBucketInteraction:
    """Verify tone bucket mapping gets correct _current_beta."""

    def test_propose_never_gets_tone(self):
        """Propose never gets _current_beta for tone."""
        runner = _make_runner(initial_beta=0.9)

        recorded = {}

        def fn(state):
            recorded["propose_beta"] = state["config"].get("_current_beta")
            return state

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = fn
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = lambda s: s
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = lambda s: s

        state = {"config": {}}
        runner._run_round_with_pid(state, round_num=1)

        # Propose NEVER gets tone beta
        assert recorded["propose_beta"] is None

    def test_critique_and_revise_get_tone(self):
        """Critique and revise get _current_beta for tone injection."""
        runner = _make_runner(initial_beta=0.9)

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {}}
        runner._run_round_with_pid(state, round_num=1)

        assert recorded["critique_beta"] == pytest.approx(0.9)
        assert recorded["revise_beta"] == pytest.approx(0.9)
