"""Tests for per-phase PID toggles (pid_propose, pid_critique, pid_revise).

Verifies that the per-phase toggle flags in DebateConfig correctly control
which phases get PID-adjusted agreeableness (β) vs. the original static value.

The runner's _run_round_with_pid() method checks each flag:
    - pid_propose=False (default): propose uses original agreeableness, beta=None
    - pid_critique=True (default):  critique uses PID β, _current_beta=β
    - pid_revise=True (default):    revise uses PID β, _current_beta=β

RAudit references:
    - Section 3.5 (p.4): Per-phase PID control toggles
    - Table 1: β modulates critique/revise tone but not propose
"""

import pytest
from unittest.mock import MagicMock, patch, call

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.runner import MultiAgentRunner


def _make_runner(
    pid_propose: bool = False,
    pid_critique: bool = True,
    pid_revise: bool = True,
    agreeableness: float = 0.3,
    initial_beta: float = 0.7,
) -> MultiAgentRunner:
    """Create a mock-mode runner with PID enabled and specific toggle flags."""
    config = DebateConfig(
        mock=True,
        roles=[AgentRole.MACRO, AgentRole.VALUE],
        max_rounds=2,
        agreeableness=agreeableness,
        _pid_enabled_flag=True,
        pid_kp=0.15,
        pid_ki=0.0,
        pid_kd=0.0,
        pid_rho_star=0.8,
        initial_beta=initial_beta,
        pid_propose=pid_propose,
        pid_critique=pid_critique,
        pid_revise=pid_revise,
    )
    return MultiAgentRunner(config)


class TestPerPhaseAgreeablenessRouting:
    """Verify that each phase gets the correct agreeableness value."""

    def test_default_toggles_propose_uses_original(self):
        """Default: pid_propose=False → propose gets original agreeableness."""
        runner = _make_runner(pid_propose=False, agreeableness=0.3, initial_beta=0.7)

        # Capture what agreeableness each phase receives
        recorded = {}

        def mock_propose_invoke(state):
            recorded["propose_a"] = state["config"]["agreeableness"]
            recorded["propose_beta"] = state["config"].get("_current_beta")
            return state

        def mock_critique_invoke(state):
            recorded["critique_a"] = state["config"]["agreeableness"]
            recorded["critique_beta"] = state["config"].get("_current_beta")
            return state

        def mock_revise_invoke(state):
            recorded["revise_a"] = state["config"]["agreeableness"]
            recorded["revise_beta"] = state["config"].get("_current_beta")
            return state

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = mock_propose_invoke
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = mock_critique_invoke
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = mock_revise_invoke

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # Propose should use ORIGINAL agreeableness (0.3), not PID beta (0.7)
        assert recorded["propose_a"] == pytest.approx(0.3)
        assert recorded["propose_beta"] is None  # No tone for propose

        # Critique + Revise should use PID beta (0.7)
        assert recorded["critique_a"] == pytest.approx(0.7)
        assert recorded["critique_beta"] == pytest.approx(0.7)
        assert recorded["revise_a"] == pytest.approx(0.7)
        assert recorded["revise_beta"] == pytest.approx(0.7)

    def test_all_toggles_on(self):
        """pid_propose=True → ALL phases get PID beta."""
        runner = _make_runner(
            pid_propose=True, pid_critique=True, pid_revise=True,
            agreeableness=0.3, initial_beta=0.7,
        )

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # ALL phases should use PID beta
        assert recorded["propose_a"] == pytest.approx(0.7)
        # Propose still gets _current_beta=None (tone is never injected for propose)
        assert recorded["propose_beta"] is None
        assert recorded["critique_a"] == pytest.approx(0.7)
        assert recorded["critique_beta"] == pytest.approx(0.7)
        assert recorded["revise_a"] == pytest.approx(0.7)
        assert recorded["revise_beta"] == pytest.approx(0.7)

    def test_all_toggles_off(self):
        """All toggles off → ALL phases get original agreeableness."""
        runner = _make_runner(
            pid_propose=False, pid_critique=False, pid_revise=False,
            agreeableness=0.3, initial_beta=0.7,
        )

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # ALL phases should use original agreeableness
        assert recorded["propose_a"] == pytest.approx(0.3)
        assert recorded["propose_beta"] is None
        assert recorded["critique_a"] == pytest.approx(0.3)
        assert recorded["critique_beta"] is None  # pid_critique=False → no tone
        assert recorded["revise_a"] == pytest.approx(0.3)
        assert recorded["revise_beta"] is None  # pid_revise=False → no tone

    def test_only_critique_on(self):
        """Only pid_critique=True → only critique gets PID beta."""
        runner = _make_runner(
            pid_propose=False, pid_critique=True, pid_revise=False,
            agreeableness=0.3, initial_beta=0.7,
        )

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        assert recorded["propose_a"] == pytest.approx(0.3)
        assert recorded["propose_beta"] is None
        assert recorded["critique_a"] == pytest.approx(0.7)
        assert recorded["critique_beta"] == pytest.approx(0.7)
        assert recorded["revise_a"] == pytest.approx(0.3)
        assert recorded["revise_beta"] is None

    def test_only_revise_on(self):
        """Only pid_revise=True → only revise gets PID beta."""
        runner = _make_runner(
            pid_propose=False, pid_critique=False, pid_revise=True,
            agreeableness=0.3, initial_beta=0.7,
        )

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        assert recorded["propose_a"] == pytest.approx(0.3)
        assert recorded["propose_beta"] is None
        assert recorded["critique_a"] == pytest.approx(0.3)
        assert recorded["critique_beta"] is None
        assert recorded["revise_a"] == pytest.approx(0.7)
        assert recorded["revise_beta"] == pytest.approx(0.7)


class TestBetaValuePropagation:
    """Verify that the actual PID controller β value is what gets propagated."""

    def test_beta_changes_after_pid_step(self):
        """After a PID step changes β, the new β is used in the next round."""
        runner = _make_runner(agreeableness=0.3, initial_beta=0.5)

        # Manually change controller beta to simulate a PID step
        runner._pid_controller.state.beta = 0.85

        recorded = {}

        def capture(phase):
            def fn(state):
                recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                return state
            return fn

        runner._propose_graph = MagicMock()
        runner._propose_graph.invoke = capture("propose")
        runner._critique_graph = MagicMock()
        runner._critique_graph.invoke = capture("critique")
        runner._revise_graph = MagicMock()
        runner._revise_graph.invoke = capture("revise")

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=2)

        # Propose uses original (pid_propose=False by default)
        assert recorded["propose_a"] == pytest.approx(0.3)
        # Critique + Revise use the updated beta from controller
        assert recorded["critique_a"] == pytest.approx(0.85)
        assert recorded["revise_a"] == pytest.approx(0.85)

    def test_beta_at_extremes(self):
        """β at 0.0 and 1.0 extremes are correctly propagated."""
        for beta_val in [0.0, 1.0]:
            runner = _make_runner(agreeableness=0.3, initial_beta=0.5)
            runner._pid_controller.state.beta = beta_val

            recorded = {}

            def capture(phase):
                def fn(state):
                    recorded[f"{phase}_a"] = state["config"]["agreeableness"]
                    recorded[f"{phase}_beta"] = state["config"].get("_current_beta")
                    return state
                return fn

            runner._propose_graph = MagicMock()
            runner._propose_graph.invoke = capture("propose")
            runner._critique_graph = MagicMock()
            runner._critique_graph.invoke = capture("critique")
            runner._revise_graph = MagicMock()
            runner._revise_graph.invoke = capture("revise")

            state = {"config": {"agreeableness": 0.3}}
            runner._run_round_with_pid(state, round_num=1)

            assert recorded["critique_a"] == pytest.approx(beta_val)
            assert recorded["critique_beta"] == pytest.approx(beta_val)
            assert recorded["revise_a"] == pytest.approx(beta_val)
            assert recorded["revise_beta"] == pytest.approx(beta_val)


class TestToggleWithToneBucketInteraction:
    """Verify toggle flags interact correctly with tone bucket mapping."""

    def test_critique_off_means_no_tone_bucket(self):
        """pid_critique=False → _current_beta=None → no tone injected."""
        runner = _make_runner(pid_critique=False, initial_beta=0.9)

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

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # With pid_critique=False, _current_beta should be None
        # even though PID beta is 0.9
        assert recorded["critique_beta"] is None

    def test_revise_off_means_no_tone_bucket(self):
        """pid_revise=False → _current_beta=None → no tone injected for revise."""
        runner = _make_runner(pid_revise=False, initial_beta=0.9)

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

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # Revise should NOT get beta for tone when toggle is off
        assert recorded["revise_beta"] is None
        # But critique still should (pid_critique defaults True)
        assert recorded["critique_beta"] == pytest.approx(0.9)

    def test_propose_never_gets_tone_even_when_toggle_on(self):
        """Propose never gets _current_beta for tone, even with pid_propose=True.

        pid_propose only controls agreeableness, not tone injection.
        Propose phase is tone-free by design (RAudit Section 3.5).
        """
        runner = _make_runner(pid_propose=True, initial_beta=0.9)

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

        state = {"config": {"agreeableness": 0.3}}
        runner._run_round_with_pid(state, round_num=1)

        # Propose NEVER gets tone beta
        assert recorded["propose_beta"] is None
