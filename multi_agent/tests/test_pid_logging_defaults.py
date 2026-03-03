"""Tests for PID metrics logging defaults.

PID metrics must be logged at INFO level and enabled by default so that
PID output is always visible during simulation runs.  These tests guard
against regressions that would silently hide PID diagnostics.
"""

import logging

import pytest

from models.config import AgentConfig
from multi_agent.config import DebateConfig, AgentRole
from multi_agent.runner import pid_metrics_logger, MultiAgentRunner


# ── config defaults ──────────────────────────────────────────────────────────


class TestPidLogMetricsDefault:
    """pid_log_metrics must default to True in both AgentConfig and DebateConfig."""

    def test_agent_config_default_is_true(self):
        cfg = AgentConfig(
            agent_system="multi_agent_debate",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            temperature=0.3,
        )
        assert cfg.pid_log_metrics is True

    def test_debate_config_default_is_true(self):
        cfg = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE],
        )
        assert cfg.pid_log_metrics is True

    def test_explicit_false_still_works(self):
        cfg = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            pid_log_metrics=False,
        )
        assert cfg.pid_log_metrics is False

    def test_runner_inherits_default(self):
        cfg = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE],
        )
        runner = MultiAgentRunner(cfg)
        assert runner._log_metrics is True

    def test_runner_respects_explicit_false(self):
        cfg = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            pid_log_metrics=False,
        )
        runner = MultiAgentRunner(cfg)
        assert runner._log_metrics is False


# ── logger level ─────────────────────────────────────────────────────────────


class TestPidMetricsLoggerLevel:
    """pid.metrics logger must emit at INFO, not DEBUG."""

    def test_logger_name(self):
        assert pid_metrics_logger.name == "pid.metrics"

    def test_info_is_enabled(self):
        """INFO messages must not be filtered out by the logger's own level."""
        assert pid_metrics_logger.isEnabledFor(logging.INFO)

    def test_info_not_filtered_at_default_level(self):
        """The effective level should allow INFO through."""
        effective = pid_metrics_logger.getEffectiveLevel()
        assert effective <= logging.INFO


# ── log calls use INFO ───────────────────────────────────────────────────────


class TestPidMetricsCallsUseInfo:
    """Verify that runner code calls pid_metrics_logger.info(), not .debug().

    We grep the source to ensure no debug() calls sneak back in.
    """

    def test_no_debug_calls_on_pid_metrics_logger(self):
        """pid_metrics_logger.debug must not appear in runner.py."""
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "pid_metrics_logger.debug" not in source, (
            "pid_metrics_logger.debug found in runner.py — "
            "PID metrics must use .info() so output is visible by default"
        )

    def test_info_calls_exist(self):
        """pid_metrics_logger.info must appear in runner.py (sanity check)."""
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "pid_metrics_logger.info" in source
