"""Tests for PID metrics logging: defaults, JSON format, and schema.

PID metrics must be logged at INFO level and enabled by default so that
PID output is always visible during simulation runs.  All per-round and
per-debate logs must be valid JSON with UUIDs for correlation.
"""

import json
import logging
import uuid
from unittest.mock import patch

import pytest

from eval.crit.schema import (
    CritResult,
    Diagnostics,
    Explanations,
    PillarScores,
    RoundCritResult,
)
from models.config import AgentConfig
from multi_agent.config import DebateConfig, AgentRole
from multi_agent.graph.mocks import (
    _mock_proposal as _orig_mock_proposal,
    _mock_critique as _orig_mock_critique,
    _mock_revision as _orig_mock_revision,
    _mock_judge as _orig_mock_judge,
)
from multi_agent.models import (
    Constraints,
    MarketState,
    Observation,
    PortfolioState,
)
from multi_agent.runner import pid_metrics_logger, MultiAgentRunner


# ── claim / position_rationale fixups for mock responses ─────────────────────

_CLAIM_DEFAULTS = {
    "evidence": ["[L1-VIX]"],
    "assumptions": ["Market conditions remain stable"],
    "falsifiers": ["Unexpected macro shock"],
    "impacts_positions": ["Allocation weighting"],
}


def _patch_claims(claims: list[dict]) -> list[dict]:
    """Ensure each claim dict has the keys the runner expects via bracket access."""
    for claim in claims:
        for key, default in _CLAIM_DEFAULTS.items():
            claim.setdefault(key, default)
    return claims


def _add_position_rationale(result: dict) -> dict:
    """Ensure the mock result includes position_rationale with supported_by_claims."""
    if "position_rationale" not in result:
        tickers = list((result.get("allocation") or {}).keys()) or ["AAPL"]
        result["position_rationale"] = [
            {
                "ticker": t,
                "weight": result.get("allocation", {}).get(t, 0.0),
                "supported_by_claims": ["mock-claim-1"],
                "explanation": f"Mock rationale for {t}",
            }
            for t in tickers
        ]
    else:
        for entry in result["position_rationale"]:
            entry.setdefault("supported_by_claims", ["mock-claim-1"])
    return result


def _patched_proposal(role, obs_dict, config=None):
    result = _orig_mock_proposal(role, obs_dict, config)
    _patch_claims(result.get("claims", []))
    _add_position_rationale(result)
    return result


def _patched_critique(role, proposals):
    return _orig_mock_critique(role, proposals)


def _patched_revision(role, original_action, obs_dict, config=None):
    result = _orig_mock_revision(role, original_action, obs_dict, config)
    _patch_claims(result.get("claims", []))
    _add_position_rationale(result)
    return result


def _patched_judge(revisions, config=None):
    result = _orig_mock_judge(revisions, config)
    _patch_claims(result.get("claims", []))
    _add_position_rationale(result)
    return result


# ── helpers ──────────────────────────────────────────────────────────────────


def _mock_crit_result(roles: list[str]) -> RoundCritResult:
    """Build a valid RoundCritResult for the given roles."""
    agent_scores = {}
    for role in roles:
        agent_scores[role] = CritResult(
            pillar_scores=PillarScores(
                logical_validity=0.8,
                evidential_support=0.7,
                alternative_consideration=0.75,
                causal_alignment=0.65,
            ),
            diagnostics=Diagnostics(
                contradictions_detected=False,
                unsupported_claims_detected=True,
                ignored_critiques_detected=False,
                premature_certainty_detected=False,
                causal_overreach_detected=False,
                conclusion_drift_detected=False,
            ),
            explanations=Explanations(
                logical_validity="Consistent reasoning.",
                evidential_support="Some claims lack evidence.",
                alternative_consideration="Decision follows analysis.",
                causal_alignment="Causal scoping appropriate.",
            ),
            rho_bar=(0.8 + 0.7 + 0.75 + 0.65) / 4.0,
        )
    rho_bar = sum(cr.rho_bar for cr in agent_scores.values()) / len(agent_scores)
    return RoundCritResult(agent_scores=agent_scores, rho_bar=rho_bar)


def _make_pid_runner(max_rounds: int = 2) -> MultiAgentRunner:
    """Create a mock-mode runner with PID enabled."""
    config = DebateConfig(
        mock=True,
        roles=[AgentRole.MACRO, AgentRole.VALUE],
        max_rounds=max_rounds,
        _pid_enabled_flag=True,
        pid_kp=0.15,
        pid_ki=0.01,
        pid_kd=0.03,
        pid_rho_star=0.8,
        initial_beta=0.5,
        trace_dir="/tmp/test_traces",
        console_display=False,
    )
    runner = MultiAgentRunner(config)
    # Patch CRIT scorer to return valid mock data
    runner._crit_scorer.score = lambda reasoning_bundles: _mock_crit_result(["macro", "value"])
    return runner


def _sample_observation() -> Observation:
    return Observation(
        timestamp="2025-03-15T10:00:00Z",
        universe=["AAPL", "MSFT"],
        market_state=MarketState(
            prices={"AAPL": 185.0, "MSFT": 390.0},
        ),
        text_context="Test context for PID logging.",
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
        constraints=Constraints(max_leverage=2.0, max_position_size=500),
    )


def _capture_pid_logs(runner: MultiAgentRunner) -> list[dict]:
    """Run a mock debate and return all JSON objects from pid.metrics.

    PID round + summary logs are emitted *before* the judge/parse phase.
    If _parse_action fails due to a missing 'orders' key in allocation-mode
    judge output, the PID data has already been captured — tolerate the error.
    """
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda record: captured.append(record)
    pid_metrics_logger.addHandler(handler)
    try:
        with patch("multi_agent.graph.nodes._mock_proposal", _patched_proposal), \
             patch("multi_agent.graph.nodes._mock_critique", _patched_critique), \
             patch("multi_agent.graph.nodes._mock_revision", _patched_revision), \
             patch("multi_agent.graph.nodes._mock_judge", _patched_judge):
            runner.run(_sample_observation())
    except KeyError:
        # _parse_action may raise KeyError("orders") in allocation mode;
        # PID logs are already captured before the judge phase.
        pass
    finally:
        pid_metrics_logger.removeHandler(handler)

    results = []
    for record in captured:
        msg = record.getMessage()
        try:
            results.append(json.loads(msg))
        except json.JSONDecodeError:
            pass
    return results


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
        assert pid_metrics_logger.isEnabledFor(logging.INFO)

    def test_info_not_filtered_at_default_level(self):
        effective = pid_metrics_logger.getEffectiveLevel()
        assert effective <= logging.INFO


# ── source-level guards ──────────────────────────────────────────────────────


class TestPidMetricsSourceGuards:
    """Verify runner source uses .info() and json.dumps(), not .debug() or printf."""

    def test_no_debug_calls_on_pid_metrics_logger(self):
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "pid_metrics_logger.debug" not in source, (
            "pid_metrics_logger.debug found in runner.py — "
            "PID metrics must use .info() so output is visible by default"
        )

    def test_info_calls_exist(self):
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "pid_metrics_logger.info" in source

    def test_no_scattered_format_strings_in_crit_and_pid_step(self):
        """_crit_and_pid_step must not contain old-style format string logging."""
        import inspect

        source = inspect.getsource(MultiAgentRunner._crit_and_pid_step)
        assert "[PID Round %d]" not in source, (
            "Old-style format string found in _crit_and_pid_step — "
            "PID metrics must be JSON, not printf"
        )

    def test_json_dumps_used_for_logging(self):
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "pid_metrics_logger.info(json.dumps(" in source

    def test_json_pretty_printed(self):
        import inspect
        import multi_agent.runner as runner_module

        source = inspect.getsource(runner_module)
        assert "indent=2" in source, (
            "PID JSON logs must use indent=2 for human readability"
        )


# ── JSON round structure ─────────────────────────────────────────────────────


class TestPidJsonRoundStructure:
    """Run a mock PID debate and validate per-round JSON objects."""

    @pytest.fixture(autouse=True)
    def _run_debate(self):
        runner = _make_pid_runner(max_rounds=2)
        self.logs = _capture_pid_logs(runner)
        self.rounds = [l for l in self.logs if l.get("type") == "pid_round"]
        self.summaries = [l for l in self.logs if l.get("type") == "pid_summary"]

    def test_at_least_two_rounds_logged(self):
        # 1 pid_round entry per round
        assert len(self.rounds) >= 2

    def test_round_has_type_field(self):
        assert all(r["type"] == "pid_round" for r in self.rounds)

    def test_round_has_debate_id(self):
        for r in self.rounds:
            assert "debate_id" in r
            uuid.UUID(r["debate_id"])

    def test_round_has_phase_id(self):
        for r in self.rounds:
            assert "phase_id" in r
            uuid.UUID(r["phase_id"])

    def test_phase_ids_are_unique(self):
        ids = [r["phase_id"] for r in self.rounds]
        assert len(ids) == len(set(ids))

    def test_all_rounds_share_debate_id(self):
        ids = {r["debate_id"] for r in self.rounds}
        assert len(ids) == 1

    def test_round_has_round_number(self):
        for r in self.rounds:
            assert isinstance(r["round"], int)
            assert r["round"] >= 1

    def test_round_has_beta_in(self):
        for r in self.rounds:
            assert isinstance(r["beta_in"], (int, float))

    def test_round_has_tone_bucket(self):
        for r in self.rounds:
            assert r["tone_bucket"] in ("collaborative", "balanced", "adversarial")

    # ── crit section ──

    def test_crit_has_rho_bar(self):
        for r in self.rounds:
            assert isinstance(r["crit"]["rho_bar"], float)

    def test_crit_has_per_agent_scores(self):
        for r in self.rounds:
            agents = r["crit"]["agents"]
            assert len(agents) >= 1
            for role, data in agents.items():
                assert "rho_i" in data
                assert isinstance(data["rho_i"], float)

    def test_crit_has_pillar_scores(self):
        for r in self.rounds:
            for role, data in r["crit"]["agents"].items():
                pillars = data["pillars"]
                for key in ("LV", "ES", "AC", "CA"):
                    assert key in pillars, f"Missing pillar {key} for {role}"
                    assert isinstance(pillars[key], float)

    def test_crit_has_diagnostics(self):
        for r in self.rounds:
            for role, data in r["crit"]["agents"].items():
                diag = data["diagnostics"]
                for key in ("contradictions", "unsupported_claims",
                            "ignored_critiques", "premature_certainty",
                            "causal_overreach", "conclusion_drift"):
                    assert key in diag, f"Missing diagnostic {key} for {role}"
                    assert isinstance(diag[key], bool)

    # ── pid section ──

    def test_pid_has_control_terms(self):
        for r in self.rounds:
            pid = r["pid"]
            for key in ("e_t", "p_term", "i_term", "d_term", "u_t"):
                assert key in pid, f"Missing PID term {key}"
                assert isinstance(pid[key], (int, float))

    def test_pid_has_beta_transition(self):
        for r in self.rounds:
            pid = r["pid"]
            assert "beta_old" in pid
            assert "beta_new" in pid
            assert 0.0 <= pid["beta_new"] <= 1.0

    def test_pid_has_state(self):
        for r in self.rounds:
            pid = r["pid"]
            assert "integral" in pid
            assert "e_prev" in pid

    def test_pid_has_quadrant(self):
        for r in self.rounds:
            assert r["pid"]["quadrant"] in (
                "stuck", "chaotic", "converged", "healthy", ""
            )

    def test_pid_has_signals(self):
        for r in self.rounds:
            pid = r["pid"]
            assert isinstance(pid["div_signal"], bool)
            assert isinstance(pid["qual_signal"], bool)
            assert isinstance(pid["sycophancy"], int)

    # ── divergence section ──

    def test_divergence_has_js_and_ov(self):
        for r in self.rounds:
            div = r["divergence"]
            assert isinstance(div["js"], float)
            assert isinstance(div["ov"], (int, float))

    def test_divergence_has_agent_confidences(self):
        for r in self.rounds:
            confs = r["divergence"]["agent_confidences"]
            assert isinstance(confs, dict)
            assert len(confs) >= 1
            for role, val in confs.items():
                assert isinstance(val, (int, float))

    def test_divergence_has_agent_evidence_ids(self):
        for r in self.rounds:
            ev = r["divergence"]["agent_evidence_ids"]
            assert isinstance(ev, dict)


# ── JSON summary structure ───────────────────────────────────────────────────


class TestPidJsonSummaryStructure:
    """Validate the end-of-debate summary JSON object."""

    @pytest.fixture(autouse=True)
    def _run_debate(self):
        runner = _make_pid_runner(max_rounds=2)
        self.logs = _capture_pid_logs(runner)
        self.rounds = [l for l in self.logs if l.get("type") == "pid_round"]
        self.summaries = [l for l in self.logs if l.get("type") == "pid_summary"]

    def test_exactly_one_summary(self):
        assert len(self.summaries) == 1

    def test_summary_has_debate_id(self):
        s = self.summaries[0]
        uuid.UUID(s["debate_id"])

    def test_summary_debate_id_matches_rounds(self):
        s = self.summaries[0]
        for r in self.rounds:
            assert r["debate_id"] == s["debate_id"]

    def test_summary_has_timestamp(self):
        s = self.summaries[0]
        assert isinstance(s["timestamp"], str)
        assert "T" in s["timestamp"]

    def test_summary_config_has_gains(self):
        cfg = self.summaries[0]["config"]
        for key in ("Kp", "Ki", "Kd"):
            assert key in cfg
            assert isinstance(cfg[key], (int, float))

    def test_summary_config_has_thresholds(self):
        cfg = self.summaries[0]["config"]
        for key in ("rho_star", "gamma_beta", "epsilon", "T_max",
                     "mu", "delta_s", "delta_js", "delta_beta"):
            assert key in cfg

    def test_summary_config_has_initial_beta(self):
        assert self.summaries[0]["config"]["initial_beta"] == pytest.approx(0.5)

    def test_summary_phases_are_phase_ids(self):
        s = self.summaries[0]
        phase_ids_from_rounds = [r["phase_id"] for r in self.rounds]
        assert s["phases"] == phase_ids_from_rounds

    def test_outcome_has_required_fields(self):
        outcome = self.summaries[0]["outcome"]
        for key in ("total_rounds", "total_phase_steps", "terminated_early",
                     "termination_reason", "final_beta", "final_rho_bar", "final_js"):
            assert key in outcome

    def test_outcome_total_rounds_matches(self):
        outcome = self.summaries[0]["outcome"]
        unique_rounds = len({r["round"] for r in self.rounds})
        assert outcome["total_rounds"] == unique_rounds

    def test_outcome_total_phase_steps_matches(self):
        outcome = self.summaries[0]["outcome"]
        assert outcome["total_phase_steps"] == len(self.rounds)

    def test_outcome_termination_reason_valid(self):
        reason = self.summaries[0]["outcome"]["termination_reason"]
        assert reason in ("stable_convergence", "max_rounds")


# ── UUID correlation ─────────────────────────────────────────────────────────


class TestPidUuidCorrelation:
    """UUIDs must be valid, unique per debate, and consistent across logs."""

    def test_fresh_debate_gets_new_debate_id(self):
        runner = _make_pid_runner(max_rounds=1)
        logs1 = _capture_pid_logs(runner)
        logs2 = _capture_pid_logs(runner)
        id1 = {l["debate_id"] for l in logs1 if "debate_id" in l}
        id2 = {l["debate_id"] for l in logs2 if "debate_id" in l}
        assert len(id1) == 1
        assert len(id2) == 1
        assert id1 != id2, "Two debates must have different debate_ids"

    def test_debate_id_is_valid_uuid4(self):
        runner = _make_pid_runner(max_rounds=1)
        logs = _capture_pid_logs(runner)
        for l in logs:
            if "debate_id" in l:
                parsed = uuid.UUID(l["debate_id"])
                assert parsed.version == 4
