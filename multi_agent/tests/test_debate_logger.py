"""Tests for multi_agent.debate_logger.DebateLogger.

Covers directory structure, artifact writing, mode gating, finalization,
and diagnostic artifact generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from multi_agent.debate_logger import DebateLogger, _replace_memo_in_prompt


# ---------------------------------------------------------------------------
# Minimal stub config (mirrors DebateConfig fields DebateLogger reads)
# ---------------------------------------------------------------------------

@dataclass
class _StubPIDGains:
    Kp: float = 0.15
    Ki: float = 0.01
    Kd: float = 0.03


@dataclass
class _StubPIDConfig:
    gains: _StubPIDGains = field(default_factory=_StubPIDGains)
    rho_star: float = 0.8
    gamma_beta: float = 0.5
    epsilon: float = 0.001
    T_max: int = 10
    mu: float = 0.5
    delta_s: float = 0.1
    delta_js: float = 0.1
    delta_beta: float = 0.05


@dataclass
class _StubConfig:
    logging_mode: str = "standard"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    roles: list = field(default_factory=lambda: ["macro", "value"])
    max_rounds: int = 3
    agreeableness: float = 0.3
    initial_beta: float = 0.5
    parallel_agents: bool = True
    pid_config: _StubPIDConfig | None = None
    pid_enabled: bool = False
    convergence_window: int = 2
    delta_rho: float = 0.02

    def to_dict(self) -> dict:
        """Minimal serialisation matching DebateConfig.to_dict()."""
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_logger(tmp_path, monkeypatch):
    """Create a DebateLogger writing to a tmp directory."""
    monkeypatch.chdir(tmp_path)
    config = _StubConfig()
    return DebateLogger(config, "test_experiment")


@pytest.fixture
def tmp_debug_logger(tmp_path, monkeypatch):
    """Create a DebateLogger in debug mode."""
    monkeypatch.chdir(tmp_path)
    config = _StubConfig(logging_mode="debug")
    return DebateLogger(config, "test_experiment")


@pytest.fixture
def tmp_off_logger(tmp_path, monkeypatch):
    """Create a DebateLogger in off mode."""
    monkeypatch.chdir(tmp_path)
    config = _StubConfig(logging_mode="off")
    return DebateLogger(config, "test_experiment")


@pytest.fixture
def sample_observation():
    return {"universe": ["AAPL", "NVDA", "MSFT"], "timestamp": "2025Q1"}


@pytest.fixture
def sample_proposals():
    return [
        {
            "role": "macro",
            "action_dict": {
                "allocation": {"AAPL": 0.3, "NVDA": 0.4, "MSFT": 0.3},
                "justification": "Macro thesis: tech momentum",
                "confidence": 0.75,
            },
            "raw_response": "Full macro reasoning text here...",
        },
        {
            "role": "value",
            "action_dict": {
                "allocation": {"AAPL": 0.4, "NVDA": 0.2, "MSFT": 0.4},
                "justification": "Value thesis: AAPL undervalued",
                "confidence": 0.68,
            },
            "raw_response": "Full value reasoning text here...",
        },
    ]


@pytest.fixture
def sample_critiques():
    return [
        {
            "role": "macro",
            "critiques": [
                {"target_role": "value", "objection": "Over-concentrated in AAPL"},
            ],
            "self_critique": "My macro thesis might be too bullish",
        },
        {
            "role": "value",
            "critiques": [
                {"target_role": "macro", "objection": "Ignoring valuation metrics"},
            ],
            "self_critique": "",
        },
    ]


@pytest.fixture
def sample_revisions():
    return [
        {
            "role": "macro",
            "action_dict": {
                "allocation": {"AAPL": 0.35, "NVDA": 0.35, "MSFT": 0.3},
                "justification": "Revised macro thesis",
                "confidence": 0.80,
            },
            "raw_response": "Revised macro reasoning...",
            "revision_notes": "Incorporated value feedback",
        },
        {
            "role": "value",
            "action_dict": {
                "allocation": {"AAPL": 0.35, "NVDA": 0.25, "MSFT": 0.4},
                "justification": "Revised value thesis",
                "confidence": 0.72,
            },
            "raw_response": "Revised value reasoning...",
            "revision_notes": "Diversified away from AAPL",
        },
    ]


# ---------------------------------------------------------------------------
# Tests: init_run and directory structure
# ---------------------------------------------------------------------------

class TestInitRun:
    def test_creates_directory_tree(self, tmp_logger, sample_observation):
        tmp_logger.init_run("debate-123", sample_observation, "memo content here")
        run_dir = tmp_logger.run_dir

        assert run_dir.exists()
        assert (run_dir / "shared_context").is_dir()
        assert (run_dir / "rounds").is_dir()
        assert (run_dir / "final").is_dir()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "shared_context" / "memo.txt").exists()

    def test_manifest_has_required_fields(self, tmp_logger, sample_observation):
        tmp_logger.init_run("debate-123", sample_observation, "memo")
        manifest = json.loads((tmp_logger.run_dir / "manifest.json").read_text())

        assert manifest["experiment_name"] == "test_experiment"
        assert manifest["debate_id"] == "debate-123"
        assert manifest["started_at"] is not None
        assert manifest["completed_at"] is None  # Not finalized yet
        assert manifest["ticker_universe"] == ["AAPL", "NVDA", "MSFT"]
        assert manifest["model_name"] == "gpt-4o-mini"
        assert manifest["logging_mode"] == "standard"

    def test_manifest_uses_ticker_universe_field(self, tmp_logger, sample_observation):
        tmp_logger.init_run("debate-123", sample_observation, "memo")
        manifest = json.loads((tmp_logger.run_dir / "manifest.json").read_text())
        assert "ticker_universe" in manifest
        assert "universe" not in manifest

    def test_writes_memo(self, tmp_logger, sample_observation):
        memo = "This is the full enriched context memo..."
        tmp_logger.init_run("debate-123", sample_observation, memo)
        memo_path = tmp_logger.run_dir / "shared_context" / "memo.txt"
        assert memo_path.read_text() == memo

    def test_writes_pid_config_when_enabled(self, tmp_path, monkeypatch, sample_observation):
        monkeypatch.chdir(tmp_path)
        pid_cfg = _StubPIDConfig()
        config = _StubConfig(pid_config=pid_cfg, pid_enabled=True)
        logger = DebateLogger(config, "test_pid")
        logger.init_run("debate-pid", sample_observation, "memo")

        pid_path = logger.run_dir / "pid_config.json"
        assert pid_path.exists()
        data = json.loads(pid_path.read_text())
        assert data["Kp"] == 0.15
        assert data["Ki"] == 0.01
        assert data["rho_star"] == 0.8

    def test_no_pid_config_when_disabled(self, tmp_logger, sample_observation):
        tmp_logger.init_run("debate-123", sample_observation, "memo")
        pid_path = tmp_logger.run_dir / "pid_config.json"
        assert not pid_path.exists()

    def test_off_mode_writes_nothing(self, tmp_off_logger, sample_observation):
        tmp_off_logger.init_run("debate-123", sample_observation, "memo")
        # run_dir path is set but nothing should be created
        assert not tmp_off_logger.run_dir.exists()


# ---------------------------------------------------------------------------
# Tests: per-round artifacts
# ---------------------------------------------------------------------------

class TestRoundArtifacts:
    def test_round_dir_three_digit_padding(self, tmp_logger, sample_observation):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        assert (tmp_logger.run_dir / "rounds" / "round_001").is_dir()

        tmp_logger.start_round(10, 0.4)
        assert (tmp_logger.run_dir / "rounds" / "round_010").is_dir()

        tmp_logger.start_round(100, 0.3)
        assert (tmp_logger.run_dir / "rounds" / "round_100").is_dir()

    def test_start_round_creates_subdirs(self, tmp_logger, sample_observation):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        round_dir = tmp_logger.run_dir / "rounds" / "round_001"
        assert (round_dir / "proposals").is_dir()
        assert (round_dir / "critiques").is_dir()
        assert (round_dir / "revisions").is_dir()
        assert (round_dir / "CRIT").is_dir()
        assert (round_dir / "metrics").is_dir()

    def test_write_proposals_creates_per_agent_dirs(
        self, tmp_logger, sample_observation, sample_proposals,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_proposals(sample_proposals)

        round_dir = tmp_logger.run_dir / "rounds" / "round_001"
        macro_dir = round_dir / "proposals" / "macro"
        value_dir = round_dir / "proposals" / "value"

        assert macro_dir.is_dir()
        assert value_dir.is_dir()
        assert (macro_dir / "response.txt").exists()
        assert (macro_dir / "portfolio.json").exists()
        assert (value_dir / "response.txt").exists()
        assert (value_dir / "portfolio.json").exists()

    def test_proposal_response_is_plain_text(
        self, tmp_logger, sample_observation, sample_proposals,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_proposals(sample_proposals)

        response = (tmp_logger.run_dir / "rounds" / "round_001" / "proposals" / "macro" / "response.txt").read_text()
        assert response == "Full macro reasoning text here..."

    def test_proposal_portfolio_is_json(
        self, tmp_logger, sample_observation, sample_proposals,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_proposals(sample_proposals)

        portfolio = json.loads(
            (tmp_logger.run_dir / "rounds" / "round_001" / "proposals" / "macro" / "portfolio.json").read_text()
        )
        assert portfolio == {"AAPL": 0.3, "NVDA": 0.4, "MSFT": 0.3}

    def test_write_critiques_creates_per_agent_json(
        self, tmp_logger, sample_observation, sample_critiques,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_critiques(sample_critiques)

        macro_resp = json.loads(
            (tmp_logger.run_dir / "rounds" / "round_001" / "critiques" / "macro" / "response.json").read_text()
        )
        assert "critiques" in macro_resp
        assert len(macro_resp["critiques"]) == 1
        assert macro_resp["critiques"][0]["target_role"] == "value"
        assert macro_resp["self_critique"] == "My macro thesis might be too bullish"

    def test_write_revisions_creates_per_agent_dirs(
        self, tmp_logger, sample_observation, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_revisions(sample_revisions)

        round_dir = tmp_logger.run_dir / "rounds" / "round_001"
        macro_dir = round_dir / "revisions" / "macro"
        assert (macro_dir / "response.txt").exists()
        assert (macro_dir / "portfolio.json").exists()

        portfolio = json.loads((macro_dir / "portfolio.json").read_text())
        assert portfolio["AAPL"] == 0.35

    def test_round_state_json_has_required_fields(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)

        state = {"proposals": sample_proposals, "revisions": sample_revisions}
        metrics = {"rho_bar": 0.81, "js_divergence": 0.12}
        tmp_logger.write_round_state(state, 1, metrics)

        round_state = json.loads(
            (tmp_logger.run_dir / "rounds" / "round_001" / "round_state.json").read_text()
        )
        assert round_state["round"] == 1
        assert round_state["beta"] == 0.5
        assert "proposals" in round_state
        assert "revisions" in round_state
        assert "metrics" in round_state
        assert round_state["metrics"]["rho_bar"] == 0.81

    def test_write_crit_prompts_creates_per_agent_dirs(
        self, tmp_logger, sample_observation,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)

        captures = {
            "macro": {
                "system_prompt": "You are a blind reasoning auditor.",
                "user_prompt": "Evaluate the following reasoning bundle...",
                "raw_response": '{"pillar_scores": {"LV": 0.85}}',
            },
            "value": {
                "system_prompt": "You are a blind reasoning auditor.",
                "user_prompt": "Evaluate the following value bundle...",
                "raw_response": '{"pillar_scores": {"LV": 0.80}}',
            },
        }
        tmp_logger.write_crit_prompts(captures)

        round_dir = tmp_logger.run_dir / "rounds" / "round_001"
        macro_dir = round_dir / "CRIT" / "macro"
        value_dir = round_dir / "CRIT" / "value"

        assert macro_dir.is_dir()
        assert value_dir.is_dir()
        assert (macro_dir / "prompt.txt").exists()
        assert (macro_dir / "response.txt").exists()
        assert (value_dir / "prompt.txt").exists()
        assert (value_dir / "response.txt").exists()

        # Verify prompt.txt content
        prompt_content = (macro_dir / "prompt.txt").read_text()
        assert "=== SYSTEM PROMPT ===" in prompt_content
        assert "You are a blind reasoning auditor." in prompt_content
        assert "=== USER PROMPT ===" in prompt_content
        assert "Evaluate the following reasoning bundle..." in prompt_content

        # Verify response.txt content
        response_content = (macro_dir / "response.txt").read_text()
        assert '{"pillar_scores": {"LV": 0.85}}' in response_content

    def test_round_state_has_crit_and_pid(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)

        state = {"proposals": sample_proposals, "revisions": sample_revisions}
        metrics = {"rho_bar": 0.81}
        crit_data = {
            "rho_bar": 0.81,
            "macro": {
                "rho_i": 0.75,
                "pillars": {"LV": 1.0, "ES": 0.5, "AC": 0.8, "CA": 0.7},
            },
            "value": {
                "rho_i": 0.87,
                "pillars": {"LV": 0.9, "ES": 0.85, "AC": 0.85, "CA": 0.9},
            },
        }
        pid_data = {
            "beta_in": 0.5,
            "beta_new": 0.45,
            "tone_bucket": "balanced",
            "e_t": -0.04,
            "p_term": -0.006,
            "i_term": -0.0004,
            "d_term": -0.001,
            "u_t": -0.008,
            "quadrant": "converged",
            "sycophancy": 0,
            "convergence": {
                "stable_rounds": 0,
                "delta_rho_actual": None,
                "delta_rho_threshold": 0.02,
            },
        }
        tmp_logger.write_round_state(
            state, 1, metrics, crit_data=crit_data, pid_data=pid_data,
        )

        round_state = json.loads(
            (tmp_logger.run_dir / "rounds" / "round_001" / "round_state.json").read_text()
        )
        # CRIT data present with per-agent pillars
        assert "crit" in round_state
        assert round_state["crit"]["rho_bar"] == 0.81
        assert round_state["crit"]["macro"]["pillars"]["LV"] == 1.0
        assert round_state["crit"]["value"]["rho_i"] == 0.87

        # PID data present with full metrics
        assert "pid" in round_state
        assert round_state["pid"]["beta_in"] == 0.5
        assert round_state["pid"]["beta_new"] == 0.45
        assert round_state["pid"]["quadrant"] == "converged"
        assert round_state["pid"]["e_t"] == -0.04
        assert round_state["pid"]["convergence"]["delta_rho_threshold"] == 0.02


# ---------------------------------------------------------------------------
# Tests: prompt capture (debug vs standard)
# ---------------------------------------------------------------------------

class TestPromptCapture:
    def test_debug_mode_writes_prompts(
        self, tmp_debug_logger, sample_observation,
    ):
        tmp_debug_logger.init_run("d", sample_observation, "m")
        tmp_debug_logger.start_round(1, 0.5)
        tmp_debug_logger.write_prompt(
            "proposals", "macro", "System prompt here", "User prompt here",
        )

        prompt_path = (
            tmp_debug_logger.run_dir / "rounds" / "round_001"
            / "proposals" / "macro" / "prompt.txt"
        )
        assert prompt_path.exists()
        content = prompt_path.read_text()
        assert "=== SYSTEM PROMPT ===" in content
        assert "System prompt here" in content
        assert "=== USER PROMPT ===" in content
        assert "User prompt here" in content

    def test_standard_mode_skips_prompts(
        self, tmp_logger, sample_observation,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_prompt(
            "proposals", "macro", "System prompt", "User prompt",
        )

        prompt_path = (
            tmp_logger.run_dir / "rounds" / "round_001"
            / "proposals" / "macro" / "prompt.txt"
        )
        assert not prompt_path.exists()


# ---------------------------------------------------------------------------
# Tests: finalize
# ---------------------------------------------------------------------------

class TestFinalize:
    def _make_state(self, proposals, revisions):
        return {
            "observation": {"universe": ["AAPL", "NVDA", "MSFT"], "timestamp": "2025Q1"},
            "proposals": proposals,
            "revisions": revisions,
            "critiques": [],
            "debate_turns": [],
            "final_action": {
                "allocation": {"AAPL": 0.3, "NVDA": 0.35, "MSFT": 0.35},
                "justification": "Balanced allocation based on debate consensus",
                "confidence": 0.75,
            },
            "strongest_objection": "NVDA concentration risk",
        }

    def test_finalize_updates_manifest(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        manifest = json.loads((tmp_logger.run_dir / "manifest.json").read_text())
        assert manifest["completed_at"] is not None
        assert manifest["terminated_early"] is False
        assert manifest["termination_reason"] == "max_rounds"

    def test_finalize_writes_final_portfolio(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        portfolio = json.loads(
            (tmp_logger.run_dir / "final" / "final_portfolio.json").read_text()
        )
        assert portfolio["AAPL"] == 0.3
        assert portfolio["NVDA"] == 0.35

    def test_finalize_writes_judge_response(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)
        state["debate_turns"] = [
            {"type": "judge_decision", "raw_response": "Judge full reasoning here..."},
        ]

        tmp_logger.finalize(state, [], False, "memo")

        judge_path = tmp_logger.run_dir / "final" / "judge_response.txt"
        assert judge_path.exists()
        assert judge_path.read_text() == "Judge full reasoning here..."

    def test_finalize_no_judge_response_when_no_judge_turn(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        judge_path = tmp_logger.run_dir / "final" / "judge_response.txt"
        assert not judge_path.exists()

    def test_finalize_writes_pid_crit_all_rounds_json(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        pid_phase_data = [
            {
                "round": 1,
                "crit": {"rho_bar": 0.75, "agents": {"macro": {"rho_i": 0.75}}},
                "pid": {"beta_new": 0.5},
                "divergence": {"js": 0.1},
                "convergence": {},
            },
            {
                "round": 2,
                "crit": {"rho_bar": 0.80, "agents": {"macro": {"rho_i": 0.80}}},
                "pid": {"beta_new": 0.45},
                "divergence": {"js": 0.05},
                "convergence": {},
            },
        ]

        tmp_logger.finalize(state, pid_phase_data, False, "memo")

        path = tmp_logger.run_dir / "final" / "pid_crit_all_rounds.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["round"] == 1
        assert data[1]["round"] == 2
        assert data[1]["crit"]["rho_bar"] == 0.80

    def test_finalize_no_pid_crit_json_when_empty(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        path = tmp_logger.run_dir / "final" / "pid_crit_all_rounds.json"
        assert not path.exists()

    def test_finalize_writes_diagnostic_txt(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "enriched memo content")

        diag_path = tmp_logger.run_dir / "final" / "debate_diagnostic.txt"
        assert diag_path.exists()
        content = diag_path.read_text()
        assert content.startswith("LLM DEBATE DIAGNOSIS SCAFFOLD")
        assert "SECTION 0" in content
        assert "SECTION 1" in content
        assert "SECTION 2" in content

    def test_diagnostic_has_all_sections(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "enriched memo content")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        for i in range(10):  # Sections 0-9
            assert f"SECTION {i}" in content, f"Missing SECTION {i}"

    def test_diagnostic_has_role_prompts(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)
        state["debate_turns"] = [
            {
                "role": "macro", "type": "proposal", "round": 0,
                "raw_system_prompt": "You are a macro strategist.",
                "raw_user_prompt": "usr macro prompt",
                "raw_response": "resp macro",
            },
            {
                "role": "value", "type": "proposal", "round": 0,
                "raw_system_prompt": "You are a value analyst.",
                "raw_user_prompt": "usr value prompt",
                "raw_response": "resp value",
            },
        ]

        tmp_logger.finalize(state, [], False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        # Section 4 — ROLE PROMPTS should have role-specific system prompts
        section4_start = content.find("SECTION 4")
        section5_start = content.find("SECTION 5")
        section4_text = content[section4_start:section5_start]
        assert "MACRO" in section4_text
        assert "VALUE" in section4_text
        assert "You are a macro strategist." in section4_text
        assert "You are a value analyst." in section4_text

    def test_section6_has_crit_explanations(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        pid_phase_data = [
            {
                "round": 1,
                "crit": {
                    "rho_bar": 0.78,
                    "rho_i": {"macro": 0.80, "value": 0.76},
                    "agents": {
                        "macro": {
                            "pillars": {"LV": 0.85, "ES": 0.70, "AC": 0.80, "CA": 0.75},
                            "diagnostics": {"contradictions": False, "unsupported_claims": True,
                                            "conclusion_drift": False, "causal_overreach": False},
                            "explanations": {
                                "logical_validity": "Agent maintains consistent thesis throughout.",
                                "evidential_support": "Two claims lack citation support.",
                                "alternative_consideration": "Allocation follows from stated thesis.",
                                "causal_alignment": "Causal claims appropriately scoped.",
                            },
                        },
                        "value": {
                            "pillars": {"LV": 0.80, "ES": 0.75, "AC": 0.70, "CA": 0.78},
                            "diagnostics": {"contradictions": False, "unsupported_claims": False,
                                            "conclusion_drift": False, "causal_overreach": False},
                            "explanations": {
                                "logical_validity": "Consistent value-based reasoning.",
                                "evidential_support": "All claims well-supported.",
                                "alternative_consideration": "Minor drift in allocation rationale.",
                                "causal_alignment": "Sound causal reasoning.",
                            },
                        },
                    },
                },
                "pid": {},
                "divergence": {},
                "convergence": {},
            }
        ]

        tmp_logger.finalize(state, pid_phase_data, False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        # Section 6 — REASONING QUALITY should have explanation lines
        section6_start = content.find("SECTION 6")
        section7_start = content.find("SECTION 7")
        section6_text = content[section6_start:section7_start]
        assert "LV: Agent maintains consistent thesis throughout." in section6_text
        assert "ES: Two claims lack citation support." in section6_text
        assert "AC: Allocation follows from stated thesis." in section6_text
        assert "CA: Causal claims appropriately scoped." in section6_text
        assert "weakest_pillar:" in section6_text

    def test_section9_has_full_proposals_r1(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)
        state["debate_turns"] = [
            {
                "role": "macro", "type": "proposal", "round": 0,
                "raw_system_prompt": "sys",
                "raw_user_prompt": "usr",
                "raw_response": "Full macro reasoning output here",
            },
            {
                "role": "value", "type": "proposal", "round": 0,
                "raw_system_prompt": "sys",
                "raw_user_prompt": "usr",
                "raw_response": "Full value reasoning output here",
            },
        ]

        tmp_logger.finalize(state, [], False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        section9_start = content.find("SECTION 9")
        section9_text = content[section9_start:]
        assert "=== MACRO ===" in section9_text
        assert "=== VALUE ===" in section9_text
        assert "Full macro reasoning output here" in section9_text
        assert "Full value reasoning output here" in section9_text

    def test_section9_empty_when_no_proposals(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        section9_start = content.find("SECTION 9")
        section9_text = content[section9_start:]
        assert "(no proposal turns found)" in section9_text

    def test_diagnostic_meta_has_agents_field(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        section0_start = content.find("SECTION 0")
        section1_start = content.find("SECTION 1")
        section0_text = content[section0_start:section1_start]
        assert "agents:" in section0_text

    def test_diagnostic_has_shared_prompt_contract(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        tmp_logger.init_run("d", sample_observation, "m")
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, "memo")

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        assert "SHARED PROMPT CONTRACT" in content


# ---------------------------------------------------------------------------
# Tests: crash safety — partial rounds preserved
# ---------------------------------------------------------------------------

class TestCrashSafety:
    def test_partial_rounds_preserved(
        self, tmp_logger, sample_observation, sample_proposals,
    ):
        tmp_logger.init_run("d", sample_observation, "m")

        # Round 1 — complete
        tmp_logger.start_round(1, 0.5)
        tmp_logger.write_proposals(sample_proposals)

        # Round 2 — started but not finished (simulating crash)
        tmp_logger.start_round(2, 0.4)

        # Verify round 1 artifacts survive
        round1 = tmp_logger.run_dir / "rounds" / "round_001"
        assert (round1 / "proposals" / "macro" / "response.txt").exists()
        assert (round1 / "proposals" / "macro" / "portfolio.json").exists()

        # Round 2 dir exists but has no proposals
        round2 = tmp_logger.run_dir / "rounds" / "round_002"
        assert round2.is_dir()
        assert not (round2 / "proposals" / "macro").exists()


# ---------------------------------------------------------------------------
# Tests: memo placeholder in diagnostic
# ---------------------------------------------------------------------------

class TestMemoPlaceholder:
    def test_replace_memo_with_marker(self):
        memo = "x" * 200  # Must be >= 100 chars to trigger replacement
        prompt = f"Before\n[INFO] QUARTERLY SNAPSHOT MEMO\n{memo}\nUsing the data above, analyze..."
        result = _replace_memo_in_prompt(prompt, memo)
        assert "<<MEMO CONTENT INSERTED HERE>>" in result
        assert memo not in result
        assert "Using the data above" in result

    def test_no_marker_returns_unchanged(self):
        prompt = "No memo marker present in this prompt"
        result = _replace_memo_in_prompt(prompt, "short")
        assert result == prompt

    def test_short_context_returns_unchanged(self):
        prompt = "Some prompt text"
        result = _replace_memo_in_prompt(prompt, "short")
        assert result == prompt

    def test_diagnostic_memo_appears_once_in_section2(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        memo_text = "[INFO] QUARTERLY SNAPSHOT MEMO\nThis is the full memo content " + "x" * 200
        tmp_logger.init_run("d", sample_observation, memo_text)
        tmp_logger.start_round(1, 0.5)

        state = {
            "observation": sample_observation,
            "proposals": sample_proposals,
            "revisions": sample_revisions,
            "critiques": [],
            "debate_turns": [
                {
                    "role": "macro", "type": "proposal", "round": 0,
                    "raw_system_prompt": "system",
                    "raw_user_prompt": f"Before\n{memo_text}\nUsing the data above, do stuff",
                    "raw_response": "response",
                    "content": {"allocation": {}},
                },
            ],
            "final_action": {"allocation": {}, "justification": "", "confidence": 0.5},
            "strongest_objection": "",
        }

        tmp_logger.finalize(state, [], False, memo_text)

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        # Section 2 should contain the memo
        assert "SHARED INVESTMENT MEMO" in content
        assert "END MEMO" in content
        # The full memo text should appear exactly once (in Section 2)
        assert content.count("[INFO] QUARTERLY SNAPSHOT MEMO") == 1


# ---------------------------------------------------------------------------
# Tests: prompt manifest
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: memo trimming for token budget
# ---------------------------------------------------------------------------

class TestMemoTrimming:
    def _make_state(self, proposals, revisions):
        return {
            "observation": {"universe": ["AAPL", "NVDA", "MSFT"], "timestamp": "2025Q1"},
            "proposals": proposals,
            "revisions": revisions,
            "critiques": [],
            "debate_turns": [],
            "final_action": {
                "allocation": {"AAPL": 0.3, "NVDA": 0.35, "MSFT": 0.35},
                "justification": "Balanced allocation",
                "confidence": 0.75,
            },
            "strongest_objection": "",
        }

    def test_memo_trimmed_when_over_token_limit(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
        monkeypatch,
    ):
        """When the diagnostic exceeds the token budget, the memo is trimmed."""
        import multi_agent.debate_logger as dl_module

        # Set a very low token limit to force trimming
        monkeypatch.setattr(dl_module, "_DIAGNOSTIC_MAX_TOKENS", 100)

        large_memo = "[INFO] QUARTERLY SNAPSHOT MEMO\n" + "x" * 5000
        tmp_logger.init_run("d", sample_observation, large_memo)
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, large_memo)

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        assert "MEMO TRIMMED" in content
        assert "shared_context/memo.txt" in content
        # The full 5000-char block should NOT appear
        assert "x" * 5000 not in content

    def test_memo_not_trimmed_when_under_limit(
        self, tmp_logger, sample_observation, sample_proposals, sample_revisions,
    ):
        """When the diagnostic is under the token budget, the memo stays intact."""
        short_memo = "[INFO] QUARTERLY SNAPSHOT MEMO\nShort memo content."
        tmp_logger.init_run("d", sample_observation, short_memo)
        tmp_logger.start_round(1, 0.5)
        state = self._make_state(sample_proposals, sample_revisions)

        tmp_logger.finalize(state, [], False, short_memo)

        content = (tmp_logger.run_dir / "final" / "debate_diagnostic.txt").read_text()
        assert "MEMO TRIMMED" not in content
        assert "Short memo content." in content


# ---------------------------------------------------------------------------
# Tests: prompt manifest
# ---------------------------------------------------------------------------

class TestPromptManifest:
    def test_writes_prompt_manifest(self, tmp_logger, sample_observation):
        tmp_logger.init_run("d", sample_observation, "m")
        manifest = {"block_order": ["causal_contract", "role_system"], "role_files": {}}
        tmp_logger.write_prompt_manifest(manifest)

        path = tmp_logger.run_dir / "prompt_manifest.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["block_order"] == ["causal_contract", "role_system"]
