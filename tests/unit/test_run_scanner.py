"""Unit tests for the run_scanner module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tools.prompt_viewer.run_scanner import (
    _get_run_status,
    _load_trajectories,
    _extract_quality_metrics,
    list_experiments,
    list_runs,
    get_run_detail,
    get_pid_trajectory,
    get_crit_trajectory,
    get_portfolio_trajectory,
    get_round_detail,
    get_file_tree,
    read_run_file,
    diff_run_configs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


SAMPLE_MANIFEST = {
    "experiment_name": "default",
    "run_id": "run_2026-01-01_00-00-00",
    "started_at": "2026-01-01T00:00:00+00:00",
    "completed_at": "2026-01-01T01:00:00+00:00",
    "model_name": "gpt-5-mini",
    "roles": ["macro", "value"],
    "actual_rounds": 3,
    "max_rounds": 5,
    "terminated_early": False,
    "termination_reason": "max_rounds",
    "pid_enabled": True,
    "invest_quarter": "2025Q1",
    "initial_beta": 0.4,
    "final_beta": 0.55,
    "ticker_universe": ["AAPL", "NVDA"],
}

SAMPLE_PID_CONFIG = {"Kp": 0.35, "Ki": 0.03, "rho_star": 0.83}

SAMPLE_CRIT_SCORES = {
    "round": 1,
    "rho_bar": 0.85,
    "agent_scores": {
        "macro": {
            "rho_i": 0.9,
            "pillar_scores": {"LV": 0.88, "ES": 0.84, "AC": 0.9, "CA": 0.88},
            "diagnostics": {},
            "explanations": {},
        },
        "value": {
            "rho_i": 0.8,
            "pillar_scores": {"LV": 0.8, "ES": 0.82, "AC": 0.88, "CA": 0.86},
            "diagnostics": {},
            "explanations": {},
        },
    },
}

SAMPLE_PID_STATE = {
    "round": 1,
    "beta_in": 0.4,
    "beta_new": 0.36,
    "tone_bucket": "balanced",
    "error": {"e_t": -0.02, "integral": -0.02, "e_prev": -0.02},
    "gains": {"p_term": -0.007, "i_term": -0.001, "d_term": -0.001},
    "u_t": -0.009,
    "quadrant": "healthy",
}

SAMPLE_JS_DIVERGENCE = {
    "round": 1,
    "js_divergence": 0.25,
    "agent_confidences": {"macro": 0.7, "value": 0.65},
}


def _build_complete_run(base: Path, experiment: str = "default",
                        run_id: str = "run_2026-01-01_00-00-00") -> Path:
    """Build a complete run directory with all expected files."""
    run_dir = base / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", SAMPLE_MANIFEST)
    _write_json(run_dir / "pid_config.json", SAMPLE_PID_CONFIG)

    # Round 1
    rd1 = run_dir / "rounds" / "round_001"
    _write_json(rd1 / "round_state.json", {
        "round": 1, "beta": 0.4,
        "proposals": {"macro": {"allocation": {"AAPL": 0.6, "NVDA": 0.4}, "confidence": 0.7}},
        "revisions": {"macro": {"allocation": {"AAPL": 0.5, "NVDA": 0.5}, "confidence": 0.72}},
        "metrics": {"rho_bar": 0.85, "beta_new": 0.36, "quadrant": "healthy"},
        "pid": {"beta_in": 0.4, "beta_new": 0.36, "quadrant": "healthy",
                "tone_bucket": "balanced", "e_t": -0.02, "u_t": -0.009},
        "crit": {"rho_bar": 0.85},
    })

    # Proposals
    _write_text(rd1 / "proposals" / "macro" / "response.txt", "Macro proposal text.")
    _write_json(rd1 / "proposals" / "macro" / "portfolio.json", {"AAPL": 0.6, "NVDA": 0.4})

    # Critiques
    _write_json(rd1 / "critiques" / "macro" / "response.json", {"critiques": ["K1"]})

    # Revisions
    _write_text(rd1 / "revisions" / "macro" / "response.txt", "Macro revision text.")
    _write_json(rd1 / "revisions" / "macro" / "portfolio.json", {"AAPL": 0.5, "NVDA": 0.5})

    # Metrics
    _write_json(rd1 / "metrics" / "crit_scores.json", SAMPLE_CRIT_SCORES)
    _write_json(rd1 / "metrics" / "pid_state.json", SAMPLE_PID_STATE)
    _write_json(rd1 / "metrics" / "js_divergence.json", SAMPLE_JS_DIVERGENCE)

    # Final
    _write_json(run_dir / "final" / "final_portfolio.json", {"AAPL": 0.5, "NVDA": 0.5})
    _write_text(run_dir / "final" / "judge_response.txt", "Judge response text.")

    # pid_crit_all_rounds.json
    _write_json(run_dir / "final" / "pid_crit_all_rounds.json", [
        {
            "type": "pid_round",
            "round": 1,
            "beta_in": 0.4,
            "tone_bucket": "balanced",
            "crit": {
                "rho_bar": 0.85,
                "rho_i": {"macro": 0.9, "value": 0.8},
                "agents": {
                    "macro": {"rho_i": 0.9, "pillars": {"LV": 0.88}},
                    "value": {"rho_i": 0.8, "pillars": {"LV": 0.8}},
                },
            },
            "pid": {
                "beta_in": 0.4, "beta_new": 0.36,
                "tone_bucket": "balanced",
                "error": {"e_t": -0.02},
                "u_t": -0.009,
                "quadrant": "healthy",
            },
            "divergence": {"js_divergence": 0.25},
        },
    ])

    _write_text(run_dir / "shared_context" / "memo.txt", "Investment memo.")

    return run_dir


@pytest.fixture
def base_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Run status detection tests
# ---------------------------------------------------------------------------

class TestRunStatus:
    def test_complete_run(self, base_dir):
        run_dir = _build_complete_run(base_dir)
        assert _get_run_status(run_dir) == "complete"

    def test_no_manifest(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        assert _get_run_status(run_dir) == "incomplete"

    def test_no_final_dir(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "manifest.json", SAMPLE_MANIFEST)
        assert _get_run_status(run_dir) == "incomplete"

    def test_partial_no_portfolio(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "manifest.json", SAMPLE_MANIFEST)
        (run_dir / "final").mkdir()
        _write_text(run_dir / "final" / "judge_response.txt", "text")
        assert _get_run_status(run_dir) == "partial"


# ---------------------------------------------------------------------------
# Trajectory loading + fallback tests
# ---------------------------------------------------------------------------

class TestTrajectoryLoading:
    def test_load_from_aggregated(self, base_dir):
        run_dir = _build_complete_run(base_dir)
        traj = _load_trajectories(run_dir)
        assert len(traj) == 1
        assert traj[0]["round"] == 1
        assert traj[0]["crit"]["rho_bar"] == 0.85

    def test_fallback_to_per_round_metrics(self, base_dir):
        """When pid_crit_all_rounds.json is missing, reconstruct from per-round files."""
        run_dir = _build_complete_run(base_dir)
        (run_dir / "final" / "pid_crit_all_rounds.json").unlink()

        traj = _load_trajectories(run_dir)
        assert len(traj) == 1
        assert traj[0]["round"] == 1
        assert traj[0]["crit"]["rho_bar"] == 0.85
        assert traj[0]["pid"]["beta_new"] == 0.36

    def test_fallback_with_missing_metrics(self, base_dir):
        """Fallback works even when some metric files are missing."""
        run_dir = _build_complete_run(base_dir)
        (run_dir / "final" / "pid_crit_all_rounds.json").unlink()
        (run_dir / "rounds" / "round_001" / "metrics" / "pid_state.json").unlink()

        traj = _load_trajectories(run_dir)
        assert len(traj) == 1
        assert "crit" in traj[0]
        assert "pid" not in traj[0]

    def test_no_rounds_dir(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        assert _load_trajectories(run_dir) == []


# ---------------------------------------------------------------------------
# Quality metrics tests
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def test_extract_from_complete_run(self, base_dir):
        run_dir = _build_complete_run(base_dir)
        q = _extract_quality_metrics(run_dir)
        assert q["final_rho_bar"] == 0.85
        assert q["final_beta"] == 0.36
        assert q["reasoning_collapse"] is False

    def test_reasoning_collapse_detection(self, base_dir):
        run_dir = _build_complete_run(base_dir)
        # Modify to have declining rho_bar
        traj = [
            {"round": 1, "crit": {"rho_bar": 0.9}, "pid": {"beta_new": 0.4},
             "divergence": {"js_divergence": 0.3}},
            {"round": 2, "crit": {"rho_bar": 0.7}, "pid": {"beta_new": 0.5},
             "divergence": {"js_divergence": 0.2}},
        ]
        _write_json(run_dir / "final" / "pid_crit_all_rounds.json", traj)
        q = _extract_quality_metrics(run_dir)
        assert q["reasoning_collapse"] is True
        assert q["final_rho_bar"] == 0.7

    def test_empty_run(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        assert _extract_quality_metrics(run_dir) == {}


# ---------------------------------------------------------------------------
# list_experiments tests
# ---------------------------------------------------------------------------

class TestListExperiments:
    def test_lists_experiments(self, base_dir):
        _build_complete_run(base_dir, "exp_a")
        _build_complete_run(base_dir, "exp_b")
        exps = list_experiments(base_dir)
        assert exps == ["exp_a", "exp_b"]

    def test_no_base_dir(self, base_dir):
        assert list_experiments(base_dir / "nonexistent") == []

    def test_ignores_hidden_dirs(self, base_dir):
        _build_complete_run(base_dir, "visible")
        (base_dir / ".hidden").mkdir()
        assert list_experiments(base_dir) == ["visible"]


# ---------------------------------------------------------------------------
# list_runs tests
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_lists_runs_with_quality_metrics(self, base_dir):
        _build_complete_run(base_dir)
        runs = list_runs(base_dir, "default")
        assert len(runs) == 1
        r = runs[0]
        assert r["run_id"] == "run_2026-01-01_00-00-00"
        assert r["status"] == "complete"
        assert r["model_name"] == "gpt-5-mini"
        assert r["final_rho_bar"] == 0.85

    def test_incomplete_run_listed(self, base_dir):
        run_dir = base_dir / "default" / "run_broken"
        run_dir.mkdir(parents=True)
        runs = list_runs(base_dir, "default")
        assert len(runs) == 1
        assert runs[0]["status"] == "incomplete"
        assert "model_name" not in runs[0]

    def test_nonexistent_experiment(self, base_dir):
        assert list_runs(base_dir, "nope") == []


# ---------------------------------------------------------------------------
# get_run_detail tests
# ---------------------------------------------------------------------------

class TestGetRunDetail:
    def test_complete_run_detail(self, base_dir):
        _build_complete_run(base_dir)
        detail = get_run_detail(base_dir, "default", "run_2026-01-01_00-00-00")
        assert detail is not None
        assert detail["status"] == "complete"
        assert detail["manifest"]["model_name"] == "gpt-5-mini"
        assert detail["final_portfolio"] == {"AAPL": 0.5, "NVDA": 0.5}
        assert len(detail["round_summaries"]) == 1

    def test_nonexistent_run(self, base_dir):
        assert get_run_detail(base_dir, "default", "nope") is None

    def test_missing_optional_files(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "manifest.json", SAMPLE_MANIFEST)
        detail = get_run_detail(base_dir, "default", "run_x")
        assert detail is not None
        assert detail["pid_config"] is None
        assert detail["final_portfolio"] is None


# ---------------------------------------------------------------------------
# PID trajectory tests
# ---------------------------------------------------------------------------

class TestPIDTrajectory:
    def test_extracts_pid_fields(self, base_dir):
        _build_complete_run(base_dir)
        traj = get_pid_trajectory(base_dir, "default", "run_2026-01-01_00-00-00")
        assert len(traj) == 1
        assert traj[0]["round"] == 1
        assert traj[0]["beta_in"] == 0.4
        assert traj[0]["beta_new"] == 0.36
        assert traj[0]["quadrant"] == "healthy"
        assert traj[0]["rho_bar"] == 0.85


# ---------------------------------------------------------------------------
# CRIT trajectory tests
# ---------------------------------------------------------------------------

class TestCRITTrajectory:
    def test_extracts_crit_fields(self, base_dir):
        _build_complete_run(base_dir)
        traj = get_crit_trajectory(base_dir, "default", "run_2026-01-01_00-00-00")
        assert len(traj) == 1
        assert traj[0]["rho_bar"] == 0.85
        assert traj[0]["rho_i"]["macro"] == 0.9
        assert traj[0]["rho_i"]["value"] == 0.8


# ---------------------------------------------------------------------------
# Portfolio trajectory tests
# ---------------------------------------------------------------------------

class TestPortfolioTrajectory:
    def test_computes_consensus(self, base_dir):
        _build_complete_run(base_dir)
        traj = get_portfolio_trajectory(base_dir, "default", "run_2026-01-01_00-00-00")
        assert len(traj) == 1
        # Only macro agent has revisions in this fixture
        assert "AAPL" in traj[0]["consensus"]
        assert "NVDA" in traj[0]["consensus"]

    def test_no_rounds(self, base_dir):
        run_dir = base_dir / "default" / "run_x"
        run_dir.mkdir(parents=True)
        assert get_portfolio_trajectory(base_dir, "default", "run_x") == []


# ---------------------------------------------------------------------------
# Round detail tests
# ---------------------------------------------------------------------------

class TestRoundDetail:
    def test_loads_agent_text(self, base_dir):
        _build_complete_run(base_dir)
        detail = get_round_detail(base_dir, "default", "run_2026-01-01_00-00-00", 1)
        assert detail is not None
        assert "macro" in detail["agents"]
        assert detail["agents"]["macro"]["proposal"] == "Macro proposal text."
        assert detail["agents"]["macro"]["revision"] == "Macro revision text."
        assert detail["agents"]["macro"]["critique"] == {"critiques": ["K1"]}
        assert detail["agents"]["macro"]["proposal_portfolio"] == {"AAPL": 0.6, "NVDA": 0.4}

    def test_nonexistent_round(self, base_dir):
        _build_complete_run(base_dir)
        assert get_round_detail(base_dir, "default", "run_2026-01-01_00-00-00", 99) is None

    def test_loads_metrics(self, base_dir):
        _build_complete_run(base_dir)
        detail = get_round_detail(base_dir, "default", "run_2026-01-01_00-00-00", 1)
        assert detail["crit_scores"]["rho_bar"] == 0.85
        assert detail["pid_state"]["beta_new"] == 0.36


# ---------------------------------------------------------------------------
# File tree tests
# ---------------------------------------------------------------------------

class TestFileTree:
    def test_returns_tree_structure(self, base_dir):
        _build_complete_run(base_dir)
        tree = get_file_tree(base_dir, "default", "run_2026-01-01_00-00-00")
        assert len(tree) > 0
        names = [item["name"] for item in tree]
        assert "manifest.json" in names
        assert "rounds" in names

    def test_includes_file_sizes(self, base_dir):
        _build_complete_run(base_dir)
        tree = get_file_tree(base_dir, "default", "run_2026-01-01_00-00-00")
        files = [item for item in tree if item["type"] == "file"]
        assert all("size_bytes" in f for f in files)

    def test_empty_run(self, base_dir):
        assert get_file_tree(base_dir, "default", "nope") == []


# ---------------------------------------------------------------------------
# read_run_file tests
# ---------------------------------------------------------------------------

class TestReadRunFile:
    def test_reads_file(self, base_dir):
        _build_complete_run(base_dir)
        content = read_run_file(base_dir, "default", "run_2026-01-01_00-00-00", "manifest.json")
        data = json.loads(content)
        assert data["model_name"] == "gpt-5-mini"

    def test_path_traversal_rejected(self, base_dir):
        _build_complete_run(base_dir)
        with pytest.raises(PermissionError, match="traversal"):
            read_run_file(base_dir, "default", "run_2026-01-01_00-00-00", "../../manifest.json")

    def test_file_not_found(self, base_dir):
        _build_complete_run(base_dir)
        with pytest.raises(FileNotFoundError):
            read_run_file(base_dir, "default", "run_2026-01-01_00-00-00", "nonexistent.txt")

    def test_reads_nested_file(self, base_dir):
        _build_complete_run(base_dir)
        content = read_run_file(
            base_dir, "default", "run_2026-01-01_00-00-00",
            "rounds/round_001/proposals/macro/response.txt",
        )
        assert content == "Macro proposal text."


# ---------------------------------------------------------------------------
# Config diff tests
# ---------------------------------------------------------------------------

class TestDiffRunConfigs:
    def test_same_config_no_diffs(self, base_dir):
        _build_complete_run(base_dir, "default", "run_a")
        _build_complete_run(base_dir, "default", "run_b")
        diff = diff_run_configs(base_dir, "default", "run_a", "default", "run_b")
        assert diff["only_left"] == {}
        assert diff["only_right"] == {}
        assert diff["different"] == {}
        assert len(diff["shared"]) > 0

    def test_detects_differences(self, base_dir):
        _build_complete_run(base_dir, "default", "run_a")
        run_b = _build_complete_run(base_dir, "default", "run_b")
        m = SAMPLE_MANIFEST.copy()
        m["model_name"] = "gpt-5"
        _write_json(run_b / "manifest.json", m)

        diff = diff_run_configs(base_dir, "default", "run_a", "default", "run_b")
        assert "model_name" in diff["different"]
        assert diff["different"]["model_name"]["left"] == "gpt-5-mini"
        assert diff["different"]["model_name"]["right"] == "gpt-5"

    def test_missing_manifest(self, base_dir):
        _build_complete_run(base_dir, "default", "run_a")
        (base_dir / "default" / "run_b").mkdir(parents=True)
        diff = diff_run_configs(base_dir, "default", "run_a", "default", "run_b")
        assert len(diff["only_left"]) > 0
        assert diff["only_right"] == {}
