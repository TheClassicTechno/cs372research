"""Integration tests for the Debate Dashboard API and HTML UI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools.dashboard.server import app, RUNS_BASE


# ---------------------------------------------------------------------------
# Helpers — reuse fixture builder from unit tests
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


SAMPLE_MANIFEST = {
    "experiment_name": "test_exp",
    "run_id": "run_2026-01-01_00-00-00",
    "started_at": "2026-01-01T00:00:00+00:00",
    "completed_at": "2026-01-01T01:00:00+00:00",
    "model_name": "gpt-5-mini",
    "crit_model_name": "gpt-5",
    "roles": ["macro", "value"],
    "actual_rounds": 2,
    "max_rounds": 5,
    "terminated_early": False,
    "termination_reason": "max_rounds",
    "pid_enabled": True,
    "invest_quarter": "2025Q1",
    "initial_beta": 0.4,
    "final_beta": 0.55,
    "ticker_universe": ["AAPL", "NVDA"],
}


def _build_run(base: Path, experiment: str = "test_exp",
               run_id: str = "run_2026-01-01_00-00-00") -> Path:
    """Build a complete run with 2 rounds."""
    run_dir = base / experiment / run_id
    manifest = SAMPLE_MANIFEST.copy()
    manifest["experiment_name"] = experiment
    manifest["run_id"] = run_id
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "pid_config.json", {"Kp": 0.35, "rho_star": 0.83})

    for rn in [1, 2]:
        rd = run_dir / "rounds" / f"round_{rn:03d}"
        _write_json(rd / "round_state.json", {
            "round": rn, "beta": 0.4 + rn * 0.05,
            "proposals": {"macro": {"allocation": {"AAPL": 0.6, "NVDA": 0.4}, "confidence": 0.7}},
            "revisions": {"macro": {"allocation": {"AAPL": 0.55, "NVDA": 0.45}, "confidence": 0.72}},
            "metrics": {"rho_bar": 0.8 + rn * 0.02, "beta_new": 0.4 + rn * 0.05, "quadrant": "healthy"},
            "pid": {"beta_in": 0.4 + (rn - 1) * 0.05, "beta_new": 0.4 + rn * 0.05,
                    "quadrant": "healthy", "tone_bucket": "balanced",
                    "e_t": -0.01, "u_t": -0.005},
            "crit": {"rho_bar": 0.8 + rn * 0.02},
        })
        _write_text(rd / "proposals" / "macro" / "response.txt", f"R{rn} macro proposal.")
        _write_json(rd / "proposals" / "macro" / "portfolio.json", {"AAPL": 0.6, "NVDA": 0.4})
        _write_json(rd / "critiques" / "macro" / "response.json", {"critiques": [f"K{rn}"]})
        _write_text(rd / "revisions" / "macro" / "response.txt", f"R{rn} macro revision.")
        _write_json(rd / "revisions" / "macro" / "portfolio.json", {"AAPL": 0.55, "NVDA": 0.45})
        _write_json(rd / "metrics" / "crit_scores.json", {
            "round": rn, "rho_bar": 0.8 + rn * 0.02,
            "agent_scores": {"macro": {"rho_i": 0.85, "pillar_scores": {"LV": 0.88}}},
        })
        _write_json(rd / "metrics" / "pid_state.json", {
            "round": rn, "beta_in": 0.4 + (rn - 1) * 0.05, "beta_new": 0.4 + rn * 0.05,
            "tone_bucket": "balanced", "error": {"e_t": -0.01}, "u_t": -0.005, "quadrant": "healthy",
        })
        _write_json(rd / "metrics" / "js_divergence.json", {
            "round": rn, "js_divergence": 0.25 - rn * 0.05,
        })

    _write_json(run_dir / "final" / "final_portfolio.json", {"AAPL": 0.5, "NVDA": 0.5})
    _write_text(run_dir / "final" / "judge_response.txt", "Judge response.")
    _write_json(run_dir / "final" / "pid_crit_all_rounds.json", [
        {
            "round": rn, "beta_in": 0.4 + (rn - 1) * 0.05,
            "tone_bucket": "balanced",
            "crit": {"rho_bar": 0.8 + rn * 0.02, "rho_i": {"macro": 0.85},
                     "agents": {"macro": {"rho_i": 0.85, "pillars": {"LV": 0.88}}}},
            "pid": {"beta_in": 0.4 + (rn - 1) * 0.05, "beta_new": 0.4 + rn * 0.05,
                    "tone_bucket": "balanced", "error": {"e_t": -0.01},
                    "u_t": -0.005, "quadrant": "healthy"},
            "divergence": {"js_divergence": 0.25 - rn * 0.05},
        }
        for rn in [1, 2]
    ])
    _write_text(run_dir / "shared_context" / "memo.txt", "Test memo.")

    return run_dir


@pytest.fixture
def test_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def client(test_base, monkeypatch):
    _build_run(test_base, "test_exp", "run_001")
    _build_run(test_base, "test_exp", "run_002")
    monkeypatch.setattr("tools.dashboard.server.RUNS_BASE", test_base)
    return TestClient(app)


@pytest.fixture
def client_with_incomplete(test_base, monkeypatch):
    _build_run(test_base, "test_exp", "run_complete")
    incomplete = test_base / "test_exp" / "run_incomplete"
    incomplete.mkdir(parents=True)
    monkeypatch.setattr("tools.dashboard.server.RUNS_BASE", test_base)
    return TestClient(app)


# ---------------------------------------------------------------------------
# HTML + routing tests
# ---------------------------------------------------------------------------

class TestHTMLUI:
    def test_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_has_nav_bar(self, client):
        html = client.get("/").text
        assert 'id="nav"' in html
        assert 'href="#live"' in html
        assert 'href="#runs"' in html
        assert "Live Debate" in html

    def test_has_title(self, client):
        html = client.get("/").text
        assert "<title>Debate Dashboard</title>" in html

    def test_has_app_container(self, client):
        html = client.get("/").text
        assert 'id="app"' in html

    def test_has_routing_logic(self, client):
        """Routing logic lives in external JS module."""
        html = client.get("/").text
        # Hash targets are in the HTML nav links
        assert "#live" in html
        assert "#runs" in html
        # hashchange listener is in app.js
        js = client.get("/static/js/app.js").text
        assert "hashchange" in js

    def test_has_chart_functions(self, client):
        """Chart functions exist in external JS modules."""
        js = client.get("/static/js/components/charts.js").text
        assert "buildPIDChart" in js
        assert "buildCRITChart" in js

    def test_has_file_explorer(self, client):
        """File explorer exists in external JS modules."""
        js = client.get("/static/js/views/runDetail/fileExplorerSection.js").text
        assert "file-tree" in js

    def test_has_live_debate_view(self, client):
        """Live debate view exists in external JS module."""
        js = client.get("/static/js/views/liveView.js").text
        assert "renderLiveDebateView" in js
        assert "live-entries" in js
        # The API endpoint is in the API module
        api_js = client.get("/static/js/api/live.js").text
        assert "/api/live_debate" in api_js


# ---------------------------------------------------------------------------
# Experiments endpoint tests
# ---------------------------------------------------------------------------

class TestExperimentsAPI:
    def test_list_experiments(self, client):
        r = client.get("/runs/")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["experiment"] == "test_exp"
        assert data[0]["run_count"] == 2

    def test_empty_base(self, test_base, monkeypatch):
        monkeypatch.setattr("tools.dashboard.server.RUNS_BASE", test_base)
        c = TestClient(app)
        r = c.get("/runs/")
        assert r.json() == []


# ---------------------------------------------------------------------------
# Runs list endpoint tests
# ---------------------------------------------------------------------------

class TestRunsListAPI:
    def test_list_runs(self, client):
        r = client.get("/runs/test_exp")
        assert r.status_code == 200
        runs = r.json()
        assert len(runs) == 2
        run_ids = [run["run_id"] for run in runs]
        assert "run_001" in run_ids
        assert "run_002" in run_ids

    def test_runs_have_quality_metrics(self, client):
        runs = client.get("/runs/test_exp").json()
        for run in runs:
            assert "status" in run
            assert run["status"] == "complete"
            assert "final_rho_bar" in run
            assert "model_name" in run

    def test_nonexistent_experiment(self, client):
        r = client.get("/runs/nonexistent")
        assert r.status_code == 200
        assert r.json() == []

    def test_incomplete_run_listed(self, client_with_incomplete):
        runs = client_with_incomplete.get("/runs/test_exp").json()
        statuses = {run["run_id"]: run["status"] for run in runs}
        assert statuses["run_complete"] == "complete"
        assert statuses["run_incomplete"] == "incomplete"


# ---------------------------------------------------------------------------
# Run detail endpoint tests
# ---------------------------------------------------------------------------

class TestRunDetailAPI:
    def test_run_detail(self, client):
        r = client.get("/runs/test_exp/run_001")
        assert r.status_code == 200
        detail = r.json()
        assert detail["status"] == "complete"
        assert detail["manifest"]["model_name"] == "gpt-5-mini"
        assert detail["final_portfolio"] == {"AAPL": 0.5, "NVDA": 0.5}
        assert len(detail["round_summaries"]) == 2

    def test_nonexistent_run(self, client):
        r = client.get("/runs/test_exp/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# PID trajectory endpoint tests
# ---------------------------------------------------------------------------

class TestPIDAPI:
    def test_pid_trajectory(self, client):
        r = client.get("/runs/test_exp/run_001/pid")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["round"] == 1
        assert data[0]["quadrant"] == "healthy"
        assert "beta_new" in data[0]
        assert "rho_bar" in data[0]


# ---------------------------------------------------------------------------
# CRIT trajectory endpoint tests
# ---------------------------------------------------------------------------

class TestCRITAPI:
    def test_crit_trajectory(self, client):
        r = client.get("/runs/test_exp/run_001/crit")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["rho_bar"] is not None
        assert "rho_i" in data[0]


# ---------------------------------------------------------------------------
# Portfolio trajectory endpoint tests
# ---------------------------------------------------------------------------

class TestPortfolioAPI:
    def test_portfolio_trajectory(self, client):
        r = client.get("/runs/test_exp/run_001/portfolio")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert "consensus" in data[0]
        assert "AAPL" in data[0]["consensus"]


# ---------------------------------------------------------------------------
# Round detail endpoint tests
# ---------------------------------------------------------------------------

class TestRoundDetailAPI:
    def test_round_detail(self, client):
        r = client.get("/runs/test_exp/run_001/round/1")
        assert r.status_code == 200
        detail = r.json()
        assert detail["round"] == 1
        assert "macro" in detail["agents"]
        assert detail["agents"]["macro"]["proposal"] == "R1 macro proposal."

    def test_nonexistent_round(self, client):
        r = client.get("/runs/test_exp/run_001/round/99")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# File tree endpoint tests
# ---------------------------------------------------------------------------

class TestFileTreeAPI:
    def test_file_tree(self, client):
        r = client.get("/runs/test_exp/run_001/tree")
        assert r.status_code == 200
        tree = r.json()
        assert len(tree) > 0
        names = [item["name"] for item in tree]
        assert "manifest.json" in names

    def test_empty_run_tree(self, client):
        r = client.get("/runs/test_exp/nonexistent/tree")
        assert r.json() == []


# ---------------------------------------------------------------------------
# File read endpoint tests
# ---------------------------------------------------------------------------

class TestFileReadAPI:
    def test_read_json_file(self, client):
        r = client.get("/runs/test_exp/run_001/file", params={"path": "manifest.json"})
        assert r.status_code == 200
        data = r.json()
        assert data["truncated"] is False
        assert data["content"]["model_name"] == "gpt-5-mini"

    def test_read_text_file(self, client):
        r = client.get("/runs/test_exp/run_001/file",
                       params={"path": "shared_context/memo.txt"})
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "Test memo."
        assert data["truncated"] is False

    def test_path_traversal_rejected(self, client):
        r = client.get("/runs/test_exp/run_001/file",
                       params={"path": "../../test_exp/run_002/manifest.json"})
        assert r.status_code == 403

    def test_file_not_found(self, client):
        r = client.get("/runs/test_exp/run_001/file",
                       params={"path": "nonexistent.txt"})
        assert r.status_code == 404



# ---------------------------------------------------------------------------
# Existing prompt trace endpoints still work
# ---------------------------------------------------------------------------

class TestPromptEndpointsUnchanged:
    def test_logs_endpoint(self, client, monkeypatch, tmp_path):
        log_file = tmp_path / "traces.jsonl"
        log_file.write_text('{"model":"test","system":"s","user":"u","response":"r"}\n')
        monkeypatch.setattr("tools.dashboard.server.LOGS_PATH", log_file)
        r = client.get("/logs")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_logs_clear(self, client, monkeypatch, tmp_path):
        log_file = tmp_path / "traces.jsonl"
        log_file.write_text('{"test": true}\n')
        monkeypatch.setattr("tools.dashboard.server.LOGS_PATH", log_file)
        r = client.post("/logs/clear")
        assert r.status_code == 200
        assert not log_file.exists()
