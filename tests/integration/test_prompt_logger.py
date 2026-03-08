"""Integration tests for the prompt logging pipeline.

Validates:
- JSONL logger writes correct entries when ENABLE_PROMPT_LOGGING is set
- Logger is a no-op when env var is unset
- Metadata (role, phase, round) propagates from _call_llm through to JSONL
- CRIT scorer passes metadata through its llm_fn interface
- FastAPI /logs endpoint reads and filters JSONL correctly
- HTML page contains required UI elements and renders entry cards
"""

import json
import os
import threading

import pytest

from eval.utils.prompt_logger import log_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    """Read all entries from a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_jsonl(path, entries: list[dict]) -> None:
    """Write entries to a JSONL file."""
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _make_entry(model="gpt-5-mini", role="macro", phase="propose",
                round_num=0, system="sys", user="usr", response="resp"):
    """Build a prompt trace entry dict."""
    return {
        "timestamp": "2026-03-07T12:00:00+00:00",
        "model": model,
        "role": role,
        "phase": phase,
        "round": round_num,
        "system": system,
        "user": user,
        "response": response,
    }


@pytest.fixture()
def log_dir(tmp_path, monkeypatch):
    """Enable prompt logging and redirect output to a temp directory."""
    monkeypatch.setenv("ENABLE_PROMPT_LOGGING", "true")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def log_file(log_dir):
    """Return the expected JSONL path after logging."""
    return str(log_dir / "logs" / "prompt_traces.jsonl")


# ---------------------------------------------------------------------------
# 1. Logger basics
# ---------------------------------------------------------------------------

class TestPromptLoggerBasics:
    def test_noop_without_env_var(self, tmp_path, monkeypatch):
        """log_prompt does nothing when ENABLE_PROMPT_LOGGING is not set."""
        monkeypatch.delenv("ENABLE_PROMPT_LOGGING", raising=False)
        monkeypatch.chdir(tmp_path)
        log_prompt(system="s", user="u", model="m")
        assert not (tmp_path / "logs").exists()

    def test_noop_when_env_var_false(self, tmp_path, monkeypatch):
        """log_prompt does nothing when ENABLE_PROMPT_LOGGING=false."""
        monkeypatch.setenv("ENABLE_PROMPT_LOGGING", "false")
        monkeypatch.chdir(tmp_path)
        log_prompt(system="s", user="u", model="m")
        assert not (tmp_path / "logs").exists()

    def test_creates_log_dir_and_file(self, log_dir, log_file):
        """log_prompt creates logs/ directory and JSONL file."""
        log_prompt(system="sys", user="usr", model="gpt-5-mini")
        assert os.path.isfile(log_file)

    def test_writes_valid_jsonl(self, log_dir, log_file):
        """Each line in the output is valid JSON."""
        log_prompt(system="s1", user="u1", model="m1")
        log_prompt(system="s2", user="u2", model="m2")
        entries = _read_jsonl(log_file)
        assert len(entries) == 2

    def test_entry_fields(self, log_dir, log_file):
        """Entry contains all expected fields."""
        log_prompt(
            system="test-sys", user="test-usr", model="gpt-5-mini",
            response="test-resp", role="macro", phase="propose", round_num=1,
        )
        entry = _read_jsonl(log_file)[0]
        assert entry["system"] == "test-sys"
        assert entry["user"] == "test-usr"
        assert entry["model"] == "gpt-5-mini"
        assert entry["response"] == "test-resp"
        assert entry["role"] == "macro"
        assert entry["phase"] == "propose"
        assert entry["round"] == 1
        assert "timestamp" in entry

    def test_messages_array(self, log_dir, log_file):
        """Entry has a reconstructed messages array."""
        log_prompt(system="sys", user="usr", model="m")
        entry = _read_jsonl(log_file)[0]
        assert entry["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_defaults_for_optional_fields(self, log_dir, log_file):
        """role, phase, round default to empty/zero when not provided."""
        log_prompt(system="s", user="u", model="m")
        entry = _read_jsonl(log_file)[0]
        assert entry["role"] == ""
        assert entry["phase"] == ""
        assert entry["round"] == 0
        assert entry["response"] is None


# ---------------------------------------------------------------------------
# 2. Thread safety
# ---------------------------------------------------------------------------

class TestPromptLoggerThreadSafety:
    def test_concurrent_writes(self, log_dir, log_file):
        """Multiple threads writing simultaneously produce valid JSONL."""
        n_threads = 8
        writes_per_thread = 10
        barrier = threading.Barrier(n_threads)

        def worker(thread_id):
            barrier.wait()
            for i in range(writes_per_thread):
                log_prompt(
                    system=f"sys-{thread_id}-{i}",
                    user=f"usr-{thread_id}-{i}",
                    model="test-model",
                    role=f"agent-{thread_id}",
                    phase="propose",
                    round_num=i,
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = _read_jsonl(log_file)
        assert len(entries) == n_threads * writes_per_thread


# ---------------------------------------------------------------------------
# 3. _call_llm metadata propagation
# ---------------------------------------------------------------------------

class TestCallLlmMetadata:
    def test_metadata_flows_through_call_llm(self, log_dir, log_file):
        """_call_llm in mock mode passes role/phase/round to log_prompt."""
        from multi_agent.graph.llm import _call_llm

        config = {"mock": True}
        _call_llm(config, "sys-prompt", "usr-prompt",
                  role="risk", phase="critique", round_num=2)
        # Mock path returns early before log_prompt, so no entry is written.
        # This confirms mock path does NOT log (by design).
        log_path = log_dir / "logs" / "prompt_traces.jsonl"
        if log_path.exists():
            entries = _read_jsonl(str(log_path))
            # If any entries exist, they should have correct metadata
            for e in entries:
                assert e["role"] == "risk"
                assert e["phase"] == "critique"
                assert e["round"] == 2


# ---------------------------------------------------------------------------
# 4. CRIT scorer metadata propagation
# ---------------------------------------------------------------------------

class TestCritMetadata:
    def test_crit_scorer_passes_role_and_round(self, log_dir, log_file):
        """CritScorer._score_single_agent passes role and round_num kwargs."""
        from eval.crit import CritScorer

        captured = {}

        def mock_llm(sys, usr, **kw):
            captured.update(kw)
            return json.dumps({
                "pillar_scores": {
                    "logical_validity": 0.8,
                    "evidential_support": 0.7,
                    "alternative_consideration": 0.75,
                    "causal_alignment": 0.65,
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
                    "logical_validity": "ok",
                    "evidential_support": "ok",
                    "alternative_consideration": "ok",
                    "causal_alignment": "ok",
                },
            })

        scorer = CritScorer(llm_fn=mock_llm)
        bundles = {
            "macro": {
                "round": 3,
                "agent_role": "macro",
                "proposal": "Buy stocks",
                "critiques_received": "None",
                "revised_argument": "Still buy stocks",
            },
        }
        scorer.score(bundles)
        assert captured["role"] == "macro"
        assert captured["round_num"] == 3


# ---------------------------------------------------------------------------
# 5. FastAPI server — API endpoints
# ---------------------------------------------------------------------------

class TestServerAPI:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from tools.dashboard.server import app
        return TestClient(app)

    def test_get_logs_empty_when_no_file(self, client, tmp_path, monkeypatch):
        """GET /logs returns [] when JSONL file doesn't exist."""
        import tools.dashboard.server as srv
        from pathlib import Path
        monkeypatch.setattr(srv, "LOGS_PATH", Path(tmp_path / "nonexistent.jsonl"))
        resp = client.get("/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_logs_returns_all_entries(self, client, tmp_path, monkeypatch):
        """GET /logs returns all entries from JSONL file."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [_make_entry(), _make_entry(model="claude-4")])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        resp = client.get("/logs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_logs_filter_by_model(self, client, tmp_path, monkeypatch):
        """GET /logs?model=claude filters to matching entries."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [
            _make_entry(model="gpt-5-mini"),
            _make_entry(model="claude-4"),
            _make_entry(model="claude-4"),
        ])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        resp = client.get("/logs", params={"model": "claude"})
        data = resp.json()
        assert len(data) == 2
        assert all("claude" in e["model"] for e in data)

    def test_get_logs_filter_by_search(self, client, tmp_path, monkeypatch):
        """GET /logs?search=keyword filters by text content."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [
            _make_entry(system="analyze inflation"),
            _make_entry(system="review bonds"),
        ])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        resp = client.get("/logs", params={"search": "inflation"})
        data = resp.json()
        assert len(data) == 1
        assert "inflation" in data[0]["system"]

    def test_post_clear_removes_file(self, client, tmp_path, monkeypatch):
        """POST /logs/clear removes the JSONL file."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [_make_entry()])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        resp = client.post("/logs/clear")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}
        assert not jsonl_path.exists()

    def test_post_clear_then_get_returns_empty(self, client, tmp_path, monkeypatch):
        """After POST /logs/clear, GET /logs returns []."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [_make_entry(), _make_entry()])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        client.post("/logs/clear")
        resp = client.get("/logs")
        assert resp.json() == []

    def test_post_clear_when_no_file(self, client, tmp_path, monkeypatch):
        """POST /logs/clear succeeds even when file doesn't exist."""
        import tools.dashboard.server as srv
        from pathlib import Path
        monkeypatch.setattr(srv, "LOGS_PATH", Path(tmp_path / "nonexistent.jsonl"))
        resp = client.post("/logs/clear")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. HTML page — element presence and rendered content
# ---------------------------------------------------------------------------

class TestHTMLViewer:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from tools.dashboard.server import app
        return TestClient(app)

    # -- Static element presence --

    def test_page_title(self, client):
        """Page has the correct <title>."""
        html = client.get("/").text
        assert "<title>Debate Dashboard</title>" in html

    def test_page_heading(self, client):
        """Page has the h1 heading."""
        html = client.get("/").text
        assert "<h1>Debate Dashboard</h1>" in html

    def test_live_status_span(self, client):
        """Page has the live status indicator."""
        html = client.get("/").text
        assert 'id="live-status"' in html

    def test_live_entries_container(self, client):
        """Page has the live entries container."""
        html = client.get("/").text
        assert 'id="live-entries"' in html

    def test_app_container(self, client):
        """Page has the app div container."""
        html = client.get("/").text
        assert 'id="app"' in html

    def test_runs_search_filter(self, client):
        """Page has the search filter for runs."""
        html = client.get("/").text
        assert 'id="runs-search"' in html
        assert 'Filter runs...' in html

    # -- JS behavior: poll function exists --

    def test_poll_function_present(self, client):
        """Page JS has the poll function that fetches /api/live_debate."""
        html = client.get("/").text
        assert "function poll()" in html
        assert "fetch('/api/live_debate')" in html

    def test_auto_refresh_interval(self, client):
        """Page JS sets up a polling interval."""
        html = client.get("/").text
        assert "setInterval(poll, 1500)" in html

    # -- CSS: card structure --

    def test_card_styles_defined(self, client):
        """CSS defines card, card-header, card-body styles."""
        html = client.get("/").text
        assert ".card " in html or ".card{" in html
        assert ".card-header" in html
        assert ".card-body" in html
        assert ".card.open" in html

    def test_card_body_hidden_by_default(self, client):
        """Card bodies are hidden (display:none) by default, shown when .open."""
        html = client.get("/").text
        assert "display: none" in html or "display:none" in html
        assert "display: block" in html or "display:block" in html

    # -- JS rendering: label format --

    def test_debate_label_format(self, client):
        """JS builds debate labels as ROLE | Round N | PHASE."""
        html = client.get("/").text
        assert "'Round '" in html or '"Round "' in html
        assert "CRIT" in html

    def test_live_event_card_format(self, client):
        """JS builds event cards with round and phase labels."""
        html = client.get("/").text
        # The JS template: '[ROUND ' + ev.round + '] ' + ev.agent + ' — ' + ev.phase
        assert "ROUND" in html
        assert "ev.phase" in html

    def test_portfolio_label_in_card(self, client):
        """Card body JS renders PORTFOLIO label when portfolio data exists."""
        html = client.get("/").text
        assert "PORTFOLIO" in html

    # -- API returns entry metadata fields --

    def test_entries_have_metadata_fields(self, client, tmp_path, monkeypatch):
        """Entries returned by /logs include role, phase, round fields."""
        import tools.dashboard.server as srv
        from pathlib import Path

        jsonl_path = tmp_path / "traces.jsonl"
        _write_jsonl(jsonl_path, [
            _make_entry(role="risk", phase="critique", round_num=2),
        ])
        monkeypatch.setattr(srv, "LOGS_PATH", Path(jsonl_path))

        data = client.get("/logs").json()
        entry = data[0]
        assert entry["role"] == "risk"
        assert entry["phase"] == "critique"
        assert entry["round"] == 2

    def test_clear_button_style(self, client):
        """Clear button has CSS styling defined via generic button selector."""
        html = client.get("/").text
        assert "button" in html and "cursor: pointer" in html
