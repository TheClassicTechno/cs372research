"""Shared fixtures for Playwright dashboard integration tests.

Launches the real FastAPI dashboard server on a random port with a
**copied** test dataset so that the canonical source data is never mutated.
"""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

# Auto-apply the 'dashboard' marker to all tests in this package so they
# are excluded from the default pytest run (which uses -m 'not dashboard').
# Run them explicitly with:  pytest -m dashboard --browser chromium
pytestmark = pytest.mark.dashboard

# Canonical read-only test dataset
_CANONICAL_RUNS = Path(__file__).resolve().parents[3] / "logging" / "runs" / "test"


def _free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 10.0) -> None:
    """Poll *url* until HTTP 200 or *timeout* seconds elapse."""
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=1)
            if resp.status == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.15)
    raise RuntimeError(f"Dashboard server did not become ready at {url}")


@pytest.fixture(scope="session")
def dashboard_url(tmp_path_factory):
    """Start the real dashboard server against a copied test dataset.

    The fixture:
    1. Copies ``logging/runs/test/`` into a temp directory so tests
       never mutate the canonical source data.
    2. Patches the copied manifest to include ``agent_profiles`` so
       the agent-name-from-config test is meaningful.
    3. Starts uvicorn in a daemon thread on a random port.
    4. Waits for the server to respond with HTTP 200.
    5. Yields the base URL.
    6. Cleans up on teardown.
    """
    import tools.dashboard.server as srv

    # --- 1. Copy dataset into temp workspace ---
    tmp_root = tmp_path_factory.mktemp("dashboard_runs")
    shutil.copytree(_CANONICAL_RUNS, tmp_root / "test")

    # --- 1b. Copy ablation_summary.json to RUNS_BASE root if present ---
    ablation_src = _CANONICAL_RUNS / "ablation_summary.json"
    if ablation_src.exists():
        shutil.copy2(ablation_src, tmp_root / "ablation_summary.json")

    # --- 2. Patch manifest in the copy only ---
    manifest_path = (
        tmp_root / "test" / "run_2026-03-07_19-50-06" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["agent_profiles"] = {
        "value": "value_enriched",
        "risk": "risk_enriched",
        "technical": "technical_enriched",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- 3. Point server at our temp copy ---
    original_base = srv.RUNS_BASE
    srv.RUNS_BASE = tmp_root

    # --- 4. Start uvicorn on random port ---
    port = _free_port()
    config = uvicorn.Config(
        srv.app, host="127.0.0.1", port=port, log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(base_url)

    yield base_url

    # --- 5. Teardown ---
    srv.RUNS_BASE = original_base
    server.should_exit = True
    thread.join(timeout=5)
