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

    # --- 2b. Create stub ablation experiment dirs for cross-ablation tests ---
    for stub_name in ("vskarich_ablation_7", "vskarich_ablation_8"):
        (tmp_root / stub_name).mkdir()

    # --- 2c. Copy ablation_summary.json to RUNS_BASE level ---
    abl_src = _CANONICAL_RUNS / "ablation_summary.json"
    if abl_src.exists():
        shutil.copy2(abl_src, tmp_root / "ablation_summary.json")

    # --- 2c. Mock financial paired tests (no real simulation data) ---
    import tools.dashboard.run_scanner as scanner

    _original_compute = scanner.compute_financial_paired_tests

    def _mock_financial_tests(*_args, **_kwargs):
        return {
            "config_a": "baseline",
            "config_b": "enriched",
            "n_paired": 35,
            "source": _kwargs.get("use_mean_revisions", False)
                      and "mean_revisions" or "judge",
            "metrics": [
                {"metric": "daily_metrics_excess_return_pct",
                 "n": 35, "a_mean": 0.42, "a_sem": 0.31,
                 "b_mean": 1.33, "b_sem": 0.28,
                 "mean_diff": 0.91, "t_statistic": 2.41,
                 "p_value": 0.0205, "ci_95": [0.14, 1.68],
                 "per_scenario": []},
                {"metric": "daily_metrics_total_return_pct",
                 "n": 35, "a_mean": 5.12, "a_sem": 1.04,
                 "b_mean": 6.03, "b_sem": 0.98,
                 "mean_diff": 0.91, "t_statistic": 1.85,
                 "p_value": 0.0732, "ci_95": [-0.09, 1.91],
                 "per_scenario": []},
                {"metric": "daily_metrics_annualized_sharpe",
                 "n": 35, "a_mean": 0.881, "a_sem": 0.102,
                 "b_mean": 1.013, "b_sem": 0.095,
                 "mean_diff": 0.132, "t_statistic": 2.52,
                 "p_value": 0.0164, "ci_95": [0.025, 0.239],
                 "per_scenario": []},
                {"metric": "daily_metrics_annualized_sortino",
                 "n": 35, "a_mean": 1.204, "a_sem": 0.158,
                 "b_mean": 1.462, "b_sem": 0.149,
                 "mean_diff": 0.258, "t_statistic": 2.48,
                 "p_value": 0.0183, "ci_95": [0.046, 0.470],
                 "per_scenario": []},
                {"metric": "daily_metrics_calmar_ratio",
                 "n": 35, "a_mean": 2.31, "a_sem": 0.42,
                 "b_mean": 3.17, "b_sem": 0.39,
                 "mean_diff": 0.86, "t_statistic": 2.33,
                 "p_value": 0.0257, "ci_95": [0.11, 1.61],
                 "per_scenario": []},
                {"metric": "daily_metrics_annualized_volatility",
                 "n": 35, "a_mean": 18.42, "a_sem": 1.21,
                 "b_mean": 17.88, "b_sem": 1.15,
                 "mean_diff": -0.54, "t_statistic": -0.72,
                 "p_value": 0.4762, "ci_95": [-2.07, 0.99],
                 "per_scenario": []},
                {"metric": "daily_metrics_max_drawdown_pct",
                 "n": 35, "a_mean": -12.35, "a_sem": 1.82,
                 "b_mean": -11.92, "b_sem": 1.74,
                 "mean_diff": 0.43, "t_statistic": 0.38,
                 "p_value": 0.7068, "ci_95": [-1.88, 2.74],
                 "per_scenario": []},
                {"metric": "total_trades",
                 "n": 35, "a_mean": 8.2, "a_sem": 0.5,
                 "b_mean": 8.5, "b_sem": 0.4,
                 "mean_diff": 0.3, "t_statistic": 0.82,
                 "p_value": 0.4193, "ci_95": [-0.4, 1.0],
                 "per_scenario": []},
                {"metric": "final_cash",
                 "n": 35, "a_mean": 15190.0, "a_sem": 2100.0,
                 "b_mean": 14850.0, "b_sem": 1950.0,
                 "mean_diff": -340.0, "t_statistic": -0.25,
                 "p_value": 0.8042, "ci_95": [-3120.0, 2440.0],
                 "per_scenario": []},
            ],
        }

    scanner.compute_financial_paired_tests = _mock_financial_tests

    # --- 2d. Mock cross-ablation financial significance summary ---
    _original_significance = scanner.compute_financial_significance_summary

    def _mock_financial_significance(*_args, **_kwargs):
        mock_result = _mock_financial_tests()
        experiments = ["vskarich_ablation_7", "vskarich_ablation_8"]
        metrics_out = []
        for m in mock_result["metrics"]:
            metrics_out.append({
                "metric": m["metric"],
                "results": {
                    exp: {"mean_diff": m["mean_diff"],
                          "p_value": m["p_value"],
                          "n": m["n"]}
                    for exp in experiments
                },
            })
        return {"experiments": experiments, "metrics": metrics_out}

    scanner.compute_financial_significance_summary = _mock_financial_significance

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
    scanner.compute_financial_paired_tests = _original_compute
    scanner.compute_financial_significance_summary = _original_significance
    server.should_exit = True
    thread.join(timeout=5)
