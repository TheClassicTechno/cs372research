"""Local FastAPI server for browsing prompt traces and debate runs.

Usage:
    python tools/dashboard/server.py

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from tools.dashboard import run_scanner

app = FastAPI(title="Debate Dashboard")

LOGS_PATH = Path("logs/prompt_traces.jsonl")
STATIC_DIR = Path(__file__).parent / "static"
RUNS_BASE = Path("logging/runs")


# ------------------------------------------------------------------
# Prompt trace endpoints (existing)
# ------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the single-page viewer UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/logs")
def logs(
    model: str | None = Query(default=None),
    search: str | None = Query(default=None),
):
    """Return prompt traces as a JSON array."""
    if not LOGS_PATH.exists():
        return JSONResponse([])

    entries: list[dict] = []
    for line in LOGS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if model and model.lower() not in entry.get("model", "").lower():
            continue
        if search:
            haystack = " ".join(
                str(entry.get(k, "")) for k in ("system", "user", "response")
            )
            if search.lower() not in haystack.lower():
                continue

        entries.append(entry)

    return JSONResponse(entries)


@app.post("/logs/clear")
def clear_logs():
    """Delete all prompt trace entries."""
    if LOGS_PATH.exists():
        LOGS_PATH.unlink()
    return JSONResponse({"status": "cleared"})


# ------------------------------------------------------------------
# Live debate endpoint
# ------------------------------------------------------------------

_CLEAR_FILE = Path(__file__).resolve().parent / ".live_clear_mtime"


def _read_clear_mtime() -> float:
    try:
        return float(_CLEAR_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0.0


def _write_clear_mtime(t: float) -> None:
    _CLEAR_FILE.write_text(str(t))


@app.get("/api/live_debate")
def live_debate():
    """Return live debate events from the most recent run.

    Always reads directly from disk — no caching.
    Events cleared via POST are filtered out by mtime.
    """
    result = run_scanner.get_live_events(RUNS_BASE)
    cutoff = _read_clear_mtime()
    if cutoff > 0 and result.get("events"):
        result["events"] = [
            e for e in result["events"] if e.get("mtime", 0) > cutoff
        ]
    return JSONResponse(result)


@app.post("/api/live_debate/clear")
def clear_live_debate():
    """Mark all current live events as cleared (by mtime cutoff)."""
    import time
    _write_clear_mtime(time.time())
    return JSONResponse({"cleared": True})


# ------------------------------------------------------------------
# Run exploration endpoints
# ------------------------------------------------------------------

@app.get("/runs/")
def list_experiments():
    """List experiments with run counts."""
    experiments = run_scanner.list_experiments(RUNS_BASE)
    result = []
    for exp in experiments:
        runs = run_scanner.list_runs(RUNS_BASE, exp)
        result.append({"experiment": exp, "run_count": len(runs)})
    return JSONResponse(result)



@app.get("/runs/{experiment}")
def list_runs(experiment: str):
    """List runs in an experiment with quality summary metrics."""
    runs = run_scanner.list_runs(RUNS_BASE, experiment)
    return JSONResponse(runs)


@app.get("/runs/{experiment}/{run_id}")
def run_detail(experiment: str, run_id: str):
    """Full run detail."""
    detail = run_scanner.get_run_detail(RUNS_BASE, experiment, run_id)
    if detail is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(detail)


@app.get("/runs/{experiment}/{run_id}/performance")
def portfolio_performance(experiment: str, run_id: str):
    """Compute portfolio performance from final allocation and daily prices."""
    return JSONResponse(
        run_scanner.compute_portfolio_performance(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/performance/by-agent")
def portfolio_performance_by_agent(experiment: str, run_id: str):
    """Compute per-agent portfolio performance from final-round allocations."""
    return JSONResponse(
        run_scanner.compute_agent_performance(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/performance/by-round")
def portfolio_performance_by_round(experiment: str, run_id: str):
    """Compute per-round, per-phase, per-agent portfolio performance."""
    return JSONResponse(
        run_scanner.compute_round_performance(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/performance/debate-impact")
def debate_impact(experiment: str, run_id: str):
    """Compute debate impact: per-agent deltas and mean portfolio comparison."""
    return JSONResponse(
        run_scanner.compute_debate_impact(RUNS_BASE, experiment, run_id)
    )


@app.get("/api/ablation/debate-impact")
def ablation_debate_impact():
    """Aggregate debate impact across all runs per experiment."""
    return JSONResponse(
        run_scanner.compute_ablation_debate_impact(RUNS_BASE)
    )


@app.get("/runs/{experiment}/{run_id}/collapse")
def collapse_diagnostics(experiment: str, run_id: str):
    """Per-round agent collapse diagnostics."""
    return JSONResponse(
        run_scanner.compute_collapse_diagnostics(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/pid")
def pid_trajectory(experiment: str, run_id: str):
    """PID trajectory array."""
    return JSONResponse(run_scanner.get_pid_trajectory(RUNS_BASE, experiment, run_id))


@app.get("/runs/{experiment}/{run_id}/crit")
def crit_trajectory(experiment: str, run_id: str):
    """CRIT trajectory array."""
    return JSONResponse(run_scanner.get_crit_trajectory(RUNS_BASE, experiment, run_id))


@app.get("/runs/{experiment}/{run_id}/divergence")
def divergence_trajectory(experiment: str, run_id: str):
    """JS divergence + evidence overlap per phase per round."""
    return JSONResponse(
        run_scanner.get_divergence_trajectory(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/portfolio")
def portfolio_trajectory(experiment: str, run_id: str):
    """Portfolio trajectory across rounds."""
    return JSONResponse(
        run_scanner.get_portfolio_trajectory(RUNS_BASE, experiment, run_id)
    )


@app.get("/runs/{experiment}/{run_id}/round/{round_num}")
def round_detail(experiment: str, run_id: str, round_num: int):
    """Round detail with agent text."""
    detail = run_scanner.get_round_detail(RUNS_BASE, experiment, run_id, round_num)
    if detail is None:
        return JSONResponse({"error": "Round not found"}, status_code=404)
    return JSONResponse(detail)


@app.get("/runs/{experiment}/{run_id}/tree")
def file_tree(experiment: str, run_id: str):
    """File tree for run directory."""
    return JSONResponse(run_scanner.get_file_tree(RUNS_BASE, experiment, run_id))


@app.get("/runs/{experiment}/{run_id}/file")
def read_file(
    experiment: str,
    run_id: str,
    path: str = Query(..., description="Relative path within run directory"),
):
    """Read a file from the run directory.

    Returns 403 on path traversal, 404 on missing file.
    """
    try:
        content = run_scanner.read_run_file(RUNS_BASE, experiment, run_id, path)
    except PermissionError:
        return JSONResponse({"error": "Path traversal rejected"}, status_code=403)
    except FileNotFoundError:
        return JSONResponse({"error": "File not found"}, status_code=404)

    if path.endswith(".json"):
        try:
            parsed = json.loads(content)
            return JSONResponse({"content": parsed, "truncated": False})
        except json.JSONDecodeError:
            pass

    return JSONResponse({"content": content, "truncated": False})


# ------------------------------------------------------------------
# Ablation summary endpoints
# ------------------------------------------------------------------

@app.get("/api/ablation")
def ablation_summary():
    """Return the ablation summary JSON, or a not_generated marker."""
    path = RUNS_BASE / "ablation_summary.json"
    if not path.exists():
        return JSONResponse({"error": "not_generated", "experiments": {}})
    return JSONResponse(json.loads(path.read_text()))


@app.post("/api/ablation/regenerate")
def ablation_regenerate():
    """Re-run aggregate_metrics.py and return the result."""
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_metrics.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return JSONResponse(
            {"status": "error", "detail": result.stderr},
            status_code=500,
        )
    return JSONResponse({"status": "ok"})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
