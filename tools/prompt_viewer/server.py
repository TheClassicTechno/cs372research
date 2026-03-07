"""Local FastAPI server for browsing prompt traces.

Usage:
    python tools/prompt_viewer/server.py

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Prompt Viewer")

LOGS_PATH = Path("logs/prompt_traces.jsonl")
HTML_PATH = Path(__file__).parent / "index.html"


@app.get("/")
def index():
    """Serve the single-page viewer UI."""
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/logs")
def logs(
    model: str | None = Query(default=None),
    search: str | None = Query(default=None),
):
    """Return prompt traces as a JSON array.

    Optional filters:
    - ``model``: only entries whose model field contains this substring
    - ``search``: only entries where system, user, or response contains this substring
    """
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
