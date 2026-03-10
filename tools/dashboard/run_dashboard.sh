#!/usr/bin/env bash

# Kill any process currently using port 8000 (dashboard)
lsof -ti :8000 | xargs kill 2>/dev/null

# Start the dashboard
uv run python -m tools.dashboard.server