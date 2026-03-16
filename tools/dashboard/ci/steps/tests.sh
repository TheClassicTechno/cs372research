#!/usr/bin/env bash
# Test stage: Playwright dashboard integration tests
# Run from tools/dashboard/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DASHBOARD_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Playwright Integration Tests ==="
cd "$REPO_ROOT"
# Activate project venv if available
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi
python3 -m pytest "$DASHBOARD_DIR/tests/" -v -o "addopts="
