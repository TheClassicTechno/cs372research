#!/usr/bin/env bash
# Architecture checker stage
# Run from tools/dashboard/
set -euo pipefail

DASHBOARD_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Architecture Checker ==="
python3 "$DASHBOARD_DIR/rules/check_dashboard_architecture.py"
