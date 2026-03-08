#!/usr/bin/env bash
# Semgrep stage
# Run from tools/dashboard/
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== Semgrep ==="
semgrep --config semgrep/dashboard_rules.yml static/js
