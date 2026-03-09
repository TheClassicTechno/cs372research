#!/usr/bin/env bash
# Rewrite guard stage
# Run from tools/dashboard/
set -euo pipefail

CI_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Rewrite Guard ==="
python "$CI_DIR/check_structure.py" --check-rewrite
