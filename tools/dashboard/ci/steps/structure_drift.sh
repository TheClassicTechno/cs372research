#!/usr/bin/env bash
# Structure drift stage
# Run from tools/dashboard/
set -euo pipefail

CI_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Structure Drift Check ==="
python3 "$CI_DIR/check_structure.py" --check-structure
