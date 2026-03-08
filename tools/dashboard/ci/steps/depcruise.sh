#!/usr/bin/env bash
# Dependency Cruiser stage
# Run from tools/dashboard/
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== Dependency Cruiser ==="
npx depcruise static/js --config .dependency-cruiser.cjs
