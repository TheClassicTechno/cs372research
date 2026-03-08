#!/usr/bin/env bash
# ESLint stage
# Run from tools/dashboard/
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== ESLint ==="
npx eslint static/js
