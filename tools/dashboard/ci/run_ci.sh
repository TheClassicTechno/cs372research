#!/usr/bin/env bash
# Local CI pipeline for the Debate Dashboard.
# Runs all static checks and tests in order, fails on first failure.
#
# Usage:
#   ./ci/run_ci.sh           # from tools/dashboard/
#   tools/dashboard/ci/run_ci.sh   # from repo root
set -euo pipefail

CI_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_DIR="$(cd "$CI_DIR/.." && pwd)"
ARTIFACTS="$CI_DIR/ci_artifacts"

rm -rf "$ARTIFACTS"
mkdir -p "$ARTIFACTS"

# Track results for summary
declare -a STAGE_NAMES=()
declare -a STAGE_RESULTS=()
declare -a STAGE_TIMES=()

run_stage() {
  local name="$1"
  local script="$2"
  STAGE_NAMES+=("$name")

  echo ""
  echo "================================"
  echo "STAGE: $name"
  echo "================================"

  local start
  start=$(date +%s)

  if bash "$script" 2>&1 | tee "$ARTIFACTS/${name}.log"; then
    local end
    end=$(date +%s)
    local elapsed=$(( end - start ))
    STAGE_RESULTS+=("ok")
    STAGE_TIMES+=("${elapsed}s")
    echo ""
    echo "--- $name: PASSED (${elapsed}s) ---"
  else
    local end
    end=$(date +%s)
    local elapsed=$(( end - start ))
    STAGE_RESULTS+=("fail")
    STAGE_TIMES+=("${elapsed}s")
    echo ""
    echo "--- $name: FAILED (${elapsed}s) ---"
    print_summary
    exit 1
  fi
}

print_summary() {
  echo ""
  echo "================================"
  echo "CI SUMMARY"
  echo "================================"
  for i in "${!STAGE_NAMES[@]}"; do
    local name="${STAGE_NAMES[$i]}"
    local result="${STAGE_RESULTS[$i]}"
    local elapsed="${STAGE_TIMES[$i]}"
    if [ "$result" = "ok" ]; then
      printf "  %-24s ✓  %s\n" "$name" "$elapsed"
    else
      printf "  %-24s ✗  %s\n" "$name" "$elapsed"
    fi
  done
  echo ""
  echo "Logs: $ARTIFACTS/"
}

cd "$DASHBOARD_DIR"

TOTAL_START=$(date +%s)

run_stage "eslint"              "$CI_DIR/steps/eslint.sh"
run_stage "dependency-cruiser" "$CI_DIR/steps/depcruise.sh"
run_stage "semgrep"            "$CI_DIR/steps/semgrep.sh"
run_stage "architecture"       "$CI_DIR/steps/architecture.sh"
run_stage "structure-drift"    "$CI_DIR/steps/structure_drift.sh"
run_stage "rewrite-guard"      "$CI_DIR/steps/rewrite_guard.sh"
run_stage "tests"              "$CI_DIR/steps/tests.sh"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

print_summary
echo "Total: ${TOTAL_ELAPSED}s"
echo "All stages passed."
