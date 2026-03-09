#!/usr/bin/env bash
# run_experiment.sh — runs a simulation and saves output to a timestamped log file
#
# Usage:
#   ./run_experiment.sh --agents config/debate/experiment_role_llm_A.yaml --scenario config/scenarios/2022Q1_inflation_shock.yaml
#
# Output is saved to: logs/<config_name>_<timestamp>.log

set -euo pipefail

# Extract config name from --agents arg for the log filename
CONFIG_NAME="unknown"
prev=""
for i in "$@"; do
    case "$prev" in
        --agents)
            # e.g. config/debate/experiment_role_llm_A.yaml → experiment_role_llm_A
            CONFIG_NAME="$(basename "$i" .yaml)"
            ;;
    esac
    prev="$i"
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "=== Logging output to: $LOG_FILE ==="
uv run python run_simulation.py "$@" 2>&1 | tee "$LOG_FILE"
echo ""
echo "=== Output saved to: $LOG_FILE ==="
