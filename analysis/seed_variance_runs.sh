#!/usr/bin/env bash
# Run-to-run variance probe.
#
# Question: how much of the between-condition difference reported in Table 2
# is attributable to LLM sampling noise at FIXED configuration and FIXED
# scenario?
#
# Independent variable : repeated sampling only (temperature 0.3, same config,
#                        same scenario, same evaluation window)
# Dependent variables  : total return %, annualized Sharpe
# Control              : configuration and scenario are held constant; the only
#                        thing that varies between repeats is the sampling draw.
#
# Config is single_agent_slim, i.e. column A of Table 2, so the resulting
# spread is directly comparable to that table's effect sizes.
set -u

CONFIG="config/debate/baselines/single_agent_slim.yaml"
OUT="results/seed_variance"
REPEATS=6

SCENARIOS=(
  "config/scenarios/baseline_sweep/scenario_2022Q1.yaml"
  "config/scenarios/baseline_sweep/scenario_2022Q4_tech.yaml"
  "config/scenarios/baseline_sweep/scenario_2023Q2.yaml"
  "config/scenarios/baseline_sweep/scenario_2025Q1_tech.yaml"
  "config/scenarios/baseline_sweep/scenario_2022Q3_rand3_1.yaml"
)

mkdir -p "$OUT"
echo "start $(date -u +%FT%TZ)  config=$CONFIG  repeats=$REPEATS  scenarios=${#SCENARIOS[@]}"

for s in "${SCENARIOS[@]}"; do
  name=$(basename "$s" .yaml)
  for i in $(seq 1 "$REPEATS"); do
    dest="$OUT/${name}__rep${i}"
    if [ -f "$dest/single_agent_slim/summary.json" ]; then
      echo "skip $name rep$i (already present)"; continue
    fi
    t0=$(date +%s)
    python3 run_simulation.py --agents "$CONFIG" --scenario "$s" \
      --output-dir "$dest" --no-display > "$OUT/${name}__rep${i}.log" 2>&1
    rc=$?
    echo "$name rep$i rc=$rc $(( $(date +%s) - t0 ))s"
  done
done

echo "done $(date -u +%FT%TZ)"
