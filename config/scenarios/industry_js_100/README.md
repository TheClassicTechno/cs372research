# Industry JS Scenario Set: 100 Scenarios

Expanded industry-stratified scenario set for debate/JS-divergence experiments.

## Counts

- Tech: 34 scenarios
- Finance: 33 scenarios
- Energy: 33 scenarios
- Total: 100 scenarios

Each industry spans 14 valid invest quarters from `2021Q4` through `2025Q3`, excluding quarters without prior-quarter coverage. Scenarios rotate through industry-specific ticker baskets while respecting `max_tickers: 5`.

## Main Scenario List

Use:

```bash
config/scenarios/industry_js_100/industry_top_scenarios_100.json
```

Per-industry lists are also available:

```bash
config/scenarios/industry_js_100/tech_top_scenarios.json
config/scenarios/industry_js_100/finance_top_scenarios.json
config/scenarios/industry_js_100/energy_top_scenarios.json
```

## Run Command

Use the dedicated config alias so the 100-scenario run writes a separate tracking CSV:

```bash
python3 scripts/run_scenario_list.py \
  --scenarios config/scenarios/industry_js_100/industry_top_scenarios_100.json \
  --config config/debate/baselines/debate_1_round_no_macro_causal_out_industry_100.yaml \
  --parallel 3
```

Expected tracking file:

```bash
results_tracking_debate_1_round_no_macro_causal_out_industry_100.csv
```

## Preflight Status

Offline validation passed:

- 100 unique scenario YAML files
- 0 missing required prior/invest-quarter asset files
- 14-quarter coverage for tech, finance, and energy
- Basket variants: tech 8, finance 8, energy 5
