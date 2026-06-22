# Industry Scenario Rerun Summary

## Run Status

- Tracking file: `results_tracking_debate_1_round_no_macro_causal_out_industry_rerun.csv`
- Scenarios completed: 12 / 12
- Missing/error rows: 0
- Summary artifacts found: 12 / 12
- Model/config: `gpt-5-mini`, `debate_1_round_no_macro_causal_out_industry_rerun`

## Aggregate Results

Across the 12 industry scenarios, the rerun produced:

- Total return with cash interest: 2.72% mean, 3.49% SEM
- Daily total return: 2.72% mean, 3.49% SEM
- SPY return: 3.12% mean, 2.57% SEM
- Excess return vs SPY: -0.40 percentage points mean, 3.13 SEM
- Annualized Sharpe: 0.66 mean, 0.57 SEM
- Annualized Sortino: 1.32 mean, 0.92 SEM
- Max drawdown: 11.70% mean, 1.71 SEM
- Runtime: 220.6 seconds mean, 13.6 SEM

## Sector Breakdown

| Sector | N | Return | Excess vs SPY | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Finance | 4 | 6.26% | 0.34 pp | 1.64 | 8.02% |
| Energy | 4 | 1.87% | 0.07 pp | 0.25 | 14.65% |
| Tech | 4 | 0.03% | -1.62 pp | 0.08 | 12.42% |

## Scenario Ranking

| Scenario | Return | Excess vs SPY | SPY | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `energy_2022Q4_resilient_cashflows` | 21.97% | 17.18 pp | 4.79% | 2.76 | 11.67% |
| `tech_2024Q1_ai_expansion` | 21.88% | 10.86 pp | 11.01% | 3.21 | 4.13% |
| `finance_2023Q4_higher_for_longer` | 16.04% | 4.35 pp | 11.68% | 4.07 | 6.47% |
| `finance_2025Q3_broad_capital_markets` | 9.63% | 1.47 pp | 8.16% | 2.42 | 2.76% |
| `finance_2021Q4_pre_hike_banks` | 0.79% | -8.97 pp | 9.76% | 0.25 | 8.76% |
| `tech_2022Q4_post_selloff` | 0.60% | -4.20 pp | 4.79% | 0.12 | 9.03% |
| `energy_2023Q2_normalization` | -0.20% | -8.47 pp | 8.27% | -0.16 | 9.98% |
| `finance_2022Q3_repricing` | -1.40% | 4.52 pp | -5.93% | -0.19 | 14.11% |
| `energy_2022Q2_commodity_spike` | -4.31% | 12.04 pp | -16.35% | -0.28 | 20.40% |
| `tech_2022Q1_rate_shock` | -8.52% | -3.36 pp | -5.16% | -0.80 | 20.51% |
| `energy_2025Q2_late_cycle_mix` | -9.99% | -20.46 pp | 10.47% | -1.33 | 16.55% |
| `tech_2025Q1_ai_leadership` | -13.83% | -9.80 pp | -4.03% | -2.20 | 16.01% |

## JS Divergence

JS divergence was computed across the three agent allocations. The logged value in each run's `round_state.json` matches the revision-stage JS divergence.

- Proposal JS: 0.0411 mean, 0.0063 SEM
- Revision/final-round JS: 0.0246 mean, 0.0049 SEM
- Mean change after critique/revision: -0.0165
- Scenarios with lower JS after revision: 9 / 12

| Scenario | Proposal JS | Revision JS | Change |
| --- | ---: | ---: | ---: |
| `tech_2022Q4_post_selloff` | 0.0283 | 0.0649 | +0.0365 |
| `tech_2022Q1_rate_shock` | 0.0173 | 0.0085 | -0.0088 |
| `tech_2024Q1_ai_expansion` | 0.0826 | 0.0232 | -0.0594 |
| `finance_2021Q4_pre_hike_banks` | 0.0186 | 0.0229 | +0.0043 |
| `tech_2025Q1_ai_leadership` | 0.0337 | 0.0083 | -0.0253 |
| `finance_2022Q3_repricing` | 0.0367 | 0.0108 | -0.0260 |
| `finance_2023Q4_higher_for_longer` | 0.0694 | 0.0404 | -0.0289 |
| `finance_2025Q3_broad_capital_markets` | 0.0507 | 0.0252 | -0.0255 |
| `energy_2022Q2_commodity_spike` | 0.0320 | 0.0142 | -0.0178 |
| `energy_2022Q4_resilient_cashflows` | 0.0198 | 0.0219 | +0.0021 |
| `energy_2023Q2_normalization` | 0.0679 | 0.0418 | -0.0262 |
| `energy_2025Q2_late_cycle_mix` | 0.0365 | 0.0133 | -0.0233 |

Sector averages:

| Sector | N | Proposal JS | Revision JS | Change |
| --- | ---: | ---: | ---: | ---: |
| Energy | 4 | 0.0390 | 0.0228 | -0.0163 |
| Finance | 4 | 0.0439 | 0.0248 | -0.0190 |
| Tech | 4 | 0.0405 | 0.0262 | -0.0143 |

## Interpretation

The rerun is complete and usable. It does not show broad benchmark outperformance on average, but it does show sector-specific variation: finance was the strongest group, energy was roughly benchmark-neutral on average despite large dispersion, and tech lagged due to weak 2022Q1 and 2025Q1 scenarios. The strongest positive cases were `energy_2022Q4_resilient_cashflows`, `tech_2024Q1_ai_expansion`, and `finance_2023Q4_higher_for_longer`.

The JS divergence results suggest that the one-round debate mostly produced convergence: average disagreement fell by about 40% from proposal to revision stage. The main exception was `tech_2022Q4_post_selloff`, where revision-stage disagreement increased sharply, indicating agents became less aligned after critique.

## Validation Notes

- `tools/dashboard/tests/test_architecture.py` passes after regenerating `tools/dashboard/rules/warning_baseline.json`.
- Dashboard-wide tests still have environment/fixture failures unrelated to the industry rerun: missing `scipy`, missing `logging/runs/test/run_2026-03-07_19-50-06`, and tests expecting older canonical metrics fixtures.
- `scripts/compare_per_scenario.py` cannot run in the current local environment because `.venv_industry` has `numpy` but not `scipy`.
