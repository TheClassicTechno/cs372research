# Industry 100-Scenario Run Summary

## Run Status

- Scenario list: `config/scenarios/industry_js_100/industry_top_scenarios_100.json`
- Debate config: `config/debate/baselines/debate_1_round_no_macro_causal_out_industry_100.yaml`
- Raw tracking file: `results_tracking_debate_1_round_no_macro_causal_out_industry_100.csv`
- Clean latest-valid tracking file: `results_tracking_debate_1_round_no_macro_causal_out_industry_100_clean.csv`
- Aggregated JSON: `results/scenario_runs/aggregated_debate_1_round_no_macro_causal_out_industry_100_industry_top_scenarios_100.json`
- Completed scenarios with non-empty summaries: 100 / 100

The raw tracking CSV contains earlier failed attempts from sandbox networking and dummy-key runs. The analysis below uses the clean CSV, which keeps only the latest valid result for each scenario.

## Scenario Breakdown

The expanded experiment contains 100 total scenarios across three industry groups. Each scenario combines one investment quarter, one industry-specific stock basket, and `_CASH_`. The debate used one round with the VALUE, TECHNICAL, and RISK agents, followed by a judge revision/final allocation. Results are compared against SPY for the same investment window.

| Sector | Scenarios | Share | Representative Tickers | Basket Types |
| --- | ---: | ---: | --- | --- |
| Technology | 34 | 34% | AAPL, AMD, AVGO, MSFT, NVDA | Core mega-cap, AI/cloud, semiconductors, platform, compute, balanced baskets |
| Finance | 33 | 33% | BAC, BLK, GS, JPM, MS | Banks, asset managers, investment banks, consumer finance, diversified financial baskets |
| Energy | 33 | 33% | COP, CVX, OXY, XOM | Integrated majors, upstream producers, oil-beta baskets, balanced energy baskets |

The valid investment quarters span 2021Q4 through 2025Q3. Candidate 2024Q4 scenarios were not used because the local macro file needed for the prior quarter, `macro_2024_Q3.json`, was not available. Those slots were replaced with valid 2021Q4 alternative baskets so the final run still reached 100 complete scenarios.

## Aggregate Financial Results

Across all 100 scenarios, the final judged portfolios produced a positive mean return and outperformed SPY on average. The mean return with cash interest was 6.07%, compared with a 2.51% mean SPY return, giving an average excess return of 3.55 percentage points.

| Metric | Mean | SEM |
| --- | ---: | ---: |
| Return with cash interest | 6.07% | 1.41% |
| Daily total return | 6.07% | 1.41% |
| SPY return | 2.51% | 0.88% |
| Excess return vs SPY | 3.55 pp | 1.21 |
| Annualized Sharpe | 1.03 | 0.21 |
| Annualized Sortino | 1.97 | 0.35 |
| Annualized volatility | 24.65% | 0.85% |
| Max drawdown | 10.86% | 0.60% |
| Mean trades per scenario | 3.43 | 0.07 |
| Mean run duration | 222.46 sec | 3.74 |

## Sector Financial Results

Technology was the strongest sector in this run, with the highest mean return, highest excess return, and highest Sharpe ratio. Energy also produced a strong positive average excess return, but its results were more dependent on the 2022Q1 oil and upstream scenarios. Finance was positive overall, but its average excess return was much smaller because the sector's judged portfolios tracked SPY more closely across several quarters.

| Sector | N | Return | SPY Return | Excess vs SPY | Sharpe | Sortino | Volatility | Max Drawdown | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Technology | 34 | 8.41% | 2.62% | 5.79 pp | 1.40 | 2.87 | 27.98% | 11.22% | 3.53 |
| Finance | 33 | 2.82% | 2.46% | 0.36 pp | 0.62 | 1.22 | 20.96% | 10.77% | 3.58 |
| Energy | 33 | 6.89% | 2.46% | 4.43 pp | 1.06 | 1.81 | 24.95% | 10.57% | 3.18 |

| Sector | Return SEM | Excess SEM | Sharpe SEM | Sortino SEM | Volatility SEM | Max Drawdown SEM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Technology | 3.08 | 1.80 | 0.42 | 0.72 | 1.30 | 1.29 |
| Finance | 1.77 | 1.08 | 0.32 | 0.52 | 0.96 | 0.88 |
| Energy | 2.18 | 2.94 | 0.32 | 0.53 | 1.29 | 0.88 |

## Sector Detail

### Technology

Technology contained 34 scenarios built from large-cap platform companies, semiconductor names, cloud/AI exposure, and more balanced three- to five-stock baskets. It had the strongest average result: 8.41% return, 5.79 percentage points of excess return over SPY, and a 1.40 Sharpe ratio. The sector also had the highest average volatility, at 27.98%, which is expected given the heavier semiconductor and growth-stock exposure.

The strongest technology scenarios came from 2025Q2 and earlier bull-market growth quarters. The top result was `tech_2025Q2_semis_cloud4`, which returned 38.12% and beat SPY by 27.65 percentage points. Other strong cases included `tech_2024Q1_compute3` and `tech_2021Q4_alt_aapl_amd_avgo`, both above 30% return.

The weakest technology scenarios were concentrated in 2022Q2, when growth and semiconductor names fell sharply. `tech_2022Q2_semis_cloud4` returned -25.18%, while `tech_2022Q2_core5` returned -21.52%. Even in this poor quarter, the model's excess return was sometimes only moderately negative because SPY also declined sharply.

| Technology Scenario | Return | SPY | Excess | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tech_2025Q2_semis_cloud4` | 38.12% | 10.47% | 27.65 pp | 3.47 | 9.69% |
| `tech_2024Q1_compute3` | 33.97% | 11.01% | 22.95 pp | 4.26 | 4.87% |
| `tech_2021Q4_alt_aapl_amd_avgo` | 31.59% | 9.76% | 21.82 pp | 4.10 | 5.06% |
| `tech_2022Q2_semis_cloud4` | -25.18% | -16.35% | -8.83 pp | -2.78 | 26.62% |
| `tech_2022Q2_core5` | -21.52% | -16.35% | -5.17 pp | -2.65 | 23.58% |

### Finance

Finance contained 33 scenarios across large banks, asset managers, investment banking, and diversified financial baskets. The sector was positive overall, but much less differentiated from SPY than technology or energy. The average finance scenario returned 2.82%, while SPY returned 2.46%, producing only 0.36 percentage points of average excess return.

The strongest finance scenarios occurred in favorable market quarters for banks and capital-markets names. `finance_2025Q2_asset_markets4` returned 18.40%, while `finance_2023Q4_diversified3` and `finance_2023Q4_core5` both returned roughly 18%. These runs had strong Sharpe ratios, especially in 2023Q4 when the portfolios benefited from a broad market rally.

The weakest finance scenarios were also concentrated in 2022Q2. However, because SPY fell sharply in that quarter, several negative finance scenarios still outperformed SPY. For example, `finance_2022Q2_core5` returned -14.20% while SPY returned -16.35%, producing positive excess return despite an absolute loss.

| Finance Scenario | Return | SPY | Excess | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `finance_2025Q2_asset_markets4` | 18.40% | 10.47% | 7.93 pp | 2.15 | 13.33% |
| `finance_2023Q4_diversified3` | 18.18% | 11.68% | 6.49 pp | 4.32 | 6.43% |
| `finance_2023Q4_core5` | 18.01% | 11.68% | 6.32 pp | 3.65 | 7.26% |
| `finance_2022Q2_diversified3` | -17.09% | -16.35% | -0.75 pp | -2.91 | 17.09% |
| `finance_2022Q2_core5` | -14.20% | -16.35% | 2.15 pp | -2.23 | 15.50% |

### Energy

Energy contained 33 scenarios across integrated oil majors, upstream producers, and oil-beta baskets. The sector returned 6.89% on average and beat SPY by 4.43 percentage points. Its strongest results came from periods where energy had a very different return profile from the broader market, especially 2022Q1.

The best energy scenarios were the strongest individual sector results in the full experiment on an excess-return basis. `energy_2022Q1_core4` returned 37.81% while SPY fell -5.16%, creating 42.98 percentage points of excess return. `energy_2022Q1_upstream3` and `energy_2022Q1_integrated_upstream3` showed the same pattern, with large absolute gains during a weak broad-market quarter.

The weakest energy scenarios occurred when the broad market was strong but energy-specific baskets lagged. In 2023Q4 and 2025Q2, several energy scenarios lost money while SPY gained more than 10%. This created the largest negative excess-return examples in the run, including `energy_2023Q4_majors3` at -21.68 percentage points versus SPY.

| Energy Scenario | Return | SPY | Excess | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `energy_2022Q1_core4` | 37.81% | -5.16% | 42.98 pp | 4.74 | 7.74% |
| `energy_2022Q1_upstream3` | 37.57% | -5.16% | 42.73 pp | 4.96 | 5.93% |
| `energy_2022Q1_integrated_upstream3` | 34.07% | -5.16% | 39.23 pp | 4.55 | 8.03% |
| `energy_2023Q4_majors3` | -10.00% | 11.68% | -21.68 pp | -2.45 | 12.71% |
| `energy_2025Q2_oil_beta3` | -9.37% | 10.47% | -19.84 pp | -1.15 | 16.31% |

## JS Divergence

JS divergence was computed across agent allocations from each run's `round_state.json`. Proposal JS measures how different the agents were before debate revision; revision JS measures how different the revised allocations were after the debate step. A negative change means agents became more similar after debate.

| Metric | Mean | SEM |
| --- | ---: | ---: |
| Proposal JS | 0.0383 | 0.0029 |
| Revision JS | 0.0270 | 0.0016 |
| Change | -0.0113 | 0.0030 |

By sector:

| Sector | N | Proposal JS | Revision JS | Change |
| --- | ---: | ---: | ---: | ---: |
| Technology | 34 | 0.0491 | 0.0265 | -0.0226 |
| Finance | 33 | 0.0372 | 0.0277 | -0.0095 |
| Energy | 33 | 0.0281 | 0.0266 | -0.0015 |

Technology had the largest debate convergence: proposal JS fell from 0.0491 to 0.0265. Finance also converged, though less dramatically. Energy had almost no average convergence, suggesting the agents began closer together or that the debate step changed allocations less for energy baskets.

## Interpretation

The 100-scenario expansion is a stronger empirical base than the original 12-scenario pilot because it tests multiple baskets, sectors, and market regimes instead of a small number of hand-selected examples. The final set is approximately balanced across technology, finance, and energy, with each sector represented by 33 or 34 scenarios.

Financially, the judged portfolios were positive on average and outperformed SPY by 3.55 percentage points across the full run. Technology was the clearest contributor to average outperformance, while energy added strong but more regime-dependent excess returns. Finance was the weakest sector by excess return, although it was still slightly positive relative to SPY.

The JS divergence results suggest that debate generally made agent allocations more similar, especially in technology. This supports the idea that the debate step can reduce disagreement among agents before the final judge allocation. However, the small change in energy indicates that convergence is not uniform across sectors and may depend on how distinct the agents' initial sector views are.
