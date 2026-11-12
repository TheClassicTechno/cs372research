# Task 1: RQ1 numerical results

Fresh GPT-5 re-scoring produced 490 valid per-agent scores aggregated into 210 runs on 35 reused scenarios. A7, A8 and A10 each have 35 baseline and 35 treatment runs. This directory contains numerical tables and statistical reports; raw prompts/responses and manuscript assets are not included.

| Analysis | N | Fresh Pearson r | Scenario-clustered 95% CI |
| --- | ---: | ---: | ---: |
| Pooled score vs Sharpe | 210 | +0.0744 | [-0.1365, +0.2574] |
| Pooled score vs return | 210 | +0.0232 | [-0.1813, +0.2250] |
| Within-scenario score vs Sharpe | 210 | +0.0224 | [-0.1237, +0.1614] |
| Within-scenario score vs return | 210 | +0.0065 | [-0.1337, +0.1355] |
| Paired score change vs Sharpe change | 105 | +0.0466 | [-0.1302, +0.2104] |
| Paired score change vs return change | 105 | +0.0569 | [-0.1190, +0.2346] |

All six 95% intervals include zero. This does not prove that the population association is zero or that reasoning metrics generally are uninformative. The fresh paired-return 90% interval extends outside the post-hoc ±0.20 screen; historical screen claims must not be transferred unchanged.

Historical mean score: 0.8371; fresh mean: 0.8359. Old-versus-fresh run-score Pearson correlation: 0.7511.

## Files

- `results/per_run_results.csv` and `per_run_results_fresh.csv`: historical and fresh scores by run, scenario, ablation and condition, with unchanged Sharpe and return (percentage units).
- `results/table1_fresh_vs_historical.csv` and `table1_historical.csv`: correlation comparisons.
- `results/appendix_c1_report.txt` and `appendix_c1_fresh_report.txt`: full numerical reports. Their conventional p-values assume independence; use the clustered statistics for primary uncertainty.
- `results/fresh_clustered_stats.json` and `historical_clustered_stats.json`: clustered intervals and sensitivity analyses.
- `results/fresh_summary.json` and `fresh_scoring_status.json`: aggregate comparison and completed-scoring status. Token usage covers successful final evaluations only, excluding failed/retried requests.

## Evaluation details

The audit used main commit `18c80f90267f4e76899d2a3c98b16649125d37a7`. The original saved analysis was reproduced and retained separately from fresh scores. The reported 0.72-to-0.84 prompt-improvement result belongs to another corpus and is not independently reproduced here.

Fresh scoring requested `gpt-5` and returned `gpt-5-2025-08-07`, using archived system/user messages via Chat Completions, medium reasoning effort, a 12,000-completion-token ceiling, no temperature override, and store=false. Leading/trailing logger whitespace was stripped. Each agent score averages four pillars; fresh run scores average agents at full precision rather than historical four-decimal rounding. The surviving original runner used Responses without an explicit output ceiling, so this is a documented re-evaluation rather than a byte-identical recreation of API settings.

Clustered statistics reuse the existing main-branch method: 10,000 scenario-cluster bootstrap draws, seed 20261030. The 210 repeated-condition runs are not 210 independent scenarios.

Full evaluator provenance and scoring records remain in the local Task 1 handoff bundle. Only numerical results and this description are included in this branch.
