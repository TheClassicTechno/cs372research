# CRIT artifact audit (2026-11-15)

Question: which CRIT (reasoning-score) records behind the paper's numbers still exist, and can
those numbers be regenerated from them? Paths are relative to the repo root.

## Summary

| Result | Status (of the 4 states) | Reproducible? |
|---|---|---|
| RQ1: 210-run CRIT vs. Sharpe/return correlations (Table 1, Table 4, Fig. 3) | **State 1**: the original per-round CRIT outputs exist, and their values match the published ones | **Yes.** ρ̄ is recomputed from the raw per-round CRIT outputs; pooled r = 0.0735 (Sharpe) |
| A7 / A8 / A10 paired Sharpe effects | State 1 (same run directories) | **Yes.** +0.143 (p = 0.028), +0.044 (p = 0.530), −0.089 (p = 0.219) |
| ρ̄ 0.72 → 0.84 prompt-tier gain, Cohen's d ≈ 2.0, pillar gains (Appendix `tab:crit_gains`) | **State 3**: only aggregate numbers in the manuscript | **No.** No per-run CRIT records for these runs exist anywhere I searched |
| Table 2 / RQ2 prompt-tier financial results | n/a: these runs did not compute CRIT | Financial metrics yes (`results/scenario_runs/`); CRIT not applicable |

No state-4 reconstruction (a re-scoring with a new evaluator) was found or performed.

## Evidence

**210-run corpus: `CS372_210_debate_runs/`** (also `~/Downloads/CS372_210_debate_runs.zip`)
- `PACKAGE_MANIFEST.json`: run membership, with per-file SHA-256 checksums.
- `logging/runs/vskarich_ablation_{7,8,10}/run_*/final/pid_crit_all_rounds.json`: 210 of 210
  present. Each record is a `crit_round` with `rho_bar` and `rho_i`, and for each agent the
  `pillars` {LV, ES, AC, CA}, `diagnostics`, and the scorer's written `explanations`.
- `.../rounds/round_001/CRIT/<agent>/{prompt.txt,response.txt}`: the raw scorer prompt and response.
- `.../_dashboard/financial_metrics.json`: return, annualized Sharpe, max drawdown, and SPY return.
- Check run on 2026-11-15: the final-round `rho_bar` read from all 210 JSONs equals
  `analysis/rq1/per_run_results.csv` (max |diff| = 0). Per-agent ρ_i equals the mean of the four
  pillars. Pooled r(ρ̄, Sharpe) = 0.0735, matching `analysis/rq1/rq1_clustered_stats.json`.
- Note: `run_path` and `crit_source` in `per_run_results.csv` are relative to
  `CS372_210_debate_runs/`, not the repo root (the repo's own `logging/runs/` is nearly empty).

**0.72 → 0.84**
- `claim_evidence_map.md` lists it as "author records … No — source logs absent".
  `analysis/verify_paper_tables_report.txt` says the same.
- A search for `pid_crit_all_rounds.json` outside `CS372_210_debate_runs/` (in the repo and in
  `~/Downloads`) found 0 files. The `results/scenario_runs/*/summary.json` files have no CRIT fields.
- Conclusion: the number appears only in the manuscript. It cannot be regenerated, and which runs
  it came from is not recorded.

## Provenance details the poster now uses
- A10 threshold: `vskarich_ablations/ablation_10/debate_configs/causal_intervention_th075.yaml`,
  `causal_alignment: 0.75` for both agents. This matches the camera-ready paper. No 0.8 value
  exists for A10; 0.8 is the A7/A8 JS-ratio threshold.
- A7/A8 trigger: `js_collapse.threshold: 0.8` with `min_js_proposal: 0.10`. The rule is skipped
  when proposal divergence is < 0.10, so it never divides by zero (`eval/interventions/rules.py`).
- JSD (`eval/divergence.py::generalized_js_divergence`): the mean base-2 KL divergence of each
  agent's allocation from the mean allocation. It works for any number of agents (A8 has 3).
- Revision limit (A7/A8/A10 retries): a **prompt instruction**, not enforced in code
  (`multi_agent/runner.py` ~L1249): "change AT MOST 2 tickers; each by no more than 5% of total
  allocation". Followed in 38 of 42 retried A7 revisions (`analysis/rq1/intervention_audit.json`).
- Sharpe (`eval/financial.py`): mean daily return minus the daily risk-free rate, divided by the
  daily return SD, times √252.
- Drawdown noise floor: computed from the same 30 repeat runs (`results/seed_variance/`) and by
  the same method as the published return floor (which reproduces 0.31 pp exactly). Result:
  0.23 pp. The paper's Fig. 1 reuses the return band (0.31 pp) on its drawdown panel; the
  conclusions are unchanged either way.
