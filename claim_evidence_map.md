# Claim → Evidence Map — checklist.md §5

"Verified" means recomputed from saved artifacts this session, not read off the
manuscript. Regenerate with `python3 analysis/verify_paper_tables.py`.

| Claim | Evidence | Metric | Verified? | Wording used |
|---|---|---|---|---|
| CRIT does not predict performance | author records, n=210 | Pearson r | **No — source logs absent** | "not predictive on our scenario set" |
| Debate increases CRIT (prompting does) | author records | $\bar\rho$ 0.72→0.84, d≈2.0 | **No — source logs absent** | stated with the coupling caveat |
| Debate beats 1/N only 33% of time | author records | binomial p<0.001 | **No — source logs absent** | as stated |
| Causal scaffolding raises return/Sharpe | `results/scenario_runs`, n=35 paired | +0.47pp p=0.024 d=0.40; Sharpe +0.08 p=0.043 | **Yes** | scoped to the ensemble baseline, and explicitly *not* to the single agent (p=0.37) |
| Enriched prompts cut drawdown | same | −1.54pp, p<0.001, d=−0.92 | **Yes** | "reliably reduces risk" |
| Enriched prompts cut volatility | same | −0.04, p<0.001, d=−1.02 | **Yes** | as stated |
| RCA-regulated debate reduces risk | same, n=35 paired | vol −0.02 p<0.001 d=−0.88; DD −0.69pp p=0.002 d=−0.56 | **Yes** | "reduced risk without improving returns" |
| RCA improves return/Sharpe | same | return −0.32pp p=0.25; Sharpe −0.05 p=0.31 | **Yes — null** | reported as indistinguishable from zero |
| A7 JSD intervention improves Sharpe | author records, n=35 | +0.14, p=0.028, d=0.39 | **No — source logs absent** | scoped to two-agent config + scenario set |
| A8 effect collapses with 3rd agent | author records | +0.04, p=0.53 | **No** | as stated |
| A10 causal repair does not help | author records | Sharpe −0.09 p=0.22 | **No** | as stated |
| Multi-round debate lowers diversity | trace inspection | JSD drop after first revision; cash 10–50%→55–61% | **Qualitative** | described as a trace observation |
| Consensus reflects sycophancy | §3.5 operational definition | 3-part criterion | **Partially** | "consistent with sycophancy"; mechanism explicitly not isolated |
| Findings not model-family-specific | N=2–17 cross-provider | mixed | **No** | "supporting rather than confirmatory" |
| Run-to-run noise floor | `results/seed_variance`, 30 runs | SD 1.30pp return, 0.159 Sharpe | **Yes** | Figure 1, Appendix |

## Table-level verification

Every cell of **Table 2** (`tab:prompts`) and **Table 5** (`tab:debate_rca`)
regenerates exactly from `results/scenario_runs/` via
`scripts/compare_multiple_configs.py`, including all significance superscripts, at
n=35 common scenarios. Confirmed independently this session.

Bootstrap 95% CIs added at the **scenario** level (10,000 resamples, seed
20260829) — the correct independent unit, not agent-turn or trading-day.

## Numbers deliberately NOT used

- Any stratified-by-role or stratified-by-regime CRIT correlation. No per-round
  CRIT corpus exists in this repository; `analysis/stratified_analysis.py` correctly
  reports "no data loaded" rather than producing anything.
- Any re-derived value for the two configurations whose config files are lost.
