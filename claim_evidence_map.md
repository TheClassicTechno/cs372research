# Claim → Evidence Map — checklist.md §5

"Verified" means recomputed from saved artifacts this session, not read off the
manuscript. Regenerate with `python3 analysis/verify_paper_tables.py`.

| Claim | Evidence | Metric | Verified? | Wording used |
|---|---|---|---|---|
| No detectable CRIT–outcome association (pooled) | `CS372_210_debate_runs/` → `analysis/rq1/per_run_results.csv` (n=210, 35 scenario clusters) → `analysis/rq1/rq1_clustered_stats.json` | Pearson r, scenario-cluster bootstrap 95% CI (10,000, seed 20261030) | **Yes** (r reproduces Table 1 to 4 dp; package checksums match) | Sharpe +0.07 [−0.13, +0.25]; return +0.03 [−0.16, +0.21]; "a modest pooled correlation cannot be excluded" |
| Within-scenario correlation is near zero and bounded | same JSON (`within:*`, `fe:*`) | demeaned r, 95%/90% cluster CIs | **Yes** | Sharpe −0.02 [−0.16, +0.12]; return −0.01 [−0.12, +0.10]; "\|r\|>≈0.16 ruled out (0.21 under quarter clustering)" |
| Paired Δρ̄ does not track ΔSharpe/Δreturn | same JSON (`paired:*`), n=105 pairs | r, cluster CI | **Yes** | −0.02 [−0.22, +0.18]; +0.01 [−0.19, +0.21] |
| Equivalence at δ=0.20 | same JSON (`equiv_screen_inside_delta`) | 90% CI inside ±0.20 | **Yes, with caveat** | δ chosen post hoc; called "a screen, not a formal equivalence test"; pooled CRIT–Sharpe fails it |
| Debate increases CRIT (prompting does) | author records | $\bar\rho$ 0.72→0.84, d≈2.0 | **No — source logs absent** | stated with the coupling caveat |
| Debate beats 1/N only 33% of time | author records | binomial p<0.001 | **No — source logs absent** | as stated |
| Causal scaffolding raises return/Sharpe | `results/scenario_runs`, n=35 paired | +0.47pp p=0.024 d=0.40; Sharpe +0.08 p=0.043 | **Yes** | scoped to the ensemble baseline, and explicitly *not* to the single agent (p=0.37) |
| Enriched prompts cut drawdown | same | −1.54pp, p<0.001, d=−0.92 | **Yes** | "reliably reduces risk" |
| Enriched prompts cut volatility | same | −0.04, p<0.001, d=−1.02 | **Yes** | as stated |
| RCA-regulated debate reduces risk | same, n=35 paired | vol −0.02 p<0.001 d=−0.88; DD −0.69pp p=0.002 d=−0.56 | **Yes** | "reduced risk without improving returns" |
| RCA improves return/Sharpe | same | return −0.32pp p=0.25; Sharpe −0.05 p=0.31 | **Yes — null** | reported as indistinguishable from zero |
| A7 JSD intervention improves Sharpe | `analysis/rq1/per_run_results.csv`, n=35 paired | +0.143, t=2.300, p=0.0277, d=0.389; return +0.83pp p=0.0523 | **Yes** (rechecked 2026-10-31, paired t) | scoped to two-agent config + scenario set |
| A8 effect collapses with 3rd agent | same | Sharpe +0.044, p=0.53 | **Yes** (rechecked 2026-10-31) | as stated |
| A10 causal repair does not help | same | Sharpe −0.089, p=0.22 (MaxDD +0.53pp not rechecked: not in per-run table) | **Yes** for Sharpe (rechecked 2026-10-31) | as stated |
| Multi-round debate lowers diversity | trace inspection | JSD drop after first revision; cash 10–50%→55–61% | **Qualitative** | described as a trace observation |
| Consensus reflects sycophancy | §3.5 operational definition | 3-part criterion | **Partially** | "consistent with sycophancy"; mechanism explicitly not isolated |
| Findings not model-family-specific | N=2–17 cross-provider | mixed | **No** | "supporting rather than confirmatory" |
| Run-to-run noise floor | `results/seed_variance`, 30 runs | SD 1.30pp return, 0.159 Sharpe | **Yes** | Figure 1, Appendix |
| Uninvested cash is deliberate `_CASH_` weight, not a broker artifact | `analysis/cash_audit.py` → `analysis/cash_audit.json`; 32 of 35 scenarios per config, purchases rebuilt from entry prices | rounding ≤ ≈2 pts; deliberate ≥ 5.7–15.2 pts | **Yes** (bounded, allocations not stored) | §2: "share rounding explains at most ≈2 points and the rest is deliberate `_CASH_` weight" |
| Cash converges 10–50% → 55–61% (4-agent 5-round) | trace inspection | `_CASH_` weights | **No — logs not retained** | labelled as such in §3.5 |

## Table-level verification

Every cell of **Table 2** (`tab:prompts`) and **Table 5** (`tab:debate_rca`)
regenerates exactly from `results/scenario_runs/` via
`scripts/compare_multiple_configs.py`, including all significance superscripts, at
n=35 common scenarios. Confirmed independently this session.

Bootstrap 95% CIs added at the **scenario** level (10,000 resamples, seed
20260829) — the correct independent unit, not agent-turn or trading-day.

## Numbers deliberately NOT used

- Per-agent subgroup correlations (value −0.29, risk +0.22): from the Ablation 1 report
  (`analysis/crit_financial_correlation_abl1_report.txt` §6; 245 runs, two market quarters). Shown in
  the appendix as exploratory only; the iid p-values are not valid and are not cited as evidence.

- Any stratified-by-role or stratified-by-regime CRIT correlation. The 210-run CRIT
  corpus was recovered on 2026-10-30 (`CS372_210_debate_runs/`, not committed), but no
  stratified analysis has been run on it yet, so none is reported.
- Any re-derived value for the two configurations whose config files are lost.
