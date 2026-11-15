# Camera-ready gap plan: Verify-Agents submission 379

Audited 2026-11-15 against `neuripsworkshoppaper.tex` (local copy, modified 15 Nov 05:00, 3,235
lines). That file is Alanood's Task 1 revision. It is **not committed anywhere**, and it was
built from commit `b02c194`, so it does **not** contain Juli's 10 Nov fixes. `paper.tex` is now
stale: it is the old 10 Nov copy.

Workshop camera-ready deadline: **15 Nov 2026** (OpenReview).

---

## A. Audit: every reviewer point

✅ = fully addressed · ◐ = partly addressed · ❌ = not addressed

### Meta-review (AC ouoN)
| Point | Status | Evidence in current tex |
|---|---|---|
| n=210 CRIT analysis not reproducible from released artifacts | ◐ | Run directories recovered and checksummed. All 490 evaluations re-scored fresh (App. `app:rq1_fresh`). The per-run table `analysis/rq1/per_run_results.csv` is committed. **But:** the raw run package (`CS372_210_debate_runs/`) is gitignored and unreleased, and the paper never tells readers where the per-run table lives. |
| Repeated scenarios make pooled r less clean | ✅ | Scenario-clustered CIs, within-scenario and FE versions (Table 1, App. C.2) |
| Intervention labels mismatched | ✅ | A8/A10 rows and caption agree; A10 threshold is $P_4<0.75$ throughout |
| Uninvested cash vs "fully invested" | ✅ | §2 explains the `_CASH_` ticker; share rounding accounts for ≤2 points |
| Claim should be narrow ("this metric", not "reasoning metrics in general") | ◐ | Abstract and RQ1 conclusion narrowed. **Not narrowed:** conclusion line 641, "the dominant driver of performance is the preservation of independent informational signals", is an untested mechanism, and the A7 result is confounded. **The OpenReview abstract and TL;DR still say "essentially / nearly uncorrelated".** |

### Reviewer SnBG (7)
| Point | Status | Evidence |
|---|---|---|
| Aggregation confounds RQ1/RQ2 (judge ≠ agents; debate changes the aggregator: mean → LLM judge) | ❌ | No agent-level analysis, and no limitation sentence about the aggregator change |
| Needs scenario-clustered CIs and an equivalence bound | ◐ | Clustered CIs ✅. The equivalence analysis is a post-hoc ±0.20 screen, explicitly "not a formal equivalence test". That is honest, but it is not a formal bound. |
| A7 does not isolate its active component; needs a cap-only control | ◐ | The confound is stated, and the cap is described correctly (prompt instruction, 38/42 compliance). **Cap-only control not run.** |
| Multiple comparisons: treat the lone positive result cautiously | ❌ | No mention of multiple comparisons or correction anywhere (grep: 0 hits) |

### Reviewer JMbz (5)
| Point | Status | Evidence |
|---|---|---|
| CRIT analysis not reproducible | ◐ | Same as the meta-review row |
| Pooled r dominated by scenario variation | ✅ | Within-scenario r = −0.02 [−0.16, 0.12] |
| Inconsistencies across tables and portfolio setup | ✅ | 2.46→2.67, labels, cash, Gemini wording, cross-refs; the compile shows no undefined refs and no empty leaf sections |
| Positive intervention involves multiple changes | ◐ | Stated; no control run |
| Pretraining leakage not controlled | ◐ | Limitation stated (§5.1). No test run (e.g., a post-cutoff quarter or a masked-ticker probe). |
| Experimental section confusing | ✅ | Experiment map (App. B.1) |

### Reviewer SmPY (5)
| Point | Status | Evidence |
|---|---|---|
| CRIT logs absent | ◐ | Recovered for the 210 runs. **The 0.72→0.84 logs are still absent**; this is disclosed. |
| Depends on one rubric and one GPT-5 judge; prompts encourage what the rubric rewards | ◐ | Rubric–prompt coupling is a stated limitation. **No sentence on single judge/rubric**; no alternative judge or human audit. |
| Stronger uncertainty accounting for repeated scenarios; more independent scenarios | ◐ | Clustered CIs ✅. No new scenarios (acceptable; should be listed as future work). |

### Reviewer Hvfp (7)
| Point | Status | Evidence |
|---|---|---|
| Finance as a proxy domain (public-information ceiling, buy-and-hold) | ✅ | Conclusions discuss the public-information ceiling and "selection, not timing" |
| n=210 logs unverifiable | ◐ | Same as the meta-review row |

### Internal task list (team email)
| Task | Status | Notes |
|---|---|---|
| T1 Re-run CRIT / release per-run table / narrow claims | ◐ | Re-scored ✅; per-run table committed but not cited in the paper; conclusion and OpenReview metadata not narrowed |
| T2 Clustered CI, within-scenario, equivalence | ✅ (equivalence = screen only) | |
| T3 Inconsistencies (8 items) | ✅ | All 8 verified in the current tex |
| T4 Cap-only control, or soften + multiple-comparisons note | ❌ | Neither the run nor the multiple-comparisons sentence |
| T5 Agent-level ρ vs own-portfolio return; limitations on judge, leakage, finance | ◐ | Leakage ✅, finance ✅; agent-level ❌; single-judge limitation ❌ |

### Regressions in the current file (fixed 10 Nov, lost in the 15 Nov copy)
| Item | Status |
|---|---|
| Intro line 131: "Debate's measurable benefit is risk reduction rather than alpha" contradicts A7 (gain came from return, max DD unchanged) and the abstract | ❌ reintroduced |
| System diagram: `SystemDesign.png` has baked-in typos ("Singie LLM", "Eamings Call") | ❌ reintroduced (fix: `system_design.pdf`) |
| RQ1 scatter figure (`crit_vs_sharpe.pdf`, `fig:crit-scatter`) | ❌ dropped |
| Fig. 1 drawdown panel uses the return noise band (±0.31 pp); the correct drawdown band from the same 30 runs is ±0.23 pp | ❌ never fixed in the paper (conclusions unchanged) |
| **Page limit:** the main text runs ~10 lines onto page 10 (end of "Uneven reproducibility" plus Broader Impact) | ❌ **new, must fix** |

---

## B. Tasks, in priority order

### MUST before submitting (today)

**W1. Fix the page overflow** (writing, 15 min)
- Trim ≥12 lines from §§1–5. Candidates:
  - Contribution 3's duplicated clause "This coincides with loss of ensemble diversity and no outcome gain."
  - The RQ1 paragraph (line 272): move the fresh-rescoring sentences to App. `app:rq1_fresh` and keep only "Fresh re-scoring of all 490 evaluations reproduces this (Table~\ref{tab:rq1_fresh})."
  - "Uneven reproducibility" (line ~703): shorten the fresh-rescoring clause to one sentence.
- Verify by compiling: `sec:limitations` content plus Broader Impact must end on page 9.

**W2. Re-apply the three lost 10 Nov fixes** (writing, 10 min)
- Line 131: replace "Debate's measurable benefit is risk reduction rather than alpha." with
  "Debate's clearest benefits concern risk and risk-adjusted performance rather than consistent alpha."
- Line 831: `SystemDesign.png` → `system_design.pdf` (the file already exists in the repo root;
  source `system_design.tex`).
- After `tab:rq1_clustered` and its "Reproduced by…" paragraph, re-insert the `fig:crit-scatter`
  figure (`crit_vs_sharpe.pdf`; script `analysis/make_crit_scatter_figure.py`) and cite it in the
  RQ1 paragraph. Optional if page or appendix space is a concern; it is appendix-only.

**W3. Add a multiple-comparisons caveat for A7** (writing, 5 min). Answers SnBG and Task 4's fallback.
- §3.4 RQ3-E2 (line ~407), after "…cannot attribute the gain to disagreement preservation alone":
  "This is the only significant result among three interventions and several metrics; the
  $p$-values are uncorrected, and A7's Sharpe $p{=}0.028$ would not survive a Bonferroni
  correction across the three interventions ($\alpha{=}0.017$)."
- Abstract: append "(uncorrected)" after "$p{=}0.028$" if space allows.

**W4. Narrow the conclusion** (writing, 5 min). Answers the AC.
- Line 641: "the dominant driver of performance is the preservation of independent informational
  signals" → "the results are more consistent with preserved signal diversity than with improved
  individual reasoning quality, though we did not manipulate diversity directly".

**W5. Add a single-judge/rubric and aggregation limitation** (writing, 5 min). Answers SmPY and SnBG.
- In §5.1 add: "**One rubric, one judge, one aggregator.** All scores come from one CRIT rubric
  graded by \texttt{gpt-5}; other rubrics, judge models, or expert audits may behave differently.
  The portfolio is set by an LLM judge (debate) or a mean (ensemble), so RQ1 tests the group
  score against the aggregated decision, not each agent's own allocation, and debate–ensemble
  comparisons also change the aggregator."

**W6. Point readers to the released per-run data** (writing, 5 min). Answers all 5 reviewers.
- In "Uneven reproducibility" add: "The per-run table (scenario, ablation, condition, $\bar\rho$,
  Sharpe, return) and the fresh scores are released at
  \texttt{analysis/rq1/per\_run\_results.csv} and \texttt{analysis/task1\_camera\_ready/}."
- Coding: merge `origin/camera-ready-task1-summary-results` into `camera-ready` so those paths exist
  on the branch the paper links to.

**W7. Update OpenReview metadata** (form edit, 5 min)
- Abstract field: replace "essentially uncorrelated with financial outcomes (descriptive
  r=+0.07 …)" with the camera-ready abstract text ("no detectable association … scenario-clustered
  95% CI [−0.13, 0.25] …").
- TL;DR: "CRIT is nearly uncorrelated" → "this CRIT rubric shows no detectable association with
  market outcomes".

**W8. Sync and ship** (coding, 10 min)
- `cp neuripsworkshoppaper.tex paper.tex`. Rebuild `overleaf_camera_ready.zip` with
  `system_design.pdf` and `crit_vs_sharpe.pdf` in place of `SystemDesign.png`. Compile from a clean
  folder (0 errors, 0 undefined refs, main text ≤ 9 pages). Commit on `camera-ready` and upload.
- Check `checklist.tex` still matches (it should mention the fresh re-scoring and the per-run
  table release).

### SHOULD (if time allows; each strengthens a ◐ to ✅)

**C1. Fix the Fig. 1 drawdown noise band** (coding, 15 min)
- `analysis/make_effect_figure.py`: compute the noise band per metric from `results/seed_variance`
  (the method already exists in
  `neurips_poster_template_for_agentic_financial_one/make_poster_figures.py::seed_noise_floor`).
  Drawdown band: 0.228 pp. Regenerate `effect_vs_noise.pdf`.
- Caption edit: "Shading is the run-to-run noise floor for each metric (Appendix~\ref{app:seed-variance})."
  Add the drawdown row (pooled SD 0.955 pp, paired SEM 0.23 pp) to `tab:seed-variance` and
  `analysis/seed_variance.py` (add `max_drawdown_pct` to `METRICS`).

**C2. Agent-level CRIT vs own-portfolio outcome** (coding, ~2 h). Answers SnBG and Task 5.
- Input: `CS372_210_debate_runs/logging/runs/*/run_*/rounds/round_001/revisions/<agent>/` (each
  agent's revised allocation) and `final/pid_crit_all_rounds.json` (each agent's ρ_i).
- Backtest each agent's own allocation over the scenario quarter with the existing daily-price
  pipeline (`eval/financial.py`, the same prices the run used). This gives 490 agent-runs.
- Report pooled and within-scenario r(ρ_i, own Sharpe) with scenario-clustered CIs (reuse
  `analysis/rq1_clustered_stats.py`). Write a new App. subsection plus one sentence in RQ1.
- Kill criterion stated in advance: if the agent-level r is clearly positive (CI excludes 0), RQ1's
  null reflects judge aggregation rather than verifier invalidity, and the abstract must say so.

**C3. Formal equivalence test** (coding, 1 h). Answers SnBG.
- Pre-register the margin now (it remains post hoc; say so) and report a cluster-bootstrap TOST,
  or keep the "screen" wording. If time is short, leave it as is: the current wording is
  accurate.

### LATER (after the conference; note as future work in §5)

**E1. Cap-only control for A7.** Run `ablation_7` with the JS trigger disabled (`threshold: 0.0`)
but the retry plus `revision_limits` forced on every revision, on the same 35 scenarios
(`gpt-5-mini`; about 35 runs, same cost as A7). Compare paired Sharpe against A7 treatment.
- Falsifier: if cap-only ≈ A7 treatment, the gain is the cap, not disagreement preservation.

**E2. Leakage probe.** Rerun one configuration on a post-training-cutoff quarter, or with tickers
masked and renamed, and compare.

**E3. Second judge or rubric.** Re-score a stratified subset (e.g., 60 of the 490 evaluations)
with a different-family judge, plus a small human audit, and report agreement with gpt-5 CRIT.

**E4. Recover or regenerate the 0.72→0.84 evidence.** No CRIT records exist for those runs (see
`neurips_poster_template_for_agentic_financial_one/crit_artifact_audit.md`). Either rerun basic
vs enriched with CRIT on the 35 scenarios, or keep the "author records" caveat.

---

## C. Completion status (2026-11-15)

Final file: `neuripsworkshoppaper.tex` (mirrored to `paper.tex`), built from
`neuripsworkshoppaper copy.tex` (Alanood's Task 1 version) plus a 3-way merge of the 10 Nov fixes
(base `b02c194`; one conflict, in the RQ1 paragraph, resolved by keeping Alanood's text and adding
the figure reference). Backups of both inputs are in the session scratchpad.
Clean-folder compile of `overleaf_camera_ready.zip`: 0 errors, 0 undefined refs, main text and
Broader Impact end on p. 9.

| Item | Done | What changed |
|---|---|---|
| W1 page limit | ✅ | Trimmed duplicates: fresh-rescoring detail kept in the appendix; condensed Uneven reproducibility, cross-provider, RQ2 conclusion, Other failure modes, Broader Impact; dropped a duplicated clause in contribution 3 and the case-study intro sentence |
| W2 10 Nov fixes | ✅ | Intro "risk and risk-adjusted performance rather than consistent alpha"; `system_design.pdf`; `fig:crit-scatter` re-inserted and cited |
| W3 multiple comparisons | ✅ | §3.4 RQ3-E2: uncorrected; Sharpe p=0.028 and Sortino p=0.026 do not survive Bonferroni across 3 interventions (α=0.017). Abstract: "(p=0.028, uncorrected)" |
| W4 conclusion | ✅ | "more consistent with preserved signal diversity … though we did not manipulate diversity directly"; RQ2 conclusion paragraph also narrowed (one intervention, confounded) |
| W5 judge/rubric/aggregator | ✅ | New §5.1 item, citing the agent-level analysis |
| W6 per-run data | ✅ | §5.1 names `analysis/rq1/per_run_results.csv` and `analysis/task1_camera_ready/`; Task 1 branch merged into `camera-ready` (fast-forward). Fixed App. C.2's wrong claim that raw fresh responses are under `results/fresh_gpt5/` (they are not in the repo) |
| W7 OpenReview | ⏳ **you** | Needs your login; paste the text below |
| W8 sync, zip, checklist, commit | ✅ | `paper.tex` synced; zip rebuilt; checklist items 5, 6, 7, 8 updated (incl. a pre-existing false claim about rate limits/budget); committed locally (not pushed) |
| C1 drawdown noise band | ✅ | `seed_variance.py` adds max drawdown (pooled SD 0.96 pp, paired SEM 0.23 pp); `make_effect_figure.py` uses per-metric bands; Fig. 1 caption, noise paragraph and seed-variance caption updated. Same classification as before |
| C2 agent-level analysis | ✅ | `analysis/agent_level_crit.py` → `analysis/rq1/agent_level_{results.csv,stats.json}`; new App. C.4 (Table 6). Check passed: reproduces all 210 stored judge Sharpe/return exactly. All 12 CIs include 0, so the stated kill criterion is not met |
| C3 equivalence | ✅ | App. C.3: cluster-bootstrap TOST ⇔ 90% CI inside ±δ; smallest supported bounds 0.14 (within Sharpe), 0.10/0.12 (within return), 0.23 (pooled Sharpe), original/fresh |
| E1–E4 | ◐ | Not run (post-conference). Now listed as "Open controls from review" in App. A (future work), as this plan required |

Extra correction found during the edits: §3.1 said tiers 2–3 both contain the confounder and
reverse-causation warnings. Per `config/agent_profiles/*_enriched.yaml`, the enriched tier does not
include `system_causal_contract.txt`; it requires numbered causal claims with mechanisms and
falsifiers instead. The text now says this.

Found while building C2: `data-pipeline/final_snapshots/memo_data/` lacks some quarters (e.g.
`memo_2023_Q3.txt`), so `tools/dashboard/run_scanner.py` silently uses a 0% risk-free rate for
them. The stored metrics used in the paper are correct (they match the run's own memo);
`agent_level_crit.py` reads the run's own memo. The scanner fallback is unfixed.

### W7: OpenReview text to paste
**Abstract:**
Multi-agent LLM systems increasingly use structured debate to improve reasoning, and their outputs are commonly evaluated with internal reasoning-quality rubrics and consensus measures. Whether those proxies predict better decisions is rarely tested, because most benchmarks lack an external outcome to check them against. We use simulated portfolio allocation as a testbed where complete reasoning traces can be compared against an objective downstream outcome, evaluating specialist LLM agents under single-agent, independent-ensemble, and multi-round debate conditions, scored with a CRIT-style four-pillar rubric and Jensen–Shannon portfolio divergence against realized return, Sharpe, and drawdown. Across 210 runs on 35 reused scenarios, CRIT reasoning scores show no detectable association with financial outcomes (Sharpe: pooled r=+0.07, scenario-clustered 95% CI [-0.13, 0.25]; within-scenario r=-0.02, [-0.16, 0.12]), even though structured prompting moves those scores substantially (mean rho from 0.72 to 0.84, Cohen's d≈2.0) and debate portfolios beat an equal-weight baseline only 33% of the time. Improved outcomes align with interventions that limit consensus collapse rather than with interventions that raise reasoning scores: a divergence-collapse trigger improves Sharpe by +0.14 (p=0.028, uncorrected), while an intervention that verifiably repairs causal-reasoning scores slightly degrades performance. That trigger also constrained revision magnitude, so we cannot isolate disagreement preservation as the mechanism. Across our tested configurations the clearest benefits concern risk and risk-adjusted performance rather than consistent alpha, and gains in this particular rubric score need not accompany improved financial outcomes. These results suggest internal reasoning proxies should be validated against external task outcomes before being treated as evidence of agent decision quality.

**TL;DR:**
This CRIT reasoning rubric can be raised by prompting without better decisions: across 210 runs it shows no detectable association with market outcomes, and a convergence-limiting intervention improves Sharpe without isolating the mechanism.

### Commit history (pushed to origin/camera-ready, 2026-11-15)
Small commits `db6a229`…`a2cc812`: analysis and figures first, then Alanood's paper revision (authored by her), the Nov 10 fixes, one commit per plan item, checklist, sync, plan. Every intermediate paper version compiles with 0 errors and 0 undefined references.

**Poster:** committed after the other session (cs372research-54) went idle and its edits compiled to one clean page (`ab19263`…`9a81727`). One fix before committing: the Scope line said "21 stocks" for all 35 scenarios; it now says "3–35 stocks each" (only the 2022 Q1 scenario has 21). Backups and LaTeX build files are ignored in the poster folder's `.gitignore`.
