# CHECKLIST.md — Multi-Agent Debate NeurIPS Workshop 3-Hour Submission Sprint

## Goal

Prepare the existing Multi-Agent Debate / trading paper for a strong NeurIPS workshop submission within **3 hours**.

Primary scientific framing:

> **Do reasoning-quality metrics actually predict better downstream decisions in multi-agent systems, or can debate improve apparent reasoning quality while degrading external task performance?**

Secondary phenomenon:

> **Can repeated debate drive agents toward consensus without corresponding improvements in evidence or decision quality?**

Do **not** attempt to redesign the trading system.

Do **not** build a new debate architecture.

Do **not** add major infrastructure.

Do **not** run broad model or hyperparameter sweeps.

The goal is:

```text
sharper verification question
+
verified existing results
+
1–2 decisive analyses
+
proper debate controls
+
careful treatment of convergence/sycophancy
+
submission-ready anonymous paper
```

Scientific correctness and successful submission take priority over obtaining favorable results.

---

# 0. NON-NEGOTIABLE RULES

Before doing anything:

* [ ] Read `CLAUDE.md` completely.
* [ ] Follow its scientific-research rules throughout this sprint.
* [ ] Never invent a number.
* [ ] Never invent an experiment.
* [ ] Never invent a citation.
* [ ] Never imply an experiment was run when it was not.
* [ ] Never retroactively describe an observed pattern as a preregistered hypothesis.
* [ ] Never claim causality from correlation.
* [ ] Never equate high reasoning score with actual reasoning quality without qualification.
* [ ] Never equate consensus with correctness.
* [ ] Never call convergence "sycophancy" unless the operational definition supports that term.
* [ ] Never call CRIT score "ground truth."
* [ ] Every numerical paper claim must map to saved results/code.
* [ ] Preserve null and negative findings.
* [ ] Do not cherry-pick favorable quarters, stocks, seeds, agents, or debate rounds.
* [ ] Do not select metrics post hoc solely because they support the desired narrative.
* [ ] Do not optimize experiment settings against final outcome data and describe them as fixed.
* [ ] Maintain double-blind anonymity.
* [ ] Remove identifying repository URLs, names, institutions, emails, and metadata.
* [ ] If a result cannot be verified, weaken or remove it.

---

# 1. HARD TIME BUDGET

Use approximately:

```text
00:00–00:15   Submission + repository audit
00:15–00:35   Scientific framing / hypothesis lock
00:35–01:20   Highest-value analysis or control
01:20–01:45   Proxy-vs-outcome + convergence/error analysis
01:45–02:25   Rewrite paper
02:25–02:45   Figures/tables + claims + references audit
02:45–03:00   Anonymization + PDF + OpenReview submission
```

At every boundary:

* [ ] Save outputs.
* [ ] Stop optional work.
* [ ] Move forward.

Do not sacrifice submission readiness for one more experiment.

---

# 2. STOP CONDITIONS

Stop adding experiments when ANY applies:

* [ ] Fewer than 60 minutes remain.
* [ ] New experiment requires architectural redesign.
* [ ] Environment/debugging issue exceeds ~10 minutes.
* [ ] Existing results already support the central claim defensibly.
* [ ] Experiment requires obtaining a substantial new dataset.
* [ ] Experiment requires large API spend.
* [ ] Experiment requires broad seed/model sweep.
* [ ] Result would not materially change the paper conclusion.
* [ ] Result would be uninterpretable under both positive and null outcomes.
* [ ] Anonymization/template/submission is not yet safe.

After **T−60 minutes**:

> paper work only, unless fixing an invalid result.

After **T−20 minutes**:

> submission work only.

---

# 3. PHASE 1 — SUBMISSION SAFETY AUDIT

## First 15 minutes

### OpenReview

Verify:

* [ ] Correct workshop: Verify-Agents if that is the target.
* [ ] Deadline confirmed.
* [ ] Page limit confirmed.
* [ ] Correct NeurIPS 2026 template.
* [ ] Double-blind requirement.
* [ ] All authors entered into OpenReview.
* [ ] All authors have valid profiles.
* [ ] Reviewer nomination requirement satisfied.
* [ ] Required fields known:

  * [ ] Title
  * [ ] Keywords
  * [ ] TL;DR
  * [ ] Abstract
  * [ ] PDF
  * [ ] Supplement if used
  * [ ] Email-sharing agreement
  * [ ] Data-release agreement
  * [ ] Reviewer nomination
  * [ ] License

---

## Repository audit

Locate:

* [ ] Main `.tex` file.
* [ ] Bibliography.
* [ ] Experiment configuration files.
* [ ] Market scenario data.
* [ ] Existing debate outputs.
* [ ] Single-agent results.
* [ ] Multi-agent/no-debate results if any.
* [ ] One-round debate results.
* [ ] Multi-round debate results.
* [ ] Devil's advocate results if any.
* [ ] PID-controlled results.
* [ ] CRIT scores.
* [ ] JSD/diversity scores.
* [ ] Portfolio allocations.
* [ ] Return calculations.
* [ ] Sharpe calculations.
* [ ] Drawdown calculations.
* [ ] Existing plots/tables.
* [ ] Raw replayable traces.
* [ ] Judge/aggregation outputs.

Create:

```text
submission_audit.md
```

Record:

```text
deadline:
target workshop:
page limit:
main tex:
experiment scripts:
raw traces:
results files:
metric scripts:
existing baselines:
known methodological issues:
```

---

# 4. PHASE 2 — LOCK THE SCIENTIFIC STORY

## Target: 20 minutes

Do this before starting new experiments.

# 4.1 Central research question

Use this unless the actual evidence contradicts it:

> **Do internal reasoning-quality metrics predict downstream decision quality in multi-agent systems?**

Secondary question:

> **Does structured multi-agent debate improve decision quality, or can repeated interaction create misleading improvements in reasoning proxies?**

Do NOT center the paper on:

> "We built a multi-agent trading system."

Trading is the **controlled downstream decision environment**.

The scientific contribution is about:

* multi-agent evaluation,
* verifier validity,
* reasoning proxies,
* debate dynamics,
* convergence,
* external outcome grounding.

---

# 4.2 Primary research questions

Write into:

```text
research_questions.md
```

---

## RQ1 — Proxy validity

> Do CRIT-style reasoning scores predict external financial performance?

### H1

If CRIT captures decision-relevant reasoning quality, higher CRIT scores should positively associate with improved external outcomes.

### Strong competing explanation

CRIT may measure properties such as:

* coherence,
* completeness,
* evidential style,
* argument structure,

without measuring whether the resulting decision is actually better.

### Metrics

Internal:

* CRIT score
* CRIT pillar scores if available

External:

* raw return
* Sharpe
* drawdown
* possibly benchmark-relative return

### Falsifier

> CRIT scores show little/no positive association with downstream performance, or improve while external performance worsens.

---

## RQ2 — Does debate itself help?

> Does structured interaction improve outcomes beyond simply having multiple independent agents?

### H1

Debate may expose weak arguments and improve decisions.

### Strong competing explanation

Any improvement may come merely from:

> multiple independent perspectives + aggregation

rather than communication/debate.

### Critical control

Compare debate against:

```text
single agent

independent agents + simple average

independent agents + judge

one-round debate

multi-round debate
```

Use only conditions already available or cheap to generate.

### Falsifier

If independent ensemble performance matches/exceeds debate:

> the claim that debate itself produces the gain is weakened.

---

## RQ3 — Convergence / possible sycophancy

> Can repeated debate reduce opinion diversity without improving decision quality?

### H1

Additional debate rounds may increase consensus even when new evidence is not introduced, potentially producing socially driven convergence.

### Strong competing explanation

Convergence could represent legitimate error correction.

### Required distinction

Do NOT equate:

```text
convergence = sycophancy
```

To support a sycophancy interpretation, investigate whether agents:

* move toward the group,
* without materially adding evidence,
* without correcting a demonstrated factual/reasoning error,
* while external performance fails to improve or worsens.

### Falsifier

If convergence consistently accompanies:

* substantive evidence updates,
* correction of identifiable errors,
* and improved outcomes,

the sycophancy interpretation is weakened.

---

# 5. BUILD CLAIM → EVIDENCE MAP

Create:

```text
claim_evidence_map.md
```

Use:

| Candidate claim                             | Evidence file | Metric | Verified? | Safe wording |
| ------------------------------------------- | ------------- | -----: | --------- | ------------ |
| Debate increases CRIT                       |               |        |           |              |
| Debate improves return                      |               |        |           |              |
| Debate improves Sharpe                      |               |        |           |              |
| Multi-round debate reduces raw returns      |               |        |           |              |
| Multi-round debate lowers diversity/JSD     |               |        |           |              |
| High CRIT can coexist with worse outcomes   |               |        |           |              |
| Devil's advocate preserves useful diversity |               |        |           |              |
| PID improves reasoning quality              |               |        |           |              |
| CRIT predicts performance                   |               |        |           |              |
| Consensus reflects sycophancy               |               |        |           |              |

For every paper number:

* [ ] Locate raw result.
* [ ] Verify denominator.
* [ ] Verify scenario count.
* [ ] Verify condition.
* [ ] Verify model/version.
* [ ] Verify date/quarter tested.
* [ ] Verify metric implementation.
* [ ] Verify no stale output was used.
* [ ] Verify aggregation level.
* [ ] Recompute if cheap.

Never copy numbers from the manuscript without tracing them.

---

# 6. PHASE 3 — CHOOSE AT MOST TWO HIGH-VALUE ANALYSES

Prioritize in this order.

---

# EXPERIMENT / ANALYSIS A — CRIT vs External Outcomes

## HIGHEST PRIORITY

This is probably the single most valuable analysis for Verify-Agents.

### Question

> Does internal reasoning evaluation track actual downstream decision performance?

### Required data

For each independent experimental unit:

* scenario ID
* configuration
* CRIT score
* portfolio decision
* realized performance

Prefer one row per:

```text
scenario × experimental condition
```

rather than treating each agent-turn as an independent financial outcome.

---

## Compute

At minimum:

* [ ] CRIT vs raw return
* [ ] CRIT vs Sharpe
* [ ] CRIT vs drawdown

If reasonable:

* [ ] Pearson correlation
* [ ] Spearman rank correlation

Do not blindly report both without reason.

Spearman may be more robust when relation may be monotonic but not linear.

Report:

* [ ] correlation coefficient
* [ ] N
* [ ] uncertainty/p-value only if valid and quick
* [ ] scatterplot if useful

---

## More important than aggregate correlation

Compare configurations where:

```text
CRIT ↑
but
external performance ↓
```

Identify whether this actually occurs.

If yes, report precise examples/group-level results.

Example claim:

> Increasing debate depth improved mean CRIT score from X to Y while mean subsequent return decreased from A to B.

ONLY if verified.

---

## Interpretation

Strong result:

> Reasoning proxy improvement is not sufficient evidence of improved downstream decision quality.

Do NOT conclude:

> "CRIT is invalid."

unless evidence supports that broad statement.

Safer:

> "CRIT and downstream financial performance diverge under some tested debate configurations."

---

# EXPERIMENT / ANALYSIS B — Debate vs Independent Ensemble

## Highest-value control

### Question

> Are benefits caused by debate or simply by having multiple agents?

Ideal conditions:

```text
1. Single agent
2. Multiple independent agents + mean
3. Multiple independent agents + judge
4. One-round debate
5. Multi-round debate
6. Devil's advocate debate
```

Do NOT build all six if they do not already exist.

Minimum useful comparison:

```text
independent agents + aggregation
vs
debate
```

---

## Match as much as possible

* [ ] same models
* [ ] same market scenarios
* [ ] same input information
* [ ] same number of agents
* [ ] same role composition
* [ ] similar decision aggregation
* [ ] comparable token/compute budget if feasible

If debate has much more compute, disclose it.

---

## Measure

Internal:

* CRIT
* disagreement/JSD

External:

* return
* Sharpe
* drawdown

Optional:

* token cost
* latency

---

## Key interpretations

If debate > ensemble:

> evidence supports an interaction benefit beyond agent diversity.

If ensemble ≈ debate:

> benefits may primarily arise from diversity/aggregation.

If ensemble > debate:

> interaction may degrade performance under tested conditions.

All three outcomes are publishable findings.

---

# EXPERIMENT / ANALYSIS C — Quantify Convergence / Possible Sycophancy

## Run if traces already contain needed information

### Do NOT start with the label "sycophancy."

Start with:

> **consensus convergence**

and test whether the stronger label is justified.

---

## Operational quantities

For each agent across rounds:

### Position shift

Measure distance between portfolio allocation at round `t` and:

* its own prior allocation,
* group consensus/mean allocation.

Possible metric:

```text
L1 distance between portfolio weight vectors
```

or another already-supported distance measure.

---

### Group diversity

Use existing JSD if appropriate.

Track:

```text
JSD round 0
JSD round 1
JSD round 2
...
```

---

### Evidence change

If traces include cited evidence IDs:

measure:

* number of new evidence references,
* changed evidence,
* removed evidence.

At minimum hand-audit representative cases.

---

### Outcome

Track whether convergence leads to:

* improved return,
* worse return,
* no change.

---

## Possible operational definition

A candidate "unsupported convergence" event:

> An agent moves substantially toward group consensus following critique while introducing no material new evidence and without correcting an identifiable factual/reasoning error.

Only call this sycophantic behavior if enough evidence supports that interpretation.

Otherwise use:

> convergence consistent with social influence / sycophancy.

---

# EXPERIMENT / ANALYSIS D — Debate Depth

## Cheap if existing runs already contain multiple rounds

Compare:

```text
0 rounds
1 round
2+ rounds
```

Measure:

* CRIT
* JSD
* return
* Sharpe
* drawdown

This can produce an extremely clean figure:

```text
x-axis = debate depth

line 1 = CRIT
line 2 = external performance
line 3 = diversity
```

Use separate panels/figures if necessary for readability.

The scientific question:

> Does more debate monotonically improve reasoning proxies while external outcomes plateau or deteriorate?

Do not assume this pattern.

Measure it.

---

# 7. DO NOT IMPLEMENT THESE DURING THE 3-HOUR SPRINT

Unless already essentially complete:

* [ ] new trading simulator
* [ ] new market data provider
* [ ] new portfolio optimizer
* [ ] new LLM provider
* [ ] RL training
* [ ] new causal-inference architecture
* [ ] new CRIT evaluator
* [ ] learned verifier
* [ ] large human annotation effort
* [ ] dozens of new market quarters
* [ ] large hyperparameter sweep
* [ ] broad prompt optimization
* [ ] new PID algorithm
* [ ] new agent roles
* [ ] elaborate statistical modeling
* [ ] major codebase refactor

Record under Future Work instead.

---

# 8. CRITICAL SANITY CHECKS

Before trusting any result:

## Financial data

* [ ] Agent only sees information available at decision time.
* [ ] No future prices leak into prompts.
* [ ] No post-quarter earnings/news leaks into scenario.
* [ ] Performance window begins after decision.
* [ ] Ticker data alignment is correct.
* [ ] Missing prices handled consistently.
* [ ] Survivorship bias acknowledged if relevant.

---

## Portfolio calculations

* [ ] Weights sum correctly.
* [ ] Cash handling correct.
* [ ] No accidental leverage unless intended.
* [ ] Returns computed consistently.
* [ ] Sharpe formula verified.
* [ ] Drawdown formula verified.
* [ ] Benchmark period matches evaluation period.

---

## Debate system

* [ ] Same market scenario given to all compared conditions.
* [ ] Agent roles correctly instantiated.
* [ ] Round count correctly logged.
* [ ] Critiques correspond to correct proposals.
* [ ] Revisions correspond to correct agents.
* [ ] Judge sees only intended inputs.
* [ ] Aggregation implementation matches paper description.
* [ ] PID state resets appropriately between independent scenarios.
* [ ] Experimental configurations are not accidentally mixed.

---

## CRIT evaluator

* [ ] Correct model.
* [ ] Correct rubric.
* [ ] Correct scoring scale.
* [ ] Same evaluator across compared conditions.
* [ ] No final financial outcome given to evaluator.
* [ ] Evaluator does not see future data.
* [ ] CRIT calculation matches manuscript equation.

---

# 9. PROXY VS GROUND-TRUTH LANGUAGE

Use precise terminology.

Avoid:

> CRIT measures true reasoning quality.

Prefer:

> CRIT provides a rubric-based reasoning-quality proxy.

Avoid:

> financial return is ground truth reasoning quality.

Prefer:

> realized financial performance provides an external downstream outcome.

Important:

```text
internal reasoning proxy
≠
external decision outcome
```

The paper studies their relationship.

---

# 10. FAILURE TAXONOMY

Add a compact taxonomy if supported.

### F1 — Unsupported causal claim

Agent attributes market movement to evidence without sufficient support.

### F2 — Evidence neglect

Agent ignores important available evidence.

### F3 — Premature consensus

Agents converge before substantive objections are resolved.

### F4 — Unsupported convergence

Agent changes toward group position without meaningful new evidence or identified correction.

### F5 — Persistent disagreement

Agents remain divergent despite decisive shared evidence.

### F6 — Conclusion drift

Agent's final allocation is inconsistent with its stated reasoning.

### F7 — Reasoning/outcome mismatch

High proxy reasoning score accompanies poor external performance.

### F8 — Overconfidence

Agent expresses strong certainty unsupported by evidence.

Only include categories that are operationalized or auditable.

---

# 11. ERROR ANALYSIS

Inspect meaningful traces rather than blindly tuning.

If available, inspect:

* [ ] highest CRIT + worst-return cases
* [ ] lowest CRIT + best-return cases
* [ ] largest convergence cases
* [ ] largest performance drops with additional debate
* [ ] devil's-advocate successes/failures
* [ ] judge vs mean-aggregation disagreement cases

For each:

```text
scenario:
configuration:
round:
CRIT:
JSD:
initial positions:
final positions:
new evidence introduced:
portfolio:
return:
Sharpe:
failure category:
likely explanation:
```

Ask:

> Why does the proxy disagree with the outcome here?

These cases may become the paper's strongest qualitative evidence.

---

# 12. MAIN RESULTS TABLE

Create one high-information table.

Ideal:

| Condition            | CRIT ↑ | Diversity/JSD ↑ | Return ↑ | Sharpe ↑ | Max Drawdown ↓ |
| -------------------- | -----: | --------------: | -------: | -------: | -------------: |
| Single agent         |        |                 |          |          |                |
| Independent ensemble |        |                 |          |          |                |
| 1-round debate       |        |                 |          |          |                |
| Multi-round debate   |        |                 |          |          |                |
| Devil's advocate     |        |                 |          |          |                |

Only include existing valid conditions.

If there are too many configurations:

use the 3–4 that directly test the scientific claim.

---

# 13. ONE HIGH-VALUE FIGURE

Prioritize:

## Option A — CRIT vs External Outcome

Scatterplot:

```text
x = CRIT
y = subsequent return / Sharpe
```

Different markers by condition if readable.

Question answered:

> Does proxy reasoning quality predict external performance?

---

## Option B — Debate Depth

```text
x = debate round

CRIT
JSD
external outcome
```

Question answered:

> How do reasoning score, consensus, and downstream performance evolve with debate depth?

---

## Option C — Configuration comparison

Simple table/bar plot comparing:

```text
independent ensemble
vs
1-round debate
vs
multi-round debate
```

Do not spend >15 minutes making fancy plots.

---

# 14. STATISTICAL UNCERTAINTY

Use the correct independent unit.

Likely:

```text
market scenario / case
```

NOT:

* agent turn,
* token,
* critique sentence.

If cheap:

* [ ] bootstrap scenario-level confidence intervals.

If multiple scenario runs/seeds exist:

* [ ] report variation.

If sample size is small:

* [ ] state N.
* [ ] avoid strong significance language.

Do NOT write:

> statistically significant

unless actually tested.

---

# 15. TITLE

For Verify-Agents, strongest option if evidence supports proxy/outcome divergence:

> **When Better Reasoning Scores Produce Worse Decisions: Verifying Multi-Agent Debate with External Outcomes**

More conservative:

> **Do Reasoning Metrics Predict Decision Quality? Evaluating Multi-Agent Debate in Simulated Markets**

Use conservative title if:

* the "worse decisions" effect is weak,
* only one configuration exhibits it,
* or uncertainty is large.

For BiAlign:

> **When Consensus Hurts: Convergence Dynamics in Multi-Agent Debate**

Only use "Sycophantic Convergence" in title if operational evidence is strong.

---

# 16. ABSTRACT REWRITE

Use approximately:

### Sentence 1 — General problem

Multi-agent LLM systems increasingly use debate or deliberation to improve reasoning and decisions.

### Sentence 2 — Gap

Most evaluation focuses on internal reasoning proxies or benchmark accuracy, leaving unclear whether improved reasoning scores correspond to better downstream decisions.

### Sentence 3 — Testbed

We study this question using controlled simulated portfolio allocation, where agent reasoning traces can be compared with subsequent external outcomes.

### Sentence 4 — Method

Mention:

* specialized agents
* structured debate
* CRIT-style evaluator
* diversity/JSD
* external portfolio performance

### Sentence 5–6 — Results

Use actual verified numbers.

Lead with the most interesting result.

Example structure:

> Additional debate increased CRIT from X to Y but reduced return from A to B while lowering disagreement from C to D.

ONLY if verified.

### Final sentence

Possible:

> These results suggest that internal reasoning-quality proxies and consensus should not be treated as sufficient evidence of improved multi-agent decision quality.

Checklist:

* [ ] At least one quantitative result.
* [ ] No marketing language.
* [ ] Finance positioned as testbed.
* [ ] Verification question visible.
* [ ] No unsupported sycophancy claim.
* [ ] No universal claim about debate.

---

# 17. INTRODUCTION REWRITE

Use:

## Paragraph 1 — Problem

Multi-agent LLM systems increasingly rely on deliberation/debate.

Motivation:

```text
more discussion
does not necessarily imply
better decision-making
```

---

## Paragraph 2 — Evaluation problem

Current systems may be evaluated using:

* self-evaluation,
* reasoning rubrics,
* agreement,
* consistency,
* consensus.

But these are proxy measures.

Question:

> Do they predict actual external outcomes?

---

## Paragraph 3 — Scientific gap

Introduce distinction:

```text
internal proxy quality
vs
external decision quality
```

State why this matters for verifier reliability.

---

## Paragraph 4 — Testbed

Simulated trading gives:

* complex decisions,
* conflicting evidence,
* measurable downstream outcomes,
* complete auditable traces.

Do NOT claim trading is a realistic deployment proxy beyond what simulation supports.

---

## Paragraph 5 — Main findings

Use verified findings.

---

## Contributions

Prefer:

### Contribution 1 — Evaluation

> We evaluate whether CRIT-style reasoning scores predict externally measured downstream decision outcomes.

### Contribution 2 — Debate controls

> We compare structured debate against simpler multi-agent aggregation strategies.

Only if actually done.

### Contribution 3 — Dynamics

> We analyze how debate depth affects disagreement, reasoning proxies, and subsequent decisions.

### Contribution 4 — Failure mode

> We identify cases where increased consensus/reasoning scores do not correspond to improved external outcomes.

Avoid:

> "We built a sophisticated trading platform"

as the main contribution.

---

# 18. RELATED WORK AUDIT

Organize around:

1. multi-agent debate
2. LLM agent evaluation/verifiers
3. sycophancy/groupthink/social influence
4. reasoning-quality evaluation
5. LLM trading/decision environments

For every citation:

* [ ] Source exists.
* [ ] Claim accurately reflects source.
* [ ] No hallucinated citation.
* [ ] No placeholder.
* [ ] No unverified "state-of-the-art" claim.

Do not spend excessive time expanding literature tonight.

---

# 19. METHODS CLEANUP

Ensure paper clearly states:

## Data

* [ ] market period
* [ ] ticker universe
* [ ] scenario generation
* [ ] information available at decision time
* [ ] external evaluation horizon
* [ ] data sources

---

## Agents

* [ ] number of agents
* [ ] roles
* [ ] model
* [ ] temperature
* [ ] prompts
* [ ] available evidence

---

## Debate

Explicit phases:

```text
propose
→ critique
→ revise
→ aggregate/judge
```

State:

* [ ] number of rounds
* [ ] stopping criterion
* [ ] debate prompt behavior
* [ ] devil's advocate condition
* [ ] PID condition if used

---

## Evaluator

For CRIT:

* [ ] model
* [ ] four pillars
* [ ] scoring range
* [ ] aggregation formula
* [ ] whether evaluator is stronger model
* [ ] what information evaluator sees

---

## Diversity

Define JSD or other disagreement metric precisely.

---

## Financial outcomes

Define:

* [ ] return
* [ ] Sharpe
* [ ] max drawdown
* [ ] evaluation period
* [ ] benchmark if used

---

# 20. BASELINE FAIRNESS AUDIT

For each configuration:

* [ ] Same scenarios?
* [ ] Same information?
* [ ] Same ticker universe?
* [ ] Same model family?
* [ ] Same agent count where relevant?
* [ ] Similar role composition?
* [ ] Same portfolio constraints?
* [ ] Same evaluation horizon?
* [ ] Same outcome calculation?

If compute differs:

* [ ] state it explicitly.

Do not imply debate benefits are architecture-specific if comparison also changes model capacity.

---

# 21. RESULTS SECTION STRUCTURE

Prefer research questions:

```text
4.1 Do reasoning-quality scores predict external decision outcomes?

4.2 Does debate improve performance beyond independent aggregation?

4.3 How does debate depth affect consensus and performance?

4.4 When do reasoning proxies and outcomes disagree?
```

Use only sections supported by results.

Avoid:

```text
Experiment 1
Experiment 2
Experiment 3
```

---

# 22. SYCOPHANCY CLAIM AUDIT

Search for:

```text
sycophancy
sycophantic
groupthink
consensus
agreement
convergence
```

For each "sycophancy" claim ask:

* [ ] Was an agent's prior belief identifiable?
* [ ] Did its position move toward others?
* [ ] Was new evidence introduced?
* [ ] Was a factual/reasoning error corrected?
* [ ] Could convergence simply represent legitimate updating?
* [ ] Does outcome deteriorate or fail to improve?
* [ ] Is the phenomenon systematic rather than anecdotal?

If not enough evidence:

replace:

> sycophantic convergence

with:

> consensus convergence

or:

> convergence consistent with social influence.

Scientific credibility is more important than a catchy label.

---

# 23. CAUSAL CLAIM AUDIT

Search:

```text
causes
leads to
results in
drives
improves
degrades
```

Ask whether experiment actually manipulates the variable.

Example:

If debate depth was experimentally varied under otherwise matched conditions:

> Increasing debate depth was associated with / caused X

may be defensible depending on design.

If merely correlational:

use:

> was associated with.

Do not claim:

> higher CRIT causes worse returns

from correlation.

---

# 24. PROXY-METRIC CLAIM AUDIT

Avoid:

> Better reasoning worsens performance.

unless "better reasoning" is independently established.

Prefer:

> Higher CRIT scores sometimes coincide with worse downstream performance.

This distinction is essential.

CRIT is:

> a measured proxy.

Not:

> reasoning itself.

---

# 25. DISCUSSION

Address:

* When does debate help?
* When does it stop helping?
* Does additional debate reduce useful diversity?
* Are high reasoning scores predictive?
* What kinds of reasoning might CRIT reward?
* Does consensus reflect correction or social influence?
* Are financial outcomes too noisy to validate reasoning?
* How much can be generalized beyond finance?
* What does this imply for agent verifiers?

Key implication:

> Internal agent-evaluation signals should ideally be validated against external task outcomes when such outcomes are available.

Scope carefully.

---

# 26. LIMITATIONS

Potential true limitations:

* simulated markets
* limited ticker universe
* finite set of historical periods
* realized return is noisy
* financial outcome is not equivalent to reasoning correctness
* CRIT is LLM-based
* no human audit of all reasoning scores
* limited model diversity
* role prompting may influence behavior
* debate compute differs from single-agent baseline
* possible data-selection bias
* sample size may be limited
* market regimes may not generalize
* no live trading
* possible evaluator-agent model correlations

Include only applicable limitations.

Important:

> Poor return does not necessarily mean reasoning was invalid.

A good decision can have a bad stochastic outcome.

Acknowledge this.

---

# 27. STOCHASTIC OUTCOME CAUTION

Finance is noisy.

Do NOT equate:

```text
good trade outcome
=
correct reasoning
```

or:

```text
bad trade outcome
=
bad reasoning
```

Frame external financial performance as:

> one objective downstream outcome signal.

If possible, use multiple external metrics and multiple scenarios.

Discuss noise explicitly.

This is an important reviewer concern.

---

# 28. FIGURE AUDIT

Every figure:

* [ ] answers a scientific question
* [ ] labels axes
* [ ] defines metric
* [ ] includes N where useful
* [ ] includes uncertainty where possible
* [ ] avoids misleading scaling
* [ ] matches source results
* [ ] contains no identity leakage

Prioritize:

1. proxy vs outcome figure
2. debate-depth figure
3. main comparison table
4. architecture diagram

Remove decorative dashboard/UI figures if space is needed.

---

# 29. FINAL NUMERICAL AUDIT

Extract all numbers from:

* abstract
* introduction
* results
* conclusion
* tables
* figure captions

For each:

```text
NUMBER:
SOURCE FILE:
SCRIPT:
CONFIG:
N:
DENOMINATOR:
VERIFIED:
```

Specifically verify:

* [ ] number of scenarios
* [ ] number of stocks/tickers
* [ ] number of agents
* [ ] number of debate rounds
* [ ] CRIT values
* [ ] JSD/diversity values
* [ ] raw returns
* [ ] Sharpe
* [ ] drawdown
* [ ] baseline performance
* [ ] any PID result
* [ ] any intervention result
* [ ] any claimed correlation
* [ ] any claimed percentage improvement

If uncertain:

> remove or qualify.

---

# 30. DOUBLE-BLIND ANONYMIZATION

Remove from PDF:

* [ ] author names
* [ ] emails
* [ ] affiliations
* [ ] Stanford identification where identifying
* [ ] acknowledgements
* [ ] GitHub repository URLs
* [ ] named course references if identifying
* [ ] local paths
* [ ] author notes

Check figures for:

* usernames
* filenames
* terminal paths
* repository names

Check supplementary material too.

Real authors still belong in the OpenReview form.

---

# 31. TEMPLATE / FORMATTING

* [ ] NeurIPS 2026 template.
* [ ] Remove old NeurIPS 2024 style.
* [ ] Correct anonymous mode.
* [ ] Page limit satisfied.
* [ ] No margin/font manipulation.
* [ ] References render.
* [ ] Figures render.
* [ ] No undefined references.
* [ ] No `??`.
* [ ] No TODO.
* [ ] No FIXME.
* [ ] No visible internal team notes.
* [ ] No commented draft material accidentally printed.

Critical:

The current source previously used:

```text
\usepackage[final]{neurips_2024}
```

Do NOT submit that unchanged.

Use the workshop-required NeurIPS 2026 anonymous template.

---

# 32. REVIEWER SIMULATION

## Maximum 10 minutes

Act as a skeptical Verify-Agents reviewer.

Answer:

### Scientific question

* [ ] Is this about verification/evaluation, or merely trading?

### Novelty

* [ ] What new scientific knowledge exists?

### Proxy validity

* [ ] Is CRIT treated as a proxy rather than ground truth?

### External outcome

* [ ] Is financial performance interpreted cautiously?

### Baseline

* [ ] Is there a non-debate multi-agent baseline?

### Debate claim

* [ ] Do experiments isolate debate from diversity/compute?

### Sycophancy

* [ ] Is it operationalized or overstated?

### Statistics

* [ ] Is N clear?
* [ ] Is uncertainty handled?

### Leakage

* [ ] Is future financial information excluded?

### Generalization

* [ ] Does paper avoid claiming universal conclusions about agents?

### Reproducibility

* [ ] Could someone reconstruct the pipeline?

Write top 3 likely rejection reasons.

Fix only:

* fatal scientific flaw
* unsupported central claim
* easy missing control
* anonymity/formatting problem

No new major experiments.

---

# 33. FINAL OPENREVIEW CHECK

### PDF

* [ ] Opens.
* [ ] Correct paper.
* [ ] Anonymous.
* [ ] Correct template.
* [ ] Correct page count.
* [ ] Figures readable.
* [ ] References complete.
* [ ] No author identity.
* [ ] No draft watermark.

### Form

* [ ] Final title.
* [ ] Final abstract.
* [ ] Clear TL;DR.
* [ ] Keywords.
* [ ] Authors complete.
* [ ] Author order correct.
* [ ] Reviewer nominated.
* [ ] Required agreements checked.
* [ ] License selected.
* [ ] Supplement anonymized if uploaded.

### Final

* [ ] Submit.
* [ ] Confirm OpenReview shows submission.
* [ ] Verify uploaded PDF.
* [ ] Save exact submitted PDF.
* [ ] Save code/config commit or snapshot.
* [ ] Save result artifacts used in manuscript.

---

# 34. MINIMUM DEFENSIBLE SUBMISSION

If time collapses:

## MUST HAVE

* [ ] Correct anonymous NeurIPS PDF.
* [ ] Verified existing results.
* [ ] Clear proxy-vs-outcome research question.
* [ ] Correct CRIT description.
* [ ] Correct financial metric definitions.
* [ ] At least one external-outcome comparison.
* [ ] Debate method clearly described.
* [ ] Results framed cautiously.
* [ ] Finance positioned as testbed.
* [ ] Limitations included.
* [ ] No unsupported sycophancy claims.

## HIGH VALUE

At least ONE:

* [ ] CRIT vs external outcome analysis.

OR

* [ ] debate vs independent ensemble control.

OR

* [ ] debate-depth / convergence analysis.

One strong analysis is enough.

## OPTIONAL

* [ ] new role
* [ ] new model
* [ ] new market period
* [ ] more seeds
* [ ] extra figures
* [ ] sophisticated PID analysis

Only if submission is already safe.

---

# 35. DESIRED FINAL PAPER STORY

A reviewer should understand:

```text
PROBLEM

Multi-agent LLM systems increasingly rely on debate and internal
reasoning-quality metrics, but it is unclear whether these signals
correspond to better downstream decisions.


QUESTION

Do higher reasoning-quality scores and stronger consensus actually
predict improved multi-agent decision outcomes?


METHOD

Use a controlled simulated-market environment where specialized agents
make portfolio decisions through independent reasoning or structured
debate.

Measure:

internal signals:
CRIT reasoning score
+
diversity / JSD
+
consensus dynamics

against

external outcomes:
return
+
Sharpe
+
drawdown.


RESULT

Some debate configurations improve internal reasoning scores and/or
agreement without corresponding improvements in external performance
[ONLY if verified].

Additional debate may reduce diversity and degrade some outcomes
[ONLY if verified].


IMPLICATION

Internal reasoning proxies and consensus should not automatically be
treated as evidence of better multi-agent decision quality.


LIMITATION

Financial outcomes are noisy and do not constitute ground truth
reasoning correctness; conclusions are limited to the tested models,
scenarios, debate protocol, and simulated environment.
```

---

# 36. FINAL PRINCIPLE

During this 3-hour sprint, do NOT ask:

> "How can we make this look more sophisticated?"

Ask:

> **"What evidence would convince a skeptical reviewer that the verifier metric actually means something?"**

Prefer:

```text
one sharp question
+
one decisive control
+
objective external outcome
+
careful proxy language
+
honest negative result
```

over:

```text
complex multi-agent architecture
+
many weak comparisons
+
overclaimed sycophancy
+
unverified correlations.
```

The strongest version of this paper is not:

> "Our trading agents reason better."

It is:

> **"We test whether commonly used reasoning-quality signals actually predict better multi-agent decisions—and identify where those signals diverge from externally measured outcomes."**
