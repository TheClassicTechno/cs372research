# Research Questions — checklist.md §4

Central question: **do internal reasoning-quality proxies predict downstream
decision quality in multi-agent systems?** Finance is the testbed, not the subject.

---

## RQ1 — Proxy validity

> Do CRIT-style reasoning scores predict external financial performance?

- **H1.** If CRIT captures decision-relevant reasoning quality, higher scores
  associate with better outcomes.
- **Competing explanation.** CRIT measures coherence, completeness, evidential
  style, and argument structure without measuring whether the decision is better.
- **Primary metric.** Pearson correlation of $\bar{\rho}$ with Sharpe and return.
- **Falsifier.** Near-zero association, or rubric gains alongside flat/worse
  performance.
- **Outcome: H1 rejected.** $r=+0.07$ with Sharpe ($p=0.29$), $r=+0.03$ with return
  ($p=0.70$), $n=210$, while enriched prompting moved $\bar{\rho}$ 0.72→0.84
  ($d\approx2.0$).
- **Caveat.** The CRIT logs behind this are not in the released artifact set.

## RQ2 — Does debate itself help?

> Does structured interaction improve outcomes beyond having multiple independent
> agents?

- **H1.** Debate exposes weak arguments and improves decisions.
- **Competing explanation.** Any gain comes from multiple perspectives plus
  aggregation, not from communication.
- **Control.** Column B of Table 2 is the independent ensemble: same models, same
  35 scenarios, same roles, same aggregation, with critique and revision disabled.
- **Falsifier.** Independent ensemble matches or beats debate on matched scenarios.
- **Outcome: H1 not supported for return.** Debate's measurable benefit is risk
  reduction (volatility $-0.04$, drawdown $-1.54$pp vs single agent, both
  $p<0.001$, $d\approx-1.0$), not alpha. Return differences are inside the
  sampling-noise floor.

## RQ3 — Can interventions repair reasoning failures?

> Do targeted interventions during debate improve outcomes?

- **H1.** Detecting and repairing a reasoning failure improves the decision.
- **Competing explanation.** What helps is preserved disagreement, not better
  reasoning.
- **Falsifier.** An intervention that verifiably repairs the measured reasoning
  deficit leaves outcomes unchanged or worse.
- **Outcome: falsifier met, and it discriminates between the two explanations.**
  The JSD intervention, which preserves disagreement, is the only significant
  return gain (Sharpe $+0.14$, $p=0.028$, $d=0.39$). The CRIT intervention (A10)
  successfully repairs causal-reasoning scores yet slightly degrades performance
  (Sharpe $-0.09$; drawdown $+0.53$pp).

## Convergence: what we do and do not claim

Operational definition used in §3.5: a revision where (i) allocation moves
substantially toward group consensus, (ii) no new evidence ID is introduced, and
(iii) no identifiable error in the prior position was corrected. Where all three
hold and the outcome does not improve, we say *consistent with sycophancy*. We do
**not** claim to have isolated a social-deference mechanism: legitimate error
correction also produces convergence and the design does not separate the two.

## Robustness

A run-to-run probe (30 runs, one config, five scenarios, six repeats) gives a
pooled within-scenario SD of 1.30pp in return. Single-scenario comparisons at these
effect sizes are uninterpretable; the paired 35-scenario design is what carries the
claims. Stated in advance as symmetric: had the noise floor swamped the paired
design, the RQ2 claims would have been weakened.
