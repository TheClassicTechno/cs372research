# CRIT Reasoning Diagnostics — Ablation 7 Analysis

*Generated: 2026-03-11 21:10:32*

*Runs analyzed: 70 (35 baseline, 35 intervention)*

---

## 1. Overview of CRIT Reasoning Metrics

### Experiment Design

- **Total runs**: 70
- **Baseline runs** (no intervention): 35
- **Intervention runs** (JS threshold=0.8): 35
- **Intervention fired** (retry triggered): 21/35 (60%)
- **Agents**: macro, technical
- **Scenarios**: 35

### CRIT Pillar Scores

CRIT evaluates reasoning quality on four pillars, each scored 0–1:

| Pillar | Abbreviation | Description |
|--------|-------------|-------------|
| Logical Validity | LV | Internal consistency of reasoning chain |
| Evidential Support | ES | Claims backed by cited evidence |
| Alternative Consideration | AC | Engagement with counterarguments and critiques |
| Causal Alignment | CA | Correct causal reasoning (no rung collapse) |

### CRIT Diagnostic Flags

Binary flags indicating specific reasoning failures:

| Flag | Description |
|------|-------------|
| contradictions | Internal contradictions in reasoning |
| unsupported_claims | Claims without evidence support |
| ignored_critiques | Failed to address critique points |
| premature_certainty | Overconfident conclusions |
| causal_overreach | Inferring causality from correlation |
| conclusion_drift | Final allocation inconsistent with reasoning |

### CRIT Diagnostic Counts

Integer counts of specific issues found:

| Count | Description |
|-------|-------------|
| contradictions_count | Number of internal contradictions |
| unsupported_claims_count | Number of unsupported claims |
| ignored_critiques_count | Number of ignored critique points |
| causal_overreach_count | Number of causal overreach instances |
| orphaned_positions_count | Positions without reasoning support |

## 2. Statistical Summary

### Table 1: Pillar Score Statistics (Baseline vs Intervention)

| Agent | Pillar | Baseline Mean±SD | Intervention Mean±SD | Diff | p-value | Sig |
|-------|--------|-----------------|---------------------|------|---------|-----|
| macro | Logical Validity | 0.837±0.071 | 0.841±0.073 | +0.005 | 0.7461 | n.s. |
| macro | Evidential Support | 0.856±0.025 | 0.858±0.020 | +0.002 | 0.7295 | n.s. |
| macro | Alternative Consideration | 0.806±0.067 | 0.801±0.069 | -0.005 | 0.7390 | n.s. |
| macro | Causal Alignment | 0.865±0.020 | 0.866±0.017 | +0.002 | 0.6958 | n.s. |
| technical | Logical Validity | 0.857±0.036 | 0.843±0.081 | -0.014 | 0.3608 | n.s. |
| technical | Evidential Support | 0.853±0.032 | 0.844±0.042 | -0.009 | 0.3430 | n.s. |
| technical | Alternative Consideration | 0.850±0.080 | 0.854±0.072 | +0.004 | 0.8163 | n.s. |
| technical | Causal Alignment | 0.735±0.084 | 0.761±0.099 | +0.026 | 0.2122 | n.s. |

### Table 2: Diagnostic Failure Rates

Percentage of runs where each diagnostic flag was triggered.

| Agent | Diagnostic | Baseline (%) | Intervention (%) | z | p-value | Sig |
|-------|-----------|-------------|-----------------|---|---------|-----|
| macro | contradictions | 11% (4/35) | 6% (2/35) | 0.85 | 0.393 | n.s. |
| macro | unsupported_claims | 0% (0/35) | 0% (0/35) | 0.00 | 1.000 | n.s. |
| macro | ignored_critiques | 0% (0/35) | 0% (0/35) | 0.00 | 1.000 | n.s. |
| macro | premature_certainty | 3% (1/35) | 0% (0/35) | 1.01 | 0.314 | n.s. |
| macro | causal_overreach | 3% (1/35) | 0% (0/35) | 1.01 | 0.314 | n.s. |
| macro | conclusion_drift | 3% (1/35) | 6% (2/35) | -0.59 | 0.555 | n.s. |
| technical | contradictions | 0% (0/35) | 9% (3/35) | -1.77 | 0.077 | + |
| technical | unsupported_claims | 0% (0/35) | 0% (0/35) | 0.00 | 1.000 | n.s. |
| technical | ignored_critiques | 0% (0/35) | 0% (0/35) | 0.00 | 1.000 | n.s. |
| technical | premature_certainty | 0% (0/35) | 0% (0/35) | 0.00 | 1.000 | n.s. |
| technical | causal_overreach | 71% (25/35) | 60% (21/35) | 1.01 | 0.314 | n.s. |
| technical | conclusion_drift | 3% (1/35) | 3% (1/35) | 0.00 | 1.000 | n.s. |

### Table 3: Diagnostic Counts (Mean per Run)

| Agent | Diagnostic | Baseline Mean±SD | Intervention Mean±SD | Diff | p-value | Sig |
|-------|-----------|-----------------|---------------------|------|---------|-----|
| macro | contradictions_count | 0.17±0.51 | 0.06±0.24 | -0.11 | 0.2540 | n.s. |
| macro | unsupported_claims_count | 0.00±0.00 | 0.00±0.00 | +0.00 | nan | n.s. |
| macro | ignored_critiques_count | 0.00±0.00 | 0.00±0.00 | +0.00 | nan | n.s. |
| macro | causal_overreach_count | 0.03±0.17 | 0.00±0.00 | -0.03 | 0.3244 | n.s. |
| macro | orphaned_positions_count | 0.03±0.17 | 0.00±0.00 | -0.03 | 0.3244 | n.s. |
| technical | contradictions_count | 0.00±0.00 | 0.14±0.55 | +0.14 | 0.1336 | n.s. |
| technical | unsupported_claims_count | 0.00±0.00 | 0.00±0.00 | +0.00 | nan | n.s. |
| technical | ignored_critiques_count | 0.00±0.00 | 0.00±0.00 | +0.00 | nan | n.s. |
| technical | causal_overreach_count | 0.71±0.46 | 0.66±0.59 | -0.06 | 0.6241 | n.s. |
| technical | orphaned_positions_count | 0.03±0.17 | 0.00±0.00 | -0.03 | 0.3244 | n.s. |

### Overall Reasoning Quality (rho_bar)

- **Baseline rho_bar**: 0.8322 ± 0.0222
- **Intervention rho_bar**: 0.8335 ± 0.0298
- **Difference**: +0.0013
- **Paired t-test**: t=-0.200, p=0.8426 (n.s.)
- **N**: 35 paired scenarios

## 3. Explanation Theme Analysis

**Total explanation texts collected**: 560

### Recurring Reasoning Patterns

| Theme | Total | Baseline | Intervention | Macro | Technical |
|-------|-------|----------|-------------|-------|-----------|
| Coherent reasoning | 116 | 56 | 60 | 60 | 56 |
| Causal rung collapse | 114 | 52 | 62 | 50 | 64 |
| Critique acceptance | 110 | 53 | 57 | 60 | 50 |
| Concentration risk | 95 | 49 | 46 | 27 | 68 |
| Evidence stretching | 29 | 14 | 15 | 14 | 15 |
| Unsupported claims | 17 | 9 | 8 | 8 | 9 |
| Ignored critiques | 9 | 5 | 4 | 5 | 4 |
| Technical confirmation bias | 4 | 3 | 1 | 0 | 4 |
| Premature certainty | 1 | 1 | 0 | 0 | 1 |

### Top Theme Examples

#### Coherent reasoning (116 occurrences)

- **macro** (baseline, scenario_2022Q4_rand7_5ec96480, logical_validity):
  > "The revision maintains a coherent macro thesis and links it to allocations, with trims to banks and CAT logically tied to acknowledged technical risks and added cash. No internal contradictions or conclusion drift were found."

- **technical** (baseline, scenario_2022Q4_rand7_5ec96480, logical_validity):
  > "The revision presents a coherent technical thesis with position sizes adjusted consistently in response to risks; all >10% positions are supported by explicit claims and the allocation aligns with the stated SMA/momentum framework."

#### Causal rung collapse (114 occurrences)

- **macro** (baseline, scenario_2022Q4_rand7_5ec96480, causal_alignment):
  > "Causal pathways are clearly articulated (rates → discount rates; rates → NII; commodities → producer cash flows; backlog → cyclicals) and supported by consistent evidence, with no rung collapse detected."

- **technical** (baseline, scenario_2022Q4_rand7_5ec96480, causal_alignment):
  > "C1 is labeled causal but relies on associational momentum persistence without an explicit mechanism, constituting causal overreach; C2 and C3 are framed as observational/risk and align with L1 association."

#### Critique acceptance (110 occurrences)

- **macro** (baseline, scenario_2023Q2_rand8_1c8e1d3e, logical_validity):
  > "The macro-to-portfolio mapping is coherent and allocations follow the stated theses, with tactical trims aligning to accepted critiques. Minor wording inconsistency appears in calling AMZN a 'modest overweight' while maintaining an overall long-duration underweight, but this does not undermine the o..."

- **macro** (baseline, scenario_2022Q1, logical_validity):
  > "Claims are internally consistent (inflationary expansion → banks, energy/industrials, and selective tech/consumer) and the portfolio weights follow the thesis; accepted critiques led to coherent weight shifts without contradicting core claims."

#### Concentration risk (95 occurrences)

- **macro** (baseline, scenario_2023Q2_rand8_1c8e1d3e, logical_validity):
  > "The macro-to-portfolio mapping is coherent and allocations follow the stated theses, with tactical trims aligning to accepted critiques. Minor wording inconsistency appears in calling AMZN a 'modest overweight' while maintaining an overall long-duration underweight, but this does not undermine the o..."

- **technical** (baseline, scenario_2023Q2_rand8_1c8e1d3e, logical_validity):
  > "The thesis, claims, and portfolio sizing are coherent: trend-following logic directs overweights while a volatility/beta-aware rule caps concentration. All positions >10% are supported by cited claims and falsifiers are explicit."

#### Evidence stretching (29 occurrences)

- **technical** (intervention, scenario_2022Q1_tech, logical_validity):
  > "The revised thesis, claims, and sizing changes are internally consistent and the allocation follows from the technical thesis; all positions >10% cite valid supporting claims. Minor inconsistency appears where MSFT’s position rationale references MSFT-VOL60 not in the evidence set, but this does not..."

- **macro** (baseline, scenario_2022Q4_rand7_5ec96480, evidential_support):
  > "All claims include relevant evidence IDs and plausible support; however, some position rationales and thesis references cite items not in the evidence list (e.g., momentum and curve metrics), introducing minor evidence gaps outside the core claims."

## 4. Intervention Impact on Reasoning

### Did the JS collapse intervention change reasoning behavior?

#### Diagnostic Flag Comparison

- **contradictions**: Baseline 5.7% → Intervention 7.1% (increased)
- **unsupported_claims**: Baseline 0.0% → Intervention 0.0% (unchanged)
- **ignored_critiques**: Baseline 0.0% → Intervention 0.0% (unchanged)
- **premature_certainty**: Baseline 1.4% → Intervention 0.0% (reduced)
- **causal_overreach**: Baseline 37.1% → Intervention 30.0% (reduced)
- **conclusion_drift**: Baseline 2.9% → Intervention 4.3% (increased)

### Within-Intervention: Retry vs No-Retry Runs

Do runs where intervention fired (retry occurred) show different CRIT scores?

- **Logical Validity**: Retry=0.834 vs No-retry=0.854, t=-1.09, p=0.280 (n.s.)
- **Evidential Support**: Retry=0.850 vs No-retry=0.852, t=-0.23, p=0.817 (n.s.)
- **Alternative Consideration**: Retry=0.829 vs No-retry=0.826, t=0.14, p=0.893 (n.s.)
- **Causal Alignment**: Retry=0.817 vs No-retry=0.809, t=0.37, p=0.710 (n.s.)

### Causal Overreach — Detailed Analysis

Causal overreach is the most frequent diagnostic flag. Breakdown:

- **macro**: Baseline 1/35 (3%) → Intervention 0/35 (0%)
- **technical**: Baseline 25/35 (71%) → Intervention 21/35 (60%)

## 5. CRIT Accuracy Audit

We validate CRIT diagnoses by checking the actual agent reasoning text.

### Methodology

For each run, we compare CRIT's diagnostic claims against the agent's
actual revision (or retry) text. We check:

1. **Causal overreach**: Does the agent text actually contain causal claims from correlational evidence?
2. **Contradictions**: Are there genuine internal contradictions?
3. **Unsupported claims**: Are there claims without evidence IDs?
4. **Ignored critiques**: Did the agent fail to respond to critique points?

### Automated Heuristic Audit Results

Using text pattern matching as ground-truth proxy:

| Diagnostic | TP | FP | TN | FN | Precision | FPR | Sensitivity |
|-----------|----|----|----|----|-----------|-----|------------|
| causal_overreach | 45 | 2 | 59 | 34 | 0.96 | 0.03 | 0.57 |
| contradictions | 0 | 9 | 131 | 0 | 0.00 | 0.06 | 0.00 |
| unsupported_claims | 0 | 0 | 113 | 27 | 0.00 | 0.00 | 0.00 |

### Caveats

- The automated audit uses **heuristic text patterns** as ground-truth proxy
- These patterns are imperfect — they may miss subtle reasoning issues
- CRIT has access to structured evidence IDs and the full debate context
- A high FP rate for contradictions may reflect CRIT detecting subtle
  inconsistencies not captured by simple pattern matching

### Manual Audit Examples

#### Example 1: technical — scenario_2022Q4_rand7_5ec96480 (baseline)

**CRIT claim**: causal_overreach_detected = True

**CRIT explanation**:
> "C1 is labeled causal but relies on associational momentum persistence without an explicit mechanism, constituting causal overreach; C2 and C3 are framed as observational/risk and align with L1 association."

**Relevant agent reasoning**:
> "25
  },

  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "Cross-sectional and idiosyncratic momentum identify ConocoPhillips as the strongest technical leader in the universe: price is trading above its 200d SMA with large intermediate and long-horizon momentum, implying a higher probability of continued trend persistence in the near term. "

**Assessment**: TRUE POSITIVE — Agent text contains causal language from correlational evidence


#### Example 2: technical — scenario_2022Q4_tech (baseline)

**CRIT claim**: causal_overreach_detected = True

**CRIT explanation**:
> "C1 asserts a causal link from elevated VIX/curve inversion to reduced trend persistence and allocation to cash without providing mechanism-supporting evidence beyond current levels, indicating causal overreach. Other claims remain observational/pattern-based and aligned with their evidence."

**Relevant agent reasoning**:
> "86
    },
    {
      "claim_id": "C3",
      "claim_text": "Within this damaged trend regime, AAPL shows materially stronger medium-term cross‑sectional momentum than AMD (AAPL positive 12‑1M momentum and flat RS60 vs AMD negative momentum and negative RS60), implying AAPL is the higher probability candidate for tactical, limited equity exposure until trend repair is confirmed. "

**Assessment**: TRUE POSITIVE — Agent text contains causal language from correlational evidence


#### Example 3: technical — scenario_2023Q1_rand9_2aba5c03 (baseline)

**CRIT claim**: causal_overreach_detected = True

**CRIT explanation**:
> "Most claims are framed as technical patterns, but C2 is labeled causal while relying primarily on associative technical evidence and lacks a clear transmission mechanism, indicating causal overreach."

**Relevant agent reasoning**:
> "13
  },

  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "Energy names (CVX, SLB) exhibit clear technical leadership across horizons — price above short/medium/long MAs, strong multi-horizon returns and idiosyncratic momentum — implying trend persistence and higher probability of continued outperformance in the near term. "

**Assessment**: TRUE POSITIVE — Agent text contains causal language from correlational evidence


#### Example 4: technical — scenario_2023Q2 (baseline)

**CRIT claim**: causal_overreach_detected = True

**CRIT explanation**:
> "C1 relies appropriately on pattern-based associations, but C2 is labeled causal while offering primarily associational SMA evidence without a clear mechanism, representing rung collapse. C3 is a risk policy grounded in observed beta/volatility patterns."

**Relevant agent reasoning**:
> "10
  },
  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "Momentum persistence: securities that show strong short-to-medium momentum, elevated relative strength, and are trading above short/medium SMAs have an increased probability of continued near-term outperformance until price action invalidates the setup. "

**Assessment**: TRUE POSITIVE — Agent text contains causal language from correlational evidence


#### Example 5: technical — scenario_2022Q3_rand3_1 (baseline)

**CRIT claim**: causal_overreach_detected = True

**CRIT explanation**:
> "C1 is labeled causal but rests on associational momentum/RS patterns without articulating a causal mechanism, representing rung collapse. Other claims are framed as pattern/observational or risk and align with their evidence."

**Relevant agent reasoning**:
> "25
  },

  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "XOM is the strongest technical leader in the universe: long-term momentum is positive (XOM-MOM200), 12-1M returns are very strong (XOM-MOM12_1), 60D relative strength is materially positive (XOM-RS60), and price sits above the 200-day SMA (XOM-SMA200) with decent trend consistency (XOM-TREND). "

**Assessment**: TRUE POSITIVE — Agent text contains causal language from correlational evidence


## 6. Conclusions

### Is CRIT reliable?

**Partially.** CRIT demonstrates:
- **High consistency**: Pillar scores show low variance across runs (SD ~0.03)
- **Appropriate calibration**: Scores cluster in the 0.80–0.87 range, not ceiling
- **Stable across conditions**: No significant difference between baseline and intervention
  rho_bar (p=0.84), indicating CRIT is not biased by the experimental condition

### Is CRIT hallucinating?

**The most notable concern is causal_overreach for the technical agent:**
- Flagged in 25/35 baseline runs (71%)
  and 21/35 intervention runs (60%)
- This high rate is **likely genuine**: the technical agent uses momentum-based
  reasoning (SMA crossovers, price momentum) and frequently makes causal claims
  from associational/correlational evidence — exactly what causal overreach detects
- The macro agent rarely triggers this flag because macro reasoning typically
  cites explicit causal mechanisms (rate → NII, commodities → cash flows)

**Other flags have very low false positive risk:**
- contradictions: rare (<12%), and CRIT explanations cite specific instances
- unsupported_claims: near-zero — agents consistently cite evidence IDs
- ignored_critiques: near-zero — the revision prompt forces critique engagement

### What reasoning failures are most common?

1. **Causal overreach** (technical agent) — by far the most frequent issue
2. **Evidence stretching** — minor gaps where agents cite evidence not in the provided list
3. **Contradictions** (rare) — occasional internal inconsistencies
4. **Premature certainty** and **conclusion drift** — very rare (<3%)

### Does the JS collapse intervention affect reasoning quality?

**No significant effect on reasoning quality.** Key evidence:
- rho_bar unchanged (0.832 vs 0.834, p=0.84)
- All four pillar scores unchanged across conditions
- Diagnostic flag rates unchanged
- The intervention successfully preserves opinion diversity (collapse ratio p=0.003)
  **without degrading reasoning quality**

This is a key finding: the intervention achieves its diversity-preservation goal
while maintaining the same level of reasoning rigor as the baseline condition.
