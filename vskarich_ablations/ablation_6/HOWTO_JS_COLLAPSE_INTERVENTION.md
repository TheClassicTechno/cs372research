# How-To: JS Collapse Intervention — Ablation 6

## What This PR Adds

Ablation 6 tests the **JS collapse intervention** — a mechanism that detects
when debate agents converge too quickly during revision and forces a retry
with a diversity nudge. This ablation sweeps three intervention strength
levels across multiple agent configurations and JS collapse thresholds.

The three scenarios are **unchanged** from prior ablations:

| Scenario | File | Quarter |
|----------|------|---------|
| Inflation rotation | `scenario_configs/2022Q1_inflation_rotation.yaml` | Q4 2021 → Q1 2022 |
| Sector dispersion | `scenario_configs/2023Q3_sector_dispersion.yaml` | Q2 2023 → Q3 2023 |
| Cross-section dispersion | `scenario_configs/2025Q1_cross_section_dispersion.yaml` | Q4 2024 → Q1 2025 |

---

## When Does the JS Collapse Retry Fire?

The pipeline runs:

```
propose → critique → revise → [IE post_revision check] → CRIT → judge
```

After the **revise** phase, the Intervention Engine (IE) computes:

```
collapse_ratio = JS_revision / JS_proposal
```

- `JS_proposal` = Jensen-Shannon divergence of the proposed portfolio allocations
- `JS_revision` = Jensen-Shannon divergence of the revised portfolio allocations

If `collapse_ratio < threshold`, it means agents converged too much during
revision. The IE triggers a **retry**: agents re-revise with an injected
nudge prompt that reminds them to maintain independent reasoning.

Each retry produces a new `revisions_retry_NNN/` directory under the round,
with its own `prompt.txt`, `response.txt`, and `portfolio.json` per agent.

---

## The Three Intervention Strength Levels

| Level | `revision_limits` | Effect |
|-------|-------------------|--------|
| **soft** | None | Nudge prompt only — agents can change any number of positions by any amount |
| **moderate** (new) | `max_changed_tickers: 3`, `max_change_pct: 10` | Nudge prompt + moderate portfolio constraints |
| **strict** | `max_changed_tickers: 2`, `max_change_pct: 5` | Nudge prompt + tight portfolio constraints |

The `revision_limits` block constrains how much an agent can change its
portfolio during an intervention retry. It appears inside `intervention_config`:

```yaml
# From debate_2agent_moderate_th08_r2_t0.3.yaml
intervention_config:
  enabled: true

  # Portfolio revision limits applied to all intervention retries.
  revision_limits:
    max_changed_tickers: 3
    max_change_pct: 10

  rules:
    js_collapse:
      enabled: true
      threshold: 0.8
      ...
```

Soft configs omit `revision_limits` entirely. Strict configs use `2` / `5`.

---

## Config Naming Convention

```
debate_{agents}_{strength}_{threshold}_r2_t0.3.yaml
```

- **agents**: `2agent`, `3agent_risk`, `3agent_value`
- **strength**: `soft`, `moderate`, `strict`
- **threshold**: `th065` (0.65) or `th08` (0.8)

Example: `debate_3agent_risk_moderate_th065_r2_t0.3.yaml`

This gives 18 debate configs total (3 agent configs × 3 strengths × 2 thresholds).

### Full Config Matrix

| Agent Config | Agents |
|--------------|--------|
| `2agent` | macro, technical |
| `3agent_risk` | macro, technical, risk |
| `3agent_value` | macro, technical, value |

Each crossed with `{soft, moderate, strict}` × `{th065, th08}`.

---

## Threshold Interpretation

| Threshold | Meaning |
|-----------|---------|
| `th08` (0.8) | More sensitive — fires when agents retain less than 80% of original diversity |
| `th065` (0.65) | Less sensitive — only fires on severe collapse (< 65% diversity retained) |

A lower threshold means the intervention fires less often. A higher threshold
catches milder convergence.

---

## Running the Ablation

Each debate config is crossed with each scenario config. The runner handles
all combinations:

```bash
# Run all 54 experiments (18 debate × 3 scenario)
python vskarich_ablations/ablation_6/run_ablation.py

# Parallel execution
python vskarich_ablations/ablation_6/run_ablation.py --workers 4

# Resume after interruption (picks up where it left off)
python vskarich_ablations/ablation_6/run_ablation.py

# Start fresh
python vskarich_ablations/ablation_6/run_ablation.py --reset
```

Progress is tracked in `ablation_status.json`. All configs use `num_episodes: 1`.

---

## Dashboard Metrics for Retry Rounds

The dashboard automatically discovers and displays retry phase metrics. No
configuration change is needed — if the run produced retry files, the
dashboard shows them.

### Divergence Overview Table

On the run detail page, the **Divergence Overview** section shows a per-round
table with these phases:

| Phase | JS Divergence | Evidence Overlap |
|-------|---------------|------------------|
| Proposed | 0.3115 | 0.0581 |
| Revised | 0.0601 | 0.0594 |
| **Retry 1** | 0.2900 | 0.0732 |
| **Retry 2** | 0.2724 | 0.0843 |

Retry rows appear automatically when the metrics directory contains
`js_divergence_retry_001.json` / `evidence_overlap_retry_001.json`, etc.

### What to Look For

When reviewing runs:

1. **Did the intervention fire?** Check the `interventions/` directory in
   each round. Each `intervention_NNN.json` logs the rule, metrics, and
   nudge text.

2. **Did diversity recover?** Compare JS divergence across phases in the
   Divergence Overview. A successful intervention shows JS divergence
   increasing from Revised → Retry 1.

3. **Did revision limits constrain the portfolio?** For moderate/strict
   configs, compare the retry portfolio to the base revision. The portfolio
   diff should stay within the configured limits.

4. **CRIT scores after retry**: The CRIT auditor evaluates the agent's
   revised argument after each retry cycle. Check the PID Stats section
   for per-agent `rho_i` and pillar scores.

### API Endpoint

```
GET /runs/{experiment}/{run_id}/divergence
```

Returns the full divergence trajectory including retry phases.

---

## Run Artifact Structure (for reference)

A completed run with interventions produces this layout per round:

```
rounds/round_001/
├── proposals/{role}/          # Round 1 only
├── critiques/{role}/          # Every round
├── revisions/{role}/          # Base revision
├── revisions_retry_001/{role}/ # First retry (after intervention_000)
├── revisions_retry_002/{role}/ # Second retry (after intervention_001)
├── interventions/
│   ├── intervention_000.json  # JS collapse or reasoning quality
│   ├── intervention_001.json
│   └── intervention_002.json
├── CRIT/{role}/               # Reasoning audit
├── metrics/
│   ├── js_divergence.json
│   ├── js_divergence_propose.json
│   ├── js_divergence_retry_001.json  # ← dashboard reads these
│   ├── js_divergence_retry_002.json
│   ├── evidence_overlap.json
│   ├── evidence_overlap_propose.json
│   ├── evidence_overlap_retry_001.json
│   ├── evidence_overlap_retry_002.json
│   ├── crit_scores.json
│   └── pid_state.json
└── round_state.json
```

---

## Nudge Prompts

All three strength levels use the **same nudge prompts**. The prompts are
identical across soft/moderate/strict — only the `revision_limits` differ.

Each nudge contains:
- A general diversity protocol activation message
- A role-specific reminder (e.g., `MACRO INTERVENTION REMINDER`,
  `TECHNICAL INTERVENTION REMINDER`, `RISK INTERVENTION REMINDER`,
  `VALUE INTERVENTION REMINDER`)
- A checklist for the agent to follow before finalizing

The nudge text is injected into the retry prompt under an
`### INTERVENTION NOTICE` header.

---

## Integration Test

The new `tests/integration/test_run_artifact_integrity.py` validates that
run artifacts are complete and internally consistent. It checks intervention
→ retry causality, nudge injection into retry prompts, portfolio validity,
and cross-round invariants. Run with:

```bash
pytest tests/integration/test_run_artifact_integrity.py -v
```
