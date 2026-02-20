# 1. Overview

The Unified Debate Output Schema v2.0 defines the canonical format for all debate artifacts in the CS372 evaluation system.

It supports:

* **Post-hoc evaluation** (CRIT, RCA, T³)
* **Mid-debate intervention** (CRIT/RCA/PID in-loop control)
* **Behavioral auditing (RAudit-style experiments)**
* Multi-run experimentation
* Long-term extensibility

The schema is intentionally:

* Flexible
* Minimally restrictive
* Forward-compatible
* Control-loop aware
* Evaluator-agnostic

---

# 2. Core Design Philosophy

## 2.1 One Canonical Artifact

There is exactly one debate output format.

All evaluation modules read from the same structured artifact.

We do NOT create separate:

* CRIT schema
* RCA schema
* T³ schema
* RAudit schema

Instead, evaluators extract what they need from a shared structure.

This prevents:

* Format drift
* Experimental contamination
* Duplicate representations
* Data misalignment

---

## 2.2 Static Artifact vs Control-Layer Extension

The schema separates:

| Layer              | Purpose                        |
| ------------------ | ------------------------------ |
| Core Debate Fields | What agents said and concluded |
| In-Loop Fields     | How evaluators intervened      |
| RAudit Fields      | Behavioral regime tracking     |

The debate transcript is never overwritten.
Retries are appended as attempts.

We never mutate history.

---

## 2.3 Separation of Scenario and Execution

Two identifiers are critical:

### `debate_id`

Identifies the scenario.

Example:

```
nvda_structural_vs_cyclical_2024
```

### `run_id`

Identifies a specific execution of that scenario.

Recommended:

* UUID v4

Why separate them?

Because the same debate may be run:

* With different models
* With different prompts
* With different temperatures
* With in-loop control enabled
* With ablations

`run_id` prevents collisions and preserves reproducibility.

---

# 3. Post-Hoc vs In-Loop Modes

The schema supports two execution modes:

```
"evaluation_mode": "posthoc" | "in_loop"
```

## Posthoc Mode

* Each turn has implicit single attempt.
* `attempts` field may be omitted.
* Evaluators operate after debate completion.

## In-Loop Mode

* Turns may contain `attempts[]`.
* Each failed attempt is preserved.
* Audit outputs are recorded.
* Control signals may be logged.
* PID logic may update behavior.

This makes v2 fully compatible with mid-debate CRIT/RCA/PID control loops.

---

# 4. Major Structural Components

---

## 4.1 Top-Level Fields

### `schema_version`

Must equal `"2.0.0"`.

Used for backward compatibility.

---

### `debate_id`

Scenario identifier.

Human-readable string.

---

### `run_id`

Execution identifier.

Recommended: UUID v4.

---

### `run_metadata`

Describes generation configuration:

* generation_mode (live, mock, ablation)
* global_model_name
* temperature
* top_p
* seed
* max_tokens
* prompt_bundle_version
* notes

This enables:

* Reproducibility
* Experimental tracking
* Debugging
* RAudit comparison

---

### `metadata`

Scenario-level context:

* created_at (required)
* scenario_id
* task_type
* market_context
* notes

---

### `participants`

Defines all agents.

Required fields:

* agent_id
* role

Optional:

* model_name
* system_prompt_version

`speaker_id` in turns must match one of these agent_ids.
(Enforced at runtime, not by JSON Schema.)

---

# 5. Turn Structure

Each turn contains:

* turn_id
* turn_index (absolute order)
* speaker_id
* content

Optional:

* round_index
* turn_type
* recommendation
* rca_trace
* attempts

---

## 5.1 turn_index

This is the canonical ordering.

It must be unique and monotonic within a run.

Do not rely on array order alone.

---

# 6. Recommendation Object (Strict by Design)

The only intentionally strict object is `recommendation`.

Why strict?

Because its numeric fields are likely to be programmatically consumed later.

Required fields:

* action (BUY, SELL, SHORT, HOLD)
* position_size_pct_min
* position_size_pct_max
* horizon_days_min
* horizon_days_max
* conviction (0–1)
* raw_text

We enforce:

```
additionalProperties: false
```

here to prevent numeric drift.

All other objects remain flexible.

---

# 7. In-Loop Extension (Attempts)

When `evaluation_mode = in_loop`, each turn may include:

```
"attempts": [...]
```

Each attempt records:

* attempt_index
* timestamp
* content (raw reasoning for that attempt)
* final_answer
* optional reasoning_trace
* optional recommendation
* optional audit
* optional control_state

This structure enables:

* Retry preservation
* Intervention traceability
* PID escalation
* RAudit behavioral classification

---

# 8. Audit Object (Extensible)

Each attempt may include:

```
"audit": {
   "crit": {...},
   "rca": {...},
   "t3": {...},
   "pid": {...},
   "raud_it": {...}
}
```

All audit sub-objects are extensible.

They may include:

CRIT:

* gamma_mean
* theta_mean
* threshold_pass
* feedback

RCA:

* blind_verification
* trace_output_consistent
* violation_type
* verdict
* feedback

T3:

* required_rung
* pearl_rung_detected
* trap_type
* pass
* feedback

PID:

* e_t
* u_t
* k_p, k_i, k_d
* feedback

RAudit:

* audit_strength
* behavioral_metrics
* retry_regime
* notes

Nothing in audit is strictly enforced beyond basic types.

This allows experimentation without schema friction.

---

# 9. Control State Object

Each attempt may include:

```
"control_state": {
   "e_t": ...,
   "u_t": ...,
   "persona": ...,
   "strategy_level": ...,
   "retry_count": ...
}
```

This tracks:

* PID control signals
* Persona shifts
* Escalation strategies
* Retry counts

It does not affect debate content.

It only logs intervention state.

---

# 10. RAudit Compatibility

v2 supports RAudit-style experiments in two ways:

### Run-level

Top-level `raud_it` object for:

* audit strength
* global metrics
* regime classification

### Attempt-level

`attempt.audit.raud_it` for:

* per-attempt behavioral state
* T→F / F→T transitions
* paranoia tracking

The schema does not predefine all RAudit metrics.

It intentionally leaves this open.

---

# 11. What This Schema Enables Long-Term

* Clean post-hoc CRIT evaluation
* Clean post-hoc RCA verification
* T³ causal rung scoring
* Mid-debate rejection and retry loops
* PID controller escalation
* RAudit-style behavioral archetyping
* Multi-run ablations
* Model comparison studies
* Seed sweeps
* Paranoia-rate measurement

Without rewriting the schema.

---

# 12. Prompting vs Schema

The schema enforces structure.

It does NOT enforce reasoning quality.

Reasoning density, causal depth, rebuttal clarity, and trace completeness are controlled through prompting.

This document defines structure only.

A separate prompting standards document governs:

* Required argument density
* Counterfactual articulation
* Rival reason explicitness
* Evidence formatting discipline

