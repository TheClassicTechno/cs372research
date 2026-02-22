
---

# eval.schema.json — What It Is and Why It Exists

## 1. What is this file?

The `eval.schema.json` file defines the structure of an **evaluation artifact**.

An evaluation artifact is a JSON file that records the results of evaluating a completed debate run.

It is **not** the debate itself.
It is **not** the raw transcript.
It is a structured record of how that debate performed under different evaluation tools (CRIT, RCA, T³, control logic, etc.).

If the debate output is the “recording,” this file is the “grading sheet.”

---

# 2. Where It Fits in the Pipeline

Here is the big picture:

```
Debate System
   │
   │ produces
   ▼
debate_output.json
   (full transcript, rounds, turns, attempts)
   │
   │ fed into
   ▼
Evaluation Module
   │
   │ produces
   ▼
eval.json   ← (THIS SCHEMA DEFINES THIS FILE)
```

More explicitly:

```
+-------------------+
| Debate Team       |
|                   |
| debate_output.json|
+-------------------+
           │
           ▼
+-------------------+
| Evaluation Layer  |
|  - CRIT           |
|  - RCA            |
|  - T³             |
|  - Control logic  |
+-------------------+
           │
           ▼
+-------------------+
| eval.json         |
| (structured       |
|  evaluation)      |
+-------------------+
```

So the evaluation artifact:

* References a debate run
* Records what evaluation tools were applied
* Stores summary results
* Optionally stores per-turn results
* Optionally stores control-loop traces

It sits between raw debate behavior and structured analysis.

---

# 3. How It Connects to debate_output.json

Two fields link the evaluation artifact back to the original debate:

* `debate_id`
* `run_id`

These must match the same fields in `debate_output.json`.

This gives you a clean separation:

* **debate_output.json** → what happened
* **eval.json** → how we evaluated what happened

The evaluation file never duplicates transcript content.
It only references it.

That means:

* You can always trace evaluation results back to the exact debate run.
* You can regenerate evaluation if needed.
* You can compare multiple evaluation passes against the same debate.

---

# 4. What Is `debate_id` vs `run_id`?

This is an important distinction.

## `debate_id`

This identifies the underlying scenario or task.

Examples:

* `"nvda_q1_2026"`
* `"causal_trap_L3_case_12"`
* `"math_problem_set_A"`

If you run the same scenario multiple times, they share the same `debate_id`.

Think of this as the case identifier.

---

## `run_id`

This identifies a specific execution of that scenario.

Examples:

* `"run_0001"`
* `"run_0042"`
* `"seed_1337"`

You might run the same debate:

* With different random seeds
* With different temperatures
* With different control settings
* Before and after fixing a bug

Those all share the same `debate_id` but have different `run_id`s.

So:

* `debate_id` groups runs by scenario.
* `run_id` distinguishes individual executions.

---

# 5. Top-Level Fields Explained

Let’s walk through the main fields in plain language.

---

## `schema_version`

This indicates which version of the evaluation schema this file follows.

Why it exists:

* If the structure changes later, we can detect it.
* It prevents silent breaking changes.
* It helps with compatibility.

---

## `evaluation_mode`

Possible values:

* `"posthoc"`
* `"in_loop"`

Plain meaning:

* **posthoc** → the debate ran freely; evaluation happened afterward.
* **in_loop** → evaluation tools were active during the debate (for example, triggering retries or adjustments).

This tells you whether evaluation was observational or interactive.

---

## `evaluated_at`

Timestamp of when evaluation ran.

Why it exists:

* Reproducibility
* Audit trail
* Comparing evaluation runs over time

---

# 6. experiment_config — What Kind of Run Was This?

This block describes the experimental setup.

It answers:

> What kind of run was this?

It does not store results. It stores configuration.

---

## `label`

A free-form name for the experiment.

Examples:

* `"debate_crit_rca_pid"`
* `"latency_stress_test"`
* `"temperature_sweep_1.3"`

This is intentionally flexible.
We do not restrict experiment names with a rigid list.

---

## `category`

Optional grouping label.

Examples:

* `"core_ablation"`
* `"exploratory"`
* `"benchmark"`

This helps when organizing results later.

---

## `interventions`

This indicates which evaluation tools actively influenced the debate while it was running.

Example:

```
crit_in_loop: true
rca_in_loop: false
```

This means:

* CRIT influenced the debate.
* RCA did not.

This helps separate:

* Tools that shaped behavior
* Tools that only measured behavior afterward

---

## `control`

This is an optional, generalized control framework.

If present, it describes:

* Whether control was enabled
* What policy was used (PID, threshold, etc.)
* What signals were monitored
* What actions were allowed
* When to stop
* Whether step-by-step control logs were stored

The important part is that control is **explicitly defined**.

For example:

* You can run PID over CRIT scores.
* You can run PID over RCA trace consistency.
* You can run threshold-based retry logic.
* You can run latency experiments.
* You can experiment with custom signals.

We intentionally avoided hard-coding assumptions like “PID requires RAudit.”

Instead, the control block declares:

* What signal is used
* What actions are available

That keeps the system flexible.

---

## `extra_dimensions`

This is a flexible space for experiment-specific settings.

Examples:

* temperature
* max_tokens
* perturbation_strength
* latency_tracking

This prevents the schema from needing updates every time we try something new.

---

# 7. eval_metadata — What Code Produced This?

This block records which versions of the evaluation tools were used.

Examples:

* `crit_version`
* `rca_version`
* `pid_version`
* `raudit_version`

This helps with reproducibility.

If evaluation logic changes later, we can trace differences back to tool versions.

This block is about traceability, not scoring.

---

# 8. run_summary — High-Level Results

This is the main summary of evaluation.

It includes:

* `overall_verdict` (pass / fail / mixed)
* `crit_summary`
* `rca_summary`
* `t3_summary`
* `pid_summary`
* `raudit_summary`

Each block stores structured metrics.

For example:

* CRIT might store average reasoning scores.
* RCA might store trace consistency rate.
* PID might store retry counts.
* RAudit might store drift metrics.

Not every experiment needs every summary.
Some fields can be null.

---

# 9. turn_evaluations — Optional Per-Turn Results

This optional array stores evaluation results per turn.

It does not store raw text.
It stores metrics about each turn.

This is useful when:

* Diagnosing specific failures
* Studying instability
* Comparing early vs late rounds

If it’s null, only summary-level metrics were stored.

---

# 10. control_trace — Optional Control Logs

If control was enabled and logging was requested, this stores step-by-step records:

* What signal values were observed
* What actions were taken
* When those actions occurred

This is useful for:

* Debugging controller behavior
* Studying convergence
* Measuring oscillation

It is optional.

---

# 11. How to Read an eval.json File (Quick Walkthrough)

If you open an `eval.json` file, here’s a simple way to approach it.

### Step 1 — Identify the Run

Look at:

```
debate_id
run_id
```

Then find the matching `debate_output.json`.

---

### Step 2 — Was It In-Loop or Posthoc?

Look at:

```
evaluation_mode
```

This tells you whether evaluation influenced the debate or just measured it.

---

### Step 3 — What Was the Experiment Setup?

Look at:

```
experiment_config
```

This tells you:

* What the run was called
* Which tools influenced it
* Whether control was active
* What signals were used

This is the “how was this run configured?” section.

---

### Step 4 — What Were the Results?

Look at:

```
run_summary
```

This gives you the high-level outcome.

If you need deeper insight, check:

```
turn_evaluations
```

If you’re studying control behavior, check:

```
control_trace
```

---

# 12. Why This Schema Is Structured This Way

The design tries to balance two things:

### Structure and traceability

We want:

* Clear links to debate runs
* Clear tool summaries
* Version tracking
* Reproducibility

### Flexibility

We do not want:

* A rigid list of experiment types
* Hard-coded assumptions about which tools must exist
* Frequent schema changes when experimenting

That’s why:

* Experiment labels are free-form.
* Control is generalized.
* Extra dimensions are allowed.
* Tools are optional.

---

# 13. One-Paragraph Summary

The evaluation schema defines the structure of a JSON file that records how a debate run was evaluated. It links back to the original debate transcript, describes how the run was configured, stores summary results from evaluation tools, and optionally logs per-turn metrics and control behavior. It is designed to be structured enough for reproducibility and comparison, but flexible enough to support new experiments without constant schema changes.

---

If someone new joins the project, this file answers a simple question:

> Given a debate run, how was it evaluated, and what were the results?

That’s its purpose.
