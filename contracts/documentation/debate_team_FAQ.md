# ❓ FAQ — For the Debate Team

---

## 1️⃣ Do we need to understand the Eval Schema?

**No.**

You only need to:

* Produce a Debate Output File
* Follow `contracts/schemas/debate_output.schema.json`
* Follow the structural assumptions in
  ⭐ `contracts/documentation/debate_output_schema_design.md`

The Eval Team will handle transformation into the Canonical Eval Schema.

---

## 2️⃣ Do we need to generate evaluation fields (CRIT, RCA, T³, RAudit)?

**No.**

Do not generate:

* `crit`
* `rca`
* `t3`
* `raud_it`
* `audit`
* `control_state`
* `intervention_log`

Those are Eval-layer concerns.

Your job is debate generation only.

---

## 3️⃣ Do we need to implement retries or mid-debate intervention?

**No (for now).**

We are in:

> **post-hoc mode only**

That means:

* Exactly one attempt per turn
* No retries
* No mid-debate control
* No PID
* No gating

The schema supports those features for future experiments.

Ignore them for this phase.

---

## 4️⃣ What is the most important document for us?

This one:

> ⭐ `contracts/documentation/debate_output_schema_design.md`

That document defines:

* What a debate “round” means
* What a “turn” means
* What assumptions we’re making
* Why the debate is structured the way it is

If your mental model of the debate differs from that document, we need to align.

---

## 5️⃣ Why can’t we just output raw text instead of JSON?

Because:

* Evaluation requires structured indexing.
* We need stable `turn_id`, `round_index`, `speaker_id`.
* We need deterministic ordering.
* We need reproducibility across experiments.

The JSON file is the formal contract between Debate and Eval.

Think of it as a container — your natural language reasoning lives inside it.

---

## 6️⃣ Why is there a validation script?

Because JSON Schema alone cannot enforce:

* Unique turn IDs
* Proper round ordering
* Mode consistency
* Semantic integrity

The validation script ensures:

* Your file is structurally coherent
* Evaluation will not crash
* Experiments remain reproducible

The Eval Team runs the validator.

If something fails, we will return clear error messages.

---

## 7️⃣ What happens if validation fails?

The file will be returned with a specific error.

Common causes might include:

* Duplicate turn IDs
* Incorrect round indexing
* Missing required fields
* Multiple attempts in post-hoc mode

This is not punitive — it protects both teams from subtle bugs.

---

## 8️⃣ Are we allowed to change debate structure?

Not unilaterally.

Structural changes affect:

* CRIT extraction
* RCA trace mapping
* T³ rung detection
* Influence modeling
* Convergence metrics

If you want to modify:

* Response topology
* Speaking order
* Round definitions
* Visibility rules

We need joint discussion first.

---

## 9️⃣ Do we need to think about CRIT / RCA while designing prompts?

Yes — at a high level.

Schema ensures structure.

Prompting ensures evaluability.

Your prompts should:

* Encourage explicit claims
* Encourage mechanism clarity
* Encourage counterargument engagement
* Encourage conditional reasoning

See:

`contracts/documentation/prompting_guide_for_eval_robustness.md`

---

## 🔟 What is the single most important principle?

Keep the layers clean.

Debate Team:

> Generate high-quality reasoning and package it in the Debate Output File.

Eval Team:

> Validate, transform, and analyze.

If we respect that boundary, the system stays stable.

---

# 🧭 One-Sentence Summary

You produce structured debate reasoning in a single JSON file per debate.
We validate it and evaluate it post-hoc.
Everything else is future extensibility.
