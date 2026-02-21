# 📘 Debate Output Validation

## Purpose, Ownership, and Operational Contract

---

# 🚨 Why This File Exists

The **Debate Output Validation Script** (`validate_debate_output.py`) is a **research safety mechanism**.

It ensures that:

* Every Debate Output File conforms to the **official JSON schema**
* Every debate artifact satisfies **semantic research invariants**
* Post-hoc experiments remain uncontaminated by in-loop behavior
* Evaluation modules (CRIT, RCA, T³, RAudit) receive structurally valid inputs

Without this script, your evaluation layer is operating on trust.

With this script, your evaluation layer is operating on guarantees.

---

# 🎯 What This Script Does

The validator performs **two completely separate classes of checks**:

---

## 1️⃣ JSON Schema Validation (Structural Integrity)

Validates the Debate Output File against:

```
contracts/schemas/debate_output.schema.json
```

This ensures:

* Required fields exist
* Field types are correct
* Enumerations are valid
* Structural nesting is correct
* Arrays and objects follow expected formats

This is **shape validation only**.

It does NOT check research semantics.

---

## 2️⃣ Runtime Semantic Validation (Research Invariants)

These are constraints that JSON Schema cannot express.

Examples include:

* No duplicate `turn_id`
* No duplicate `turn_index`
* `speaker_id` must exist in participants list
* `position_size_pct_min ≤ position_size_pct_max`
* `attempt_index` must be strictly increasing
* Referential integrity across object layers

Most importantly:

> 🔥 In `mode = "posthoc"`, each turn must contain **exactly one attempt**.

This rule prevents retry contamination of baseline experiments.

Schema alone cannot enforce this.
The runtime validator must.

---

# 🧠 Why Schema Alone Is Not Enough

JSON Schema can enforce:

* Required properties
* Type correctness
* Numeric ranges
* Enum membership

It cannot enforce:

* Cross-field logical constraints
* Cross-object uniqueness
* Mode-dependent invariants
* Monotonic ordering
* Referential integrity

Therefore:

> Schema validation and runtime validation are complementary, not redundant.

---

# 👥 Who Runs This Script?

**The Eval Team runs this script.**

Not the Debate Team.

### Why?

The Debate Team generates artifacts.

The Eval Team:

* Verifies structural compliance
* Enforces research invariants
* Guards experimental integrity
* Controls the evaluation boundary

This prevents:

* Debate-side schema drift
* Accidental retry leakage
* Mode confusion
* Silent corruption of evaluation metrics

Validation is part of the evaluation boundary layer.

---

# 🕒 When Should It Be Run?

The validator must be run:

### ✅ After debate generation

### ✅ Before any evaluation (CRIT, RCA, T³, RAudit)

### ✅ Before transforming into Canonical Eval Schema

It should be considered:

> A required gate before evaluation begins.

If the script exits with HARD errors (exit code 2):

* The artifact must not proceed to evaluation
* The debate output must be corrected or regenerated

---

# 📂 Where It Lives

```
contracts/scripts/validate_debate_output.py
```

It validates files shaped like:

```
contracts/schemas/debate_output.schema.json
```

---

# 🔍 What Happens If Validation Fails?

There are two types of findings:

| Type | Meaning                                               |
| ---- | ----------------------------------------------------- |
| HARD | Research invariant violation — evaluation must halt   |
| SOFT | Warning — artifact is usable but potentially degraded |

If HARD errors exist:

```
exit code = 2
```

Evaluation must not proceed.

---

# 🔬 Why This Matters for Research Integrity

Your experiments compare:

* Natural vs Structured prompting
* Post-hoc vs In-loop reasoning
* Retry vs No-retry regimes
* Model variants
* Temperature regimes

If artifacts silently violate mode constraints, you lose:

* Experimental isolation
* Interpretability
* Reproducibility
* Publication credibility

This validator enforces:

> Experimental boundary discipline.

---

# 🧩 Relationship to the Pipeline

Full system:

```
Debate → Debate Output File → Validation → Eval Transform → Evaluation
```

The validator sits between:

* Debate generation
* Evaluation

It is the **gatekeeper layer**.

---

# 📌 What It Does NOT Do

The validator does NOT:

* Modify reasoning
* Inject corrections
* Rewrite arguments
* Score debates
* Run CRIT
* Run RCA
* Run T³
* Run RAudit

It only verifies integrity.

---

# 🚦 Operational Summary

| Responsibility              | Owner       |
| --------------------------- | ----------- |
| Generate debate             | Debate Team |
| Produce Debate Output File  | Debate Team |
| Validate Debate Output File | Eval Team   |
| Transform to Eval Schema    | Eval Team   |
| Run Evaluators              | Eval Team   |

---

# 🧠 Final Principle

This script exists to enforce one critical boundary:

> The Debate Layer may generate reasoning.
>
> The Evaluation Layer decides what is admissible.

