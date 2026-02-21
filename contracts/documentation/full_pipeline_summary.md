# 📘 CS372 Debate & Evaluation Pipeline

## Post-Hoc Phase (Current Implementation)

---

# 🚨 READ THIS FIRST (5-Minute Orientation)

If you are new to this system, here is what matters:

1. The Debate Team produces **one JSON file per debate**.

2. That file follows the schema in:

   `contracts/schemas/debate_output.schema.json`

3. The conceptual design for that file is documented in:

   ⭐ `contracts/documentation/debate_output_schema_design.md`

4. We are currently operating in **post-hoc mode only**:

   * No retries
   * No mid-debate intervention
   * No PID control
   * No RCA gating

5. The **Eval Team runs the validation script** and converts the Debate Output File into the Canonical Eval Schema.

If you read only one structural document, read:

> ⭐ `contracts/documentation/debate_output_schema_design.md`

Everything else builds on that.

---

# 🔎 Quick Navigation

| What You Want                       | Go Here                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Debate Output file format           | `contracts/schemas/debate_output.schema.json`                                |
| Debate format & shared mental model | ⭐ `contracts/documentation/debate_output_schema_design.md`                   |
| Debate topology assumptions         | `contracts/documentation/debate_topology_considerations.md`                  |
| Speaking order assumptions          | `contracts/documentation/agent_response_order_considerations.md`             |
| Transcript visibility assumptions   | `contracts/documentation/agent_full_transcript_visibility_considerations.md` |
| Prompting standards                 | `contracts/documentation/prompting_guide_for_eval_robustness.md`             |
| Eval schema contract                | `evaluation/schemas/eval.schema.json`                                        |
| Debate Output validator             | `contracts/scripts/validate_debate_output.py`                                |

---

# 1️⃣ System Overview

# 2️⃣ End-to-End Flow (Post-Hoc Mode)

The current pipeline is:

```
Scenario Definition
    ↓
Prompting Layer
    ↓
LLM Debate Agents
    ↓
Debate Output File (JSON)
    ↓
Debate Json Validation Script (Eval Team owns, checks semantic debate requirements; see validate_debate_output_doc.md)
    ↓
Eval Logic for CRIT / RCA / T³ / RAudit
    ↓
Canonical Post-Eval File (we store the evaluation results for one debate here, PR is up)
 
```

There is no feedback loop at this stage.

This is strictly:

> Debate → Artifact → Validate → Evaluate

---

# 3️⃣ Debate Output File (What Debate Team Produces)

📁 Schema:
`contracts/schemas/debate_output.schema.json`

📄 Core Conceptual Document:
⭐ `contracts/documentation/debate_output_schema_design.md`

This file defines:

* Round-based structure
* One response per agent per round
* Deterministic speaking order
* Full transcript visibility
* Emergent interaction topology
* Post-hoc baseline assumptions
* Extensibility for future in-loop mode

That design document encodes the **shared mental model of what a debate is** in this system.

If structural assumptions change, evaluation results become invalid or uninterpretable.

---

# 4️⃣ Current Mode: Post-Hoc Only

We are running:

> **mode = "posthoc"**

This means:

* Exactly one attempt per turn
* No retries
* No intervention_log
* No control_state
* No PID
* No mid-debate evaluator feedback

The schema includes those fields for future use.

They are inactive for now.

Each turn must include exactly one attempt:

```json
"attempts": [
  {
    "attempt_index": 0,
    "status": "ok",
    "content": "..."
  }
]
```

Keep it simple.

---

# 5️⃣ Why the Validation Script Is Required

📁
`contracts/scripts/validate_debate_output.py`

The JSON schema alone is not sufficient.

JSON Schema can enforce:

* Field presence
* Field types
* Numeric bounds
* Enum values

It **cannot enforce semantic consistency**.

The validation script is required because it enforces constraints such as:

* Unique `turn_id` across all turns
* Strict monotonic `turn_index`
* Proper round grouping
* Speaker consistency
* No duplicate agent IDs
* Exactly one attempt per turn in post-hoc mode
* Mode alignment (e.g., no retries in post-hoc)
* Structural integrity across nested objects

Without the validator:

* Schema-valid files could still be semantically broken
* Evaluation modules could crash or misinterpret data
* Experimental comparisons could become invalid

The validator is the **integrity firewall** between Debate and Evaluation.

---

### Important Ownership Decision

> The **Eval Team runs the validation script.**

Debate Team produces the file.

Eval Team validates it.

If validation fails, the file is returned for correction.

This ensures:

* Clean responsibility boundaries
* No schema drift
* No hidden evaluation contamination
* Reproducibility

---

# 7️⃣ Separation of Responsibilities

| Component                        | Owned By                             |
|----------------------------------|--------------------------------------|
| Prompting                        | Debate Team w/ Eval Team Suggestions |
| Debate Output File               | Debate Team                          |
| Debate Structure Documentation   | Shared                               |
| Debate Output Schema             | Shared                               |
| Validation Script                | Eval Team                            |
| Evaluators (CRIT/RCA/T³/RAudit)  | Eval Team                            |
| Eval Results Schema)             | Eval Team                            |

This separation prevents:

* Schema drift
* Evaluation contamination
* Artifact mutation
* Responsibility confusion

---

# 8️⃣ Why Schema Alone Is Not Enough

It is critical to understand:

* **Schema guarantees structure.**
* **Prompting guarantees reasoning quality.**
* **Validator guarantees semantic integrity.**
* **Evaluators measure reasoning.**

If any of those layers are removed:

* Results become unstable.
* Metrics become uninterpretable.
* Experiments lose validity.

This layered architecture is intentional.

---

# 9️⃣ Mental Model (Current Phase)

We are intentionally keeping the system simple:

```
Debate → Debate Output File → Validation → Eval Schema → Evaluation
```

No retries.
No intervention.
No in-loop gating.

Just clean generation and clean post-hoc analysis.

