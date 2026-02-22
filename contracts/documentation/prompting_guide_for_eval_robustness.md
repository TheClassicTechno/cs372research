# 📘 Debate Prompting Guidelines

## Producing Evaluation-Ready, Information-Dense Debate Content

*(For Post-Hoc CRIT, RCA, and T³ Analysis)*

---

# 1. Purpose of This Guide

This document defines prompting standards for debate agents whose outputs will be evaluated post hoc under:

* **CRIT** (argument quality evaluation)
* **RCA** (reasoning trace → conclusion consistency)
* **T³** (causal reasoning level analysis)

The **JSON schema defines structural format**.

This guide defines **reasoning expectations**.

These are separate concerns.

---

# 2. Why Prompting Matters (Even With a Strong Schema)

The Unified Debate Output Schema guarantees:

* Structural consistency
* Proper field alignment
* Reproducibility
* Artifact integrity

However, the schema does **not** guarantee:

* Clear claims
* Explicit mechanisms
* Causal discipline
* Counterargument engagement
* Logical coherence

A debate artifact can be perfectly schema-compliant and still be unusable for:

* CRIT (if claims are vague)
* RCA (if conclusions drift from reasoning)
* T³ (if causal structure is implicit or conflated)

In other words:

> The schema enforces structure.
> Prompting enforces reasoning density.

Without disciplined prompting, evaluators are forced to infer structure that may not exist. That reduces reliability and increases evaluator noise.

Prompting is therefore not cosmetic — it is upstream evaluator enablement.

---

# 3. Shared Reasoning Expectations (All Runs)

Regardless of reasoning regime, each debate turn must:

1. State a clear **central claim**
2. Provide multiple **distinct supporting reasons**
3. Engage at least one **counter-argument**
4. State key **assumptions**
5. End with a clear **final position**
6. Avoid silent stance shifts
7. Avoid self-labeling evaluation constructs
8. Not output JSON reasoning traces

These are baseline requirements for evaluator compatibility.

---

# 4. The Two Reasoning Regimes

In addition to baseline standards, we intentionally support two reasoning regimes.

This allows us to study whether structured discipline improves evaluation reliability.

Each run should specify:

```json
"reasoning_regime": "natural" | "structured"
```

in `run_metadata`.

The regimes differ in the level of structural enforcement, not in the core expectations above.

---

## Regime A — Natural Reasoning (Minimal Scaffolding)

**Purpose:**
Capture spontaneous reasoning quality with moderate structure.

**Characteristics:**

* Clear claim
* 2–4 identifiable supporting reasons
* Counterargument engagement
* Assumptions stated
* Clear final decision
* Causal reasoning expressed naturally

**Not required:**

* Explicit falsifiers per reason
* Formal epistemic separation
* Strict section ordering
* Mandatory conditional statements

**Use when:**

* Establishing baseline reasoning quality
* Measuring natural causal collapse frequency
* Comparing spontaneous vs structured reasoning

---

## Regime B — Structured Reasoning (Causal Discipline)

**Purpose:**
Enforce explicit reasoning discipline to reduce ambiguity in post-hoc evaluation.

**Characteristics:**

* Required section ordering
* Explicit causal mechanisms
* At least one conditional per reason
* Falsifiers per reason
* Operational assumptions
* Explicit epistemic clarity
* Internal consistency checks

Still **not allowed:**

* Self-labeling Pearl levels
* Self-scoring CRIT metrics
* Referencing evaluation frameworks
* Outputting JSON reasoning traces

**Use when:**

* Maximizing evaluator reliability
* Reducing RCA inconsistencies
* Improving T³ classification clarity
* Preparing for future in-loop control experiments

---

# 5. Supporting CRIT

CRIT depends on:

* Extractable claims
* Separable reasons
* Explicit warrants
* Counterargument engagement
* Stated assumptions

If reasoning is vague or rhetorical, CRIT scoring becomes unstable.

Structured Regime improves extractability; Natural Regime tests spontaneous reasoning quality.

---

# 6. Supporting RCA

RCA evaluates whether the conclusion logically follows from the reasoning trace.

To support RCA:

* Avoid hidden stance shifts
* Explain revisions explicitly
* Restate the final decision clearly

Structured Regime reduces accidental inconsistencies.
Natural Regime may expose them.

---

# 7. Supporting T³

T³ evaluates causal reasoning level post hoc.

We do not ask agents to label Pearl levels.

Instead:

* Natural Regime: causal reasoning emerges organically.
* Structured Regime: conditionals and epistemic clarity are enforced.

This enables comparison between spontaneous and disciplined causal reasoning.

---

# 8. What We Explicitly Avoid

Across all regimes:

* No Pearl rung self-labeling
* No self-scoring of CRIT metrics
* No trap classification
* No JSON reasoning traces
* No evaluator references in the body

Evaluation is strictly post hoc.

---

# 9. Why This Matters

High-quality evaluation requires:

* Clear reasoning
* Extractable structure
* Consistent conclusions
* Explicit causal framing

The JSON schema ensures structural integrity.

Prompting ensures intellectual integrity.

Both are required.

Without prompting discipline, the schema produces clean but shallow artifacts.

With prompting discipline, the artifact becomes evaluable.
