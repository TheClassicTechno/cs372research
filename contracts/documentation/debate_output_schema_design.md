# Debate Transcript Output JSON Spec v2.0

# 📚 Table of Contents

## 1. Role of This Artifact in the System

* Pipeline Overview
* Single Source of Truth Principle

## 2. Debate Assumptions and Shared Mental Model

1. Round-Based Structure
2. One Holistic Response Per Agent Per Round
3. Deterministic Speaking Order
4. Full Transcript Visibility
5. Emergent Interaction Topology
6. Explicit Referencing of Opposing Claims
7. No Mid-Debate Intervention (Current Phase)
8. Single Final Position Per Turn
9. No Structural Enforcement of Convergence
10. Clean Separation Between Debate and Evaluation

* Shared Debate Visualization
* Why These Assumptions Matter

## 3. Design Goals

* Post-Hoc Evaluation Support
* Retry & Intervention Infrastructure
* Extensibility & Research Robustness

## 4. 🚨 Debate Team Requirements (Post-Hoc Evaluation Only)

**Attention Debate Team:**
This section defines exactly what must be implemented for this phase.
We are in post-hoc evaluation mode only.
No retries, no interventions, no PID, no controller state.
Just clean structured transcript generation with one attempt per turn.

* Required Top-Level Fields
* Required Round Structure
* Required Turn Structure
* Required Attempt Structure
* One-Attempt Rule (Critical)
* What Can Be Ignored for This Phase

## 5. Major Design Decisions & Tradeoffs

A. JSON vs. Markdown
B. One Debate = One File
C. Explicit Attempt Logging
D. Emergent Interaction Topology
E. Schema vs. Runtime Validation

## 6. Role of the Transcript Validator

* Structural Integrity Checks
* Attempt Semantics
* Mode Rules
* Intervention Consistency
* Round Consistency
* Recommendation to Debate Team

## 7. Top-Level Fields

Details about each top-level field and its purpose within the system.

---

# 1. Role of This Artifact in the System

This JSON object is the **primary transport artifact** produced by the Debate Team and consumed by the Evaluation layer.

It is **not** the canonical evaluation schema used by CRIT / RCA / T³.
Instead, it is the *raw, structured record* of a debate run.

Pipeline overview:

```
Debate Runner
    ↓
Transcript JSON (this spec)
    ↓
Transcript Validator (semantic checks)
    ↓
Eval Transformer
    ↓
Canonical Eval Schema
    ↓
CRIT / RCA / T³ / PID / RAudit
```

This artifact must:

* Preserve raw natural-language debate content
* Log retries and failed attempts
* Log mid-debate interventions
* Remain extensible for new control mechanisms
* Be simple enough for the Debate Team to produce
* Be robust enough for research-grade evaluation

It is designed to be the **single source of truth** for debate execution.

One debate = one file.

Before we discuss the schema details, let's align on debate structure!

# 🧠 Debate Assumptions and Shared Mental Model

This section clarifies what we mean by “a debate” in this system. The Evaluation module makes certain structural assumptions about how debates are conducted. These assumptions are not arbitrary — they are necessary to ensure that post-hoc reasoning evaluation is meaningful and consistent.

The goal of this section is alignment. We want the Debate Team and Evaluation Team to share the same picture of what a debate run looks like.

---

# 1️⃣ Debate Is Round-Based

A debate consists of a sequence of **rounds**.

Each round:

* Has an index (`round_index`)
* Contains one contribution from each active agent
* Is ordered (Round 1, Round 2, Round 3, …)

We are assuming:

> Debate proceeds in discrete, synchronous rounds.

Agents are not interleaving messages arbitrarily.
We are not modeling asynchronous chat.

This makes:

* Turn ordering deterministic
* Convergence measurable
* Influence tracking coherent

---

# 2️⃣ One Holistic Response Per Agent Per Round

Each agent produces:

> **Exactly one response per round.**

That response may:

* Introduce new arguments
* Rebut opposing claims
* Refine its own position
* Address multiple agents

But it is one unified response.

We are not:

* Making separate API calls per opponent
* Doing n(n−1) pairwise cross-examination
* Forcing per-opponent response slots

Each agent sees the transcript and produces one structured argument.

---

# 3️⃣ Deterministic Speaking Order

We assume a **deterministic rotating speaking order** across rounds.

Example (3 agents):

* Round 1: A → B → C
* Round 2: B → C → A
* Round 3: C → A → B

This ensures:

* No persistent anchoring bias
* Fair distribution of “first mover” advantage
* Reproducibility across runs

Order is not random and not adaptive (for now).

---

# 4️⃣ Full Transcript Visibility (Baseline)

Each agent sees:

> **The full transcript up to that point.**

We are not using:

* Prior-round-only visibility
* Sliding window context
* Partial transcript exposure

This preserves:

* Path dependence
* Convergence dynamics
* Sycophancy effects
* Long-memory reasoning

If we later run ablations, visibility may change.
But the baseline assumes full history access.

---

# 5️⃣ Emergent Interaction Topology

Agents are **not required** to respond to all other agents.

They are expected to:

> Address materially relevant opposing claims.

But they are not forced to:

* Respond to every agent
* Respond to exactly one agent
* Follow fixed rebuttal slots

Interaction structure is allowed to emerge naturally.

This preserves:

* Dominance patterns
* Marginalization effects
* Silence-as-acceptance signals
* Realistic convergence dynamics

Eval will infer the rebuttal graph post-hoc.

---

# 6️⃣ Explicit Referencing of Opposing Claims

When responding to another agent’s argument, agents must:

* Identify the agent
* Identify the claim being addressed

This is critical for:

* Influence graph extraction
* Sycophancy detection
* CRIT analysis
* T³ claim alignment

We are not forcing structured JSON rebuttal slots.
But we are requiring explicit naming in natural language.

---

# 7️⃣ No Mid-Debate Intervention (Current Phase)

For this phase:

* No RCA gating
* No CRIT injection
* No PID controller
* No forced retries
* No mid-round modifications

The debate runs uninterrupted.

Evaluation happens afterward.

---

# 8️⃣ Single Final Position Per Turn

Each turn should:

* Maintain internal consistency
* Commit to a clear stance
* Not oscillate within the same turn

We assume:

> A turn represents a coherent position update.

Eval will penalize stance shifts inside a single turn.

---

# 9️⃣ No Structural Enforcement of Convergence

We do not assume:

* Debate must converge
* Agents must agree
* A final consensus is required

Debate may:

* Converge
* Polarize
* Stabilize
* Oscillate

That behavior is part of what we are studying.

---

# 🔟 Clean Separation Between Debate and Evaluation

Debate’s job:

* Produce structured, high-quality natural language reasoning.
* Respect the round structure.
* Respect one response per agent per round.
* Maintain clarity and explicit engagement.

Evaluation’s job:

* Extract structure.
* Detect inconsistencies.
* Score reasoning.
* Analyze convergence.
* Run CRIT / RCA / T³.

Debate does not need to think about scoring metrics.

Eval does not interfere with generation (in this phase).

---

# 🧭 The Shared Picture

If we visualize the debate:

```id="kq9h3t"
Round 1:
    Agent A
    Agent B
    Agent C

Round 2:
    Agent B
    Agent C
    Agent A

Round 3:
    Agent C
    Agent A
    Agent B
```

Each agent:

* Sees full transcript.
* Produces one response.
* May address any subset of prior claims.
* Commits to a position.
* Does not retry.
* Is not corrected mid-stream.

That is the debate.

---

# 📌 Why These Assumptions Matter

These assumptions ensure:

* Clean extraction of argument structure
* Fair speaking influence
* Stable convergence measurement
* Reliable CRIT signal
* Meaningful T³ rung detection
* Reproducible experimentation

If debate structure deviates from this mental model, evaluation results become confounded.

---

# ✅ Summary

The debate we are assuming is:

* Round-based
* Synchronous
* Deterministically ordered
* Full-transcript visible
* One response per agent per round
* Emergent interaction structure
* Explicit claim referencing
* No in-loop correction (for now)



---

# 2. Design Goals

This spec is designed to support:

* Post-hoc evaluation (CRIT, RCA, T³)
* In-loop controller experiments (PID-style)
* Retry logging
* Intervention tracking
* Agent dropout / partial rounds
* Prompt regime ablations
* Sycophancy / convergence analysis
* Future RAudit-style metrics


# 🔒 Current Phase: Post-Hoc Evaluation Only (No In-Loop Control)

Before anything else:

> **We are not running mid-debate intervention right now.**

Yes, the schema supports:

* Retries
* PID control
* RCA gating
* Intervention logging
* RAudit integration
* Attempt-level control state
* Dropouts
* Controller snapshots

But for this phase:

> **All evaluation is post-hoc.**

That means:

* No mid-debate corrections
* No controller-triggered retries
* No intervention injections
* No PID state tracking
* No RCA blocking
* No dynamic debate modification

The infrastructure exists for future experiments, but we are not using it.

---

# ✅ What the Debate Team Actually Needs to Do (Right Now)

For post-hoc mode, your responsibilities are minimal and clean.

You must produce a Transcript JSON object with:

### **Mandatory Top-Level Fields**

* `schema_version = "2.0.0"`
* `debate_id`
* `run_id`
* `mode = "posthoc"`
* `created_at`
* `run_metadata`
* `debate_metadata`
* `participants`
* `rounds`

That’s it at the top level.

You may omit:

* `intervention_log`
* `participant_state_log`
* `raud_it`
* `integrity`

---

# 🧱 What Must Be Present Inside `rounds`

Each round must contain:

* `round_index`
* `turns[]`

Each turn must contain:

* `turn_id`
* `round_index`
* `turn_index_in_round`
* `speaker_id`
* `attempts[]`

Each `attempt` must contain:

* `attempt_index`
* `timestamp`
* `status`
* `content`

---

# 🚨 Extremely Important Rule for Post-Hoc Mode

Because we are in `"posthoc"` mode:

> **Each turn must have exactly one attempt.**

That means:

```json
"attempts": [
  {
    "attempt_index": 0,
    "timestamp": "...",
    "status": "ok",
    "content": "..."
  }
]
```

No retries.
No `"retry"` status.
No `"failed"` status.
No `control_state`.
No `interventions_applied`.

**One attempt per turn. Always.**

---

# 🧠 What You Can Ignore (For Now)

You can safely ignore:

* `intervention_log`
* `control_state`
* `diagnostics`
* `interventions_applied`
* `participant_state_log`
* `recommendation_hint`
* `final_answer`
* `reasoning_trace`
* PID parameters
* Controller snapshots
* RAudit containers
* Retry semantics
* Mode switching
* Dropout handling
* Round status flags (unless you want to log them)

None of that is required for post-hoc evaluation.

---

# 🎯 What You Should Focus On

Focus on:

* Producing high-quality natural language debate responses
* Correct round structure
* Correct speaker ordering
* Accurate metadata
* Clean, single-attempt turns
* Proper agent identification
* Explicit referencing of opposing claims (as specified in prompting docs)

That’s it.

---

# 🧪 Why This Is Enough

For post-hoc evaluation:

* Eval will extract reasoning structure from `attempt.content`
* Eval will transform transcript into canonical schema
* Eval will run CRIT, RCA, T³ afterward
* Eval will not modify your debate during generation

So from your perspective:

> You are just generating a structured transcript of a clean debate.

Everything else happens later.

---

# 📌 Mental Model

Think of it this way:

Right now, the Transcript JSON is just:

> A structured wrapper around natural language debate turns.

Not a control system.
Not a PID experiment.
Not a retry engine.

Just a structured transcript.

---

# 🔮 Why the Extra Infrastructure Exists

The schema supports:

* In-loop RCA gating
* PID stabilization
* Sycophancy experiments
* Multi-attempt retries
* Human overrides
* RAudit-style metrics

But those are future extensions.

We designed the schema to avoid needing a breaking revision later.

That does **not** mean you need to implement those features now.

---

# 🟢 Summary for the Debate Team

For this phase:

* **Set `mode = "posthoc"`**
* **One attempt per turn**
* **No interventions**
* **No retries**
* **No controller state**
* **Focus on debate quality**

# 3. Major Design Decisions & Tradeoffs

## A. JSON Instead of Markdown

We chose JSON for:

* Machine stability
* Direct transformability into eval schema
* Easier validation
* Cleaner retry/intervention modeling
* Single unified object (no cross-linking artifacts)

Human readability is handled via:

* A renderer script (JSON → Markdown)
* Notebook viewers

Tradeoff:

* Slightly less ergonomic to read raw
* Dramatically simpler for tooling

---

## B. One Debate = One File

Pros:

* Clean experiment unit
* No cross-file ID resolution
* Easy archival
* Easy S3/zip sharing
* Easy integrity checking

---

## C. Explicit Attempt Logging

We deliberately log:

* All retry attempts
* Failed attempts
* Controller overrides

This makes:

* RCA reproducible
* PID experiments analyzable
* Intervention effects measurable

Tradeoff:

* Larger file size
* But complete reproducibility

---

## D. Emergent Interaction Topology

We do NOT enforce:

* Fully connected rebuttals
* One-target-per-turn

Instead:

* We allow emergent topology
* We optionally log `addresses[]` edges
* Eval infers debate graph post-hoc

This preserves ecological validity.

---

## E. Separation of Schema vs Runtime Validation

JSON Schema enforces:

* Structure
* Required fields
* Types
* Enums

It cannot enforce:

* Sequential attempt indices
* Exactly one accepted attempt
* Retry semantics consistency
* Intervention reference integrity
* Round completeness rules
* Mode-specific constraints

Therefore:

> A Python runtime validation script is mandatory.

Schema + Validator together form the full contract.

They complement each other.

---

# 4. Role of the Transcript Validator

The validation script enforces:

### Structural Integrity

* Unique `turn_id`
* Unique `intervention_id`
* Valid speaker references
* Valid round indices

### Attempt Semantics

* attempt_index must be sequential (0..n)
* Exactly one final attempt per turn
* final_attempt_index valid
* Retry_count consistent with attempts

### Mode Rules

* posthoc mode → no retries allowed
* in_loop mode → retries allowed
* intervention presence consistent with mode

### Intervention Consistency

* intervention references valid turn/round
* intervention ordering monotonic
* interventions_applied reference valid intervention_id

### Round Consistency

* round_index strictly increasing
* turn.round_index matches parent round
* No duplicate speaker in same round (if round complete)

### Optional Checks

* Monotonic timestamps
* Controller state sanity
* Partial round logic

---

### Recommendation

Yes — the Debate Team should run the validator before handing artifacts to Eval.

This:

* Prevents downstream pipeline failure
* Reduces debugging time
* Ensures contract compliance

---

# 5. Top-Level Fields

---

## `schema_version`

Purpose:

* Locks contract version
* Prevents silent incompatibilities

Eval usage:

* Ensures compatible transformer logic

---

## `debate_id`

Stable experiment identifier.

Used for:

* Aggregation across reruns
* Cross-ablation comparisons

---

## `run_id`

Unique identifier for this run.

Used for:

* Exact reproduction
* Logging
* Archival

---

## `mode`

Values:

* `"posthoc"`
* `"in_loop"`

Controls:

* Retry allowance
* Intervention expectations

Eval uses this to:

* Gate controller analysis
* Interpret retry behavior

---

## `created_at`

ISO timestamp.

Used for:

* Logging
* Ordering
* Artifact integrity

---

# 6. `run_metadata`

Run-level generation configuration.

Contains:

* model name
* temperature
* seed
* prompt regime
* generation mode
* prompt bundle version

Eval uses this for:

* Ablation studies
* Hyperparameter analysis
* Reproducibility

---

# 7. `debate_metadata`

Task-level metadata.

Examples:

* task_type
* scenario_id
* asset_symbol
* market_context

Used by Eval to:

* Route to correct scoring rubric
* Apply correct T³ rung
* Compare across scenarios

---

# 8. `participants`

Defines all expected agents.

Used for:

* Round completeness checks
* Influence analysis
* Convergence metrics

---

# 9. `participant_state_log`

Optional.

Tracks:

* Agent dropped
* Timeout
* Restart

Critical for:

* Robust experiment logging
* Avoiding silent missing-turn bugs

---

# 10. `rounds`

Core debate container.

Each round contains:

* round_index
* speaking order (optional)
* visibility regime
* turns

Supports:

* Deterministic rotation
* Transcript visibility ablations
* Partial round modeling

---

# 11. `turn`

Represents one agent’s contribution in one round.

Contains:

* turn_id
* round_index
* speaker_id
* attempts[]
* turn_type
* addresses[]
* metadata

Eval uses:

* content → CRIT / RCA input
* addresses[] → debate graph
* attempts[] → retry analysis
* control_state → PID analysis

---

# 12. `attempt`

Represents one model generation attempt.

Includes:

* attempt_index
* timestamp
* status: ok | retry | failed
* content (raw text)
* optional reasoning_trace
* control_state
* diagnostics

Critical for:

* RCA blind verification
* PID stability studies
* Failure analysis

---

# 13. `intervention_log`

First-class mid-debate intervention tracking.

Logs:

* trigger
* instruction_sent
* target scope
* controller_snapshot

Supports:

* RCA gating
* CRIT injection
* PID escalation
* Human override logging

---

# 14. `raud_it`

Optional run-level container.

Reserved for:

* RAudit-style metrics
* Future process-level evaluation

Not required by Debate Team.

---

# 15. `integrity`

Optional checksum block.

Useful for:

* Large artifact sharing
* S3 transfers
* Corruption detection

---

# 16. How Eval Uses This

Eval will:

1. Validate transcript (schema + runtime script)
2. Transform turns into canonical eval schema
3. Extract:

   * recommendations
   * reasoning traces
   * rebuttal structure
4. Run:

   * CRIT
   * RCA
   * T³
   * PID metrics
   * RAudit metrics

The Debate Team does not need to produce canonical eval JSON.

Eval owns transformation.

---

# 17. Stability Commitment

Transcript JSON Spec v2.0 is intended to be:

* Forward extensible
* Backward compatible via versioning
* Stable for the remainder of the project

Changes should increment:

* Minor version for additive fields
* Major version for breaking changes

---

# 18. Summary

This artifact:

* Preserves raw debate behavior
* Logs retries and failures
* Supports mid-debate control
* Enables research-grade evaluation
* Separates structure from scoring
* Is hardened via runtime validation
