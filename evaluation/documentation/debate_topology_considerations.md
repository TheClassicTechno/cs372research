# Debate Interaction Protocol — Current Consensus

## Purpose

This document summarizes the agreed-upon structure for agent interaction in the multi-agent debate system. The goal is to balance ecological realism with evaluability, without introducing artificial topology that contaminates reasoning measurements.

---

# 1. Core Engineering Assumption

**One API call per agent per round.**

* Each agent receives the full transcript up to that point.
* Each agent produces a single holistic response.
* We do *not* make separate calls per opponent.
* We do *not* use pairwise cross-examination calls (no n(n−1) scaling).

This keeps runtime and cost linear in the number of agents.

---

# 2. Interaction Topology Assumption

Agents are **not required** to respond to:

* All other agents
* Exactly one specific agent
* Any fixed set of opponents

Instead:

> Agents may respond to any opposing arguments they judge materially relevant to their recommendation.

We include a soft constraint in the prompt:

> “Address any opposing arguments that materially affect your recommendation. If you choose not to engage a claim, your silence may be interpreted as implicit acceptance.”

This creates engagement pressure without enforcing structural symmetry.

---

# 3. Explicit Referencing Requirement (New Clarification)

To improve post-hoc evaluability without forcing connectivity, we now also require:

> When engaging an opposing view, explicitly identify the agent and the specific claim being addressed.

This does **not** require agents to respond to all opponents.
It simply ensures that when they *do* respond, the interaction edge is traceable.

### Rationale

* Improves interpretability of influence dynamics
* Allows cleaner post-hoc reconstruction of the debate graph
* Strengthens CRIT rival-reason identification
* Improves sycophancy and convergence measurement

This preserves emergent topology while reducing ambiguity in interaction mapping.

---

# 4. What We Explicitly Chose NOT To Do

We are not enforcing:

* Fully connected rebuttal graphs (each agent must respond to every other agent each round).
* Exactly-one-target reply constraints.
* Fixed per-opponent response slots in natural debate mode.

These options were considered but rejected.

---

# 5. Tradeoffs Considered

## Option: Fully Connected Per Round

Pros:

* Cleaner rebuttal graph
* Easier CRIT extraction
* Stronger guaranteed rival exposure

Cons:

* Artificial interaction density
* Suppresses natural dominance patterns
* Reduces ecological validity
* Partially “solves” the debate structure for agents

Conclusion: Too structurally rigid for our research goals.

---

## Option: Exactly One Target Per Round

Pros:

* Predictable sparse topology

Cons:

* Artificial sparsity
* Order-dependent dynamics
* Biased interaction graph
* Reduced cross-pressure

Conclusion: Worse than both full connectivity and free response.

---

## Chosen Design: Emergent Topology with Explicit Referencing

Pros:

* Natural debate dynamics
* Allows dominance and ignoring to emerge
* Preserves groupthink and sycophancy signals
* Improves traceability of interaction edges
* Ecologically realistic

Cons:

* Still requires post-hoc graph inference
* Less deterministic than forced symmetry

Conclusion: Best alignment with research goals and measurement needs.

---

# 6. Why This Matches Our Research Questions

We are studying:

* Groupthink
* Sycophancy
* Convergence dynamics
* Entropy collapse
* Trace–output consistency
* Control interventions (CRIT, RCA, PID)

Artificially enforcing rebuttal symmetry would distort:

* Natural convergence behavior
* Silence-as-acceptance signals
* Dominance and marginalization patterns

At the same time, requiring explicit referencing improves our ability to:

* Detect influence propagation
* Identify ignored claims
* Measure directed rebuttal density
* Reconstruct debate interaction graphs

Thus, we preserve ecological validity while improving evaluability.

---

# 7. Summary Statement

We use one holistic response per agent per round, encourage engagement with materially relevant opposing claims, require explicit identification of referenced agents when responding, and allow the debate graph to emerge naturally rather than forcing symmetric rebuttal structure.
