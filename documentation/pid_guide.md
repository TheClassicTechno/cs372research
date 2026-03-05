# PID Controller for Multi-Agent Debate — User's Guide

## What is this?

The PID controller is an in-loop feedback system that automatically regulates debate quality during multi-agent trading discussions. It watches how well the agents are reasoning, and adjusts their behavior between rounds to prevent two failure modes:

1. **Groupthink / sycophancy** — agents converge too quickly, agreeing without actually reasoning well.
2. **Low reasoning quality** — agents produce internally inconsistent, unsupported, or causally sloppy arguments.

The name comes from control theory: **P**roportional-**I**ntegral-**D**erivative. It's the same family of controller used in thermostats, cruise control, and industrial process regulation — adapted here for LLM debate quality.

---

## How it works (the 30-second version)

Each debate round:

1. Agents produce proposals, critiques, and revisions (the normal debate flow).
2. **CRIT**, a blind reasoning auditor, scores **each agent individually** across four pillars of reasoning quality (`ρ_i`), then averages into `ρ̄ = 1/n Σ ρ_i`.
3. The **PID controller** compares `ρ̄` against a target and computes a correction signal.
4. That correction adjusts `agreeableness` — the dial that controls how confrontational vs. deferential agents are in the next round.

High reasoning quality → ease off, let agents converge naturally.
Low reasoning quality → push harder, make agents more critical.

---

## Code walkthrough

This section traces exactly what happens in the code when PID is enabled, so you can follow along file-by-file.

### File map

**Configuration layer** — how PID gets turned on:

| File | What it does |
|------|-------------|
| `config/debate_memo_pid.yaml` | YAML config where `pid_enabled: true` and gain values live |
| `models/config.py` → `AgentConfig` | Pydantic model that receives YAML fields (`pid_kp`, `pid_ki`, etc.) |
| `agents/multi_agent_debate.py` → `DebateAgentSystem.__init__` | Adapter that passes `AgentConfig` PID fields into `DebateConfig` |
| `multi_agent/config.py` → `DebateConfig` | Dataclass that constructs a `PIDConfig` object in `__post_init__` |

**Debate runner** — the main execution loop:

| File | What it does |
|------|-------------|
| `multi_agent/runner.py` → `MultiAgentRunner.__init__` | Creates `PIDController`, `CritScorer`, and per-phase sub-graphs |
| `multi_agent/runner.py` → `_run_round_with_pid()` | Runs propose/critique/revise as 3 separate sub-graph invocations, setting `_current_beta` between each |
| `multi_agent/runner.py` → `_pid_phase_step()` | Calls CRIT scorer, computes divergence, runs PID controller, records events |

**CRIT scorer** — the blind reasoning auditor:

| File | What it does |
|------|-------------|
| `eval/crit/scorer.py` → `CritScorer.score()` | Iterates over agents, makes one LLM call per agent, aggregates scores |
| `eval/crit/prompts.py` → `CRIT_SYSTEM_PROMPT` | System prompt telling the LLM how to evaluate reasoning quality |
| `eval/crit/prompts.py` → `build_crit_single_agent_prompt()` | Builds per-agent user prompt with that agent's traces and decision |
| `eval/crit/schema.py` | `CritResult`, `RoundCritResult`, `PillarScores`, `Diagnostics` data models |

**PID controller** — the feedback math:

| File | What it does |
|------|-------------|
| `eval/PID/types.py` | `PIDGains`, `PIDConfig`, `PIDState`, `PIDStepResult` dataclasses |
| `eval/PID/controller.py` → `PIDController.step()` | Runs one full PID iteration: sycophancy → error → PID output → beta update |
| `eval/PID/controller.py` → `compute_error()` | `e_t = (rho_star - rho_bar) + mu * s_t` |
| `eval/PID/controller.py` → `compute_pid_output()` | `u_t = Kp*e_t + Ki*integral + Kd*(e_t - e_prev)` |
| `eval/PID/beta_dynamics.py` → `update_beta_clipped()` | `beta_new = clip(gamma_beta * beta + u_t, 0, 1)` |
| `eval/PID/sycophancy.py` → `compute_sycophancy_signal()` | Detects fake convergence: JS drops but evidence overlap also drops |
| `eval/PID/stability.py` → `validate_gains()` | Checks that gains won't cause instability before the debate starts |
| `eval/PID/termination.py` → `check_convergence()` | Returns true if JS divergence drops below epsilon (early stop) |

**Prompts that PID controls** — the text that changes as beta changes:

| File | What it does |
|------|-------------|
| `multi_agent/prompts/tone/critique_adversarial.txt` | Injected when β ≥ 0.67: push agents to challenge assumptions |
| `multi_agent/prompts/tone/critique_balanced.txt` | Injected when 0.33 ≤ β < 0.67: balanced evaluation |
| `multi_agent/prompts/tone/critique_collaborative.txt` | Injected when β < 0.33: find common ground |
| `multi_agent/prompts/tone/revise_adversarial.txt` | Same thresholds, for revise phase |
| `multi_agent/prompts/tone/revise_balanced.txt` | Same thresholds, for revise phase |
| `multi_agent/prompts/tone/revise_collaborative.txt` | Same thresholds, for revise phase |
| `multi_agent/prompts/registry.py` → `PromptRegistry.build()` | Assembles system prompt from role + preamble + tone blocks |
| `multi_agent/prompts/registry.py` → `resolve_beta()` | Unifies PID β and static agreeableness: `β = 1.0 - agreeableness` |

**Data models for PID events** — what gets stored in the trace:

| File | What it does |
|------|-------------|
| `multi_agent/models.py` → `PIDEvent` | One event per round: metrics, CRIT result, PID step details, controller output |
| `multi_agent/models.py` → `RoundMetrics` | `rho_bar`, `js_divergence`, `ov_overlap` for one round |
| `multi_agent/models.py` → `ControllerOutput` | `new_agreeableness` value to use in the next round |
| `multi_agent/models.py` → `AgentTrace.pid_events` | Optional list of `PIDEvent` on the trace (null when PID disabled) |

### Step-by-step: what happens during one PID-enabled debate round

Here's the exact sequence of function calls for round `t` of a PID-enabled debate:

**Step 1 — Config flows in.**
`config/debate.yaml` sets `pid_enabled: true` with gain values. `SimulationConfig.from_yaml()` parses this into `AgentConfig`. The adapter (`DebateAgentSystem.__init__`) passes the fields to `DebateConfig`, whose `__post_init__` constructs a `PIDConfig` object from the flat YAML fields.

**Step 2 — Runner initializes PID components.**
`MultiAgentRunner.__init__` sees `config.pid_enabled == True` and:
- Compiles 3 per-phase sub-graphs (propose, critique, revise) — one per debate phase, so agreeableness can differ between them
- Creates a `CritScorer` instance with the debate LLM as its backend
- Creates a `PIDController(config, initial_beta)` with initial state
- Calls `validate_gains()` to ensure the gains won't cause instability

**Step 3 — Round starts: `_run_round_with_pid(state, round_num)`.**
The runner reads the current `beta` from the controller's state:
```
beta = self._pid_controller.beta
```

**Step 4 — Propose phase.**
The runner checks `config.pid_propose`. If true, sets `state["config"]["agreeableness"] = beta`. If false, uses the original static agreeableness. Sets `state["config"]["_current_beta"] = None` (no tone injection for proposals). Then invokes `self._propose_graph.invoke(state)`.

Inside the propose graph, each agent gets a system prompt assembled by `PromptRegistry.build(role, "propose", beta=None)` — which includes the role identity prompt (e.g., `roles/macro_slim.txt`) and optionally the causal contract, but no tone block. The user prompt is built by `build_proposal_user_prompt()`. Proposals aren't affected by tone (which is why `pid_propose` defaults to false).

**Step 5 — Critique phase.**
Same toggle check with `config.pid_critique` (defaults to true, so beta IS used). Sets `state["config"]["_current_beta"] = beta`. Invokes `self._critique_graph.invoke(state)`.

Inside the critique graph, `resolve_beta(config, "critique")` returns the current PID β, and `PromptRegistry.build()` maps it to a tone bucket via `beta_to_bucket()`:
- β ≥ 0.67 → adversarial (`tone/critique_adversarial.txt`)
- 0.33 ≤ β < 0.67 → balanced (`tone/critique_balanced.txt`)
- β < 0.33 → collaborative (`tone/critique_collaborative.txt`)

The tone text is placed LAST in the system prompt for maximum LLM attention (recency bias).

**Step 6 — Revise phase.**
Same toggle check with `config.pid_revise` (defaults to true). Invokes `self._revise_graph.invoke(state)`. The same β → tone mapping applies, using `tone/revise_*.txt` files.

**Step 7 — CRIT scores the round: `_pid_phase_step(state, round_num, phase)`.**
The runner calls `self._crit_scorer.score()` with:
- `case_data`: the enriched market context (what agents saw)
- `agent_traces`: all debate turns from `state["debate_turns"]`
- `decisions`: the revisions (or proposals if no revisions yet)

`CritScorer.score()` iterates over each agent role, makes a separate LLM call per agent using `CRIT_SYSTEM_PROMPT` + `build_crit_single_agent_prompt()`, and aggregates into `rho_bar = mean(all agent rho_i)`.

**Step 8 — Divergence signals.**
The runner computes Jensen-Shannon divergence from agent confidence values:
```
js = jensen_shannon_divergence(confidences)
```
This measures how much agents disagree — it drops as agents converge.

**Step 9 — PID controller step: `self._pid_controller.step(rho_bar, js, ov)`.**
Inside `PIDController.step()`, the following happens in order:

1. **Sycophancy check**: `compute_sycophancy_signal()` — fires (`s_t = 1`) if JS dropped sharply AND evidence overlap also dropped (agents agreeing without sharing evidence = groupthink).
2. **Error**: `compute_error()` — `e_t = (rho_star - rho_bar) + mu * s_t`. Positive = below target.
3. **Integral accumulation**: `integral += e_t` (running sum for I-term).
4. **PID output**: `compute_pid_output()` — `u_t = Kp*e_t + Ki*integral + Kd*(e_t - e_prev)`.
5. **Beta update**: `update_beta_clipped()` — `beta_new = clip(gamma_beta * beta + u_t, 0, 1)`.
6. **State advance**: saves `e_prev = e_t`, `beta = beta_new`, increments round counter.

**Step 10 — Event recording.**
The runner packages everything into a `PIDEvent` and appends it to `self._pid_events`. This list ends up in `AgentTrace.pid_events` in the final output.

**Step 11 — Feedback loop.**
On the next round, Step 3 reads `self._pid_controller.beta` — which is now the updated `beta_new` from Step 9. The cycle repeats.

**Step 12 — Termination check.**
`should_terminate()` calls `check_convergence(js, epsilon)`. If JS divergence dropped below epsilon (default 0.01), the debate stops early — agents have genuinely converged.

### The CRIT prompts

CRIT is the LLM-based reasoning auditor. It makes one LLM call per agent per round. Here's what those calls look like:

**System prompt** (`eval/crit/prompts.py:CRIT_SYSTEM_PROMPT`):
Tells the LLM to evaluate reasoning quality across 4 pillars, score each 0.0–1.0, and return JSON. The LLM is explicitly told it must NOT evaluate whether the agent's prediction was correct — only whether the reasoning is internally sound.

**User prompt** (`eval/crit/prompts.py:build_crit_single_agent_prompt()`):
Built per-agent with these sections:
- `## Case Context` — what the agents saw (market data, news, portfolio)
- `## Agent Under Evaluation: {ROLE}` — identifies which agent
- `### Reasoning Trace` — that agent's proposal, critiques, and revision from the round
- `### Trading Decision` — the agent's final action (orders, confidence, justification)
- `## Instructions` — evaluate the 4 pillars, return JSON

**CRIT output format** (what the LLM returns):
```json
{
  "pillar_scores": {
    "logical_validity": 0.85,
    "evidential_support": 0.75,
    "alternative_consideration": 0.90,
    "causal_alignment": 0.70
  },
  "diagnostics": {
    "contradictions_detected": false,
    "unsupported_claims_detected": false,
    "ignored_critiques_detected": false,
    "premature_certainty_detected": false,
    "causal_overreach_detected": false,
    "conclusion_drift_detected": false
  },
  "explanations": {
    "logical_validity": "Reasoning logically supports the conclusion...",
    "evidential_support": "Most claims cite case data...",
    "alternative_consideration": "Competing explanations considered...",
    "causal_alignment": "L2 claims are properly scoped..."
  }
}
```

The per-agent `rho_i` is the mean of the 4 pillar scores. The round-level `rho_bar` is the mean of all agents' `rho_i`. This `rho_bar` feeds the PID controller's error signal.

### How β changes the prompts

All debate phases route through `PromptRegistry.build()`, which assembles system prompts from composable blocks in this order:

1. **Causal contract** (optional) — shared reasoning rules from `system_contract/system_causal_contract.txt`
2. **Role identity** — agent expertise from `roles/{role}.txt` or `roles/{role}_slim.txt`
3. **Phase preamble** — brief instruction (e.g., "Provide explicit, substantive critiques.")
4. **Tone** (critique/revise only) — β-driven modifier from `tone/{phase}_{bucket}.txt`

Tone is placed LAST in the system prompt for maximum LLM attention (recency bias in transformer attention patterns).

When PID adjusts β, it changes which tone file gets injected into critique and revise system messages:

| β range | Bucket | Tone file | Effect |
|---------|--------|-----------|--------|
| ≥ 0.67 | adversarial | `tone/critique_adversarial.txt` | Push agents to challenge assumptions, explore alternatives |
| 0.33–0.67 | balanced | `tone/critique_balanced.txt` | Evaluate on merits, balanced critique |
| < 0.33 | collaborative | `tone/critique_collaborative.txt` | Find common ground, ease off, converge |

The same mapping applies for revise-phase tone files (`tone/revise_*.txt`).

**β ↔ agreeableness conversion**: The legacy `agreeableness` knob (0 = confrontational, 1 = agreeable) has inverted semantics from β (RAudit-aligned: 0 = collaborative, 1 = adversarial). `resolve_beta()` in `registry.py` handles unification:
- When PID is active: uses `_current_beta` directly (set by the runner)
- When PID is off: derives β from static agreeableness via `β = 1.0 - agreeableness`
- For propose/judge phases: always returns `None` (no tone injection)

So when CRIT reports low reasoning quality → PID increases β → agents get the adversarial tone → they push back harder → reasoning quality (hopefully) improves → CRIT reports higher quality → PID decreases β → collaborative tone.

### Key data structures

| Type | File | What it holds |
|------|------|--------------|
| `PIDGains` | `eval/PID/types.py` | `Kp`, `Ki`, `Kd` — the three tuning knobs |
| `PIDConfig` | `eval/PID/types.py` | Gains + `rho_star`, `gamma_beta`, `mu`, `delta_s`, `T_max`, `epsilon` |
| `PIDState` | `eval/PID/types.py` | Mutable state: `beta`, `integral`, `e_prev`, `t`, JS/OV history |
| `PIDStepResult` | `eval/PID/types.py` | One step's output: `e_t`, `u_t`, `beta_new`, P/I/D terms, `s_t` |
| `CritResult` | `eval/crit/schema.py` | Per-agent: pillar scores, diagnostics, explanations, `rho_bar` |
| `RoundCritResult` | `eval/crit/schema.py` | All agents: `agent_scores` dict + aggregated `rho_bar` |
| `PIDEvent` | `multi_agent/models.py` | Full event: round metrics + CRIT result + PID step + controller output |
| `AgentTrace` | `multi_agent/models.py` | Debate trace with optional `pid_events: list[PIDEvent]` |

---

## Architecture

```
                            Debate Round t
                            ══════════════

    beta_new from                           Each phase is a separate
    previous round                          sub-graph invocation so
          │                                 agreeableness can differ
          ▼                                 between them.

    ┌─────────────┐   pid_propose toggle
    │   Propose   │   ──────────────────▶  _current_beta = None (no tone)
    │ (all agents)│
    └──────┬──────┘
           ▼
    ┌─────────────┐   pid_critique toggle
    │  Critique   │   ──────────────────▶  if true:  _current_beta = β
    │ (all agents)│                        if false: _current_beta = None
    └──────┬──────┘
           ▼
    ┌─────────────┐   pid_revise toggle
    │   Revise    │   ──────────────────▶  if true:  _current_beta = β
    │ (all agents)│                        if false: _current_beta = None
    └──────┬──────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │                     CRIT Scorer                          │
    │              (per-agent, blind audit)                     │
    │                                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
    │  │  MACRO   │  │  VALUE   │  │   RISK   │  │TECHNICAL│ │
    │  │ IC ES TA │  │ IC ES TA │  │ IC ES TA │  │IC ES TA │ │
    │  │ CI → ρ_m │  │ CI → ρ_v │  │ CI → ρ_r │  │CI → ρ_t│ │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬─────┘ │
    │       │              │              │             │       │
    │       └──────┬───────┴──────┬───────┘             │       │
    │              │              │                      │       │
    │              ▼              ▼                      │       │
    │        ρ̄ = (ρ_m + ρ_v + ρ_r + ρ_t) / 4 ◀────────┘       │
    └────────────────────────┬─────────────────────────────────┘
                             │ ρ̄
                             ▼
    ┌──────────────────────────────────────────┐
    │              PID Controller               │
    │                                          │
    │  e_t = (ρ* − ρ̄) + μ · s_t              │
    │  u_t = Kp·e_t + Ki·∫e + Kd·Δe           │
    │  β_new = clip(γ·β + u_t,  0,  1)        │
    └────────────────────┬─────────────────────┘
                         │ β_new
                         ▼
    Feed into round t+1
    (each phase reads its toggle
     to decide whether to use
     β_new or original)
```

### Per-phase decomposition

Without PID, each debate round runs as a single graph invocation (propose → critique → revise atomically). With PID enabled, each round is decomposed into **three separate sub-graph invocations**. Between each invocation, the runner sets `_current_beta` based on that phase's toggle:

| Phase     | Config toggle   | Default | Why |
|-----------|----------------|---------|-----|
| Propose   | `pid_propose`  | `false` | Proposals: no tone injection (β always None) |
| Critique  | `pid_critique` | `true`  | Critique tone is most sensitive to β |
| Revise    | `pid_revise`   | `true`  | Revision deference is also sensitive to β |

When a phase's toggle is `true`, PID's computed β is set as `_current_beta` on the config, and `resolve_beta()` returns it for tone bucket selection. When `false`, `_current_beta` is set to `None` and `resolve_beta()` falls back to deriving β from the static agreeableness knob. All agents within a phase share the same β — the per-phase granularity is at the phase level, not the individual agent level.

---

## CRIT: The Blind Reasoning Auditor

CRIT (Critique of Reasoning Integrity and Traceability) is an LLM-based scorer that evaluates reasoning quality **without access to ground truth**. It never sees market outcomes, impact scores, or whether the agents' predictions were correct. It evaluates only logical integrity.

### Per-agent scoring (RAudit Section 3.3)

Per the RAudit paper, CRIT scores **each agent individually**. In a 4-agent debate, CRIT makes 4 separate LLM calls — one per agent. Each call receives only that agent's reasoning trace and decision (plus the shared case context). This ensures one agent's weak reasoning cannot inflate another's score.

```
Round t completes → CRIT evaluates each agent:

   MACRO  agent → ρ_macro  = mean(IC, ES, TA, CI)
   VALUE  agent → ρ_value  = mean(IC, ES, TA, CI)
   RISK   agent → ρ_risk   = mean(IC, ES, TA, CI)
   TECHNICAL    → ρ_tech   = mean(IC, ES, TA, CI)

   ρ̄ = (ρ_macro + ρ_value + ρ_risk + ρ_tech) / 4  →  PID controller
```

The aggregate `ρ̄` feeds the PID controller. Per-agent `ρ_i` scores are preserved in the `RoundCritResult` for diagnostics and logging.

### What is per-agent vs. global?

| Scope | What | Paper reference |
|-------|------|-----------------|
| **Per-agent** | CRIT scores (ρ_i), reasoning traces, belief distributions, evidence spans | Section 3.3, Algorithm 1 lines 7-8 |
| **Global** | β (agreeableness), ρ* (target), PID controller, sycophancy signal s_t | Section 3.5, Algorithm 1 line 5 |

All agents share the same β within a phase. The per-phase toggles (propose/critique/revise) control which phases use PID's β vs. the original agreeableness — but within a phase, all agents get the same value.

### Four pillars

| Pillar | What it measures | Failure mode |
|--------|-----------------|-------------|
| **Internal Consistency** | Are the agent's claims logically compatible with each other? | Contradictions within a single argument |
| **Evidence Support** | Are factual claims backed by cited evidence from the case context? | Unsupported assertions |
| **Trace Alignment** | Does the final trading decision follow from the reasoning presented? | Conclusion drifts from argument |
| **Causal Integrity** | Are causal claims properly scoped (Pearl L1/L2/L3)? | Causal overreach |

Each pillar is scored 0.0–1.0. The per-agent composite `ρ_i` is the mean of all four pillars. The round-level `ρ̄` is the mean of all per-agent `ρ_i` scores.

### Diagnostics

CRIT produces binary diagnostic flags **per agent**:
- `contradictions_detected`
- `unsupported_claims_detected`
- `conclusion_drift_detected`
- `causal_overreach_detected`

These are logged per-agent when PID metrics logging is enabled and appear in the trace output.

### Source files

```
eval/crit/
    __init__.py     # Public API: CritScorer, RoundCritResult, CritResult
    scorer.py       # CritScorer class (per-agent iteration + aggregation)
    prompts.py      # System prompt + single-agent user prompt template
    schema.py       # RoundCritResult, CritResult, PillarScores, Diagnostics
```

---

## The PID control loop

### Error signal (Eq 5)

```
e_t = (rho_star - rho_bar) + mu * s_t
```

- `rho_star` — target quality score (default 0.8)
- `rho_bar` — actual quality from CRIT this round (mean of per-agent ρ_i)
- `mu` — sycophancy penalty weight (default 1.0)
- `s_t` — binary sycophancy indicator (0 or 1)

Positive error = quality below target. Negative = quality exceeds target.

### PID output (Eq 6)

```
u_t = Kp * e_t  +  Ki * integral  +  Kd * (e_t - e_prev)
      ─────────    ──────────────    ─────────────────────
      P-term       I-term            D-term
```

| Term | What it does | Analogy |
|------|-------------|---------|
| **P** (Proportional) | Reacts to the error right now | "Quality is below target, push proportionally" |
| **I** (Integral) | Reacts to accumulated past errors | "We've been below target for 5 rounds, push harder" |
| **D** (Derivative) | Reacts to rate of change of error | "Error is improving, ease off to avoid overshoot" |

### Beta update (Eq 14)

```
beta_new = clip(gamma_beta * beta_old + u_t,  0,  1)
```

- `gamma_beta` — momentum factor (default 0.9). Higher = slower change.
- The clip keeps beta in [0, 1].

### Sycophancy detection (Eq 4)

The sycophancy detector fires (`s_t = 1`) when **both** of these happen in one round:
- Agent disagreement (JS divergence) drops sharply (by more than `delta_s`)
- Evidence overlap also drops

This pattern means agents are agreeing on conclusions without agreeing on underlying reasoning — a red flag for groupthink. When detected, the error signal gets a penalty of `+mu`, forcing the controller to push agents toward more critical reasoning.

### Termination

The debate can terminate early when JS divergence drops below `epsilon` (default 0.01), indicating genuine convergence. Otherwise it runs for `max_rounds`.

### Stability validation

Before the debate starts, the runner validates that PID gains won't cause instability. The check (Eq 17) ensures the worst-case correction signal can't overwhelm the momentum term:

```
e_max * (Kp + T_max * Ki + 2 * Kd) < 1 / gamma_beta
```

If this fails, `GainInstabilityError` is raised at initialization — fix by lowering gains or reducing `T_max`.

### Source files

```
eval/PID/
    controller.py      # PIDController class, compute_error(), compute_pid_output()
    types.py           # PIDGains, PIDConfig, PIDState, PIDStepResult
    beta_dynamics.py   # update_beta_clipped(), update_beta_unclipped()
    sycophancy.py      # jensen_shannon_divergence(), evidence_overlap(), compute_sycophancy_signal()
    stability.py       # validate_gains(), check_stability(), check_non_oscillation()
    termination.py     # termination_bound(), check_convergence(), simulate_contraction()
```

---

## Configuration

### Enabling PID via YAML

In `config/debate_memo_pid.yaml`:

```yaml
agent:
  agent_system: "multi_agent_debate"
  llm_provider: "openai"
  llm_model: "gpt-4o-mini"
  temperature: 0.3

  # Enable PID controller
  pid_enabled: true

  # PID gains (how aggressively the controller reacts)
  pid_kp: 0.15          # Proportional gain
  pid_ki: 0.01          # Integral gain
  pid_kd: 0.03          # Derivative gain

  # Target quality score (0.0 - 1.0)
  pid_rho_star: 0.8

  # Starting beta for PID (0.0 - 1.0)
  pid_initial_beta: 0.5

  # Per-phase toggles (which phases use PID's beta)
  pid_propose: false     # Proposals: no tone injection
  pid_critique: true     # Critiques: β → tone bucket
  pid_revise: true       # Revisions: β → tone bucket

  # Logging
  pid_log_metrics: true
  log_prompt_manifest: true
```

### Enabling PID programmatically

```python
from eval.PID.types import PIDConfig, PIDGains
from multi_agent.config import DebateConfig
from multi_agent.runner import MultiAgentRunner

config = DebateConfig(
    pid_config=PIDConfig(
        gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
        rho_star=0.8,
    ),
    initial_beta=0.5,
    pid_propose=False,
    pid_critique=True,
    pid_revise=True,
    mock=True,
)

runner = MultiAgentRunner(config)
action, trace = runner.run(observation)

# PID events are in the trace
for event in trace.pid_events:
    print(f"Round {event.round_index}: rho_bar={event.metrics.rho_bar:.3f}, "
          f"beta_new={event.pid_step['beta_new']:.3f}")
```

### All PID parameters

| Parameter | YAML key | Default | Description |
|-----------|----------|---------|-------------|
| Enabled | `pid_enabled` | `false` | Master switch for PID |
| Kp | `pid_kp` | `0.15` | Proportional gain — reaction to current error |
| Ki | `pid_ki` | `0.01` | Integral gain — reaction to accumulated error |
| Kd | `pid_kd` | `0.03` | Derivative gain — reaction to error rate of change |
| rho_star | `pid_rho_star` | `0.8` | Target reasoning quality score |
| initial_beta | `pid_initial_beta` | `0.5` | Starting agreeableness value |
| Propose toggle | `pid_propose` | `false` | Whether PID controls propose phase |
| Critique toggle | `pid_critique` | `true` | Whether PID controls critique phase |
| Revise toggle | `pid_revise` | `true` | Whether PID controls revise phase |
| Metrics logging | `pid_log_metrics` | `false` | Log scalar PID stats each round |
| LLM logging | `pid_log_llm_calls` | `false` | Log full CRIT prompts/responses |
| Prompt manifest | `log_prompt_manifest` | `false` | Log prompt file names once per round |

Parameters not exposed in YAML (set via `PIDConfig` programmatically):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_beta` | `0.9` | Momentum factor for beta updates |
| `mu` | `1.0` | Sycophancy penalty weight |
| `delta_s` | `0.05` | JS drop threshold for sycophancy detection |
| `T_max` | `20` | Maximum rounds (used in stability check) |
| `epsilon` | `0.01` | Convergence tolerance for early termination |

---

## Logging

Three independent logging channels can be enabled for PID debugging:

### Scalar metrics (`pid_log_metrics: true`)

Logs per-phase structured JSON to the `pid.metrics` logger at INFO level. One entry per phase (propose, critique, revise) per round:

```json
{
  "type": "pid_phase",
  "debate_id": "...",
  "round": 1,
  "phase": "critique",
  "beta_in": 0.5,
  "tone_bucket": "balanced",
  "crit": {
    "rho_bar": 0.75,
    "agents": {
      "macro": { "rho_i": 0.80, "pillars": { "IC": 0.85, "ES": 0.75, "TA": 0.90, "CI": 0.70 } },
      "value": { "rho_i": 0.73, "pillars": { "IC": 0.80, "ES": 0.70, "TA": 0.80, "CI": 0.60 } }
    }
  },
  "pid": {
    "e_t": 0.05, "p_term": 0.0075, "i_term": 0.0005, "d_term": 0.0015,
    "u_t": 0.0095, "beta_old": 0.5, "beta_new": 0.4595,
    "quadrant": "healthy", "sycophancy": 0
  },
  "divergence": { "js": 0.001, "ov": 0.15 }
}
```

At the end of the debate, a `pid_summary` entry is emitted with config, outcome, and convergence info.

### LLM call logging (`pid_log_llm_calls: true`)

Logs full CRIT prompts and responses to the `pid.llm` logger at DEBUG level:

```
[CRIT LLM REQUEST]
===== SYSTEM PROMPT =====
You are a reasoning quality auditor (CRIT). Your job is to evaluate...
===== USER PROMPT =====
## Case Context
...
## Agent Arguments
...

[CRIT LLM RESPONSE]
{"pillar_scores": {"logical_validity": 0.8, ...}, ...}
```

### Prompt file manifest (`log_prompt_manifest: true`)

Logs a compact summary of all prompt files used, emitted **once per round** at the start (not per-agent). Uses the `debate.prompts` logger at INFO level:

```
========================================================================
  [Prompt Manifest] Round 1 | Snapshot: 2025Q1 (AAPL, NVDA, MSFT, TSLA, JPM, GOOG, AMZN, META, XOM, LLY)
------------------------------------------------------------------------
  System blocks: causal_contract → role_system → phase_preamble → tone
    causal_contract: system_contract/system_causal_contract.txt
    role_files: macro_slim.txt, value_slim.txt, risk_slim.txt, technical_slim.txt
    tone (β=0.50, balanced): critique_balanced.txt, revise_balanced.txt
  Phase templates: proposal_allocation.txt, critique_allocation.txt, revision_allocation.txt, judge_allocation.txt
========================================================================
```

This shows:
- **System block order** — the assembly order of system prompt blocks
- **Causal contract** — which shared reasoning contract file is used (if enabled)
- **Role files** — which role identity prompt each agent gets
- **Tone** — current β value, bucket, and which tone files are injected into critique/revise
- **Phase templates** — which Jinja2 template files build the user prompts
- **Snapshot identifier** — unique identifier for the quarterly memo (invest quarter + tickers)

The snapshot identifier replaces logging the full 80K+ char memo content, enabling traceability without log bloat.

Implementation: `build_prompt_manifest()` in `registry.py` resolves all file names (same logic as `PromptRegistry.build()` but without loading content). `_extract_snapshot_id()` in `llm.py` extracts quarter and tickers from state. The runner calls `_log_prompt_manifest()` at the top of each round iteration.

### Enabling logging

In `debate.yaml`:

```yaml
agent:
  pid_enabled: true
  pid_log_metrics: true
  pid_log_llm_calls: true
  log_prompt_manifest: true
```

To see the output, configure Python's logging system to show the relevant loggers. For example:

```python
import logging
logging.getLogger("pid.metrics").setLevel(logging.DEBUG)
logging.getLogger("pid.llm").setLevel(logging.DEBUG)
logging.getLogger("debate.prompts").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)
```

---

## Output artifacts

When PID is enabled, every output artifact includes PID events.

### In `episode_log.json`

```json
{
  "decision_point_logs": [{
    "agent_output": {
      "debate_trace": {
        "pid_events": [{
          "round_index": 1,
          "metrics": {
            "rho_bar": 0.75,
            "js_divergence": 0.0,
            "ov_overlap": 0.0
          },
          "crit_result": {
            "agent_scores": {
              "macro": {
                "pillar_scores": { "logical_validity": 0.85, ... },
                "diagnostics": { ... },
                "explanations": { ... },
                "rho_bar": 0.80
              },
              "value": { ... },
              "risk": { ... },
              "technical": { ... }
            },
            "rho_bar": 0.75
          },
          "pid_step": {
            "e_t": 0.05,
            "u_t": 0.0095,
            "beta_new": 0.4595,
            "p_term": 0.0075,
            "i_term": 0.0005,
            "d_term": 0.0015,
            "s_t": 0
          },
          "controller_output": {
            "new_agreeableness": 0.4595,
            "force_extra_round": false
          }
        }]
      }
    }
  }]
}
```

### In `reasoning/case_NNN.txt`

Same structure as above, serialized as JSON.

### In `simulation_log.json` and `summary.json`

PID events flow through the full pipeline and are present in all output files.

### When PID is disabled

All artifacts have `"pid_events": null`.

---

## What changed from the previous codebase

### New files

| File | Purpose |
|------|---------|
| `eval/crit/__init__.py` | Public API exports |
| `eval/crit/scorer.py` | CRIT scorer — per-agent scoring with aggregation |
| `eval/crit/prompts.py` | CRIT system prompt + single-agent user prompt template |
| `eval/crit/schema.py` | RoundCritResult, CritResult, PillarScores, Diagnostics |

### Modified files

| File | What changed |
|------|-------------|
| `multi_agent/runner.py` | Decomposed round loop into per-phase sub-graphs when PID is active. Added CRIT scoring + PID step after each phase. Added `_reset_per_invocation_state()` to prevent state leaking between decision points. Added PID logging infrastructure with instance-level flag gating. Added `_log_prompt_manifest()` for once-per-round prompt file logging. |
| `multi_agent/graph/nodes.py` | (Decomposed from `graph.py`.) All 7 node functions (propose, critique, revise × sequential/parallel + judge) route through `PromptRegistry.build()`. No more `if is_modular_mode(config):` branches. |
| `multi_agent/graph/llm.py` | (Decomposed from `graph.py`.) LLM call infrastructure, prompt logging (`_log_prompt`), memo compaction (`_compact_user_prompt`), snapshot ID extraction (`_extract_snapshot_id`). |
| `multi_agent/prompts/registry.py` | Unified prompt builder. `PromptRegistry.build()` assembles system prompts from composable blocks. `resolve_beta()` unifies PID β and static agreeableness. `build_prompt_manifest()` resolves file names for manifest logging. |
| `multi_agent/prompts/__init__.py` | Removed `get_agreeableness_modifier()` and 5-bucket agreeableness system. Tone is now handled by the system prompt via `PromptRegistry.build()`, not the user prompt template. |
| `multi_agent/models.py` | Added `RoundMetrics`, `ControllerOutput`, `PIDEvent` Pydantic models. Added `pid_events` field to `AgentTrace`. |
| `multi_agent/config.py` | Added PID fields to `DebateConfig`: `pid_config`, `_pid_enabled_flag`, gain fields, `initial_beta`, per-phase toggles, logging flags, `log_prompt_manifest`. Added `pid_enabled` and `evaluation_mode` properties. Added validation in `__post_init__` for PID gains and `initial_beta`. |
| `models/config.py` | Added PID fields and `log_prompt_manifest` to `AgentConfig` for YAML-driven configuration. |
| `agents/multi_agent_debate.py` | Wires PID fields and `log_prompt_manifest` from `AgentConfig` through to `DebateConfig`. |
| `config/debate_memo_pid.yaml` | PID configuration with `log_prompt_manifest: true`. |

### Deleted files

| File | Why |
|------|-----|
| `multi_agent/prompts/agreeableness/*.txt` | Replaced by 3-bucket tone system in `multi_agent/prompts/tone/`. |

### Backward compatibility

- PID is **disabled by default**. All existing behavior is unchanged when `pid_enabled: false`.
- When PID is disabled, the runner uses the original single-round graph path (`run_single_round()`). The decomposed per-phase path is only used when PID is enabled.
- When PID is disabled, `resolve_beta()` falls back to `β = 1.0 - agreeableness`, so the static agreeableness knob still works through the unified 3-bucket tone system.
- All existing tests continue to pass without modification.
- `pid_events` is `null` in all output artifacts when PID is disabled.

---

## Tuning guide

### Starting gains

The defaults (`Kp=0.15, Ki=0.01, Kd=0.03`) are conservative. They produce gentle corrections that won't cause oscillation.

### If reasoning quality is consistently low

- Increase `Kp` to react more strongly to the current gap
- Increase `Ki` if the problem persists across many rounds (accumulated error)
- Keep `Kd` moderate to avoid overshoot

### If agents are oscillating (quality bounces high/low)

- Decrease `Kp` and `Ki`
- Increase `Kd` for more damping
- Increase `gamma_beta` for more momentum (slower changes)

### If sycophancy is a problem

- The default `mu=1.0` adds a significant penalty when detected
- Lower `delta_s` (default 0.05) to make detection more sensitive
- These are only configurable programmatically via `PIDConfig`

### If the stability check fails

The error `GainInstabilityError` means your gains are too aggressive. Fix by:
- Lowering `Kp`, `Ki`, or `Kd`
- Reducing `T_max` (fewer rounds = less integral accumulation)
- Increasing `gamma_beta` (more momentum = harder to destabilize)

### Observing the effect

Enable `pid_log_metrics: true` and `log_prompt_manifest: true`, then watch:
- `e_t` — is it trending toward 0?
- `beta_new` — is it settling to a stable value or oscillating?
- `s_t` — is sycophancy being detected when it shouldn't be?
- Per-pillar scores — which aspect of reasoning is weakest?
- Prompt manifest — which tone bucket (adversarial/balanced/collaborative) is active each round? Which role files and templates are in use?
