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
2. **CRIT**, a blind reasoning auditor, scores the round's reasoning quality across four pillars.
3. The **PID controller** compares that quality score against a target and computes a correction signal.
4. That correction adjusts `agreeableness` — the dial that controls how confrontational vs. deferential agents are in the next round.

High reasoning quality → ease off, let agents converge naturally.
Low reasoning quality → push harder, make agents more critical.

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────┐
 │                   Debate Round t                         │
 │                                                         │
 │   ┌──────────┐   ┌───────────┐   ┌──────────┐          │
 │   │ Propose  │──▶│ Critique  │──▶│  Revise  │          │
 │   │ (β_prop) │   │ (β_crit)  │   │ (β_rev)  │          │
 │   └──────────┘   └───────────┘   └──────────┘          │
 │                                                         │
 └────────────────────────┬────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  CRIT Scorer  │  ← Blind reasoning audit
                  │  (4 pillars)  │     No ground truth access
                  └───────┬───────┘
                          │ rho_bar = mean(IC, ES, TA, CI)
                          ▼
                  ┌───────────────┐
                  │PID Controller │  ← Compare rho_bar vs target
                  │  step()       │     Compute correction u_t
                  └───────┬───────┘
                          │ beta_new
                          ▼
                  Adjust agreeableness
                  for round t+1
```

### Per-phase decomposition

Without PID, each debate round runs as a single graph invocation (propose → critique → revise). With PID enabled, each round is decomposed into **three separate sub-graph invocations** so that agreeableness can be set independently for each phase:

| Phase     | Config toggle   | Default | Why |
|-----------|----------------|---------|-----|
| Propose   | `pid_propose`  | `false` | Proposals aren't strongly affected by agreeableness |
| Critique  | `pid_critique` | `true`  | Critique tone is most sensitive to agreeableness |
| Revise    | `pid_revise`   | `true`  | Revision deference is also sensitive |

When a phase's toggle is `true`, PID's computed `beta` is used as agreeableness for that phase. When `false`, the original static `agreeableness` value (from `DebateConfig`) is used.

---

## CRIT: The Blind Reasoning Auditor

CRIT (Critique of Reasoning Integrity and Traceability) is an LLM-based scorer that evaluates reasoning quality **without access to ground truth**. It never sees market outcomes, impact scores, or whether the agents' predictions were correct. It evaluates only logical integrity.

### Four pillars

| Pillar | What it measures | Failure mode |
|--------|-----------------|-------------|
| **Internal Consistency** | Are the agent's claims logically compatible with each other? | Contradictions within a single argument |
| **Evidence Support** | Are factual claims backed by cited evidence from the case context? | Unsupported assertions |
| **Trace Alignment** | Does the final trading decision follow from the reasoning presented? | Conclusion drifts from argument |
| **Causal Integrity** | Are causal claims properly scoped (Pearl L1/L2/L3)? | Causal overreach |

Each pillar is scored 0.0–1.0. The composite score **rho_bar** is the mean of all four pillars. This is the signal that feeds the PID controller.

### Diagnostics

CRIT also produces binary diagnostic flags:
- `contradictions_detected`
- `unsupported_claims_detected`
- `conclusion_drift_detected`
- `causal_overreach_detected`

These are logged when PID metrics logging is enabled and appear in the trace output.

### Source files

```
eval/crit/
    __init__.py     # Public API: CritScorer, CritResult
    scorer.py       # CritScorer class (dependency-injected LLM function)
    prompts.py      # System prompt + user prompt template
    schema.py       # PillarScores, Diagnostics, Explanations, CritResult
```

---

## The PID control loop

### Error signal (Eq 5)

```
e_t = (rho_star - rho_bar) + mu * s_t
```

- `rho_star` — target quality score (default 0.8)
- `rho_bar` — actual quality from CRIT this round
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

In `config/debate.yaml`:

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

  # Starting agreeableness for PID (0.0 - 1.0)
  pid_initial_beta: 0.5

  # Per-phase toggles (which phases use PID's beta)
  pid_propose: false     # Proposals use original agreeableness
  pid_critique: true     # Critiques use PID-adjusted agreeableness
  pid_revise: true       # Revisions use PID-adjusted agreeableness
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

Two independent logging channels can be enabled for PID debugging:

### Scalar metrics (`pid_log_metrics: true`)

Logs per-round stats to the `pid.metrics` logger at DEBUG level:

```
[PID Round 1]
  CRIT:     rho_bar=0.7500  rho_star=0.8000
  Pillars:  IC=0.800  ES=0.700  TA=0.900  CI=0.600
  Error:    e_t=0.050000  integral=0.050000  e_prev=0.000000
  PID:      p_term=0.007500  i_term=0.000500  d_term=0.001500  u_t=0.009500
  Gains:    Kp=0.1500  Ki=0.0100  Kd=0.0300
  Beta:     old=0.5000  new=0.4595  gamma_beta=0.9000
  Syco:     s_t=0  mu=1.0000  delta_s=0.0500
  Diverg:   JS=0.000000  OV=0.000000
  Phase toggles: propose=False  critique=True  revise=True

[PID Round 1] Per-phase agreeableness:
  propose=0.3000  critique=0.4595  revise=0.4595  (beta=0.4595, original=0.3000)

[PID Round 1] CRIT diagnostics:
  contradictions=False  unsupported_claims=False
  conclusion_drift=False  causal_overreach=False
```

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
{"pillar_scores": {"internal_consistency": 0.8, ...}, ...}
```

### Enabling logging

In `debate.yaml`:

```yaml
agent:
  pid_enabled: true
  pid_log_metrics: true
  pid_log_llm_calls: true
```

To see the output, configure Python's logging system to show DEBUG messages for the `pid.*` loggers. For example:

```python
import logging
logging.getLogger("pid.metrics").setLevel(logging.DEBUG)
logging.getLogger("pid.llm").setLevel(logging.DEBUG)
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
            "pillar_scores": { ... },
            "diagnostics": { ... },
            "explanations": { ... },
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
| `eval/crit/scorer.py` | CRIT scorer (blind reasoning auditor) |
| `eval/crit/prompts.py` | CRIT system and user prompt templates |
| `eval/crit/schema.py` | CritResult, PillarScores, Diagnostics, Explanations |

### Modified files

| File | What changed |
|------|-------------|
| `multi_agent/runner.py` | Decomposed round loop into per-phase sub-graphs when PID is active. Added CRIT scoring + PID step after each round. Added `_reset_per_invocation_state()` to prevent state leaking between decision points. Added PID logging infrastructure with instance-level flag gating. |
| `multi_agent/graph.py` | Added 6 per-phase sub-graph builders (propose/critique/revise, sequential and parallel variants) + 6 compile wrappers. Added `ParallelRoundState` TypedDict. Fixed `StopIteration` crash in critique/revise nodes (safe `next()` with default). Added error logging in `_call_llm`. |
| `multi_agent/models.py` | Added `RoundMetrics`, `ControllerOutput`, `PIDEvent` Pydantic models. Added `pid_events` field to `AgentTrace`. |
| `multi_agent/config.py` | Added PID fields to `DebateConfig`: `pid_config`, `_pid_enabled_flag`, gain fields, `initial_beta`, per-phase toggles, logging flags. Added `pid_enabled` and `evaluation_mode` properties. Added validation in `__post_init__` for PID gains and `initial_beta`. |
| `models/config.py` | Added PID fields to `AgentConfig` for YAML-driven configuration. |
| `agents/multi_agent_debate.py` | Wires PID fields from `AgentConfig` through to `DebateConfig`. |
| `config/debate.yaml` | Added PID configuration section with commented-out defaults. |

### Backward compatibility

- PID is **disabled by default**. All existing behavior is unchanged when `pid_enabled: false`.
- When PID is disabled, the runner uses the original single-round graph path (`run_single_round()`). The decomposed per-phase path is only used when PID is enabled.
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

Enable `pid_log_metrics: true` and watch:
- `e_t` — is it trending toward 0?
- `beta_new` — is it settling to a stable value or oscillating?
- `s_t` — is sycophancy being detected when it shouldn't be?
- Per-pillar scores — which aspect of reasoning is weakest?
