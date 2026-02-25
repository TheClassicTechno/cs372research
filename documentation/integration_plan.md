
# Plan: Wire Multi-Agent Debate into the Simulation Pipeline (v4)

---

## Context

The multi-agent debate system (`multi_agent/`) and the simulation pipeline (`simulation/`) are currently disconnected.

Running:

```bash
python run_simulation.py --config config/example.yaml
```

only knows about the `single_llm` agent.

### Goal

Make:

```bash
python run_simulation.py --config config/debate.yaml
```

run the **full pipeline end-to-end**, with a clean **4-layer separation** that:

* Keeps the system PID-ready
* Avoids future refactoring
* Preserves strict architectural boundaries

---

# 1. Architecture Diagram (4 Layers)

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT LAYER  (simulation/)                                    │
│                                                                      │
│  data/cases/*.json → case_loader → runner._run_episode()             │
│       │                                                              │
│       ├── feature_engineering.build_observation(case) → Observation  │
│       │                                                              │
│  ┌────┼──────────────────────────────────────────────────────────┐   │
│  │  GENERATION LAYER  (agents/ + multi_agent/)                   │   │
│  │    │                                                          │   │
│  │    ▼                                                          │   │
│  │  agents/multi_agent_debate.py  (AgentSystem adapter)         │   │
│  │    │                                                          │   │
│  │    ▼                                                          │   │
│  │  multi_agent/runner.py  (debate orchestrator)                │   │
│  │    │                                                          │   │
│  │    │  ┌─ per round ──────────────────────────────────┐        │   │
│  │    │  │  propose → critique → revise                 │        │   │
│  │    │  │       │                                      │        │   │
│  │    │  │       ▼                                      │        │   │
│  │    │  │  round_hook(RoundMetrics)                    │◄───┐   │   │
│  │    │  │       │                                      │    │   │   │
│  │    │  │       ▼                    ┌─────────────────┼────┘   │   │
│  │    │  │  apply ControllerOutput    │ CONTROLLER LAYER│        │   │
│  │    │  │  (adjust agreeableness,    │ (eval/PID/)     │        │   │
│  │    │  │   inject prompt modifier,  │ PIDController   │        │   │
│  │    │  │   force extra round)       │   .step()       │        │   │
│  │    │  └──────────────────────────────────────────────┘        │   │
│  │    │                                                          │   │
│  │    ▼                                                          │   │
│  │  judge → (Action, AgentTrace)                                 │   │
│  │    │                                                          │   │
│  │    ▼                                                          │   │
│  │  adapter translates Action → Decision + trace dict            │   │
│  │  returns AgentInvocationResult(decision, trace)               │   │
│  │                                                               │   │
│  │  NOTE: Agent does NOT call broker. Runner does.               │   │
│  └───────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  runner calls broker.execute_decision(decision, case)                │
│  runner collects trace into trace_store                               │
│  runner writes episode_log, reasoning/, trades.json                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EVALUATION LAYER  (eval/)                                           │
│                                                                      │
│  eval/posthoc_pipeline.py  (NEW)                                     │
│       │                                                              │
│       ├── eval/consistency.py  (existing RCA judge)                  │
│       │                                                              │
│       ├── aggregate: consistency_rate, sycophancy_rate, etc.         │
│       │                                                              │
│       ▼                                                              │
│  eval.schema.json v1.2.0 artifact → results/{run}/eval.json         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

# 2. Layer Responsibilities

---

## Generation Layer (`agents/ + multi_agent/`)

Takes a prepared `Observation`, runs debate, returns:

* `Decision`
* `trace` dict

### Must NOT:

* Compute financial features
* Call the broker
* Evaluate its own reasoning
* Write eval artifacts

| Module                         | Role                                              |
| ------------------------------ | ------------------------------------------------- |
| `multi_agent/graph.py`         | LangGraph nodes: propose, critique, revise, judge |
| `multi_agent/runner.py`        | Orchestrates graph, exposes per-round hook        |
| `agents/multi_agent_debate.py` | Adapter: Observation ↔ debate, Action → Decision  |

---

## Environment Layer (`simulation/`)

Loads data, builds Observation, calls agent, executes broker, logs results.

| Module                                    | Role                                           |
| ----------------------------------------- | ---------------------------------------------- |
| `simulation/feature_engineering.py` (NEW) | `build_observation(case)`                      |
| `simulation/runner.py`                    | Episode loop + trace collection + posthoc eval |
| `simulation/broker.py`                    | Validate + execute trades                      |
| `simulation/case_loader.py`               | Load JSON cases                                |
| `simulation/sim_logging.py`               | Write logs + artifacts                         |

---

## Evaluation Layer (`eval/`)

Post-hoc only. Blind reasoning scoring.

| Module                           | Role                                 |
| -------------------------------- | ------------------------------------ |
| `eval/consistency.py`            | RCA per-turn verdicts                |
| `eval/posthoc_pipeline.py` (NEW) | Aggregate + build `eval.schema.json` |

---

## Controller Layer (`eval/PID/`)

Mid-debate intervention.

| Module                   | Role                             |
| ------------------------ | -------------------------------- |
| `eval/PID/controller.py` | `PIDController.step(...)`        |
| `eval/PID/sycophancy.py` | JS divergence + evidence overlap |
| `eval/PID/types.py`      | PIDConfig, PIDState, etc.        |

Controller runs **between debate rounds**, not post-episode.

---

# 3. RoundMetrics and ControllerOutput

## RoundMetrics

```python
@dataclass
class RoundMetrics:
    round_index: int
    proposal_vectors: list[dict]
    claim_levels: list[str]
    divergence_score: float
    consistency_proxy: float
```

* `divergence_score` → Jensen-Shannon divergence across proposals
* `consistency_proxy` → cheap heuristic (no LLM)

---

## ControllerOutput

```python
@dataclass
class ControllerOutput:
    new_agreeableness: float | None = None
    force_extra_round: bool = False
    inject_prompt_modifier: str | None = None
```

PID can modify:

1. Agreeableness
2. Extra rounds
3. Prompt modifiers

PID cannot modify:

* Agent roster
* Model choice
* Temperature
* Observation data

---

## PID Event Logging

```python
@dataclass
class PIDEvent:
    round_index: int
    metrics: RoundMetrics
    pid_result: PIDStepResult
    output: ControllerOutput
```

Stored in:

```python
AgentTrace.pid_events: list[PIDEvent] | None
```

---

# 4. Data Flow (Updated)

```
data/cases/nvda_001.json
   │
case_loader.load_case_templates(...)
   │
Case
   │
build_observation(case)
   │
Observation
   │
agent.invoke(...)
   │
MultiAgentRunner.run()
   │
propose → critique → revise
   │
round_hook(RoundMetrics)
   │
PIDController.step(...)
   │
ControllerOutput applied
   │
judge → Action
   │
Action → Decision
   │
runner calls broker.execute_decision(...)
   │
trace_store.append(trace)
   │
Episode complete
   │
run_posthoc_eval(trace_store)
   │
eval.schema.json written
```

---

# 5. File-by-File Change List

## New Files

### `simulation/feature_engineering.py`

* `build_observation(case)`
* Computes returns + volatility
* Flattens text_context
* Converts portfolio to float

---

### `agents/multi_agent_debate.py`

* Adapter class
* Handles Observation → debate
* Action → Decision
* Returns trace
* Does NOT call broker

---

### `eval/posthoc_pipeline.py`

* Runs RCA
* Aggregates verdicts
* Writes `eval.schema.json`

---

### `config/debate.yaml`

* Sets `agent_system: multi_agent_debate`
* Includes debate + optional pid config

---

### `data/cases/nvda_001.json`

* Sample case for testing

---

## Modified Files

### `models/agents.py`

* Add `observation` to `AgentInvocation`
* Add `trace` to `AgentInvocationResult`

---

### `models/config.py`

* Add `DebateAgentConfig`
* Add `PIDAgentConfig`
* Add `debate_config` and `pid_config`

---

### `agents/registry.py`

* Import `agents.multi_agent_debate`

---

### `simulation/runner.py`

* Build Observation before `agent.invoke`
* Runner calls broker (not agent)
* Collect traces
* Run posthoc eval

---

### `multi_agent/runner.py`

* Add `round_hook`
* Compute RoundMetrics
* Apply ControllerOutput
* Log PIDEvent

---

### `multi_agent/models.py`

* Change `Order.size` from float → int
* Add `pid_events` to `AgentTrace`

---

# 6. Schema Notes for PID

No schema changes required.

Use:

* `evaluation_mode: "in_loop"` if PID active
* `control_trace` → derived from PIDEvent
* `experiment_config.control.policy.type: "pid"`

---

# 7. No Silent Rounding — Order Type Resolution

Change:

```python
multi_agent.models.Order.size: float
```

To:

```python
size: int
```

Eliminates rounding mismatch at the source.

---

# 8. Runner Behavior: Debate vs Single LLM

### Single LLM Flow

1. Create submit_decision tool
2. Agent calls tool
3. Broker executes inside tool

### Debate Flow

1. Agent returns `Decision`
2. Runner calls broker directly
3. No tool used

Clear separation.

---

# 9. Implementation Order

1. Friction fixes (types + models)
2. Feature engineering
3. Agent adapter
4. Runner integration
5. Post-hoc eval pipeline
6. End-to-end test

---

# 10. Verification

```bash
pytest --ignore=.venv -v
python run_simulation.py --config config/debate.yaml
cat results/*/eval.json | python -m json.tool
```

Layer separation checks:

```bash
grep agents/multi_agent_debate.py broker
grep agents/multi_agent_debate.py consistency
grep eval/posthoc_pipeline.py broker
```

