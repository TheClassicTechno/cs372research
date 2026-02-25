# CS372 — End-to-End Simulation & Multi-Agent Debate Pipeline

This document explains how to run the **entire trading simulation pipeline**, how the multi-agent debate integrates into it, and the exact data formats passed between components.

The goal is to make it possible for a new contributor to:

1. Run the full system from the CLI.
2. Understand every data handoff.
3. Know where reasoning traces and execution logs are written.
4. Extend or replace the agent architecture safely.

---

# 1️⃣ One Command to Run Everything

```bash
python run_simulation.py --config config/example.yaml
```

This executes:

```
Case Templates (JSON)
        ↓
SimulationRunner
        ↓
AgentSystem (single or multi-agent)
        ↓
submit_decision tool
        ↓
Broker (execution)
        ↓
Logging + Trace Output
```

All outputs are written to:

```
results/{run_name}/
```

---

# 2️⃣ Top-Level Architecture

```
┌──────────────────────────────────────────────┐
│ run_simulation.py (CLI, async entrypoint)   │
└──────────────────┬───────────────────────────┘
                   │
           ┌───────▼────────┐
           │ SimulationRunner│
           └───────┬────────┘
                   │
      ┌────────────▼────────────┐
      │      AgentSystem        │
      │ (single_llm, debate, …) │
      └────────────┬────────────┘
                   │
          submit_decision tool
                   │
           ┌───────▼────────┐
           │     Broker     │
           └────────────────┘
```

The simulation runner does not know how reasoning works.
The agent system does not know how trades execute.
The broker does not know how reasoning works.

Each layer has a strict data contract.

---

# 3️⃣ Step-by-Step Data Flow

---

## STEP 1 — Case Templates (Disk → Memory)

### Motivation

Separate static market data from live portfolio state.
Ensure experiments are reproducible and deterministic.

---

### Supported Dataset Formats

`dataset_path` in YAML may be:

| Type       | Example       | Meaning                         |
| ---------- | ------------- | ------------------------------- |
| Directory  | `data/cases/` | Recursively loads `.json` files |
| JSON Array | `cases.json`  | Array of cases                  |
| JSONL      | `cases.jsonl` | One case per line               |

---

### On-Disk Case Format

Each file represents one **decision point** (typically quarterly earnings).

```json
{
  "case_data": {
    "items": [
      {"kind": "earnings", "content": "..."},
      {"kind": "news", "content": "..."}
    ]
  },
  "stock_data": {
    "AAPL": {
      "ticker": "AAPL",
      "current_price": 185.50,
      "daily_bars": [
        {"timestamp": "2024-10-01", "close": 175.0}
      ]
    }
  }
}
```

---

### Runtime Transformation

At runtime, the runner stamps in portfolio state:

```python
Case(
    case_data: CaseData,
    stock_data: dict[str, StockData],
    portfolio_snapshot: PortfolioSnapshot
)
```

This becomes the input to the agent.

---

## STEP 2 — SimulationRunner

File:

```
simulation/runner.py
```

Class:

```
AsyncSimulationRunner
```

---

### Responsibilities

Per episode:

1. Initialize broker.
2. Load case templates.
3. For each case:

   * Snapshot portfolio.
   * Build `Case`.
   * Create `submit_decision` tool.
   * Invoke agent.
   * Execute trades.
   * Log everything.

---

### Data Handoff to Agent

The runner constructs:

```python
AgentInvocation(
    case: Case,
    episode_id: str,
    agent_id: str,
    steps_remaining: int | None
)
```

This is passed into:

```python
await agent.invoke(invocation)
```

---

## STEP 3 — Agent System

Agent systems are registered via:

```python
@register("agent_name")
class MyAgent(AgentSystem):
```

Configured in YAML:

```yaml
agent:
  agent_system: "single_llm"
  llm_provider: "openai"
  llm_model: "gpt-4o"
```

---

### Agent Interface Contract

Every agent must implement:

| Method       | Purpose                            |
| ------------ | ---------------------------------- |
| `__init__`   | Receive config                     |
| `bind_tools` | Receive fresh submit_decision tool |
| `invoke`     | Analyze case and submit orders     |

---

# 4️⃣ submit_decision Tool Contract

### Motivation

Prevent agents from bypassing broker rules.
Force trade execution through validation layer.

---

### Tool Input Format

```json
{
  "orders": [
    {
      "ticker": "AAPL",
      "side": "buy",
      "quantity": 10
    }
  ]
}
```

---

### Tool Response Format

```json
{
  "status": "accepted"
}
```

OR

```json
{
  "status": "rejected",
  "reason": "Insufficient cash"
}
```

---

### Execution Semantics

* Sells execute before buys.
* All-or-nothing execution.
* Ticker must exist.
* Cannot oversell shares.
* Cannot overspend cash.
* Agent may retry up to `max_retries`.

---

# 5️⃣ Multi-Agent Debate Integration

If `agent_system` is set to debate (Python LangGraph system), then:

```
AgentInvocation
      ↓
MultiAgentRunner
      ↓
LangGraph Debate
      ↓
Judge
      ↓
Decision
```

---

## Debate Internal Data Types

Shared Pydantic models:

| Type          | Purpose                     |
| ------------- | --------------------------- |
| `Observation` | Simulator → debate input    |
| `Action`      | Proposal output             |
| `Claim`       | Structured causal reasoning |
| `DebateTurn`  | Round record                |
| `AgentTrace`  | Final auditable artifact    |

---

## Debate Graph Structure

```
START
  → news_digest (optional)
  → data_analysis (optional)
  → build_context
  → propose
  → critique
  → revise
  → judge
  → build_trace
END
```

---

### Proposal Format

```python
Action(
    orders: list[Order],
    justification: str,
    confidence: float,
    claims: list[Claim]
)
```

---

### Claim Format

```python
Claim(
    pearl_level: "L1" | "L2" | "L3",
    statement: str
)
```

---

### Judge Output

```python
Decision(
    orders: list[Order],
    confidence: float
)
```

Plus structured `AgentTrace`.

---

# 6️⃣ Broker Execution Layer

File:

```
simulation/broker.py
```

Input:

```
Decision.orders
```

Output:

```
ExecutedTrade
DecisionResult
```

Portfolio state updated in memory.

---

# 7️⃣ External Services

---

## OpenAI

Via:

```python
langchain_openai.ChatOpenAI
```

Message format:

```python
[
  SystemMessage(content="..."),
  HumanMessage(content="...")
]
```

Response:

```python
AIMessage(content="...")
```

---

## Anthropic

Via:

```python
langchain_anthropic.ChatAnthropic
```

Same message abstraction.

---

## Environment Variables

```
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

Loaded via standard environment resolution.

---

# 8️⃣ Output Directory Structure

Each run produces:

```
results/{run_name}/
├── config.yaml
├── simulation_log.json
├── summary.json
└── episodes/
    └── ep_000/
        ├── episode_log.json
        ├── trades.json
        └── reasoning/
```

---

## simulation_log.json

Full experiment reconstruction artifact.

---

## episode_log.json

Contains per-decision-point:

* Portfolio before
* Decision submitted
* Execution result
* Portfolio after
* Timing

---

## reasoning/

Contains:

* Raw LLM outputs
* Debate transcripts (if multi-agent)
* Structured trace files

---

# 9️⃣ Why This Structure Exists

| Layer       | Why                            |
| ----------- | ------------------------------ |
| Case Loader | Reproducibility                |
| Runner      | Episode orchestration          |
| AgentSystem | Architecture swap flexibility  |
| Tool        | Enforced execution constraints |
| Broker      | Market realism                 |
| Logs        | Full auditability              |

The system is intentionally modular to support:

* Agent ablations
* Prompt experiments
* Debate vs single-agent comparisons
* Economic performance evaluation
* Reasoning trace analysis

---

# 🔟 Full Mental Model

When you run:

```bash
python run_simulation.py --config config/example.yaml
```

The system:

1. Loads config.
2. Loads case templates.
3. Starts episode.
4. Builds Case.
5. Invokes AgentSystem.
6. Agent calls submit_decision.
7. Broker validates and executes.
8. Logs reasoning + trades.
9. Repeats for next case.
10. Writes full audit artifacts.

Final result:

* Portfolio evolution
* Full trade history
* Structured reasoning traces
* Complete experiment reconstruction data
