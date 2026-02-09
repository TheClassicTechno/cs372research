# Data Models for Multi-Agent Trading Simulation

This document defines the Pydantic data models used by the Simulation Environment, Multi-Agent System, and Evaluation System. All three groups import from `models` to ensure interoperability.

---

## Simulation-Invokes-Agent Loop

At each decision point:

1. **Simulation** builds a `Case` (incorporating current portfolio state).
2. **Simulation** invokes the agent with the Case in the prompt (no `get_case` tool).
3. **Agent** reasons and calls `submit_decision(Decision)`.
4. **Simulation** extracts the Decision from the tool call, executes it, updates state.
5. **Simulation** advances to the next decision point.

**Critical detail:** Each Case must include the **current** portfolio (set dynamically). The loop builds the Case each iteration:

```python
for decision_point_idx in range(num_decision_points):
    case_id = f"{episode_id}:{decision_point_idx}"  # case_id is required and must be set
    case_template = get_case_template(decision_point_idx, config)
    case = build_case(case_template, state.portfolio, case_id=case_id)
    steps_remaining = num_decision_points - decision_point_idx - 1  # optional
    result = agent.invoke(AgentInvocation(case=case, episode_id=..., agent_id=..., steps_remaining=steps_remaining))
    decision = extract_decision(result)
    executed_trades, state = execute_decision(decision, case, state)  # case needed for ticker validation
    trade_history.extend(executed_trades)
```

**Execution:** The simulation runs `execute_decision(decision, case, state)` after invoke (sells before buys). Execution is **all-or-nothing**: every order must be valid (ticker in `Case.tickers`, etc.) or the whole decision is rejected. The `submit_decision` tool returns `DecisionResult` to the agent; when rejected, `message` must explain the reason (e.g. which tickers were not in the case universe).

**Reasoning traces:** Handled by LangChain. The agent should write reasoning traces to a standardized location if needed for evaluation.

---

## File Structure

```
models/
├── README.md           # This file
├── __init__.py         # Re-exports all models
├── case.py             # Case, CaseData, CaseDataItem, StockData, PricePoint, IntervalPriceSummary
├── portfolio.py        # PortfolioSnapshot
├── decision.py         # Order, Decision, ExecutedTrade, DecisionResult
├── episode.py          # EpisodeConfig, EpisodeResult, TradeHistory
├── agents.py           # AgentInvocation, AgentInvocationResult
└── experiment.py       # SimulationLog, EpisodeLog, DecisionPointLog (storage)
```

---

## Model Catalog by File

### `case.py` — Case & Market Data

| Model | Purpose |
|-------|---------|
| `Case` | Wraps case_data, stock_data, portfolio; **case_id** is required (e.g. `f"{episode_id}:{decision_point_idx}"`). Metadata (decision_point_idx, information_cutoff_timestamp) stored but not passed to the agent. Use `case.for_agent()` for the agent payload. |
| `CaseData` | Variable-length list of information items (earnings, news, etc.). |
| `CaseDataItem` | Single item: kind (earnings/news/other), content. |
| `StockData` | Per-ticker: current_price, daily_bars (each with timestamp), and optional **interval_summary** (lighter payload for the agent). |
| `PricePoint` | One day's OHLCV bar with **timestamp** (e.g. YYYY-MM-DD). |
| `IntervalPriceSummary` | Optional open/close/high/low/volume summary for the interval; use in agent payload to reduce token usage. |

Portfolio on Case is set dynamically by the simulation; other case attributes can be pre-computed.

---

### `portfolio.py` — Portfolio State

| Model | Purpose |
|-------|---------|
| `PortfolioSnapshot` | Cash + positions (ticker → shares) at a decision point. Used as an attribute of Case, set dynamically. |

---

### `decision.py` — Agent Output & Execution

| Model | Purpose |
|-------|---------|
| `Order` | Single order: ticker, side, quantity. |
| `Decision` | Agent output: list of orders; empty = hold. |
| `ExecutedTrade` | Single executed fill; **order_index** links to `Decision.orders[i]`. Simulation produces one ExecutedTrade per Order when accepted. |
| `DecisionResult` | **All-or-nothing**: status is `accepted` or `rejected`. When rejected, `message` must state the reason (e.g. "ticker XYZ not in Case.tickers"). `executed_trades` non-empty only when accepted. |

---

### `episode.py` — Episode-Level

| Model | Purpose |
|-------|---------|
| `EpisodeConfig` | Episode parameters: episode_id, agent_id, tickers, initial_cash. |
| `EpisodeResult` | Final outcome: trade_history, final_portfolio. |
| `TradeHistory` | List of ExecutedTrade for an episode. |

---

### `agents.py` — Agent Interface

| Model | Purpose |
|-------|---------|
| `AgentInvocation` | Input when invoking agent: case, episode_id, agent_id. Optional **steps_remaining** (int) so the agent can know how many decision steps remain. |
| `AgentInvocationResult` | Parsed output: decision. |

---

### `experiment.py` — Logging & Experiment Storage

| Model | Purpose |
|-------|---------|
| `DecisionPointLog` | Per-decision audit: case_id, decision_point_idx, agent_output, extracted_decision, execution_result. |
| `EpisodeLog` | Full episode audit: config, decision_point_logs, episode_result. |
| `SimulationLog` | Run-level log: run_id, episode_logs, errors. Optional **simulation_version**, **agent_version**, **evaluation_version** for reproducibility. |

**Storage layout (example):**

```
experiments/
├── {run_id}/
│   ├── config.json           # EpisodeConfig(s)
│   ├── episode_log.jsonl     # EpisodeLog per line
│   └── trade_history.json
```

---

## Data Flow

```
EpisodeConfig
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  for each decision_point:                               │
│    Case (portfolio set dynamically)                     │
│         │                                               │
│         ▼                                               │
│    AgentInvocation(case, episode_id, agent_id)          │
│         │                    │                          │
│         ▼                    ▼                          │
│    Agent.invoke() ──► submit_decision(Decision)         │
│         │                    │                          │
│         ▼                    ▼                          │
│    extract_decision() ──► Decision                       │
│         │                                               │
│         ▼                                               │
│    execute_decision() ──► ExecutedTrade(s)               │
│         │                                               │
│         ▼                                               │
│    DecisionPointLog (case_id, decision, result)         │
│         │                                               │
│         ▼                                               │
│    state.portfolio updated                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
EpisodeResult (TradeHistory, final_portfolio)
    │
    ▼
EpisodeLog ──► persisted to experiments/{run_id}/
```

---

## Tool Contract

| Tool | Args | Returns |
|------|------|---------|
| `submit_decision` | `Decision` | `DecisionResult` |

The Case is passed in the prompt (use `case.for_agent()` for the agent-facing payload); no `get_case` tool.

---

## Extracting Decision from Agent Output

When parsing the agent's response:

- **No tool call:** Default to `Decision(orders=[])` (hold).
- **Multiple tool calls:** Use the last `submit_decision` call.
- **Invalid Decision:** Any order whose ticker is not in `Case.tickers` (or that violates execution rules) causes the whole decision to be **rejected**. Return `DecisionResult(status="rejected", message="...")` with the reason (e.g. "ticker XYZ not in case universe"); the simulation does not execute. Fall back to hold for the step.

The simulation need only produce a `Decision` for execution and optionally log raw output in `DecisionPointLog.agent_output` for debugging.

---

## Agent–Simulation Communication and LangChain Feasibility

**How they communicate**

- The **simulation** is the driver: it runs in a normal Python loop (no special runtime). At each decision point it builds a `Case`, builds an `AgentInvocation`, and **invokes** the agent (e.g. a LangChain runnable).
- The **agent** receives the case via the **prompt** (e.g. the prompt includes `case.for_agent()` as structured text or JSON). The agent has exactly one tool: `submit_decision(Decision)`.
- When the agent calls that tool, the **simulation** (as the caller) receives the tool call, extracts the `Decision`, runs `execute_decision(decision, case, state)`, and then **returns** a `DecisionResult` as the tool’s return value. The agent sees that return value (e.g. in the next message or step).
- So the flow is: **Simulation → invoke agent with prompt → Agent reasons and calls tool → Simulation handles tool, executes, returns DecisionResult → Agent receives result.** There is no separate “environment server”; the simulation process both runs the loop and implements the tool handler.

**Feasibility in LangChain**

This is straightforward to implement in LangChain:

1. **Agent**: Use an LLM with a single tool whose name is `submit_decision`, description explains the `Decision` schema, and whose implementation is a function that your simulation code provides. That function can either (a) return the `DecisionResult` directly to the agent (if the framework supports synchronous tool execution by the caller), or (b) store the decision in a shared structure and return a placeholder, and have the simulation run execution after `invoke()` and pass the result in a follow-up (depending on how you structure the run).

2. **Recommended pattern**: Bind the tool so that when the agent calls `submit_decision`, the **tool function** is executed by the same process that runs the simulation. The tool function can:
   - Accept the parsed `Decision`,
   - Call your `execute_decision(decision, case, state)` (you pass `case` and `state` in closure or context),
   - Return the resulting `DecisionResult` (e.g. as a JSON string or dict that LangChain passes back to the LLM). The simulation then updates `state` and `trade_history` after `invoke()` returns, using the same decision it already executed inside the tool (or the tool can update shared state so the loop only advances). Either way, a single process holds both the loop and the tool implementation.

3. **No special env protocol**: You do not need LangGraph’s environment step, a separate simulator process, or custom message types. A simple loop + LLM with one tool + a tool handler that runs execution and returns `DecisionResult` is enough. If you use LangGraph, the “simulation” can be the graph’s caller that invokes the graph once per decision point and supplies the tool; the graph’s tool node can run execution and return the result.

4. **Validation**: Before executing, the simulation (inside the tool or immediately after extracting the decision) validates that every order’s ticker is in `Case.tickers`; if not, it returns `DecisionResult(status="rejected", message="...")` and does not update state. All-or-nothing execution keeps the contract simple.
