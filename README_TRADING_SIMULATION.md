# CS 372 — LLM Trading Agent Simulation

A research framework for evaluating LLM-based trading agents in a controlled market simulation. Agents receive market data (prices, news, earnings) and a live portfolio, then submit buy/sell/hold decisions through a tool-calling interface. The simulation executes trades via an in-process broker with all-or-nothing semantics and logs full audit trails for evaluation.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  run_simulation.py                    │
│              (CLI entrypoint, async)                  │
└──────────────────┬───────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  SimulationRunner   │  ← config (YAML)
         │  (simulation/)      │  ← case templates (JSON)
         └──┬─────────────┬───┘
            │             │
   ┌────────▼───┐   ┌────▼──────────┐
   │   Broker   │   │  AgentSystem  │
   │ (portfolio │   │  (pluggable)  │
   │  & trades) │   └────┬──────────┘
   └────────▲───┘        │
            │            │  submit_decision tool
            └────────────┘
```

**Data flow per decision point:**

1. Runner snapshots the portfolio and builds a `Case` (market data + portfolio).
2. A fresh `submit_decision` tool is created, bound to the current broker state and case.
3. The agent receives the case in its prompt and reasons about it.
4. The agent calls `submit_decision` with a list of orders.
5. The broker validates and executes the orders (sells before buys, all-or-nothing).
6. The result (accepted/rejected) is returned to the agent; if rejected, the agent may revise and retry.
7. Portfolio snapshots, decisions, and execution results are logged.

## Project Structure

```
cs372research/
├── run_simulation.py           # CLI entrypoint
├── requirements.txt            # Python dependencies
├── config/
│   └── example.yaml            # Example experiment configuration
├── models/                     # Pydantic data models (shared contracts)
│   ├── case.py                 # Case, CaseData, StockData, PricePoint
│   ├── portfolio.py            # PortfolioSnapshot
│   ├── decision.py             # Order, Decision, ExecutedTrade, DecisionResult
│   ├── agents.py               # AgentInvocation, AgentInvocationResult
│   ├── config.py               # SimulationConfig, AgentConfig, BrokerConfig
│   └── log.py                  # SimulationLog, EpisodeLog, DecisionPointLog
├── simulation/                 # Simulation harness
│   ├── runner.py               # AsyncSimulationRunner (main loop)
│   ├── broker.py               # Broker (portfolio mgmt & trade execution)
│   ├── case_loader.py          # Load case templates from disk
│   └── sim_logging.py          # Output persistence & directory management
└── agents/                     # Pluggable agent systems
    ├── base.py                 # AgentSystem abstract base class
    ├── registry.py             # Agent registry & factory
    ├── single_llm.py           # SingleLLMAgent (built-in ReAct agent)
    └── tools.py                # submit_decision tool factory
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:** Pydantic, PyYAML, LangChain Core, LangGraph, LangChain OpenAI, LangChain Anthropic.

### 2. Set API keys

The framework uses LangChain chat models, so set the appropriate environment variable for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 3. Prepare a case dataset

The simulation reads case templates from disk. Each case contains market data (`case_data`, `stock_data`) that the agent will see. See [Case Data Format](#case-data-format) below.

## Running a Simulation

```bash
python run_simulation.py --config config/example.yaml
```

### CLI Arguments

| Argument        | Required | Default    | Description                              |
|-----------------|----------|------------|------------------------------------------|
| `--config`      | Yes      | —          | Path to the YAML configuration file.     |
| `--output-dir`  | No       | `results/` | Directory where run outputs are written. |
| `--log-level`   | No       | `INFO`     | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### Examples

```bash
# Basic run
python run_simulation.py --config config/example.yaml

# Custom output directory with verbose logging
python run_simulation.py --config config/example.yaml --output-dir experiments/ --log-level DEBUG
```

The **run name** is derived automatically from the config filename (e.g., `example.yaml` produces the run name `example`).

## Configuration Reference

Experiments are defined in YAML config files. Here is the full schema with all options:

```yaml
# Path to the case dataset.
# Accepts: directory of .json files, single .json array, or .jsonl file.
dataset_path: "data/cases"

# Universe of ticker symbols available in this simulation.
tickers:
  - AAPL
  - MSFT
  - GOOGL

# Number of independent episodes (each starts with a fresh portfolio).
num_episodes: 1

# Agent configuration
agent:
  agent_system: "single_llm"       # Registered agent system name
  llm_provider: "openai"           # "openai" or "anthropic"
  llm_model: "gpt-4o"              # Model name
  temperature: 0.7                 # Sampling temperature (0.0–2.0)
  max_retries: 3                   # Max submit_decision retries per case
  # system_prompt_override: null   # Optional: replace the default system prompt

# Broker configuration
broker:
  initial_cash: 100000.0           # Starting cash balance
```

### Setting Up a New Experiment

1. Create a new YAML file in `config/` (e.g., `config/my_experiment.yaml`).
2. Point `dataset_path` to your case data.
3. Set `tickers` to the universe of symbols your cases cover.
4. Configure the agent (model, temperature, prompt, etc.).
5. Set `num_episodes` for how many independent runs you want.
6. Run it:

```bash
python run_simulation.py --config config/my_experiment.yaml
```

Results will be written to `results/my_experiment/`.

## Building a New Agent

The framework uses a **registry pattern** so you can add custom agent architectures without modifying the simulation runner.

### Step 1: Create a new file in `agents/`

Create `agents/my_agent.py`:

```python
from __future__ import annotations

import logging
from typing import Any, Callable

from agents.base import AgentSystem
from agents.registry import register
from models.agents import AgentInvocation, AgentInvocationResult
from models.config import AgentConfig
from models.decision import Decision

logger = logging.getLogger(__name__)


@register("my_agent")  # This name is used in the YAML config
class MyAgent(AgentSystem):
    """Custom agent implementation."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        # Initialize your LLM, graph, or any resources here.
        # self.config gives you access to llm_provider, llm_model,
        # temperature, max_retries, system_prompt_override, etc.

    def bind_tools(self, submit_decision_fn: Callable[..., Any]) -> None:
        """Called before each invoke() with a fresh submit_decision tool.

        Store the tool so your agent can call it during invoke().
        The tool is a LangChain StructuredTool instance.
        """
        self._tool = submit_decision_fn

    async def invoke(self, invocation: AgentInvocation) -> AgentInvocationResult:
        """Run the agent for one decision point.

        Available on the invocation:
          - invocation.case          : Case (market data + portfolio)
          - invocation.episode_id    : str
          - invocation.agent_id      : str
          - invocation.steps_remaining : int | None

        The agent should analyze the case and call self._tool (submit_decision)
        with a list of orders. Return the Decision in an AgentInvocationResult.
        """
        # Your agent logic here...
        # Example: call the tool with orders
        # result_json = self._tool.invoke({"orders": [
        #     {"ticker": "AAPL", "side": "buy", "quantity": 10}
        # ]})

        # Extract decision from tool state
        decision = self._extract_decision()
        return AgentInvocationResult(decision=decision)

    def _extract_decision(self) -> Decision:
        """Extract the last decision from the tool's stored state."""
        if self._tool is None:
            return Decision(orders=[])
        func = self._tool.func
        last_decision = getattr(func, "_last_decision", None)
        if last_decision is None:
            return Decision(orders=[])
        return last_decision
```

### Step 2: Register the import

Add your module to `agents/registry.py` so it gets loaded automatically:

```python
def _ensure_builtins_loaded() -> None:
    import agents.single_llm  # noqa: F401
    import agents.my_agent    # noqa: F401  <-- add this line
```

### Step 3: Use it in a config

```yaml
agent:
  agent_system: "my_agent"    # matches the @register name
  llm_provider: "openai"
  llm_model: "gpt-4o"
  temperature: 0.5
```

### Agent Interface Summary

Every agent must implement the `AgentSystem` abstract base class:

| Method         | Called When                        | Purpose                                    |
|----------------|-----------------------------------|--------------------------------------------|
| `__init__`     | Once per episode                  | Receive `AgentConfig`, initialize resources |
| `bind_tools`   | Once per decision point (case)    | Receive the `submit_decision` tool          |
| `invoke`       | Once per decision point           | Analyze the case and submit a decision      |

### The `submit_decision` Tool

The tool accepts an `orders` list where each order has:

| Field      | Type              | Description                       |
|------------|-------------------|-----------------------------------|
| `ticker`   | `str`             | Stock ticker symbol               |
| `side`     | `"buy"` or `"sell"` | Order direction                 |
| `quantity` | `int` (positive)  | Number of shares                  |

An empty list means **hold** (no trades). The tool returns a JSON response with status `"accepted"` or `"rejected"` (with a message explaining why).

## Case Data Format

Each case represents a **single ticker at a specific date** — typically aligned with a quarterly earnings report. A case bundles together:

- **Earnings data** — the earnings report for the quarter.
- **News articles** — relevant news from the period.
- **Price history** — daily OHLCV bars covering the interval since the previous case (roughly one quarter).
- **Current price** — the execution price available to the agent at this decision point.

Cases are loaded from disk as templates and stamped with the live portfolio at runtime. On disk, each case needs `case_data` and `stock_data`:

```json
{
  "case_data": {
    "items": [
      {"kind": "earnings", "content": "AAPL Q4 2024 — Revenue: $94.9B (+6% YoY), EPS: $1.64 (beat est. $1.58). Services revenue hit record $23.1B. Guided Q1 revenue $89-93B."},
      {"kind": "news", "content": "Apple Vision Pro launched in February with tepid initial demand..."},
      {"kind": "news", "content": "iPhone 16 sales in China slowed amid Huawei competition..."},
      {"kind": "news", "content": "Apple announces $110B share buyback, largest in US history..."}
    ]
  },
  "stock_data": {
    "AAPL": {
      "ticker": "AAPL",
      "current_price": 185.50,
      "daily_bars": [
        {"timestamp": "2024-10-01", "open": 174.0, "high": 175.5, "low": 173.2, "close": 175.0, "volume": 4800000},
        {"timestamp": "2024-10-02", "open": 175.1, "high": 176.8, "low": 174.5, "close": 176.3, "volume": 5100000},
        "... (daily bars for the full quarter) ...",
        {"timestamp": "2024-12-31", "open": 184.0, "high": 186.0, "low": 183.5, "close": 185.5, "volume": 5200000}
      ]
    }
  }
}
```

> **Note:** Each case's `stock_data` contains a single ticker entry. The `daily_bars` array covers the price movement from the previous decision point (or the start of the dataset) up to the current earnings date.

### Dataset directory structure

The recommended layout separates each ticker into its own sub-directory. Within each sub-directory, files are named so they sort chronologically (e.g., by year and quarter):

```
data/cases/
├── AAPL/
│   ├── 2023_Q1.json
│   ├── 2023_Q2.json
│   ├── 2023_Q3.json
│   ├── 2023_Q4.json
│   ├── 2024_Q1.json
│   ├── 2024_Q2.json
│   ├── 2024_Q3.json
│   └── 2024_Q4.json
├── GOOGL/
│   ├── 2023_Q1.json
│   └── ...
└── MSFT/
    ├── 2023_Q1.json
    └── ...
```

The case loader **recursively** searches for `.json` files and sorts them by their path relative to the dataset root. With the layout above, the simulation processes all of AAPL's quarters in order, then GOOGL's, then MSFT's — giving the agent a chronological view of each stock's history.

### Supported dataset formats

| Format                          | `dataset_path` value | Description                                                  |
|---------------------------------|----------------------|--------------------------------------------------------------|
| Directory (with sub-dirs)       | `data/cases/`        | Recursively finds all `.json` files, sorted by relative path |
| Single JSON array               | `data/cases.json`    | A JSON file containing an array of cases                     |
| JSON-lines                      | `data/cases.jsonl`   | One case per line                                            |

## Output Structure

Each run produces a self-contained output directory:

```
results/{run_name}/
├── config.yaml              # Copy of the input config (for reproducibility)
├── simulation_log.json      # Full run-level log with all episodes
├── summary.json             # Lightweight summary (per-episode P&L, trade counts)
└── episodes/
    ├── ep_000/
    │   ├── episode_log.json # Full episode audit trail
    │   ├── trades.json      # Flat list of executed trades
    │   └── reasoning/
    │       ├── case_000.txt # Agent reasoning trace for decision point 0
    │       └── case_001.txt # Agent reasoning trace for decision point 1
    └── ep_001/
        └── ...
```

### Key output files

- **`simulation_log.json`** — Complete run log with embedded config and all episode data. Sufficient to reconstruct the entire experiment.
- **`summary.json`** — Quick overview: initial/final cash, positions, and trade counts per episode.
- **`episode_log.json`** — Per-decision-point audit trail including portfolio before/after, the extracted decision, execution result, and timing.
- **`trades.json`** — Flat list of all executed trades in the episode with prices and quantities.
- **`reasoning/`** — Raw agent output / reasoning traces for each decision point (useful for qualitative analysis).

## Execution Semantics

The broker enforces several rules during trade execution:

- **All-or-nothing**: If any order in a decision fails validation, the entire decision is rejected. No partial fills.
- **Sells before buys**: Sell orders execute first so that freed cash is available for subsequent buys.
- **Ticker validation**: Orders must reference tickers present in both the case data and the simulation's ticker universe.
- **Position checks**: Cannot sell more shares than currently held.
- **Cash checks**: Cannot buy more than available cash allows (including proceeds from sells in the same decision).
- **Retry on rejection**: If a decision is rejected, the agent receives the rejection reason and may revise and resubmit (up to `max_retries` times). If all retries are exhausted, the agent defaults to hold.