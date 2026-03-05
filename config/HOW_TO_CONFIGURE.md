# How to Configure Agents, Scenarios, and Prompts

This guide covers everything you need to create custom agent configs, scenarios, and prompts for the multi-agent debate simulation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Agent Configuration](#agent-configuration)
4. [Scenario Configuration](#scenario-configuration)
5. [How Agents and Scenarios Merge](#how-agents-and-scenarios-merge)
6. [Prompt System](#prompt-system)
7. [Sector Constraints](#sector-constraints)
8. [Allocation Constraints](#allocation-constraints)
9. [Snapshot Generation](#snapshot-generation)
10. [PID Controller](#pid-controller)
11. [Logging](#logging)
12. [Parallel Execution and Rate Limiting](#parallel-execution-and-rate-limiting)
13. [Validation Rules Reference](#validation-rules-reference)
14. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

A simulation run is configured by two YAML files:

- **Agent config** (`config/agents/*.yaml`) -- defines the LLM, debate structure, prompt style, logging, and PID settings. This is the "how" of the debate.
- **Scenario config** (`config/scenarios/*.yaml`) -- defines the investment universe: which quarter, which tickers, sector constraints, and allocation rules. This is the "what" of the debate.

```
python run_simulation.py \
  --agents config/agents/debate_diverse_agents.yaml \
  --scenario config/scenarios/2022Q1_inflation_shock.yaml
```

The scenario overlays onto the agent config via deep merge (scenario wins on conflicts). The merged result is validated as a `SimulationConfig`.

---

## Quick Start

### Run with an existing scenario

```bash
python run_simulation.py \
  --agents config/agents/debate_diverse_agents.yaml \
  --scenario config/scenarios/sector_constrained.yaml
```

### Run with agent config only (no scenario overlay)

```bash
python run_simulation.py --agents config/agents/debate_diverse_agents.yaml
```

### Create a minimal custom scenario

```yaml
# config/scenarios/my_scenario.yaml
invest_quarter: "2025Q1"

tickers:
  - AAPL
  - NVDA
  - JPM
  - XOM
  - UNH

allocation_constraints:
  max_weight: 0.30
  min_holdings: 4
  fully_invested: true
  max_tickers: 10
```

### Create a minimal custom agent config

```yaml
# config/agents/my_agents.yaml
case_format: "memo"
dataset_path: "data-pipeline/final_snapshots"
memo_format: "text"
invest_quarter: "2025Q1"

tickers:
  - AAPL
  - NVDA
  - JPM

allocation_constraints:
  max_weight: 0.40
  min_holdings: 3
  fully_invested: true
  max_tickers: 10

num_episodes: 1

agent:
  agent_system: "multi_agent_debate"
  llm_provider: "openai"
  llm_model: "gpt-4o-mini"
  temperature: 0.3
  max_retries: 3
  max_rounds: 5
  prompt_profile: "diverse_agents"
  logging_mode: "standard"

broker:
  initial_cash: 100000.0
```

---

## Agent Configuration

The agent config lives under `config/agents/` and contains the full `SimulationConfig`. Every field is documented below.

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | string | **required** | Path to data. Use `"data-pipeline/final_snapshots"` for memo mode |
| `case_format` | string | `"memo"` | Always `"memo"` for current pipeline |
| `memo_format` | `"text"` or `"json"` | `"text"` | Memo payload format |
| `invest_quarter` | string | **required** | Target quarter, e.g. `"2025Q1"` |
| `tickers` | list | **required** | Ticker universe (non-empty) |
| `num_episodes` | int | `1` | Number of simulation episodes |

### The `agent:` Block

This configures the debate system itself.

#### LLM Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_system` | string | **required** | Use `"multi_agent_debate"` |
| `llm_provider` | string | **required** | `"openai"` or `"anthropic"` |
| `llm_model` | string | **required** | Model name (e.g. `"gpt-4o-mini"`, `"claude-sonnet-4-20250514"`) |
| `temperature` | float | `0.7` | Sampling temperature (0.0 - 2.0) |
| `max_retries` | int | `3` | Max retries per case |

#### Debate Structure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_rounds` | int | `1` | Number of critique-revision cycles. 5-7 recommended with PID |
| `debate_roles` | list | `null` | Agent roles. Defaults to `["macro", "value", "risk", "technical"]` |
| `agreeableness` | float | `0.3` | Sycophancy knob: 0.0 = confrontational, 1.0 = agreeable |
| `enable_adversarial` | bool | `false` | Add a devil's advocate agent |

Available roles: `macro`, `value`, `risk`, `technical`, `sentiment`, `devils_advocate`

#### Prompt Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_profile` | string | `""` | Profile name: `"default"`, `"diverse_agents"`, `"minimal"`, `"no_scaffold"` |
| `prompt_file_overrides` | dict | `null` | Override specific prompt .txt files (see [Prompt System](#prompt-system)) |
| `use_system_causal_contract` | bool | `false` | Move causal scaffolding to shared system contract |
| `system_prompt_block_order` | list | `null` | Custom order of system prompt blocks |
| `user_prompt_section_order` | list | `null` | Custom order of user prompt sections |

#### Parallel Execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parallel_agents` | bool | `true` | Run agents in parallel |
| `no_rate_limit` | bool | `false` | Disable stagger (fire all calls at once) |
| `llm_stagger_ms` | int | `200` | Milliseconds between parallel LLM call starts |
| `max_concurrent_llm` | int | `0` | Max concurrent LLM calls (0 = unlimited) |

#### PID Controller

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pid_enabled` | bool | `false` | Enable PID-based debate quality regulation |
| `pid_kp` | float | `0.15` | Proportional gain |
| `pid_ki` | float | `0.01` | Integral gain |
| `pid_kd` | float | `0.03` | Derivative gain |
| `pid_rho_star` | float | `0.8` | Target reasonableness score (0.0 - 1.0) |
| `pid_initial_beta` | float | `0.5` | Initial contentiousness (0.0 - 1.0) |
| `pid_epsilon` | float | `0.001` | JS divergence convergence tolerance |
| `convergence_window` | int | `2` | Consecutive stable rounds before early stop |
| `delta_rho` | float | `0.02` | Rho-bar plateau tolerance |

#### Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logging_mode` | string | `"off"` | `"off"`, `"standard"`, or `"debug"` |
| `pid_log_metrics` | bool | `true` | Log PID scalar metrics each round |
| `pid_log_llm_calls` | bool | `false` | Log full CRIT LLM prompts/responses (verbose) |
| `log_rendered_prompts` | bool | `false` | Log full prompts via `debate.prompts` logger |
| `log_prompt_manifest` | bool | `false` | Log prompt file names per round |
| `experiment_name` | string | `null` | Custom experiment name for logging directory |
| `console_display` | bool | `true` | Show Rich-formatted terminal output |

### The `broker:` Block

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_cash` | float | `100000.0` | Starting cash balance |

### Complete Agent Config Example

```yaml
# config/agents/my_custom_agents.yaml
case_format: "memo"
dataset_path: "data-pipeline/final_snapshots"
memo_format: "text"
invest_quarter: "2025Q1"

tickers:
  - TSLA
  - NFLX
  - JPM
  - GS
  - BAC
  - UNH
  - XOM
  - CAT
  - DAL
  - COST
  - WMT

allocation_constraints:
  max_weight: 0.20
  min_holdings: 5
  fully_invested: true
  max_tickers: 20

num_episodes: 1

agent:
  agent_system: "multi_agent_debate"
  llm_provider: "openai"
  llm_model: "gpt-4o-mini"
  temperature: 0.3
  max_retries: 3
  max_rounds: 7

  # Prompt style
  prompt_profile: "diverse_agents"
  prompt_file_overrides:
    role_macro: "roles/macro_diverse.txt"
    role_value: "roles/value_diverse.txt"
    role_technical: "roles/technical_diverse.txt"
    role_risk: "roles/risk_diverse.txt"
    output_allocation: "output_format/allocation_output_diverse.txt"

  agreeableness: 0.2

  # PID controller
  pid_enabled: true
  pid_kp: 0.15
  pid_ki: 0.01
  pid_kd: 0.03
  pid_rho_star: 0.8
  pid_initial_beta: 0.5
  pid_epsilon: 0.001

  # Logging
  logging_mode: "debug"
  pid_log_metrics: true
  pid_log_llm_calls: false
  log_rendered_prompts: true
  log_prompt_manifest: true

  # Parallel execution
  parallel_agents: true
  llm_stagger_ms: 500
  max_concurrent_llm: 2

broker:
  initial_cash: 100000.0
```

---

## Scenario Configuration

Scenarios live under `config/scenarios/` and define only the fields that vary per experiment. Everything else comes from the agent config.

### Scenario Fields

| Field | Required | Description |
|-------|----------|-------------|
| `invest_quarter` | yes | Target quarter (e.g. `"2022Q2"`) |
| `tickers` | yes | Ticker universe for this scenario |
| `allocation_constraints` | yes | Allocation rules (see [Allocation Constraints](#allocation-constraints)) |
| `sectors` | no | Ticker-to-sector mapping |
| `sector_limits` | no | Per-sector min/max bounds (requires `sectors`) |
| `agent_sector_permissions` | no | Per-role allowed sectors (requires `sectors`) |
| `max_sector_weight` | no | Blanket sector cap (requires `sectors`) |

### Simple Scenario (no sectors)

```yaml
# config/scenarios/my_simple_scenario.yaml
invest_quarter: "2025Q1"

tickers:
  - AAPL
  - NVDA
  - JPM
  - GS
  - XOM
  - UNH

allocation_constraints:
  max_weight: 0.25
  min_holdings: 4
  fully_invested: true
  max_tickers: 10
```

### Sector-Constrained Scenario

```yaml
# config/scenarios/my_sector_scenario.yaml
invest_quarter: "2022Q2"

tickers:
  - AAPL
  - NVDA
  - JPM
  - GS
  - XOM
  - CVX
  - WMT
  - COST
  - AMZN
  - UNH
  - CAT

sectors:
  tech:
    - AAPL
    - NVDA
  financials:
    - JPM
    - GS
  energy:
    - XOM
    - CVX
  defensive:
    - WMT
    - COST
  consumer:
    - AMZN
  healthcare:
    - UNH
  industrials:
    - CAT

# Per-sector exposure bounds (applied to judge's final allocation)
sector_limits:
  tech:
    min: 0.10
    max: 0.35
  energy:
    min: 0.05
    max: 0.25
  financials:
    min: 0.05
    max: 0.25
  defensive:
    min: 0.05
    max: 0.25
  consumer:
    min: 0.05
    max: 0.25
  healthcare:
    min: 0.05
    max: 0.20
  industrials:
    min: 0.05
    max: 0.20

# Which sectors each agent role can allocate to
# Use ["*"] for unrestricted access
agent_sector_permissions:
  macro:
    - energy
    - financials
    - industrials
    - tech
  value:
    - financials
    - industrials
    - consumer
    - energy
  technical:
    - "*"
  risk:
    - defensive
    - healthcare
    - financials

# No single sector may exceed 40%
max_sector_weight: 0.40

allocation_constraints:
  max_weight: 0.20
  min_holdings: 5
  fully_invested: true
  max_tickers: 25
```

### Historical Scenario (pre-2025 data)

For historical scenarios, you need upstream pipeline data (macro + asset details) for the prior quarter. The `invest_quarter` determines what data agents see:

- `invest_quarter: "2022Q2"` means agents see Q1 2022 data and allocate for Q2 2022
- The snapshot builder generates snapshots for both the invest quarter and the prior quarter

If upstream data is missing, the simulation will fail with a `FileNotFoundError` listing exactly what's needed. Generate it with:

```bash
cd data-pipeline
python macro/macro_quarter_builder.py --start 2022Q1 --end 2022Q1
python quarterly_asset_details/asset_quarter_builder.py \
  --start 2022Q1 --end 2022Q1 \
  --tickers AAPL,NVDA,JPM,GS,XOM
```

---

## How Agents and Scenarios Merge

When you pass both `--agents` and `--scenario`, the two YAML files are deep-merged:

```
base (agents) + overlay (scenario) = merged config
```

**Merge rules:**
- Scenario values **override** base values at every key
- For nested dicts (like `allocation_constraints`, `agent`), merge is recursive: scenario keys override matching base keys, but base keys not in the scenario are preserved
- For non-dict values (like `tickers`, `invest_quarter`), scenario completely replaces base

**Example:** Agent config has `tickers: [TSLA, NFLX, JPM]` and scenario has `tickers: [AAPL, NVDA]`. Result: `tickers: [AAPL, NVDA]` (scenario wins entirely -- no concatenation).

**What typically comes from where:**

| From agent config | From scenario |
|---|---|
| `agent:` (LLM, prompts, PID, logging) | `invest_quarter` |
| `broker:` | `tickers` |
| `dataset_path` | `allocation_constraints` |
| `case_format`, `memo_format` | `sectors`, `sector_limits` |
| `num_episodes` | `agent_sector_permissions` |
| | `max_sector_weight` |

### CLI Overrides

These flags override the merged config:

```bash
python run_simulation.py \
  --agents config/agents/my_agents.yaml \
  --scenario config/scenarios/my_scenario.yaml \
  --logging-mode debug \      # override logging_mode
  --no-parallel \             # set parallel_agents=false
  --no-rate-limit \           # set no_rate_limit=true
  --stagger-ms 1000 \         # set llm_stagger_ms
  --no-display                # disable Rich terminal output
```

---

## Prompt System

The prompt system assembles system and user prompts from composable text blocks. Everything lives under `multi_agent/prompts/`.

### Directory Structure

```
multi_agent/prompts/
    profiles/            # What blocks/sections to include
        default.yaml
        diverse_agents.yaml
        minimal.yaml
        no_scaffold.yaml
    roles/               # Role identity prompts
        macro.txt              macro_slim.txt         macro_diverse.txt
        value.txt              value_slim.txt         value_diverse.txt
        risk.txt               risk_slim.txt          risk_diverse.txt
        technical.txt          technical_slim.txt     technical_diverse.txt
        sentiment.txt          sentiment_slim.txt
        devils_advocate.txt    devils_advocate_slim.txt
    phases/              # Phase-specific user prompt templates
        proposal_allocation.txt
        critique_allocation.txt
        revision_allocation.txt
        judge_allocation.txt
    output_format/       # Output format instructions
        allocation_output_instructions.txt
        allocation_output_diverse.txt
    tone/                # PID beta-driven tone blocks
        critique_adversarial.txt    critique_balanced.txt    critique_collaborative.txt
        revise_adversarial.txt      revise_balanced.txt      revise_collaborative.txt
    scaffolding/         # Causal reasoning scaffolding
        causal_claim_format.txt
        forced_uncertainty.txt
        trap_awareness.txt
    system_contract/     # Shared system-level causal contract
        system_causal_contract.txt
```

### Prompt Profiles

A profile defines **which** blocks appear in the system prompt and **which** sections appear in the user prompt. Set via `prompt_profile` in the agent config.

**`default`** -- Full scaffolding with causal contract and reasoning aids:

```yaml
system_blocks:      [causal_contract, role_system, phase_preamble, tone]
user_sections:      [preamble, context, agent_data, task, scaffolding, output_format]
```

**`diverse_agents`** -- No shared scaffolding (reasoning is defined per-role in the role prompts):

```yaml
system_blocks:      [role_system, phase_preamble, tone]
user_sections:      [preamble, context, agent_data, task, output_format]
```

**`minimal`** -- Bare minimum:

```yaml
system_blocks:      [role_system, phase_preamble]
user_sections:      [context, task, output_allocation]
```

**`no_scaffold`** -- Keeps tone but removes scaffolding:

```yaml
system_blocks:      [role_system, phase_preamble, tone]
user_sections:      [preamble, context, agent_data, task, output_allocation]
```

### Role Prompt Variants

Each role has up to 3 variants:

| Variant | File | Use When |
|---------|------|----------|
| Standard | `roles/macro.txt` | `prompt_profile: "default"` with inline scaffolding |
| Slim | `roles/macro_slim.txt` | `use_system_causal_contract: true` (scaffolding is in shared contract) |
| Diverse | `roles/macro_diverse.txt` | `prompt_profile: "diverse_agents"` (self-contained reasoning methodology) |

**Standard** role prompts define: Analytical Domain, How to Reason (Step by Step), Evidence Citation (MANDATORY), Common Reasoning Traps.

**Diverse** role prompts define: Primary Decision Lens, Evidence Priorities, Decision Objective, Conviction Ranking, Portfolio Differentiation, How You Critique Others, Active Critique Requirement, Anti-Convergence Rule, Disagreement Tracking, What Would Change Your Mind.

### Prompt File Overrides

Use `prompt_file_overrides` to swap specific .txt files without changing the profile. Keys and their default files:

```yaml
prompt_file_overrides:
  # Role prompts (one per agent role)
  role_macro: "roles/macro.txt"
  role_value: "roles/value.txt"
  role_technical: "roles/technical.txt"
  role_risk: "roles/risk.txt"
  role_sentiment: "roles/sentiment.txt"
  role_devils_advocate: "roles/devils_advocate.txt"

  # Output format
  output_allocation: "output_format/allocation_output_instructions.txt"

  # Phase templates
  proposal_template: "phases/proposal_allocation.txt"
  critique_template: "phases/critique_allocation.txt"
  revision_template: "phases/revision_allocation.txt"
  judge_template: "phases/judge_allocation.txt"

  # Tone blocks (selected by PID beta)
  tone_critique_collaborative: "tone/critique_collaborative.txt"
  tone_critique_balanced: "tone/critique_balanced.txt"
  tone_critique_adversarial: "tone/critique_adversarial.txt"
  tone_revise_collaborative: "tone/revise_collaborative.txt"
  tone_revise_balanced: "tone/revise_balanced.txt"
  tone_revise_adversarial: "tone/revise_adversarial.txt"

  # System contract
  causal_contract: "system_contract/system_causal_contract.txt"
```

Paths are relative to `multi_agent/prompts/`.

### How Prompts Are Assembled

1. The **profile** decides which blocks/sections are included
2. `prompt_file_overrides` decides which .txt file loads for each named block
3. The **system prompt** is assembled by concatenating blocks in order: `[causal_contract, role_system, phase_preamble, tone]`
4. The **user prompt** is assembled by extracting `---SECTION: name---` delimited sections from the phase template and including only the sections listed in the profile's `user_sections`

**Tone injection:** Only during `critique` and `revise` phases (never `propose` or `judge`). The tone block is always placed **last** in the system prompt for maximum LLM recency-bias attention.

**Beta-to-tone mapping (from PID or static agreeableness):**

| Beta Range | Tone Bucket | Behavior |
|------------|-------------|----------|
| `>= 0.67` | adversarial | Challenge claims aggressively |
| `0.33 - 0.67` | balanced | Fair but rigorous scrutiny |
| `< 0.33` | collaborative | Look for common ground |

Static agreeableness is converted: `beta = 1.0 - agreeableness`. So `agreeableness: 0.2` gives `beta = 0.8` (adversarial).

### Writing Custom Role Prompts

Create a new .txt file in `multi_agent/prompts/roles/` and reference it via `prompt_file_overrides`:

```
# multi_agent/prompts/roles/my_quant_role.txt

You are a QUANTITATIVE ANALYST specializing in statistical arbitrage
and factor-based investing.

## Primary Decision Lens
You evaluate investments through quantitative metrics: factor exposures,
statistical significance of returns, mean-reversion signals, and
cross-sectional momentum.

## Evidence Priorities
1. Price momentum (3m, 6m, 12m returns)
2. Valuation ratios vs sector median
3. Earnings revision momentum
4. Volatility-adjusted returns

## Decision Objective
Maximize risk-adjusted returns using systematic, rules-based analysis.
Minimize exposure to narrative-driven reasoning.
```

Then in your agent config:

```yaml
agent:
  debate_roles: ["macro", "quant", "risk", "technical"]
  prompt_file_overrides:
    role_quant: "roles/my_quant_role.txt"
```

### Writing Custom Phase Templates

Phase templates use Jinja2 and `---SECTION: name---` delimiters:

```
---SECTION: context---
## Market Context
{{ context }}

---SECTION: task---
## Your Task
Propose a portfolio allocation for the following tickers.

---SECTION: output_format---
{{ allocation_output_instructions }}
```

Available template variables depend on the phase:

| Phase | Variables |
|-------|-----------|
| Proposal | `context`, `causal_claim_format`, `forced_uncertainty`, `trap_awareness`, `allocation_output_instructions` |
| Critique | `role`, `context`, `my_proposal`, `others_text` |
| Revision | `role`, `context`, `my_proposal`, `critiques_text`, `causal_claim_format`, `forced_uncertainty` |
| Judge | `context`, `revisions_text`, `all_critiques_text`, `disagreements_section`, `causal_claim_format` |

---

## Sector Constraints

Sector constraints restrict how the final allocation distributes weight across industry sectors.

### Defining Sectors

Every ticker must belong to exactly one sector. All tickers in your `tickers` list must appear in the sector mapping, and vice versa.

```yaml
sectors:
  tech:
    - AAPL
    - NVDA
  energy:
    - XOM
    - CVX
  financials:
    - JPM
    - GS
```

### Per-Sector Limits

Enforced on the judge's final allocation:

```yaml
sector_limits:
  tech:
    min: 0.10    # Tech must be at least 10%
    max: 0.35    # Tech can't exceed 35%
  energy:
    min: 0.05
    max: 0.25
```

**Constraints:**
- Sum of all sector minimums must be <= 1.0
- Sum of all sector maximums must be >= 1.0
- Sectors not listed are unconstrained

### Agent Sector Permissions

Restrict which sectors each agent role can allocate to during the debate. The judge ignores these (it sees all sectors).

```yaml
agent_sector_permissions:
  macro:
    - energy
    - financials
    - tech
  value:
    - financials
    - consumer
    - energy
  technical:
    - "*"           # Can allocate to any sector
  risk:
    - defensive
    - healthcare
```

Valid role names: `macro`, `value`, `risk`, `technical`, `sentiment`, `devils_advocate`

### Blanket Sector Cap

A single number that caps any sector's weight. Composes with `sector_limits` -- the stricter bound wins:

```yaml
max_sector_weight: 0.40
```

If `sector_limits` says tech max is 0.60 and `max_sector_weight` is 0.40, tech is capped at 0.40.

---

## Allocation Constraints

These rules are enforced on the final portfolio output.

```yaml
allocation_constraints:
  max_weight: 0.20       # No single ticker can exceed 20%
  min_holdings: 5        # At least 5 tickers must have weight > 0
  fully_invested: true   # Weights must sum to 1.0
  max_tickers: 25        # Upper bound on universe size (validation only)
```

### Feasibility Rule

The config validates that the constraints are mathematically satisfiable:

```
max_weight * min_holdings >= 1.0
```

If this fails, you get:
```
Impossible allocation constraints: max_weight (0.30) * min_holdings (3) = 0.90 < 1.0
```

**Fix:** Either increase `max_weight` or decrease `min_holdings`.

| max_weight | min_holdings | Product | Valid? |
|------------|-------------|---------|--------|
| 0.20 | 5 | 1.00 | yes |
| 0.25 | 4 | 1.00 | yes |
| 0.30 | 3 | 0.90 | **no** |
| 0.35 | 3 | 1.05 | yes |
| 0.40 | 3 | 1.20 | yes |

### Max Tickers

This is a validation-only check. If your `tickers` list has more entries than `max_tickers`, the config fails:

```
Too many tickers (22) for allocation mode (max 20)
```

**Fix:** Increase `max_tickers` to accommodate your ticker universe.

---

## Snapshot Generation

### How It Works

When `dataset_path` contains `"final_snapshots"` (which all agent configs under `config/agents/` use), the simulation **automatically generates snapshots** before running:

1. Parses `invest_quarter` and `tickers` from the merged config
2. Calls `data-pipeline/final_snapshots/snapshot_builder.py` as a subprocess
3. The builder generates snapshot JSON and memo files for the invest quarter and its prior quarter
4. The simulation then reads these files as agent context

**Snapshots are regenerated every run.** They are overwritten with the current scenario's tickers, so you don't need to manage them manually.

### What's In a Snapshot

Each snapshot (`snapshot_{year}_{quarter}.json`) merges:

- **EDGAR filing summaries** -- 10-Q/10-K analysis, 8-K event filings
- **Sentiment data** -- News sentiment (optional, requires paid Finnhub API for pre-2025)
- **Macro economic data** -- Fed rates, inflation, VIX, yield curves, etc.
- **Asset feature details** -- Price, fundamentals, valuation ratios, momentum, volatility

### Missing Upstream Data

If upstream data hasn't been generated for your scenario's quarter, you'll get:

```
FileNotFoundError: Missing upstream data required for snapshot generation.

Required files not found:
  macro/data/macro_2022_Q3.json
  quarterly_asset_details/data/AAPL/2022_Q3.json
  ...

Run the data pipeline to generate missing data:
  cd data-pipeline
  python quarterly_asset_details/asset_quarter_builder.py --start 2022Q2 --end 2022Q3 --tickers AAPL,NVDA,...
  python macro/macro_quarter_builder.py --start 2022Q2 --end 2022Q3
```

Run the exact commands from the error message. Macro and asset data are fetched from public APIs (Yahoo Finance, yfinance fundamentals) and don't require API keys. EDGAR and sentiment data are optional (warnings only).

### Generating Data for All Scenarios

Use the bulk pipeline script:

```bash
python data-pipeline/run_historical_stages_1_4.py
```

Or validate all scenarios have the data they need:

```bash
python data-pipeline/validate_scenarios.py
python data-pipeline/validate_scenarios.py --verbose
python data-pipeline/validate_scenarios.py --scenario 2022Q1_inflation_shock
```

---

## PID Controller

The PID controller dynamically adjusts debate contentiousness based on measured reasoning quality (CRIT scores).

### How It Works

Each round:
1. Agents propose/critique/revise with the current tone (set by beta)
2. CRIT evaluates each agent's reasoning quality, producing a score (rho)
3. PID computes error: `e = rho_star - rho_bar` (target vs actual quality)
4. PID updates beta (contentiousness) for the next round
5. Beta selects the tone bucket (adversarial/balanced/collaborative)

**High beta** (adversarial tone) pushes agents to challenge each other harder.
**Low beta** (collaborative tone) lets agents converge.

### Recommended Settings

```yaml
agent:
  pid_enabled: true
  max_rounds: 7          # Give PID room to adjust
  pid_kp: 0.15           # Proportional: react to current error
  pid_ki: 0.01           # Integral: correct persistent bias
  pid_kd: 0.03           # Derivative: dampen oscillation
  pid_rho_star: 0.8      # Target quality score
  pid_initial_beta: 0.5  # Start in balanced zone
  pid_epsilon: 0.001     # Convergence threshold (JS divergence)
  convergence_window: 2  # Consecutive stable rounds to stop early
  delta_rho: 0.02        # Quality plateau tolerance
```

### Tuning Tips

- **Higher `pid_kp`**: Stronger reaction to quality deviations. Can cause oscillation.
- **Higher `pid_ki`**: Corrects persistent under/over-performance. Can cause overshoot.
- **Higher `pid_kd`**: Dampens oscillation. Can slow convergence.
- **Higher `pid_rho_star`**: Demands higher quality reasoning. May keep debate adversarial longer.
- **More `max_rounds`**: Gives PID more iterations to converge. 5-7 is the sweet spot.

### Early Termination

The debate stops early when ALL three conditions hold for `convergence_window` consecutive rounds:

1. Quality is high and diversity is low (quadrant = "converged")
2. JS divergence < `pid_epsilon` (agents agree on allocations)
3. Quality plateau: `|rho_bar(t) - rho_bar(t-1)| < delta_rho`

---

## Logging

### Logging Modes

Set via `logging_mode` in the agent config or `--logging-mode` CLI flag.

**`"off"` (default):** No debate log files written.

**`"standard"`:** Writes to `logging/runs/<experiment>/run_<timestamp>/`:

```
manifest.json                        # Run metadata
pid_config.json                      # PID settings
prompt_manifest.json                 # Prompt files used
shared_context/memo.txt              # Agent context memo
rounds/round_001/
    round_state.json                 # Round metadata
    proposals/<agent>/
        response.txt                 # Agent's proposal text
        portfolio.json               # Proposed allocation
    critiques/<agent>/
        response.json                # Agent's critiques
    revisions/<agent>/
        response.txt                 # Revised proposal
        portfolio.json               # Revised allocation
    CRIT/<agent>/
        prompt.txt                   # CRIT evaluation prompt
        response.txt                 # CRIT score response
    metrics/
        crit_scores.json             # Per-agent CRIT scores
        js_divergence.json           # Portfolio agreement metric
        evidence_overlap.json        # Cross-agent evidence usage
        pid_state.json               # PID controller state
final/
    final_portfolio.json             # Judge's final allocation
    judge_response.txt               # Judge's reasoning
    debate_diagnostic.txt            # 9-step diagnostic scaffold
```

**`"debug"`:** Everything in standard **plus** `prompt.txt` files showing the full system + user prompts sent to each agent in every phase (proposals, critiques, revisions).

### Additional Logging Flags

```yaml
agent:
  pid_log_metrics: true       # PID scalar metrics each round (to console + files)
  pid_log_llm_calls: false    # Full CRIT LLM prompts/responses (very verbose)
  log_rendered_prompts: false  # Full prompts via debate.prompts Python logger
  log_prompt_manifest: false   # Which prompt .txt files were used per round
```

### Viewing Logs

```bash
# Find the latest run
ls -lt logging/runs/*/

# View the final allocation
cat logging/runs/<experiment>/run_<timestamp>/final/final_portfolio.json

# View CRIT scores over rounds
cat logging/runs/<experiment>/run_<timestamp>/rounds/round_*/metrics/crit_scores.json

# View the debate diagnostic
cat logging/runs/<experiment>/run_<timestamp>/final/debate_diagnostic.txt
```

---

## Parallel Execution and Rate Limiting

### How Parallel Execution Works

When `parallel_agents: true`, all agents in each phase (propose, critique, revise) execute their LLM calls simultaneously using LangGraph fan-out. With 4 agents, this means 4 concurrent API calls per phase.

### Stagger

The stagger mechanism serializes call **starts** to avoid hitting rate limits:

```yaml
agent:
  parallel_agents: true
  llm_stagger_ms: 500     # 500ms between call starts
```

With 4 agents and 500ms stagger, calls start at t=0, t=500ms, t=1000ms, t=1500ms. They still overlap during execution (each call takes several seconds), so total phase time only increases by ~1500ms vs sequential.

### Concurrency Limit

```yaml
agent:
  max_concurrent_llm: 2   # Max 2 simultaneous LLM calls
```

When set to 2 with 4 agents: agents 1+2 start, agent 3 waits until one finishes, then agent 4 waits. This is a hard cap via semaphore.

### Rate Limit Handling

The system has built-in retry logic with exponential backoff:

- **Max retries:** 6
- **Rate limit (429) errors:** Uses the API's `retry-after` header if provided, plus 0.5s buffer. Fallback: exponential backoff (2s, 4s, 8s, 16s, 32s, 64s).
- **Other errors:** Exponential backoff (1s, 2s, 4s, 8s, 16s, 32s).
- After all retries exhausted: returns empty JSON `"{}"`.

### If You're Getting Rate Limited

**Option 1: Increase stagger**

```yaml
agent:
  llm_stagger_ms: 1000    # 1 second between calls
```

**Option 2: Limit concurrency**

```yaml
agent:
  max_concurrent_llm: 2   # Only 2 calls at once
```

**Option 3: Both (recommended for strict rate limits)**

```yaml
agent:
  llm_stagger_ms: 500
  max_concurrent_llm: 2
```

**Option 4: Disable parallelism entirely**

```yaml
agent:
  parallel_agents: false   # Sequential execution
```

Or via CLI: `--no-parallel`

**Option 5: Disable stagger (if you have high rate limits)**

```yaml
agent:
  no_rate_limit: true      # Fire all calls at once
```

Or via CLI: `--no-rate-limit`

---

## Validation Rules Reference

The config is validated by Pydantic on construction. Here's every validation rule and its error message.

### Required Fields

| Rule | Error |
|------|-------|
| `tickers` must not be empty | `tickers must not be empty.` |
| `invest_quarter` must be set | `invest_quarter is required.` |

### Allocation Constraints

| Rule | Error |
|------|-------|
| `len(tickers) <= max_tickers` | `Too many tickers (N) for allocation mode (max M).` |
| `max_weight * min_holdings >= 1.0` | `Impossible allocation constraints: max_weight (X) * min_holdings (Y) = Z < 1.0.` |

### Sector Constraints

| Rule | Error |
|------|-------|
| `sector_limits` requires `sectors` | `sector_limits requires 'sectors' to be defined.` |
| `agent_sector_permissions` requires `sectors` | `agent_sector_permissions requires 'sectors' to be defined.` |
| `max_sector_weight` requires `sectors` | `max_sector_weight requires 'sectors' to be defined.` |
| Ticker in multiple sectors | `Ticker {t} appears in multiple sectors: '{s1}' and '{s2}'` |
| Ticker missing from sector map | `Tickers missing from sector mapping: [...]` |
| Sector references unknown ticker | `Sector '{s}' references unknown tickers: [...]` |
| Sum of sector mins > 1.0 | `Sum of sector minimums ({sum}) exceeds 1.0 — infeasible.` |
| Sum of sector maxes < 1.0 | `Sum of sector maximums ({sum}) is less than 1.0 — cannot reach fully invested.` |
| Unknown role in permissions | `agent_sector_permissions references unknown role: '{role}'` |
| Unknown sector in permissions | `Role '{role}' references unknown sector: '{s}'` |

Valid roles: `macro`, `value`, `risk`, `technical`, `sentiment`, `devils_advocate`

---

## Troubleshooting

### "invest_quarter is required"

Every config must have `invest_quarter` set. Add it to your scenario or agent config:

```yaml
invest_quarter: "2025Q1"
```

### "Too many tickers (22) for allocation mode (max 20)"

Your ticker list exceeds `max_tickers`. Increase it:

```yaml
allocation_constraints:
  max_tickers: 25
```

### "Impossible allocation constraints"

`max_weight * min_holdings` must be >= 1.0. See the [feasibility table](#feasibility-rule).

### "agent_sector_permissions references unknown role: 'growth'"

Only these roles are valid: `macro`, `value`, `risk`, `technical`, `sentiment`, `devils_advocate`. Check your scenario's `agent_sector_permissions` for typos or nonexistent roles.

### "Missing upstream data required for snapshot generation"

Run the exact commands from the error message. They generate macro and asset data from public APIs:

```bash
cd data-pipeline
python macro/macro_quarter_builder.py --start 2022Q1 --end 2022Q1
python quarterly_asset_details/asset_quarter_builder.py \
  --start 2022Q1 --end 2022Q1 --tickers AAPL,NVDA,JPM
```

### Rate limit errors (429)

See [Parallel Execution and Rate Limiting](#parallel-execution-and-rate-limiting). Quick fix:

```bash
python run_simulation.py --agents my.yaml --stagger-ms 1000
# or
python run_simulation.py --agents my.yaml --no-parallel
```

### "Tickers missing from sector mapping"

Every ticker in your `tickers` list must appear in exactly one sector. Check for typos or missing tickers in your `sectors` definition.

### Sentiment data warnings

Sentiment warnings are non-fatal. Sentiment data from Finnhub's free tier only covers recent news (2025+). Historical scenarios will show warnings like:

```
WARNING: sentiment/data/AAPL/2022_Q2.json — missing (optional)
```

These are safe to ignore. To get historical sentiment, you need a paid Finnhub API key.
