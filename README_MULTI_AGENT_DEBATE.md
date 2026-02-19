# CS372 Multi-Agent Trading System — Agent Team

This module implements the agent interface for the CS372 trading simulator: **single-agent**, **majority vote**, and (future) **multi-agent debate** architectures. It produces auditable reasoning traces for T³/CRIT-style evaluation.

## Quick Start

```bash
# From repo root (CS 372 folder)
cd Final_project

# Run demo with MOCK mode (no API key needed)
npm run demo:mock

# Run demo with live OpenAI (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... npm run demo
```

Or from the parent directory:

```bash
npx tsx Final_project/agents/demo.ts
MOCK=1 npx tsx Final_project/agents/demo.ts
```

## Shared Types (Contract)

| Type | Purpose |
|------|---------|
| `Observation` | Input from simulator: timestamp, universe, market_state, text_context, portfolio_state, constraints |
| `Action` | Output to broker: orders, justification, confidence, claims |
| `Claim` | Machine-readable causal claim (pearl_level L1/L2/L3) for T³ scoring |
| `AgentTrace` | Auditable trace: what_i_saw, hypothesis, decision, risks_or_falsifiers |
| `DebateTurn` | For multi-agent: round, agent_id, proposal, critique, revision |

## Architecture

### 1. SingleAgentRunner
- Reads `Observation` → produces `Action`
- Logs structured trace to `./traces/single_*.json`
- Anti-failure rules: forced "what would falsify this", causal claim formatting, trap-awareness

### 2. MajorityVoteRunner
- 3 role agents (macro, value, risk) each propose `Action`
- Aggregator: vote on direction, median size, log disagreements
- Preserves strongest objection even if overruled

### 3. DebateRunner
- **Round 0**: Each agent writes proposal + causal claims
- **Round 1**: Each agent critiques at least 2 others (explicit objections, falsifiers)
- **Round 2**: Each agent revises based on critiques received
- **Judge**: Decides final trades, writes audited memo, preserves strongest objection

## Coordination

- **Simulation team**: Provide dummy Observation generator + stub broker API
- **Evals team**: Specify desired fields in `AgentTrace` and `Claims` (we ship a best guess first)

## Multi-Agent Debate Orchestrator with LangGraph — Juli

Full multi-agent debate system built in Python with LangGraph, extending Deveen's single-agent baseline into a research-grade orchestration framework for ablation experiments.

### What was built

- **6 specialized role agents** with deeply enriched prompts designed to elicit Pearl L2 (intervention) and L3 (counterfactual) causal claims, not just L1 associations:
  - **Macro Strategist** — Fed policy, inflation, yield curves, economic regime analysis
  - **Value/Fundamentals Analyst** — earnings, valuation multiples, balance sheet, intrinsic value
  - **Risk Manager** — volatility regimes, VaR, position sizing, tail risk, correlation dynamics
  - **Technical Analyst** — price action, momentum, support/resistance, volume confirmation
  - **Sentiment Analyst** (new) — news sentiment, narrative shifts, information freshness, market psychology
  - **Devil's Advocate** (new) — adversarial agent that stress-tests the emerging consensus to prevent groupthink

- **LangGraph debate orchestrator** — a directed graph with configurable structure:
  - Pipeline preprocessing nodes (NewsDigester + DataAnalyst) run in parallel before debate
  - Propose → Critique → Revise loop with configurable N rounds
  - Judge/Aggregator synthesizes final decision with audited memo and disagreement preservation
  - Conditional edges allow the critique-revision cycle to repeat for deeper deliberation

- **Sycophancy/agreeableness knob** (0.0–1.0) — a tunable parameter injected into critique prompts that controls how confrontational vs. agreeable agents are during debate, supporting RQ3 (does debate reduce failure modes like overconfidence and sycophancy):
  - 0.0–0.2: Maximally confrontational (challenges every assumption)
  - 0.2–0.4: Skeptical (demands evidence)
  - 0.4–0.6: Balanced (critiques on merit)
  - 0.6–0.8: Collaborative (finds common ground)
  - 0.8–1.0: Highly agreeable (seeks consensus)

- **Adversarial/devil's advocate mode** — when `enable_adversarial=True`, a devil's advocate agent is auto-injected into the debate roster to argue against the majority and surface hidden risks

- **Pipeline preprocessing agents** — optional NewsDigester and DataAnalyst nodes that run before the debate to produce structured intelligence (sentiment scores, momentum signals, volatility regime classification) that enriches the context for debate agents

- **Configurable `DebateConfig`** for ablation experiments — all experimental knobs in one place:
  - Which roles participate, number of debate rounds, agreeableness level
  - Toggle adversarial mode, toggle pipeline preprocessing
  - Model selection, temperature, mock mode for testing without API keys

- **Pydantic models** matching Deveen's TypeScript types for cross-language compatibility (Observation, Action, Claim, AgentTrace, etc.)

- **61 passing tests** covering models, config, prompts, mock generators, individual graph nodes, full end-to-end graph execution, configuration ablations, graph structure validation, and edge cases

- **Demo script** with 8 different configurations showcasing default debate, sentiment agent, adversarial mode, high/low agreeableness, multiple rounds, no pipeline, and a risk-off scenario

- **Comprehensive test suite** with 61 passing tests validating:
  - Pydantic model creation, serialization, and roundtrip conversions
  - Config defaults, customization, and agent role enumeration
  - Prompt generation for all 6 roles (Pearl L2/L3 requirements, falsifier checks, JSON format validation)
  - Mock generators for proposals, critiques, revisions, judge decisions, and pipeline outputs
  - Individual LangGraph node execution (propose, critique, revise, judge, pipeline, trace builder)
  - Full end-to-end graph execution with different debate configurations
  - Configuration ablations (adversarial mode, sentiment agent, multiple rounds, pipeline toggles)
  - Graph structure validation (node existence, edge correctness, conditional loops)
  - Edge cases (empty universe, single role, no text context, extreme agreeableness values)
  - All tests run in mock mode without requiring API keys

- **Research literature review** — curated `papers.txt` with relevant multi-agent systems papers:
  - **Agyn** (2026): Team-based autonomous software engineering with role specialization, peer review, and GitHub-native workflows — scored 72.2% on SWE-bench 500
  - **FullStack-Agent** (2026): Multi-agent full-stack web development with repository back-translation for training data generation, comprehensive testing across frontend/backend/database layers
  - Key architectural insights: explicit role separation, manager-coordinated workflow, iterative self-improvement via synthetic data augmentation, importance of organizational design alongside model improvements

### Quick Start (Python / Multi-Agent)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no API key needed)
python -m pytest multi_agent/tests/test_multi_agent.py -v

# Run demo in mock mode (no API key needed)
python -m multi_agent.demo

# Run demo with live OpenAI
python -m multi_agent.demo --live
```

### LangGraph Architecture

```
[START]
  |---> [news_digest]     (parallel, optional)
  |---> [data_analysis]   (parallel, optional)
  +---> [build_context]
          |
          v
        [propose]       all role agents generate proposals
          |
          v
        [critique]      all role agents cross-critique
          |
          v
        [revise]        all role agents revise based on critiques
          |
          v
        [should_continue?] --loop--> [critique]  (if rounds remain)
          |
          +--done------> [judge]    synthesize final decision
                            |
                            v
                        [build_trace]  construct AgentTrace for eval team
                            |
                            v
                          [END]
```

## Files

```
agents/                          # Deveen's TypeScript baseline
├── types.ts                     # Observation, Action, Claim, DebateTurn, AgentTrace
├── prompts.ts                   # Role prompts, debate prompts, anti-failure rules
├── single-agent-runner.ts
├── majority-vote-runner.ts
├── debate-runner.ts             # Round 0→1→2 + Judge
├── demo.ts                      # 5-timestep demo
└── index.ts

multi_agent/                     # Juli's Python + LangGraph orchestrator
├── __init__.py                  # Package exports
├── models.py                    # Pydantic models (matches TS types)
├── config.py                    # DebateConfig with all ablation knobs
├── prompts.py                   # 6 enriched role prompts + debate phase prompts
├── graph.py                     # LangGraph nodes, mocks, graph construction
├── runner.py                    # MultiAgentRunner (clean interface)
├── demo.py                      # 8-config demo script
└── tests/
    └── test_multi_agent.py      # 61 tests

traces/                          # Written at runtime
package.json                     # Node.js (TypeScript agents)
requirements.txt                 # Python (LangGraph agents)
```