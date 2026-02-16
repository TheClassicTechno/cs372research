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

## Files

```
Final_project/
├── agents/
│   ├── types.ts           # Observation, Action, Claim, DebateTurn, AgentTrace
│   ├── prompts.ts         # Role prompts, debate prompts, anti-failure rules
│   ├── single-agent-runner.ts
│   ├── majority-vote-runner.ts
│   ├── debate-runner.ts   # Round 0→1→2 + Judge
│   ├── demo.ts            # 5-timestep demo
│   └── index.ts
├── traces/                # Written at runtime
├── package.json
└── README.md
```
