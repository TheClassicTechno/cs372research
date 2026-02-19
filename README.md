# CS372 Research Project: Multi-Agent Debate for Explainable Trading

Juli Huang, Alanood Alrassan, Deveen Harischandra, Theodore Wu, Veljko Skarich, Matthew Hayes  
Department of Engineering, Stanford University

---

## Watch the Debate Unfold in Real-Time

We added a **verbose mode** so you can actually see what every agent is thinking during the debate. Instead of just getting a final answer, you can watch each agent propose a trade, critique the others, revise their position, and then see how the judge synthesizes everything into a final decision. It prints right in the terminal with box-drawing formatting so it's easy to follow.

### Quick Start: Run a Debate

You don't need an API key to try it out. Mock mode uses synthetic responses so you can see the full flow without any setup:

```bash
# See the full debate with verbose output:
python -m multi_agent.demo --verbose

# Run quietly (just progress markers, no debate content):
python -m multi_agent.demo
```

If you have an OpenAI API key, you can run the real thing with live LLM responses:

```bash
# Run all 8 configurations (takes about 3-5 minutes):
python -m multi_agent.demo --live

# Run a single config with full debate output (recommended to start here):
python -m multi_agent.demo --live --verbose --config 1

# Run specific configs (e.g., compare high vs low agreeableness):
python -m multi_agent.demo --live --config 4,5
```

### What You'll See in Verbose Mode

When you run with `--verbose`, you'll see each phase of the debate printed to the terminal:

**1. Proposals** — Each agent presents their trading thesis:

```
┌─── MACRO STRATEGIST proposes ───
│ Orders: BUY 50 AAPL
│ Confidence: 75%
│ Thesis: Fed rate cuts signal bullish environment...
│ Claim [L2]: If rate cuts materialize, tech valuations expand
│ Falsifier: Inflation data reversal would invalidate this thesis
└──────────────────────────────────────────────────
```

**2. Critiques** — Agents challenge each other's reasoning:

```
┌─── RISK MANAGER critiques ───
│ → MACRO STRATEGIST: Your L2 claim ignores recent volatility spike
│   Falsifier: VIX > 25 contradicts low-risk environment assumption
│ Self-critique: May be overweighting short-term volatility
└──────────────────────────────────────────────────
```

**3. Revisions** — Agents update their positions based on the feedback they received:

```
┌─── MACRO STRATEGIST revises ───
│ Orders: BUY 30 AAPL
│ Confidence: 65%
│ Revision: Reducing position size to account for volatility concerns
└──────────────────────────────────────────────────
```

**4. Judge Decision** — The judge reads everything and writes a final audited memo, preserving any unresolved disagreements:

```
╔══════════════════════════════════════════════════
║  JUDGE FINAL DECISION
╠══════════════════════════════════════════════════
║  Orders: BUY 30 AAPL
║  Confidence: 68%
║  Consensus: Bullish macro environment supports tech exposure
║  but position sizing reflects elevated volatility concerns
║
║  Strongest objection preserved:
║  Risk Manager: Geopolitical tensions remain unresolved
╚══════════════════════════════════════════════════
```

### Demo Configurations

The demo includes 8 different configurations so you can compare how the system behaves under different settings. This is what we use for our ablation experiments:

| Config | Description | What it tests |
|--------|-------------|---------------|
| 1 | Default 4-agent debate | Baseline with Macro, Value, Risk, Technical |
| 2 | 5 agents (+Sentiment) | Does adding a sentiment agent change the outcome? |
| 3 | Adversarial mode | Devil's Advocate actively tries to break the consensus |
| 4 | High agreeableness (0.9) | Agents tend to agree quickly — tests for sycophancy |
| 5 | Low agreeableness (0.1) | Agents push back hard — tests confrontational debate |
| 6 | 3 debate rounds | More rounds of critique and revision |
| 7 | No pipeline preprocessing | Agents work with raw observations only |
| 8 | Full system, risk-off scenario | Everything enabled during a market crisis |

### Testing

We have 63 tests covering models, config, prompts, mock generators, node functions, full graph runs, config ablations, graph structure, and edge cases:

```bash
pytest multi_agent/tests/test_multi_agent.py -v
```

### What Changed (Implementation Notes)

Verbose mode was added by wiring 4 display helper functions into the LangGraph node functions in `graph.py`:

- `_verbose_proposal()` prints each agent's proposed orders, confidence, thesis, causal claims, and falsifiers
- `_verbose_critique()` prints each agent's critiques of the other agents, including what evidence would change their mind
- `_verbose_revision()` prints how each agent updated their position after hearing critiques
- `_verbose_judge()` prints the judge's final decision in a box with the audited memo and strongest preserved objection

These get called inside `propose_node`, `critique_node`, `revise_node`, and `judge_node` when `config.verbose` is true. The `--verbose` flag is parsed in `demo.py` and passed into every `DebateConfig` instance.

Non-verbose mode is completely unchanged — you just see progress markers without any debate content.
