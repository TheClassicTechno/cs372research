# Juli's Presentation Slides - Multi-Agent Debate Orchestrator

**Presentation Time: ~1 minute (32-40 seconds script + buffer)**

---

## Slide 1: Multi-Agent Debate Orchestrator

### Title
**Python Multi-Agent Debate Orchestrator with LangGraph**

### Visual Elements
- LangGraph workflow diagram showing the debate flow
- Icons/badges for 6 specialized agents

### Content
**6 Specialized Trading Agents:**
- 🌍 Macro Strategist — Fed policy, inflation, yield curves
- 💰 Value/Fundamentals Analyst — earnings, valuation multiples
- ⚠️ Risk Manager — volatility, VaR, position sizing
- 📈 Technical Analyst — price action, momentum, support/resistance
- 📰 Sentiment Analyst (new) — news sentiment, market psychology
- 😈 Devil's Advocate (new) — challenges consensus, prevents groupthink

**Key Feature: Pearl L2/L3 Causal Reasoning**
- Prompts elicit intervention (L2) and counterfactual (L3) claims, not just correlations

**Workflow:**
```
Propose → Critique → Revise → Judge
```

**Validation:**
✅ 61 passing tests across 9 test suites:
- **TestModels** (8 tests): Pydantic model creation & serialization
- **TestConfig** (4 tests): Configuration defaults & customization  
- **TestPrompts** (10 tests): Prompt generation & Pearl L2/L3 requirements
- **TestMockGenerators** (6 tests): Mock data for proposals, critiques, revisions
- **TestNodes** (12 tests): Individual LangGraph node execution
- **TestFullGraph** (6 tests): End-to-end graph execution
- **TestConfigAblations** (5 tests): Different debate configurations
- **TestGraphStructure** (4 tests): Graph validation & edge correctness
- **TestEdgeCases** (6 tests): Edge cases (empty universe, single role, etc.)

---

## Slide 2: Key Innovation - Agreeableness Knob

### Title
**Tunable Agent Behavior for Ablation Experiments**

### Visual Elements
- Horizontal scale/slider showing 0.0 to 1.0
- Example critique outputs at different settings

### Content

**Agreeableness Scale (0.0 - 1.0):**

| Range | Behavior | Description |
|-------|----------|-------------|
| 0.0-0.2 | 🔥 Confrontational | Challenges every assumption |
| 0.2-0.4 | 🤔 Skeptical | Demands evidence |
| 0.4-0.6 | ⚖️ Balanced | Critiques on merit |
| 0.6-0.8 | 🤝 Collaborative | Finds common ground |
| 0.8-1.0 | 🎯 Agreeable | Seeks consensus |

**Research Question (RQ3):**
Does debate reduce failure modes like overconfidence and sycophancy?

**Why it matters:**
Enables controlled ablation experiments to test debate effectiveness

---

## Slide 3 (OPTIONAL - Only if time permits)

### Title
**Research-Grade Platform**

### Content
- **Mock Mode:** Rapid iteration without API costs
- **Configurable:** Debate rounds, agent roles, pipeline preprocessing
- **Research-Backed:** Architecture insights from Agyn & FullStack-Agent papers
- **Cross-Compatible:** Pydantic models match TypeScript baseline
- **8 Demo Configurations:** Default, sentiment agent, adversarial mode, high/low agreeableness, multiple rounds
- **Judge with Audited Memo:** Final decision includes disagreement preservation for research transparency

---

## Speaking Script (81 words, ~35 seconds)

> "I built a Python multi-agent debate orchestrator using LangGraph with **6 specialized trading agents**. The key innovation is a tunable **'agreeableness knob'** from 0 to 1 that controls how confrontational versus collaborative agents are during debate - this lets us test whether debate actually reduces failure modes like overconfidence. The system follows a **propose-critique-revise-judge workflow** with configurable rounds, and includes **61 passing tests** across 9 test suites validating everything from individual nodes to full end-to-end execution. This gives us a research-grade platform for ablation experiments."

---

## Delivery Tips

1. **Slides 1-2 are ESSENTIAL** - use these for your ~1 minute
2. **Slide 3 is OPTIONAL** - only if the team has extra time
3. **Emphasize bold terms** in your script for impact
4. **Point to visuals** - the agreeableness scale and workflow diagram
5. **Practice transition phrases:**
   - "The key innovation here is..."
   - "This enables us to..."
   - "What makes this research-grade is..."

---

## Key Talking Points (If Q&A)

**Q: Why 6 agents specifically?**
- A: Each represents a distinct trading perspective: macro economics, fundamental analysis, risk management, technical patterns, sentiment analysis, plus a devil's advocate to prevent groupthink.

**Q: What does the agreeableness knob actually do?**
- A: It modifies the critique prompts to make agents more confrontational or collaborative, letting us test whether debate structure reduces overconfidence compared to just agreeable consensus-seeking.

**Q: How does this integrate with the team's work?**
- A: It extends the TypeScript baseline into a research platform for ablation experiments. We maintain the same Observation/Action/Claim types for compatibility with the simulation and evaluation teams.

**Q: What's the significance of 61 tests?**
- A: Complete validation across 9 test suites covering unit tests (Pydantic models, prompts), integration tests (graph node execution, end-to-end workflows), configuration ablations (adversarial mode, different agent combinations), and edge cases (empty data, extreme parameters). Everything runs in mock mode for rapid iteration without API costs.

---

## Coordination with Team

Make sure your slides don't overlap with:
- Deveen's TypeScript baseline architecture
- Simulation team's market data generation
- Evaluation team's T³/CRIT scoring methodology
- Other team members' specific contributions

Focus on: **LangGraph orchestration** + **agreeableness knob innovation** + **testing rigor**
