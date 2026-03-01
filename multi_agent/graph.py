"""
LangGraph-based multi-agent debate orchestrator.

This module defines ALL node functions and graph builders for the debate
system.  It serves two execution paths:

  1. MONOLITHIC GRAPH (original, preserved for backward compatibility):
       build_debate_graph / compile_debate_graph
       Runs the full pipeline → propose → critique/revise loop → judge → trace
       as a single LangGraph invocation.  The internal should_continue edge
       decides when to stop looping.

  2. DECOMPOSED SUB-GRAPHS (new, used by MultiAgentRunner):
       build_pipeline_graph   — news → data → build_context → END
       build_single_round_graph — propose → critique → revise → END
       build_finalize_graph   — judge → build_trace → END
       The runner calls each sub-graph separately, owning the iteration
       loop itself so external controllers (PID, agreeableness tuners)
       can intervene between rounds.

WHY THIS DECOMPOSITION EXISTS
------------------------------
The monolithic graph encapsulates the iteration loop inside LangGraph's
conditional edge (should_continue).  This makes it impossible for external
code to inspect or modify state between rounds — there is no seam to hook
into.  By splitting into sub-graphs, the runner's for-loop becomes that
seam:

    for t in range(max_rounds):
        state = runner.run_single_round(state)
        # <-- future: controller_step(state) goes here
        if runner.should_terminate(state):
            break

INVARIANTS PRESERVED
--------------------
- All sub-graphs use StateGraph(DebateState), so LangGraph's reducers
  (Annotated[list, operator.add] for debate_turns) are applied correctly
  at every phase boundary.  No manual state.update(node(state)) anywhere.

- propose_node has an idempotency guard: if proposals already exist in
  state, it returns {} (a no-op).  This lets the single-round graph
  include propose in every round without duplicating proposals.  The guard
  is harmless for the monolithic graph where propose runs exactly once.

- current_round is KEPT in propose_node and revise_node returns.  The
  monolithic graph's should_continue edge depends on this.  The runner
  also sets current_round before each round invocation; the two sources
  agree because the runner sets t+1 and propose returns 1 (first round
  only), and revise returns current_round+1 (matching the runner's
  next iteration value).

- proposals (plain list, no reducer) is set once by propose_node in
  round 1, then preserved in subsequent rounds because propose returns {}.
  critiques and revisions are REPLACED each round (plain list, no reducer),
  matching the monolithic graph's behavior.

- debate_turns (Annotated[list, operator.add]) accumulates correctly:
  pipeline produces no turns, each round appends that round's turns,
  finalize appends the judge turn.  Between sub-graph invocations the
  runner passes the full state dict, so the accumulated list carries over.

- All existing exports (build_debate_graph, compile_debate_graph,
  should_continue, all node functions, all mock functions) remain
  unchanged and functional.  Existing tests import them directly.

Graph structures:

  Monolithic (original):
    [START]
      -> [news_digest]  (optional, parallel)
      -> [data_analysis] (optional, parallel)
      -> [build_context]
      -> [propose]
      -> [critique]
      -> [revise]
      -> [should_continue?]  (loop back to critique or proceed to judge)
      -> [judge]
      -> [build_trace]
    [END]

  Decomposed (new):
    Pipeline:      [START] -> [news] -> [data] -> [build_context] -> [END]
    Single round:  [START] -> [propose] -> [critique] -> [revise] -> [END]
    Finalize:      [START] -> [judge] -> [build_trace] -> [END]
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()  # auto-load .env file if present
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import AgentRole, DebateConfig
from .models import Observation
from .prompts import (
    DATA_ANALYSIS_SYSTEM_PROMPT,
    NEWS_DIGEST_SYSTEM_PROMPT,
    ROLE_SYSTEM_PROMPTS,
    build_critique_prompt,
    build_judge_prompt,
    build_observation_context,
    build_proposal_user_prompt,
    build_revision_prompt,
    get_agreeableness_modifier,
)


# =============================================================================
# STATE DEFINITION
# =============================================================================


class DebateState(TypedDict):
    """State that flows through the LangGraph debate graph."""

    # --- Inputs (set once at invocation) ---
    observation: dict
    config: dict

    # --- Pipeline outputs (set by pipeline nodes) ---
    news_digest: str
    data_analysis: str
    enriched_context: str

    # --- Debate state (replaced each round) ---
    proposals: list  # [{role, action_dict, raw_response}]
    critiques: list  # [{role, critiques: [...], self_critique}]
    revisions: list  # [{role, action_dict, revision_notes}]
    current_round: int

    # --- Accumulated across all rounds (append-only) ---
    debate_turns: Annotated[list, operator.add]

    # --- Final outputs ---
    final_action: dict
    strongest_objection: str
    audited_memo: str
    trace: dict


class ParallelRoundState(TypedDict):
    """State for parallel single-round graph.

    Identical to DebateState but with Annotated[list, operator.add] on
    proposals, critiques, and revisions.  This is required because
    LangGraph raises InvalidUpdateError when parallel nodes write to a
    non-annotated field.  Each per-agent node returns a single-element
    list, and operator.add merges them at the sync barrier.

    debate_turns already uses operator.add in DebateState — unchanged here.
    """

    # --- Inputs (set once at invocation) ---
    observation: dict
    config: dict

    # --- Pipeline outputs (set by pipeline nodes) ---
    news_digest: str
    data_analysis: str
    enriched_context: str

    # --- Debate state (parallel: accumulated via operator.add) ---
    proposals: Annotated[list, operator.add]
    critiques: Annotated[list, operator.add]
    revisions: Annotated[list, operator.add]
    current_round: int

    # --- Accumulated across all rounds (append-only) ---
    debate_turns: Annotated[list, operator.add]

    # --- Final outputs ---
    final_action: dict
    strongest_objection: str
    audited_memo: str
    trace: dict


# =============================================================================
# LLM CALL HELPER
# =============================================================================


def _call_llm(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Call LLM with the given system and user prompts. Returns raw text.

    Retries up to 3 times with exponential backoff on transient errors
    (connection errors, rate limits, timeouts).
    """
    if config.get("mock", False):
        return "{}"

    import time

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import-not-found]

    llm = ChatOpenAI(
        model=config.get("model_name", "gpt-4o-mini"),
        temperature=config.get("temperature", 0.3),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-dummy"),
        request_timeout=60,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return response.content
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            if attempt < max_retries - 1:
                print(f"  [LLM RETRY] {type(e).__name__} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})...", flush=True)
                logger.warning(
                    "_call_llm retry %d/%d: %s: %s",
                    attempt + 1, max_retries, type(e).__name__, e,
                )
                time.sleep(wait)
            else:
                print(f"  [LLM ERROR] {type(e).__name__}: {e} (all {max_retries} attempts failed)", flush=True)
                logger.error(
                    "_call_llm failed after %d attempts: %s: %s — returning empty JSON",
                    max_retries, type(e).__name__, e,
                )
                return "{}"
    return "{}"


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = match.group(1) if match else text
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


# =============================================================================
# MOCK RESPONSE GENERATORS (deterministic, for testing without API keys)
# =============================================================================


def _mock_proposal(role: str, obs_dict: dict) -> dict:
    """Generate a deterministic mock proposal for a given role."""
    tickers = obs_dict.get("universe", ["AAPL"])
    returns = obs_dict.get("market_state", {}).get("returns") or {}
    t = tickers[0] if tickers else "AAPL"
    r = returns.get(t, 0.0)

    # Each role has a different bias to create realistic disagreements
    bias_map = {
        "macro": 0.02,
        "value": -0.01,
        "risk": -0.03,
        "technical": 0.015,
        "sentiment": 0.005,
        "devils_advocate": -0.025,
    }
    bias = bias_map.get(role, 0.0)
    threshold = r + bias

    if threshold > 0.01:
        orders = [{"ticker": t, "side": "buy", "size": 10}]
        direction = "Bullish"
    elif threshold < -0.01:
        orders = [{"ticker": t, "side": "sell", "size": 10}]
        direction = "Bearish"
    else:
        orders = []
        direction = "Neutral"

    pearl = "L2" if role != "devils_advocate" else "L3"

    return {
        "what_i_saw": f"[{role}] Observed {t} return {r * 100:.2f}%",
        "hypothesis": f"[{role}] {direction} based on {role} analysis",
        "orders": orders,
        "justification": f"[{role} mock] threshold={threshold * 100:.2f}%",
        "confidence": round(min(0.9, 0.4 + abs(threshold) * 5), 2),
        "risks_or_falsifiers": f"A reversal in {role} signals would change this view",
        "claims": [
            {
                "claim_text": f"If {role} signals persist, {t} will {'rise' if threshold > 0 else 'decline'}",
                "pearl_level": pearl,
                "variables": [t, f"{role}_signal"],
                "assumptions": [f"{role} analytical framework is valid for current regime"],
                "confidence": 0.55,
            }
        ],
    }


def _mock_critique(role: str, proposals: list) -> dict:
    """Generate a deterministic mock critique."""
    others = [p for p in proposals if p.get("role") != role][:2]
    return {
        "critiques": [
            {
                "target_role": o.get("role", "unknown"),
                "objection": (
                    f"[mock] {role} challenges {o.get('role')}'s core assumption; "
                    f"possible confounder: regime change could invalidate their signal"
                ),
                "alternative_explanation": f"Market noise rather than genuine {o.get('role')} signal",
                "falsifier": "Unexpected macro shock or data revision",
                "objection_confidence": 0.6,
            }
            for o in others
        ],
        "self_critique": f"[mock] {role} may be overweighting recent data; sample size is small",
    }


def _mock_revision(role: str, original_action: dict, obs_dict: dict) -> dict:
    """Generate a deterministic mock revision."""
    proposal = _mock_proposal(role, obs_dict)
    proposal["confidence"] = max(0.25, proposal["confidence"] - 0.1)
    proposal["revision_notes"] = f"[mock] {role} reduced confidence after considering critiques"
    return proposal


def _mock_judge(revisions: list) -> dict:
    """Generate a deterministic mock judge decision."""
    buy_count = 0
    sell_count = 0
    ticker = "AAPL"
    for r in revisions:
        action = r.get("action_dict", {})
        for o in action.get("orders", []):
            if o.get("side") == "buy":
                buy_count += 1
                ticker = o.get("ticker", ticker)
            elif o.get("side") == "sell":
                sell_count += 1
                ticker = o.get("ticker", ticker)

    if buy_count > sell_count:
        orders = [{"ticker": ticker, "side": "buy", "size": 10}]
    elif sell_count > buy_count:
        orders = [{"ticker": ticker, "side": "sell", "size": 10}]
    else:
        orders = []

    return {
        "orders": orders,
        "audited_memo": (
            f"[Judge mock] {len(revisions)} agents debated. "
            f"Buy votes: {buy_count}, Sell votes: {sell_count}. "
            f"Decision based on majority direction."
        ),
        "strongest_objection": "Risk agent raised volatility regime concerns",
        "confidence": 0.55,
        "risks_or_falsifiers": "Unexpected correlation breakdown or macro shock",
        "claims": [
            {
                "claim_text": "Debate consensus reflects weighted multi-perspective analysis",
                "pearl_level": "L2",
                "variables": ["consensus", "trade_direction"],
                "assumptions": ["Multi-agent deliberation reduces individual bias"],
                "confidence": 0.6,
            }
        ],
    }


def _mock_pipeline(agent_type: str, obs_dict: dict) -> dict:
    """Generate a deterministic mock pipeline output."""
    if agent_type == "news_digest":
        return {
            "summary": f"Mixed sentiment for {', '.join(obs_dict.get('universe', []))}. Key drivers: macro policy and earnings.",
            "sentiment_score": 0.15,
            "key_signals": [
                "Fed signals potential rate adjustment",
                "Tech sector earnings mixed",
                "Geopolitical uncertainty elevated",
            ],
            "causal_implications": "Rate policy shift could benefit growth stocks if confirmed",
            "information_freshness": "new",
            "narrative_shift": False,
            "confidence": 0.6,
        }
    else:
        returns = obs_dict.get("market_state", {}).get("returns") or {}
        avg_return = sum(returns.values()) / max(len(returns), 1) if returns else 0.0
        return {
            "summary": "Market data shows moderate momentum with contained volatility",
            "momentum_signal": "positive" if avg_return > 0 else "negative" if avg_return < 0 else "neutral",
            "volatility_regime": "medium",
            "relative_strength": returns,
            "risk_assessment": "Moderate risk; no extreme readings detected",
            "key_levels": [],
            "confidence": 0.6,
        }


# =============================================================================
# VERBOSE DISPLAY HELPERS
# =============================================================================

_ROLE_LABELS = {
    "macro": "MACRO STRATEGIST",
    "value": "VALUE ANALYST",
    "risk": "RISK MANAGER",
    "technical": "TECHNICAL ANALYST",
    "sentiment": "SENTIMENT ANALYST",
    "devils_advocate": "DEVIL'S ADVOCATE",
}


def _verbose_proposal(role: str, result: dict) -> None:
    """Print a formatted proposal for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    hyp = result.get("hypothesis", result.get("justification", ""))[:200]
    claims = result.get("claims", [])

    print(f"\n    ┌─── {label} proposes ───", flush=True)
    print(f"    │ Orders: {orders_str}", flush=True)
    print(f"    │ Confidence: {conf:.0%}", flush=True)
    print(f"    │ Thesis: {hyp}", flush=True)
    for c in claims[:2]:
        lvl = c.get("pearl_level", "?")
        txt = c.get("claim_text", "")[:120]
        print(f"    │ Claim [{lvl}]: {txt}", flush=True)
    falsifiers = result.get("risks_or_falsifiers", "")
    if falsifiers:
        falsifiers_str = str(falsifiers)
        print(f"    │ Falsifier: {falsifiers_str[:120]}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_critique(role: str, result: dict) -> None:
    """Print a formatted critique for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    critiques = result.get("critiques", [])
    self_crit = result.get("self_critique", "")

    print(f"\n    ┌─── {label} critiques ───", flush=True)
    for c in critiques[:3]:
        target = _ROLE_LABELS.get(c.get("target_role", ""), c.get("target_role", "?"))
        obj = c.get("objection", "")[:150]
        print(f"    │ → {target}: {obj}", flush=True)
        falsifier = c.get("falsifier", "")
        if falsifier:
            print(f"    │   Falsifier: {falsifier[:100]}", flush=True)
    if self_crit:
        print(f"    │ Self-critique: {self_crit[:120]}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_revision(role: str, result: dict) -> None:
    """Print a formatted revision for verbose mode."""
    label = _ROLE_LABELS.get(role, role.upper())
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    notes = result.get("revision_notes", result.get("justification", ""))[:200]

    print(f"\n    ┌─── {label} revises ───", flush=True)
    print(f"    │ Orders: {orders_str}", flush=True)
    print(f"    │ Confidence: {conf:.0%}", flush=True)
    if notes:
        print(f"    │ Revision: {notes}", flush=True)
    print(f"    └{'─' * 50}", flush=True)


def _verbose_judge(result: dict) -> None:
    """Print a formatted judge decision for verbose mode."""
    orders = result.get("orders", [])
    orders_str = ", ".join(f"{o.get('side')} {o.get('size')} {o.get('ticker')}" for o in orders) or "HOLD"
    conf = result.get("confidence", 0.5)
    memo = result.get("audited_memo", result.get("justification", ""))[:300]
    objection = result.get("strongest_objection", "")[:200]

    print(f"\n    ╔{'═' * 54}", flush=True)
    print(f"    ║  JUDGE FINAL DECISION", flush=True)
    print(f"    ╠{'═' * 54}", flush=True)
    print(f"    ║  Orders: {orders_str}", flush=True)
    print(f"    ║  Confidence: {conf:.0%}", flush=True)
    if memo:
        # Wrap long memo text
        for line in [memo[i:i+70] for i in range(0, len(memo), 70)]:
            print(f"    ║  {line}", flush=True)
    if objection:
        print(f"    ║", flush=True)
        print(f"    ║  Strongest objection preserved:", flush=True)
        print(f"    ║  {objection}", flush=True)
    print(f"    ╚{'═' * 54}", flush=True)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================


def news_digest_node(state: DebateState) -> dict:
    """Pipeline node: digest news/text context into structured signals."""
    print("  [Pipeline] News digest agent processing...", flush=True)
    obs = state["observation"]
    config = state["config"]
    text_context = obs.get("text_context", "")

    if not text_context:
        return {"news_digest": "No news context provided."}

    if config.get("mock", False):
        result = _mock_pipeline("news_digest", obs)
        return {"news_digest": json.dumps(result, indent=2)}

    raw = _call_llm(
        config,
        NEWS_DIGEST_SYSTEM_PROMPT,
        f"Text context to analyze:\n{text_context}\n\nUniverse: {', '.join(obs.get('universe', []))}",
    )
    return {"news_digest": raw}


def data_analysis_node(state: DebateState) -> dict:
    """Pipeline node: analyze market data into structured signals."""
    print("  [Pipeline] Data analysis agent processing...", flush=True)
    obs = state["observation"]
    config = state["config"]

    market_str = json.dumps(obs.get("market_state", {}), indent=2)
    portfolio_str = json.dumps(obs.get("portfolio_state", {}), indent=2)

    if config.get("mock", False):
        result = _mock_pipeline("data_analysis", obs)
        return {"data_analysis": json.dumps(result, indent=2)}

    raw = _call_llm(
        config,
        DATA_ANALYSIS_SYSTEM_PROMPT,
        f"Market data:\n{market_str}\n\nPortfolio:\n{portfolio_str}\n\nUniverse: {', '.join(obs.get('universe', []))}",
    )
    return {"data_analysis": raw}


def build_context_node(state: DebateState) -> dict:
    """Combine observation + pipeline outputs into enriched context string."""
    obs_model = Observation(**state["observation"])

    pipeline_parts = []
    news = state.get("news_digest", "")
    if news and news != "No news context provided.":
        pipeline_parts.append(f"### News Intelligence\n{news}")
    data = state.get("data_analysis", "")
    if data:
        pipeline_parts.append(f"### Data Analysis\n{data}")

    pipeline_context = "\n\n".join(pipeline_parts)
    enriched = build_observation_context(obs_model, pipeline_context)
    return {"enriched_context": enriched}


def propose_node(state: DebateState) -> dict:
    """All role agents generate their initial proposals.

    Idempotency guard: when the runner calls single_round_graph multiple
    times, propose is in the graph every round but should only execute
    once (round 1).  On subsequent rounds proposals already exist in
    state, so we return {} — a no-op that leaves all state fields
    untouched.  Uses len(...) > 0 for robustness (handles None, empty
    list).  This guard is harmless for the monolithic graph where propose
    runs exactly once.
    """
    if len(state.get("proposals") or []) > 0:
        return {}

    config = state["config"]
    context = state["enriched_context"]
    obs = state["observation"]
    roles = config.get("roles", ["macro", "value", "risk"])
    is_mock = config.get("mock", False)

    proposals = []
    turns = []

    for i, role in enumerate(roles):
        print(f"  [Round 0 - Propose] {role.upper()} agent ({i+1}/{len(roles)})...", flush=True)
        role_system = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS.get(AgentRole.MACRO, ""))
        user_prompt = build_proposal_user_prompt(context)

        if config.get("log_system_prompts"):
            print(f"  [Round 0 - Propose] {role.upper()} system prompt:\n{role_system}", flush=True)
        if config.get("log_user_prompts"):
            print(f"  [Round 0 - Propose] {role.upper()} user prompt:\n{user_prompt}", flush=True)

        raw_text = None  # Raw LLM output for eval module
        if is_mock:
            result = _mock_proposal(role, obs)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, role_system, user_prompt)
            result = _parse_json(raw_text)

        if config.get("log_llm_responses"):
            print(f"  [Round 0 - Propose] {role.upper()} raw LLM response:\n{raw_text}", flush=True)

        action_dict = {
            "orders": result.get("orders", []),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_proposal(role, result)

        proposals.append({
            "role": role,
            "action_dict": action_dict,
            "raw_response": raw_text,
        })

        turns.append({
            "round": 0,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "proposal",
            "content": result,
            "raw_system_prompt": role_system,
            "raw_user_prompt": user_prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
            },
        })

    print(f"  [Round 0 - Propose] All {len(roles)} proposals complete.", flush=True)
    return {
        "proposals": proposals,
        "debate_turns": turns,
        "current_round": 1,
    }


def critique_node(state: DebateState) -> dict:
    """All role agents critique each other's proposals (or prior revisions)."""
    config = state["config"]
    context = state["enriched_context"]
    current_round = state.get("current_round", 1)
    agreeableness = config.get("agreeableness", 0.3)
    is_mock = config.get("mock", False)

    # After first round, critique the revisions; otherwise the proposals
    source = state.get("revisions") if state.get("revisions") else state["proposals"]

    all_proposals_for_critique = [
        {
            "role": p["role"],
            "proposal": json.dumps(p.get("action_dict", {})),
        }
        for p in source
    ]

    critiques = []
    turns = []

    for i, p in enumerate(source):
        role = p["role"]
        print(f"  [Round {current_round} - Critique] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        # Build prompts unconditionally so eval module can inspect them even in mock mode
        prompt = build_critique_prompt(
            role, context, all_proposals_for_critique, my_proposal, agreeableness,
        )
        system_msg = (
            f"You are the {role.upper()} agent. Provide explicit, substantive critiques."
            + get_agreeableness_modifier(agreeableness)
        )

        raw_text = None  # Raw LLM output for eval module
        if is_mock:
            result = _mock_critique(role, source)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, system_msg, prompt)
            result = _parse_json(raw_text)

        if config.get("verbose"):
            _verbose_critique(role, result)

        critiques.append({
            "role": role,
            "critiques": result.get("critiques", []),
            "self_critique": result.get("self_critique", ""),
        })

        turns.append({
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "critique",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "all_proposals_for_critique": all_proposals_for_critique,
                "my_proposal": my_proposal,
            }
        })

    print(f"  [Round {current_round} - Critique] All critiques complete.", flush=True)
    return {
        "critiques": critiques,
        "debate_turns": turns,
    }


def revise_node(state: DebateState) -> dict:
    """All role agents revise their proposals based on critiques received."""
    config = state["config"]
    context = state["enriched_context"]
    current_round = state.get("current_round", 1)
    agreeableness = config.get("agreeableness", 0.3)
    is_mock = config.get("mock", False)
    obs = state["observation"]
    all_critiques = state.get("critiques", [])

    source = state.get("revisions") if state.get("revisions") else state["proposals"]

    revisions = []
    turns = []

    for i, p in enumerate(source):
        role = p["role"]
        print(f"  [Round {current_round} - Revise] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        # Collect critiques targeted at this role
        critiques_received = []
        for c in all_critiques:
            for crit in c.get("critiques", []):
                if crit.get("target_role") == role:
                    critiques_received.append({
                        "from_role": c["role"],
                        "objection": crit.get("objection", ""),
                        "falsifier": crit.get("falsifier"),
                    })

        # Build prompts unconditionally so eval module can inspect them even in mock mode
        prompt = build_revision_prompt(
            role, context, my_proposal, critiques_received, agreeableness,
        )
        system_msg = f"You are the {role.upper()} agent. Revise your proposal based on critiques."

        if is_mock:
            result = _mock_revision(role, p.get("action_dict", {}), obs)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, system_msg, prompt)
            result = _parse_json(raw_text)

        action_dict = {
            "orders": result.get("orders", p.get("action_dict", {}).get("orders", [])),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_revision(role, result)

        revisions.append({
            "role": role,
            "action_dict": action_dict,
            "revision_notes": result.get("revision_notes", ""),
        })

        turns.append({
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "revision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "my_proposal": my_proposal,
                "critiques_received": critiques_received,
            }
        })

    print(f"  [Round {current_round} - Revise] All revisions complete.", flush=True)
    return {
        "revisions": revisions,
        "debate_turns": turns,
        "current_round": current_round + 1,
    }


# =============================================================================
# PER-AGENT NODE FACTORIES (for parallel single-round graph)
# =============================================================================
#
# Each factory returns a closure that handles ONE agent in a fan-out
# pattern.  The closure returns single-element lists so that
# ParallelRoundState's operator.add reducers can merge outputs from
# all parallel nodes at the sync barrier.
#
# Key differences from the batch node functions above:
#   - Per-agent nodes do NOT return current_round (avoids parallel
#     write conflict on a non-annotated int; the runner manages it).
#   - Return values are single-element lists, not full lists.
#   - Each closure captures `role` from the factory argument.
# =============================================================================


def _sync_noop(state: ParallelRoundState) -> dict:
    """No-op sync barrier node for fan-in.

    LangGraph merges all upstream outputs before running downstream
    nodes, so this node just passes through without modifying state.
    """
    return {}


def make_propose_node(role: str):
    """Factory: create a per-agent propose node for the given role.

    Extracts the loop body from propose_node.  Returns single-element
    lists for proposals and debate_turns.  Includes the same
    idempotency guard: if proposals already exist, returns {}.
    Does NOT return current_round.
    """

    def _propose(state: ParallelRoundState) -> dict:
        # Idempotency guard: skip if proposals already exist
        if len(state.get("proposals") or []) > 0:
            return {}

        config = state["config"]
        context = state["enriched_context"]
        obs = state["observation"]
        roles = config.get("roles", ["macro", "value", "risk"])
        is_mock = config.get("mock", False)

        i = roles.index(role) if role in roles else 0
        print(f"  [Round 0 - Propose] {role.upper()} agent ({i+1}/{len(roles)})...", flush=True)

        role_system = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS.get(AgentRole.MACRO, ""))
        user_prompt = build_proposal_user_prompt(context)

        if config.get("log_system_prompts"):
            print(f"  [Round 0 - Propose] {role.upper()} system prompt:\n{role_system}", flush=True)
        if config.get("log_user_prompts"):
            print(f"  [Round 0 - Propose] {role.upper()} user prompt:\n{user_prompt}", flush=True)

        raw_text = None
        if is_mock:
            result = _mock_proposal(role, obs)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, role_system, user_prompt)
            result = _parse_json(raw_text)

        if config.get("log_llm_responses"):
            print(f"  [Round 0 - Propose] {role.upper()} raw LLM response:\n{raw_text}", flush=True)

        action_dict = {
            "orders": result.get("orders", []),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_proposal(role, result)

        proposal = {
            "role": role,
            "action_dict": action_dict,
            "raw_response": raw_text,
        }

        turn = {
            "round": 0,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "proposal",
            "content": result,
            "raw_system_prompt": role_system,
            "raw_user_prompt": user_prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
            },
        }

        return {
            "proposals": [proposal],
            "debate_turns": [turn],
        }

    _propose.__name__ = f"propose_{role}"
    return _propose


def make_critique_node(role: str):
    """Factory: create a per-agent critique node for the given role.

    Extracts the loop body from critique_node.  Finds own entry in
    the source list by role field.  Reads ALL proposals for the
    critique prompt.  Returns single-element lists.
    """

    def _critique(state: ParallelRoundState) -> dict:
        config = state["config"]
        context = state["enriched_context"]
        current_round = state.get("current_round", 1)
        agreeableness = config.get("agreeableness", 0.3)
        is_mock = config.get("mock", False)

        # After first round, critique the revisions; otherwise the proposals.
        # Sort by config role order for deterministic behavior (operator.add
        # merge order is non-deterministic).
        roles = config.get("roles", ["macro", "value", "risk"])
        role_order = {r: i for i, r in enumerate(roles)}
        raw_source = state.get("revisions") if state.get("revisions") else state["proposals"]
        source = sorted(raw_source, key=lambda e: role_order.get(e["role"], len(roles)))

        # Find own entry by role (safe lookup — missing role returns empty dict)
        p = next((entry for entry in source if entry["role"] == role), {})

        all_proposals_for_critique = [
            {
                "role": entry["role"],
                "proposal": json.dumps(entry.get("action_dict", {})),
            }
            for entry in source
        ]

        roles = config.get("roles", ["macro", "value", "risk"])
        i = roles.index(role) if role in roles else 0
        print(f"  [Round {current_round} - Critique] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        prompt = build_critique_prompt(
            role, context, all_proposals_for_critique, my_proposal, agreeableness,
        )
        system_msg = (
            f"You are the {role.upper()} agent. Provide explicit, substantive critiques."
            + get_agreeableness_modifier(agreeableness)
        )

        raw_text = None
        if is_mock:
            result = _mock_critique(role, source)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, system_msg, prompt)
            result = _parse_json(raw_text)

        if config.get("verbose"):
            _verbose_critique(role, result)

        critique = {
            "role": role,
            "critiques": result.get("critiques", []),
            "self_critique": result.get("self_critique", ""),
        }

        turn = {
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "critique",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "all_proposals_for_critique": all_proposals_for_critique,
                "my_proposal": my_proposal,
            },
        }

        return {
            "critiques": [critique],
            "debate_turns": [turn],
        }

    _critique.__name__ = f"critique_{role}"
    return _critique


def make_revise_node(role: str):
    """Factory: create a per-agent revise node for the given role.

    Extracts the loop body from revise_node.  Collects critiques
    targeted at this role.  Returns single-element lists.
    Does NOT return current_round.
    """

    def _revise(state: ParallelRoundState) -> dict:
        config = state["config"]
        context = state["enriched_context"]
        current_round = state.get("current_round", 1)
        agreeableness = config.get("agreeableness", 0.3)
        is_mock = config.get("mock", False)
        obs = state["observation"]
        all_critiques = state.get("critiques", [])

        # Sort by config role order for deterministic behavior (operator.add
        # merge order is non-deterministic).
        roles = config.get("roles", ["macro", "value", "risk"])
        role_order = {r: i for i, r in enumerate(roles)}
        raw_source = state.get("revisions") if state.get("revisions") else state["proposals"]
        source = sorted(raw_source, key=lambda e: role_order.get(e["role"], len(roles)))

        # Find own entry by role (safe lookup — missing role returns empty dict)
        p = next((entry for entry in source if entry["role"] == role), {})

        roles = config.get("roles", ["macro", "value", "risk"])
        i = roles.index(role) if role in roles else 0
        print(f"  [Round {current_round} - Revise] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        # Collect critiques targeted at this role
        critiques_received = []
        for c in all_critiques:
            for crit in c.get("critiques", []):
                if crit.get("target_role") == role:
                    critiques_received.append({
                        "from_role": c["role"],
                        "objection": crit.get("objection", ""),
                        "falsifier": crit.get("falsifier"),
                    })

        prompt = build_revision_prompt(
            role, context, my_proposal, critiques_received, agreeableness,
        )
        system_msg = f"You are the {role.upper()} agent. Revise your proposal based on critiques."

        if is_mock:
            result = _mock_revision(role, p.get("action_dict", {}), obs)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, system_msg, prompt)
            result = _parse_json(raw_text)

        action_dict = {
            "orders": result.get("orders", p.get("action_dict", {}).get("orders", [])),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_revision(role, result)

        revision = {
            "role": role,
            "action_dict": action_dict,
            "revision_notes": result.get("revision_notes", ""),
        }

        turn = {
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "revision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "my_proposal": my_proposal,
                "critiques_received": critiques_received,
            },
        }

        return {
            "revisions": [revision],
            "debate_turns": [turn],
        }

    _revise.__name__ = f"revise_{role}"
    return _revise


# =============================================================================
# PARALLEL SINGLE-ROUND GRAPH CONSTRUCTION
# =============================================================================


def build_parallel_single_round_graph(config: DebateConfig) -> StateGraph:
    """Parallel single round: per-agent fan-out → sync → fan-out → sync → fan-out → END.

    Target topology for 3 agents (macro, value, risk):

        [START] ┬→ [propose_macro]  ┐
                ├→ [propose_value]  ├→ [sync_propose] ┬→ [critique_macro]  ┐
                └→ [propose_risk]   ┘                  ├→ [critique_value]  ├→ [sync_critique] ┬→ [revise_macro]  ┐
                                                       └→ [critique_risk]   ┘                  ├→ [revise_value]  ├→ [END]
                                                                                               └→ [revise_risk]   ┘

    Uses ParallelRoundState with operator.add reducers on proposals,
    critiques, and revisions so that parallel nodes can each contribute
    single-element lists that get merged at the sync barriers.

    The runner resets critiques/revisions to [] between rounds because
    operator.add would otherwise accumulate across rounds.
    """
    roles = [r.value for r in config.roles]
    graph = StateGraph(ParallelRoundState)

    # Sync barriers (no-op pass-throughs for fan-in)
    graph.add_node("sync_propose", _sync_noop)
    graph.add_node("sync_critique", _sync_noop)

    # Per-agent nodes + edges
    for role in roles:
        graph.add_node(f"propose_{role}", make_propose_node(role))
        graph.add_edge(START, f"propose_{role}")
        graph.add_edge(f"propose_{role}", "sync_propose")

        graph.add_node(f"critique_{role}", make_critique_node(role))
        graph.add_edge("sync_propose", f"critique_{role}")
        graph.add_edge(f"critique_{role}", "sync_critique")

        graph.add_node(f"revise_{role}", make_revise_node(role))
        graph.add_edge("sync_critique", f"revise_{role}")
        graph.add_edge(f"revise_{role}", END)

    return graph


def compile_parallel_single_round_graph(config: DebateConfig):
    """Build and compile the parallel single-round graph, ready for invocation."""
    return build_parallel_single_round_graph(config).compile()


# =============================================================================
# PER-PHASE SUB-GRAPHS (used by PID controller for per-phase intervention)
# =============================================================================
#
# When the PID controller is active, it needs to adjust agreeableness
# BETWEEN propose, critique, and revise phases within a single round.
# The existing single-round graphs run all three atomically, so we
# provide per-phase graphs that the runner can invoke individually.
#
# Sequential (for parallel_agents=False):
#   build_propose_graph   — START → propose → END
#   build_critique_graph  — START → critique → END
#   build_revise_graph    — START → revise → END
#
# Parallel (for parallel_agents=True):
#   build_parallel_propose_graph   — START → propose_* → END
#   build_parallel_critique_graph  — START → critique_* → END
#   build_parallel_revise_graph    — START → revise_* → END
# =============================================================================


def build_propose_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: propose only.

    START → propose → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("propose", propose_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", END)
    return graph


def build_critique_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: critique only.

    START → critique → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("critique", critique_node)
    graph.add_edge(START, "critique")
    graph.add_edge("critique", END)
    return graph


def build_revise_graph(config: DebateConfig) -> StateGraph:
    """Single-phase graph: revise only.

    START → revise → END.  Uses DebateState (sequential, batch node).
    """
    graph = StateGraph(DebateState)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "revise")
    graph.add_edge("revise", END)
    return graph


def build_parallel_propose_graph(config: DebateConfig) -> StateGraph:
    """Parallel propose: per-agent fan-out → END.

    START ┬→ propose_macro ┐
          ├→ propose_value ├→ END
          └→ propose_risk  ┘

    Uses ParallelRoundState for operator.add on proposals.
    """
    roles = [r.value for r in config.roles]
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"propose_{role}", make_propose_node(role))
        graph.add_edge(START, f"propose_{role}")
        graph.add_edge(f"propose_{role}", END)
    return graph


def build_parallel_critique_graph(config: DebateConfig) -> StateGraph:
    """Parallel critique: per-agent fan-out → END.

    Uses ParallelRoundState for operator.add on critiques.
    """
    roles = [r.value for r in config.roles]
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"critique_{role}", make_critique_node(role))
        graph.add_edge(START, f"critique_{role}")
        graph.add_edge(f"critique_{role}", END)
    return graph


def build_parallel_revise_graph(config: DebateConfig) -> StateGraph:
    """Parallel revise: per-agent fan-out → END.

    Uses ParallelRoundState for operator.add on revisions.
    """
    roles = [r.value for r in config.roles]
    graph = StateGraph(ParallelRoundState)
    for role in roles:
        graph.add_node(f"revise_{role}", make_revise_node(role))
        graph.add_edge(START, f"revise_{role}")
        graph.add_edge(f"revise_{role}", END)
    return graph


def compile_propose_graph(config: DebateConfig):
    """Build and compile the sequential propose graph."""
    return build_propose_graph(config).compile()


def compile_critique_graph(config: DebateConfig):
    """Build and compile the sequential critique graph."""
    return build_critique_graph(config).compile()


def compile_revise_graph(config: DebateConfig):
    """Build and compile the sequential revise graph."""
    return build_revise_graph(config).compile()


def compile_parallel_propose_graph(config: DebateConfig):
    """Build and compile the parallel propose graph."""
    return build_parallel_propose_graph(config).compile()


def compile_parallel_critique_graph(config: DebateConfig):
    """Build and compile the parallel critique graph."""
    return build_parallel_critique_graph(config).compile()


def compile_parallel_revise_graph(config: DebateConfig):
    """Build and compile the parallel revise graph."""
    return build_parallel_revise_graph(config).compile()


def should_continue(state: DebateState) -> str:
    """Conditional edge: loop back to critique or proceed to judge."""
    current_round = state.get("current_round", 2)
    max_rounds = state.get("config", {}).get("max_rounds", 1)
    if current_round <= max_rounds:
        return "critique"
    return "judge"


def judge_node(state: DebateState) -> dict:
    """Judge synthesizes the debate into a single final trading decision."""
    print("  [Judge] Synthesizing final decision...", flush=True)
    config = state["config"]
    context = state["enriched_context"]
    revisions = state.get("revisions", state.get("proposals", []))
    all_critiques = state.get("critiques", [])
    is_mock = config.get("mock", False)

    # Format critiques for the judge
    critiques_text = "\n".join(
        f"[{c['role']} -> {crit.get('target_role', '?')}]: {crit.get('objection', '')}"
        for c in all_critiques
        for crit in c.get("critiques", [])
    )

    revisions_for_judge = [
        {
            "role": r["role"],
            "action": json.dumps(r.get("action_dict", {})),
            "confidence": r.get("action_dict", {}).get("confidence", 0.5),
        }
        for r in revisions
    ]

    # Build prompts unconditionally so eval module can inspect them even in mock mode
    prompt = build_judge_prompt(context, revisions_for_judge, critiques_text)
    system_msg = "You are the Judge. Synthesize the debate and produce final orders with an audited memo."

    raw_text = None  # Raw LLM output for eval module
    if is_mock:
        result = _mock_judge(revisions)
        raw_text = json.dumps(result, indent=2)
    else:
        raw_text = _call_llm(config, system_msg, prompt)
        result = _parse_json(raw_text)

    if config.get("verbose"):
        _verbose_judge(result)

    final_action = {
        "orders": result.get("orders", []),
        "justification": result.get("audited_memo", result.get("justification", "")),
        "confidence": result.get("confidence", 0.5),
        "claims": result.get("claims", []),
    }

    turns = [
        {
            "round": state.get("current_round", 2),
            "agent_id": "judge",
            "role": "judge",
            "type": "judge_decision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "revisions_for_judge": revisions_for_judge,
                "critiques_text": critiques_text
            }
        }
    ]

    return {
        "final_action": final_action,
        "strongest_objection": result.get("strongest_objection", ""),
        "audited_memo": result.get("audited_memo", ""),
        "debate_turns": turns,
    }


def _get_vote_direction(proposals: list, ticker: str) -> str:
    """Count buy/sell votes for a ticker across all proposals. Majority wins; ties = hold.

    Each agent gets exactly one vote per ticker (first matching order wins)
    to prevent multi-order double-counting.
    """
    buy_count = 0
    sell_count = 0
    for p in proposals:
        for o in p.get("action_dict", {}).get("orders", []):
            if o.get("ticker") == ticker:
                side = o.get("side")
                if side == "buy":
                    buy_count += 1
                elif side == "sell":
                    sell_count += 1
                break  # one vote per agent per ticker
    if buy_count > sell_count:
        return "buy"
    elif sell_count > buy_count:
        return "sell"
    return "hold"


def _get_median_size(proposals: list, ticker: str, side: str) -> float:
    """Median order size for a ticker+direction across proposals."""
    sizes = []
    for p in proposals:
        for o in p.get("action_dict", {}).get("orders", []):
            if o.get("ticker") == ticker and o.get("side") == side:
                sizes.append(o.get("size", 0))
    if not sizes:
        return 0.0
    sizes.sort()
    n = len(sizes)
    if n % 2 == 1:
        return float(sizes[n // 2])
    return (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0


def aggregate_proposals_node(state: DebateState) -> dict:
    """LangGraph node: aggregate proposals by majority vote + median sizing."""
    proposals = state.get("proposals", [])
    obs = state["observation"]
    tickers = obs.get("universe", [])

    # Build aggregated orders
    orders = []
    for ticker in tickers:
        direction = _get_vote_direction(proposals, ticker)
        if direction in ("buy", "sell"):
            size = _get_median_size(proposals, ticker, direction)
            if size > 0:
                orders.append({"ticker": ticker, "side": direction, "size": size})

    # Detect disagreements
    disagreements = []
    for ticker in tickers:
        votes = []
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker:
                    votes.append(f"{p['role']}:{o['side']}{o.get('size', 0)}")
        sides = set()
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker:
                    sides.add(o.get("side"))
        if len(sides) > 1:
            disagreements.append(f"{ticker}: {' vs '.join(votes)}")

    # Find strongest objection: lowest-confidence agent who disagreed with consensus
    strongest_objection = ""
    min_conf = 1.0
    for ticker in tickers:
        direction = _get_vote_direction(proposals, ticker)
        if direction == "hold":
            continue  # no consensus to dissent from
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker and o.get("side") != direction:
                    conf = p.get("action_dict", {}).get("confidence", 0.5)
                    if conf < min_conf:
                        min_conf = conf
                        justification = p.get("action_dict", {}).get("justification", "")
                        strongest_objection = (
                            f"[{p['role']}] (conf={conf:.2f}) dissented: {justification}"
                        )

    # Merge justifications
    merged_justification = " | ".join(
        f"[{p['role']}] {p.get('action_dict', {}).get('justification', '')}"
        for p in proposals
    )

    # Merge claims (deduplicated by claim_text)
    seen_claims = set()
    merged_claims = []
    for p in proposals:
        for c in p.get("action_dict", {}).get("claims", []):
            text = c.get("claim_text", "")
            if text and text not in seen_claims:
                seen_claims.add(text)
                merged_claims.append(c)

    # Average confidence
    confidences = [p.get("action_dict", {}).get("confidence", 0.5) for p in proposals]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    final_action = {
        "orders": orders,
        "justification": merged_justification,
        "confidence": round(avg_confidence, 4),
        "claims": merged_claims,
    }

    return {
        "final_action": final_action,
        "strongest_objection": strongest_objection,
        "audited_memo": (
            f"Majority vote: {len(proposals)} agents. "
            + (f"Disagreements: {'; '.join(disagreements)}" if disagreements else "Unanimous.")
        ),
    }


def build_mv_trace_node(state: DebateState) -> dict:
    """LangGraph node: build AgentTrace for majority_vote architecture."""
    print("  [Trace] Building majority-vote trace...", flush=True)
    obs = state["observation"]
    final = state.get("final_action", {})
    config = state.get("config", {})

    orders_desc = (
        "; ".join(
            f"{o.get('side', '?')} {o.get('size', 0)} {o.get('ticker', '?')}"
            for o in final.get("orders", [])
        )
        or "Hold"
    )

    roles = config.get("roles", [])
    trace = {
        "observation_timestamp": obs.get("timestamp", ""),
        "architecture": "majority_vote",
        "what_i_saw": state.get("enriched_context", "")[:500] + "...",
        "hypothesis": (
            f"Majority vote: {len(roles)} agents ({', '.join(roles)}) "
            f"with vote-based aggregation and median sizing"
        ),
        "decision": orders_desc,
        "risks_or_falsifiers": state.get("audited_memo", "")[:500],
        "strongest_objection": state.get("strongest_objection", ""),
        "debate_turns": state.get("debate_turns", []),
        "action": final,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"trace": trace}


def build_trace_node(state: DebateState) -> dict:
    """Construct the final AgentTrace from the accumulated debate state."""
    print("  [Trace] Building auditable trace...", flush=True)
    obs = state["observation"]
    final = state.get("final_action", {})
    config = state.get("config", {})

    orders_desc = (
        "; ".join(
            f"{o.get('side', '?')} {o.get('size', 0)} {o.get('ticker', '?')}"
            for o in final.get("orders", [])
        )
        or "Hold"
    )

    roles = config.get("roles", [])
    trace = {
        "observation_timestamp": obs.get("timestamp", ""),
        "architecture": "debate",
        "what_i_saw": state.get("enriched_context", "")[:500] + "...",
        "hypothesis": (
            f"Multi-agent debate: {len(roles)} agents ({', '.join(roles)}), "
            f"agreeableness={config.get('agreeableness', 0.3)}, "
            f"rounds={config.get('max_rounds', 1)}"
        ),
        "decision": orders_desc,
        "risks_or_falsifiers": state.get("audited_memo", "")[:500],
        "strongest_objection": state.get("strongest_objection", ""),
        "debate_turns": state.get("debate_turns", []),
        "action": final,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"trace": trace}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def build_debate_graph(config: DebateConfig) -> StateGraph:
    """
    Build the LangGraph debate graph based on configuration.

    The graph structure adapts to config:
      - Pipeline nodes are included only if enabled
      - The critique->revise loop runs for config.max_rounds iterations
      - Adversarial agent is injected into the roles list
    """
    graph = StateGraph(DebateState)

    # --- Add all nodes ---
    has_news = config.enable_news_pipeline
    has_data = config.enable_data_pipeline

    if has_news:
        graph.add_node("news_digest", news_digest_node)
    if has_data:
        graph.add_node("data_analysis", data_analysis_node)

    graph.add_node("build_context", build_context_node)
    graph.add_node("propose", propose_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_node("judge", judge_node)
    graph.add_node("build_trace", build_trace_node)

    # --- Edges: START -> pipeline (parallel) -> build_context ---
    if has_news and has_data:
        graph.add_edge(START, "news_digest")
        graph.add_edge(START, "data_analysis")
        graph.add_edge("news_digest", "build_context")
        graph.add_edge("data_analysis", "build_context")
    elif has_news:
        graph.add_edge(START, "news_digest")
        graph.add_edge("news_digest", "build_context")
    elif has_data:
        graph.add_edge(START, "data_analysis")
        graph.add_edge("data_analysis", "build_context")
    else:
        graph.add_edge(START, "build_context")

    # --- Edges: debate flow ---
    graph.add_edge("build_context", "propose")
    graph.add_edge("propose", "critique")
    graph.add_edge("critique", "revise")

    # Conditional: more rounds or go to judge
    graph.add_conditional_edges(
        "revise",
        should_continue,
        {"critique": "critique", "judge": "judge"},
    )

    graph.add_edge("judge", "build_trace")
    graph.add_edge("build_trace", END)

    return graph


def compile_debate_graph(config: DebateConfig):
    """Build and compile the debate graph, ready for invocation."""
    graph = build_debate_graph(config)
    return graph.compile()


# =============================================================================
# SINGLE-ROUND SUB-GRAPH CONSTRUCTION
# =============================================================================
#
# These three builders decompose the monolithic debate graph into phases
# that the runner can invoke independently.  This is the key architectural
# change: by moving the iteration loop out of LangGraph and into the
# runner, we create a seam where external controllers can observe and
# modify state between debate rounds.
#
# All three use StateGraph(DebateState) so that LangGraph's reducers
# (Annotated[list, operator.add] for debate_turns) are properly applied.
# The runner never does manual state.update(node(state)) — every state
# transition goes through graph.invoke().
#
# Equivalence with the monolithic graph is verified by tests in
# test_round_exposure.py which run both paths on identical inputs and
# assert identical outputs (final_action, debate_turns, trace, etc.).
# =============================================================================


def build_pipeline_graph(config: DebateConfig) -> StateGraph:
    """Pipeline: sequential news → data → build_context → END.

    Runs preprocessing stages that enrich the observation with news
    signals and data analysis before the debate begins.

    Uses SEQUENTIAL edges (not parallel fan-in) because this graph is
    invoked as a standalone phase.  The monolithic graph uses parallel
    fan-in for news+data, but since these nodes don't depend on each
    other's outputs, the sequential ordering produces identical results.
    Equivalence tests (with pipelines disabled) confirm this doesn't
    affect debate output.
    """
    graph = StateGraph(DebateState)
    has_news = config.enable_news_pipeline
    has_data = config.enable_data_pipeline

    if has_news:
        graph.add_node("news_digest", news_digest_node)
    if has_data:
        graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("build_context", build_context_node)

    # SEQUENTIAL chain — no parallel fan-in from START
    if has_news and has_data:
        graph.add_edge(START, "news_digest")
        graph.add_edge("news_digest", "data_analysis")
        graph.add_edge("data_analysis", "build_context")
    elif has_news:
        graph.add_edge(START, "news_digest")
        graph.add_edge("news_digest", "build_context")
    elif has_data:
        graph.add_edge(START, "data_analysis")
        graph.add_edge("data_analysis", "build_context")
    else:
        graph.add_edge(START, "build_context")

    graph.add_edge("build_context", END)
    return graph


def build_single_round_graph(config: DebateConfig) -> StateGraph:
    """One debate round: propose → critique → revise → END.

    The runner calls this once per round.  propose_node is idempotent:
    on round 1 it generates proposals; on rounds 2+ it detects existing
    proposals and returns {} (no-op), so critique and revise operate on
    the latest revisions.

    State flow per round:
      - debate_turns: appended via operator.add reducer (accumulates)
      - proposals: set once in round 1, preserved thereafter (plain list)
      - critiques/revisions: replaced each round (plain list, no reducer)
      - current_round: set by runner before invocation, updated by
        revise_node to current_round+1 (kept for monolithic compat)
    """
    graph = StateGraph(DebateState)
    graph.add_node("propose", propose_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "critique")
    graph.add_edge("critique", "revise")
    graph.add_edge("revise", END)
    return graph


def build_finalize_graph(config: DebateConfig) -> StateGraph:
    """Judge synthesis + trace: judge → build_trace → END.

    Called once after all debate rounds complete.  The judge sees the
    final revisions and critiques from the last round and produces the
    final trading decision.  build_trace constructs the auditable trace
    including all accumulated debate_turns.
    """
    graph = StateGraph(DebateState)
    graph.add_node("judge", judge_node)
    graph.add_node("build_trace", build_trace_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", "build_trace")
    graph.add_edge("build_trace", END)
    return graph


def compile_pipeline_graph(config: DebateConfig):
    """Build and compile the pipeline graph, ready for invocation."""
    return build_pipeline_graph(config).compile()


def compile_single_round_graph(config: DebateConfig):
    """Build and compile the single-round graph, ready for invocation."""
    return build_single_round_graph(config).compile()


def compile_finalize_graph(config: DebateConfig):
    """Build and compile the finalize graph, ready for invocation."""
    return build_finalize_graph(config).compile()


# =============================================================================
# MAJORITY VOTE GRAPH CONSTRUCTION
# =============================================================================


def build_majority_vote_graph(config: DebateConfig) -> StateGraph:
    """
    Build a LangGraph majority-vote graph.

    Graph structure (no critique/revise/judge):
      [START] -> [pipeline nodes] -> [build_context] -> [propose]
              -> [aggregate] -> [build_mv_trace] -> [END]
    """
    graph = StateGraph(DebateState)

    has_news = config.enable_news_pipeline
    has_data = config.enable_data_pipeline

    if has_news:
        graph.add_node("news_digest", news_digest_node)
    if has_data:
        graph.add_node("data_analysis", data_analysis_node)

    graph.add_node("build_context", build_context_node)
    graph.add_node("propose", propose_node)
    graph.add_node("aggregate", aggregate_proposals_node)
    graph.add_node("build_mv_trace", build_mv_trace_node)

    # --- Edges: START -> pipeline (parallel) -> build_context ---
    if has_news and has_data:
        graph.add_edge(START, "news_digest")
        graph.add_edge(START, "data_analysis")
        graph.add_edge("news_digest", "build_context")
        graph.add_edge("data_analysis", "build_context")
    elif has_news:
        graph.add_edge(START, "news_digest")
        graph.add_edge("news_digest", "build_context")
    elif has_data:
        graph.add_edge(START, "data_analysis")
        graph.add_edge("data_analysis", "build_context")
    else:
        graph.add_edge(START, "build_context")

    graph.add_edge("build_context", "propose")
    graph.add_edge("propose", "aggregate")
    graph.add_edge("aggregate", "build_mv_trace")
    graph.add_edge("build_mv_trace", END)

    return graph


def compile_majority_vote_graph(config: DebateConfig):
    """Build and compile the majority-vote graph, ready for invocation."""
    graph = build_majority_vote_graph(config)
    return graph.compile()
