"""
LangGraph-based multi-agent debate orchestrator.

Graph structure:
  [START]
    -> [news_digest]  (optional, parallel)
    -> [data_analysis] (optional, parallel)
    -> [build_context]
    -> [propose]      (all role agents generate proposals)
    -> [critique]     (all role agents critique each other)
    -> [revise]       (all role agents revise based on critiques)
    -> [should_continue?]  (loop back to critique or proceed to judge)
    -> [judge]        (synthesize final decision)
    -> [build_trace]  (construct AgentTrace for eval team)
  [END]
"""

from __future__ import annotations

import json
import operator
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

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
                time.sleep(wait)
            else:
                print(f"  [LLM ERROR] {type(e).__name__}: {e} (all {max_retries} attempts failed)", flush=True)
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
    """All role agents generate their initial proposals."""
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

        raw_text = None  # Raw LLM output for eval module
        if is_mock:
            result = _mock_proposal(role, obs)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm(config, role_system, user_prompt)
            result = _parse_json(raw_text)

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
        })

    print(f"  [Round {current_round} - Revise] All revisions complete.", flush=True)
    return {
        "revisions": revisions,
        "debate_turns": turns,
        "current_round": current_round + 1,
    }


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

    if is_mock:
        result = _mock_judge(revisions)
    else:
        prompt = build_judge_prompt(context, revisions_for_judge, critiques_text)
        system_msg = "You are the Judge. Synthesize the debate and produce final orders with an audited memo."
        raw = _call_llm(config, system_msg, prompt)
        result = _parse_json(raw)

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
        }
    ]

    return {
        "final_action": final_action,
        "strongest_objection": result.get("strongest_objection", ""),
        "audited_memo": result.get("audited_memo", ""),
        "debate_turns": turns,
    }


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
