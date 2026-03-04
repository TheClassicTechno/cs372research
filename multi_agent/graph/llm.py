"""LLM call infrastructure for the debate orchestrator.

Handles LLM invocation with retries, prompt logging, and JSON parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading

from dotenv import load_dotenv

load_dotenv()  # auto-load .env file if present

# Hardcode logger name so it matches the original module regardless of
# where this sub-module lives on disk.
logger = logging.getLogger("multi_agent.graph")

# Dedicated prompt logger — lets users filter rendered prompts separately.
# Activated by log_rendered_prompts=True in config.
prompt_logger = logging.getLogger("debate.prompts")
prompt_logger.setLevel(logging.INFO)

# Throttle concurrent LLM calls to avoid 429 rate limits.
# Parallel agents fire 4 calls at once per phase; this semaphore caps
# concurrency to 2 (still 2x faster than sequential, avoids bursts).
_LLM_SEMAPHORE = threading.Semaphore(2)


def _call_llm(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Call LLM with the given system and user prompts. Returns raw text.

    Retries up to 3 times with exponential backoff on transient errors
    (connection errors, rate limits, timeouts).
    """
    if config.get("mock", False):
        # Return a valid single-agent CRIT response.
        # Debate graph nodes never reach _call_llm in mock mode (they use
        # their own mock shortcuts). Only the CRIT scorer hits this path.
        import json as _json
        return _json.dumps({
            "pillar_scores": {
                "internal_consistency": 0.8,
                "evidence_support": 0.7,
                "trace_alignment": 0.75,
                "causal_integrity": 0.65,
            },
            "diagnostics": {
                "contradictions_detected": False,
                "unsupported_claims_detected": False,
                "conclusion_drift_detected": False,
                "causal_overreach_detected": False,
            },
            "explanations": {
                "internal_consistency": "Mock: consistent.",
                "evidence_support": "Mock: supported.",
                "trace_alignment": "Mock: aligned.",
                "causal_integrity": "Mock: sound.",
            },
        })

    import time

    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import-not-found]

    model_name = config.get("model_name", "gpt-4o-mini")
    temperature = config.get("temperature", 0.3)

    if model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        llm = ChatAnthropic(model=model_name, temperature=temperature, timeout=60)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY", "sk-dummy"),
            request_timeout=60,
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with _LLM_SEMAPHORE:
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


def _extract_snapshot_id(enriched_context: str, observation: dict) -> str:
    """Extract a unique snapshot identifier from debate state.

    Returns a string like ``"2025Q1 (AAPL, NVDA, MSFT, ...)"`` suitable for
    compact log entries.  Uses the observation's universe for tickers and
    regex extraction from enriched_context for the quarter/date.

    Args:
        enriched_context: The enriched context string (contains memo header).
        observation: The observation dict (has ``universe`` key).

    Returns:
        Human-readable snapshot identifier, or ``"N/A"`` if nothing found.
    """
    # Tickers from observation
    universe_list = observation.get("universe", [])
    universe = ", ".join(t.upper() for t in universe_list)
    if not universe:
        m = re.search(r"Allocation universe:\s*(.+)", enriched_context)
        if m:
            universe = m.group(1).strip()

    # Quarter from enriched context header
    quarter = ""
    m2 = re.search(r"As-of:\s*(\S+)", enriched_context)
    if m2:
        quarter = m2.group(1)

    if quarter and universe:
        return f"{quarter} ({universe})"
    if quarter:
        return quarter
    if universe:
        return f"({universe})"
    return "N/A"


def _compact_user_prompt(user_prompt: str, config: dict) -> str:
    """Replace the memo body in the user prompt with a compact placeholder.

    Keeps everything before and after the memo (allocation instructions,
    causal scaffolding, JSON schema) but swaps the large memo text for a
    one-line summary with tickers and quarter.
    """
    # Detect memo start marker
    memo_start = user_prompt.find("[INFO] QUARTERLY SNAPSHOT MEMO")
    if memo_start == -1:
        # Non-memo mode or no memo marker — return as-is
        return user_prompt

    # Everything before the memo (header: cash, universe, as-of)
    before = user_prompt[:memo_start]

    # Find the end of the memo: the next section that starts with a known
    # template block.  These follow the {{ context }} variable.
    candidates = [
        "CRITICAL — Evidence citation rules:",
        "Using the data above",
        "## Causal Claim Requirements",
        "## Mandatory Uncertainty Disclosure",
        "## Causal Reasoning Traps",
        "## Your Task",
        "## Financial Context",
        "Respond with valid JSON",
    ]
    memo_end = len(user_prompt)
    for marker in candidates:
        idx = user_prompt.find(marker, memo_start)
        if idx != -1 and idx < memo_end:
            memo_end = idx

    after = user_prompt[memo_end:]

    # Build compact placeholder
    roles = config.get("roles", [])
    universe = ", ".join(t.upper() for t in config.get("_universe", []))
    if not universe:
        # Extract universe from the header if not in config
        import re as _re
        m = _re.search(r"Allocation universe:\s*(.+)", before)
        if m:
            universe = m.group(1).strip()

    quarter = config.get("_invest_quarter", "")
    if not quarter:
        m2 = re.search(r"As-of:\s*(\S+)", before)
        if m2:
            quarter = m2.group(1)

    placeholder = (
        f"[MEMO CONTENT — {universe or 'N/A'}, as-of {quarter or 'N/A'}]\n"
        f"(Full memo omitted from log — {len(user_prompt[memo_start:memo_end]):,} chars)\n\n"
    )

    return before + placeholder + after


def _log_prompt(
    config: dict,
    role: str,
    phase: str,
    round_num: int,
    system_prompt: str,
    user_prompt: str,
) -> None:
    """Log the rendered system + user prompt via the debate.prompts logger.

    Activated by config["log_rendered_prompts"] = True.  Emits at INFO level
    so it's visible with default logging.  The dedicated logger name
    ("debate.prompts") lets users filter prompt output separately from
    debate progress and PID metrics.

    The memo body in user prompts is replaced with a compact placeholder
    (tickers + quarter) to avoid flooding the log with 80K+ chars.
    """
    if not config.get("log_rendered_prompts"):
        return
    display_prompt = _compact_user_prompt(user_prompt, config)
    prompt_logger.info(
        "\n%s\n"
        "  %s | Round %d | %s\n"
        "%s\n"
        "--- SYSTEM PROMPT ---\n"
        "%s\n"
        "--- USER PROMPT (%d chars, showing compact) ---\n"
        "%s\n"
        "%s",
        "=" * 72,
        phase.upper(),
        round_num,
        role.upper(),
        "-" * 72,
        system_prompt,
        len(user_prompt),
        display_prompt,
        "=" * 72,
    )


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = match.group(1) if match else text
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
