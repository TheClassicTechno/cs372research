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

# Stagger concurrent LLM calls to avoid 429 rate limits.
# Parallel agents fire 4 calls at once per phase; this stagger spreads
# call starts by a configurable interval (default 200ms).  All calls
# still overlap during execution — total phase time increases by only
# (N-1) * stagger_ms, which is negligible vs LLM latency.
_LLM_STAGGER_LOCK = threading.Lock()
_LLM_LAST_CALL: float = 0.0
_DEFAULT_STAGGER_MS: int = 500


def _stagger_wait(stagger_ms: int) -> None:
    """Wait until at least ``stagger_ms`` since the last LLM call start.

    Serializes call *starts* so parallel threads don't all hit the API
    at t=0.  The actual API calls still overlap during execution.
    """
    import time

    global _LLM_LAST_CALL
    with _LLM_STAGGER_LOCK:
        now = time.monotonic()
        delay = (stagger_ms / 1000) - (now - _LLM_LAST_CALL)
        if delay > 0:
            time.sleep(delay)
        _LLM_LAST_CALL = time.monotonic()


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract retry-after hint from an API rate-limit error message.

    OpenAI errors include text like "Please try again in 610ms" or
    "Please try again in 1.2s".  Returns seconds, or None if not found.
    """
    msg = str(exc)
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", msg, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2).lower() == "ms":
        value /= 1000.0
    return value


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error."""
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    if hasattr(exc, "status_code") and exc.status_code == 429:
        return True
    if "429" in str(exc)[:200]:
        return True
    return False


def _retry_wait(exc: Exception, attempt: int) -> float:
    """Compute wait time for a retry attempt.

    For rate-limit errors: use the API's retry-after hint (+ 0.5s buffer),
    falling back to longer exponential backoff (2s, 4s, 8s, 16s, 32s).
    For other errors: standard exponential backoff (1s, 2s, 4s, 8s, 16s, 32s).
    """
    if _is_rate_limit_error(exc):
        hint = _parse_retry_after(exc)
        if hint is not None:
            return hint + 0.5  # add buffer
        return 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
    return 2 ** attempt  # 1s, 2s, 4s, 8s, 16s, 32s


def _call_llm(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Call LLM with the given system and user prompts. Returns raw text.

    Retries up to 6 times on transient errors.  For rate-limit (429)
    errors, uses the API's retry-after hint when available; otherwise
    falls back to exponential backoff.
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

    max_retries = 6
    stagger_ms = 0 if config.get("no_rate_limit", False) else config.get("llm_stagger_ms", _DEFAULT_STAGGER_MS)
    for attempt in range(max_retries):
        try:
            if stagger_ms > 0:
                _stagger_wait(stagger_ms)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return response.content
        except Exception as e:
            wait = _retry_wait(e, attempt)
            if attempt < max_retries - 1:
                print(f"  [LLM RETRY] {type(e).__name__} — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...", flush=True)
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


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles: ```json ... ```, ``` ... ```, and unclosed ``` prefixes.
    """
    # Closed fence: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Unclosed fence: ```json\n{...  (no closing ```)
    match = re.match(r"^\s*```(?:json)?\s*\n([\s\S]*)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    json_str = _strip_code_fences(text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
