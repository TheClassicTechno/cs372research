"""JSONL prompt logger for LLM call inspection.

Gated by the ENABLE_PROMPT_LOGGING environment variable.  When the env var
is not set (or not ``"true"``), ``log_prompt()`` is a no-op — safe to import
and call unconditionally.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def log_prompt(
    system: str,
    user: str,
    model: str,
    response: str | None = None,
    *,
    role: str = "",
    phase: str = "",
    round_num: int = 0,
) -> None:
    """Append one prompt trace to ``logs/prompt_traces.jsonl``.

    No-op unless ``ENABLE_PROMPT_LOGGING=true`` is set in the environment.
    Thread-safe via append mode.
    """
    if os.environ.get("ENABLE_PROMPT_LOGGING", "").lower() != "true":
        return

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "role": role,
        "phase": phase,
        "round": round_num,
        "system": system,
        "user": user,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response": response,
    }

    with open(os.path.join(log_dir, "prompt_traces.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
