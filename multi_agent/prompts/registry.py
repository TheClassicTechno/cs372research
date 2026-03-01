"""Modular prompt registry for RAudit PID-controlled debates.

This module provides the PromptRegistry — a layered prompt builder that
separates concerns (role identity, phase preamble, tone injection) into
composable blocks.  It is activated ONLY when PID is enabled
(``is_modular_mode`` returns True); legacy prompts are untouched.

==========================================================================
β → TONE MAPPING (CORRECTED RAudit SEMANTICS)
==========================================================================

RAudit paper references:
    - Section 3.5 (p.4): "Contentiousness (β): Modulates adversarial prompting."
    - Algorithm 1, line 5 (p.19): "Generate traces with contentiousness β^(t-1)"
    - Table 1 (p.4): Stuck → β↑ EXPLORE (adversarial); Converged → β↓ (collaborative)

The CORRECT mapping:
    high β  →  adversarial   (push agents harder, more exploration)
    low  β  →  collaborative (ease off, converge)

This FIXES the legacy inversion where high β → high agreeableness → agreeable tone.

==========================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AgentRole
from . import ROLE_SYSTEM_PROMPTS

logger = logging.getLogger("prompt.build")

_TONE_DIR = Path(__file__).resolve().parent / "tone"

# Cache for loaded tone files (module-level singleton)
_tone_cache: dict[str, str] = {}


def reset_registry_cache() -> None:
    """Clear the tone file cache. Call at the start of each run()."""
    _tone_cache.clear()


def _load_tone(filename: str) -> str:
    """Load and cache a tone file from the tone/ directory."""
    if filename not in _tone_cache:
        path = _TONE_DIR / filename
        if not path.exists():
            logger.warning("Tone file not found: %s", path)
            return ""
        _tone_cache[filename] = path.read_text()
    return _tone_cache[filename]


def beta_to_bucket(beta: float) -> str:
    """Map PID β to tone bucket (RAudit-aligned).

    RAudit paper references:
        - Section 3.5 (p.4): "Contentiousness (β): Modulates adversarial prompting."
        - Algorithm 1, line 5 (p.19): "Generate traces R_i and beliefs p_i
          with contentiousness β^(t-1)"
        - Table 1 (p.4): Stuck regime increases β for EXPLORE (adversarial),
          Converged/Healthy decrease β via decay (collaborative).

    RAudit semantics: high β = contentious/explore, low β = collaborative/exploit.
    Therefore:
        high β → adversarial   (push agents harder)
        low β  → collaborative (ease off, converge)

    Args:
        beta: Current contentiousness dial value in [0, 1].

    Returns:
        One of "collaborative", "balanced", or "adversarial".
    """
    if beta < 0.33:
        return "collaborative"
    if beta < 0.67:
        return "balanced"
    return "adversarial"


def is_modular_mode(config: dict) -> bool:
    """Check whether the modular prompt path should be used.

    Modular mode activates when PID is enabled in the config dict.
    When False, graph.py falls through to legacy prompt builders.

    Args:
        config: The LangGraph state config dict (state["config"]).
    """
    return config.get("pid_enabled", False)


# =========================================================================
# Prompt recipe definitions
# =========================================================================

# Each recipe defines which system-prompt blocks to assemble for a given
# (role_pattern, phase) combination.  Blocks are assembled in order:
#   role_system → phase_preamble → tone (tone LAST for maximum LLM attention)

_CRITIQUE_PREAMBLE = "Provide explicit, substantive critiques."
_REVISE_PREAMBLE = "Revise your proposal based on critiques."

# Tone file naming convention: {phase}_{bucket}.txt
_TONE_FILES = {
    ("critique", "adversarial"):  "critique_adversarial.txt",
    ("critique", "balanced"):     "critique_balanced.txt",
    ("critique", "collaborative"): "critique_collaborative.txt",
    ("revise", "adversarial"):    "revise_adversarial.txt",
    ("revise", "balanced"):       "revise_balanced.txt",
    ("revise", "collaborative"):  "revise_collaborative.txt",
}


@dataclass
class PromptBuildResult:
    """Result of building a modular prompt."""
    system_prompt: str
    user_prompt: str
    tone_file: str = ""
    beta_bucket: str = ""
    blocks_used: list[str] = field(default_factory=list)


class PromptRegistry:
    """Layered prompt builder for PID-controlled debates.

    Assembles system prompts from composable blocks:
        1. Role system prompt (identity + expertise)
        2. Phase preamble (critique/revise instructions)
        3. Tone injection (β-driven adversarial/balanced/collaborative)

    Tone is placed LAST in the system prompt for maximum LLM attention
    (recency bias in transformer attention patterns).
    """

    def __init__(self, prompt_logging: dict | None = None):
        self._prompt_logging = prompt_logging or {}

    def build(
        self,
        role: str,
        phase: str,
        beta: float | None = None,
        user_prompt: str = "",
    ) -> PromptBuildResult:
        """Build a modular prompt for a given role and phase.

        Args:
            role: Agent role (e.g. "macro", "value", "risk").
            phase: Debate phase ("propose", "critique", "revise", "judge").
            beta: Current PID β value. None = no tone injection.
            user_prompt: The user/task prompt (already rendered by legacy builder).

        Returns:
            PromptBuildResult with assembled system_prompt and forwarded user_prompt.
        """
        blocks_used = []
        parts = []

        # Block 1: Role system prompt
        role_system = ROLE_SYSTEM_PROMPTS.get(
            role, ROLE_SYSTEM_PROMPTS.get(AgentRole.MACRO, "")
        )
        if phase in ("critique", "revise"):
            # For critique/revise, use the standard role + phase preamble
            preamble = _CRITIQUE_PREAMBLE if phase == "critique" else _REVISE_PREAMBLE
            parts.append(f"You are the {role.upper()} agent. {preamble}")
            blocks_used.append("role_preamble")
        elif phase == "judge":
            parts.append(
                "You are the Judge. Synthesize the debate and produce final orders with an audited memo."
            )
            blocks_used.append("judge_system")
        else:
            # propose: use full role system prompt
            parts.append(role_system)
            blocks_used.append("role_system")

        # Block 2: Tone injection (critique/revise only, when β is provided)
        tone_file = ""
        beta_bucket = ""
        if beta is not None and phase in ("critique", "revise"):
            beta_bucket = beta_to_bucket(beta)
            tone_key = (phase, beta_bucket)
            tone_filename = _TONE_FILES.get(tone_key, "")
            if tone_filename:
                tone_text = _load_tone(tone_filename)
                if tone_text:
                    parts.append(tone_text)
                    tone_file = tone_filename
                    blocks_used.append("tone")

        system_prompt = "\n".join(parts)

        result = PromptBuildResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tone_file=tone_file,
            beta_bucket=beta_bucket,
            blocks_used=blocks_used,
        )

        # Logging (config-gated)
        if self._prompt_logging.get("enabled"):
            self._log_build(role, phase, result)

        return result

    def _log_build(self, role: str, phase: str, result: PromptBuildResult) -> None:
        """Log prompt build details when prompt_logging is enabled."""
        pl = self._prompt_logging

        msg_parts = [
            f"[PromptBuild]",
            f"Role: {role}",
            f"Phase: {phase}",
        ]

        if pl.get("log_beta_bucket") and result.beta_bucket:
            msg_parts.append(f"Beta bucket: {result.beta_bucket}")
            if result.tone_file:
                msg_parts.append(f"Tone file: tone/{result.tone_file}")

        if pl.get("log_selected_blocks"):
            msg_parts.append(f"System blocks: {result.blocks_used}")

        if pl.get("log_rendered_prompt"):
            max_chars = pl.get("max_prompt_log_chars", 2000)
            sys_preview = result.system_prompt[:max_chars]
            usr_preview = result.user_prompt[:max_chars]
            msg_parts.append(f"--- SYSTEM PROMPT ---\n{sys_preview}")
            msg_parts.append(f"--- USER PROMPT ---\n{usr_preview}")

        logger.info("\n".join(msg_parts))


# Module-level singleton registry (lazy-initialized)
_registry: PromptRegistry | None = None


def get_registry(config: dict) -> PromptRegistry:
    """Get or create the module-level PromptRegistry singleton.

    Args:
        config: LangGraph state config dict. Reads 'prompt_logging' key.
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry(
            prompt_logging=config.get("prompt_logging"),
        )
    return _registry
