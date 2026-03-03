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
from . import ROLE_SYSTEM_PROMPTS, SYSTEM_CAUSAL_CONTRACT, get_role_prompts

logger = logging.getLogger("prompt.build")

_PROMPT_DIR = Path(__file__).resolve().parent
_TONE_DIR = _PROMPT_DIR / "tone"

_DEFAULT_BLOCK_ORDER = ["causal_contract", "role_system", "phase_preamble", "tone"]

# Cache for loaded tone files (module-level singleton)
_tone_cache: dict[str, str] = {}


def reset_registry_cache() -> None:
    """Clear the tone file cache. Call at the start of each run()."""
    _tone_cache.clear()


def _load_prompt_file(filename: str) -> str:
    """Load a prompt file by name from the prompts directory.

    Looks first in the main prompts dir, then in tone/ subdirectory.
    Returns empty string if not found.
    """
    path = _PROMPT_DIR / filename
    if path.exists():
        return path.read_text()
    # Fallback: check tone subdirectory
    tone_path = _TONE_DIR / filename
    if tone_path.exists():
        return tone_path.read_text()
    logger.warning("Prompt file not found: %s", filename)
    return ""


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
        use_system_causal_contract: bool = False,
        block_order: list[str] | None = None,
        prompt_file_overrides: dict[str, str] | None = None,
    ) -> PromptBuildResult:
        """Build a modular prompt for a given role and phase.

        Args:
            role: Agent role (e.g. "macro", "value", "risk").
            phase: Debate phase ("propose", "critique", "revise", "judge").
            beta: Current PID β value. None = no tone injection.
            user_prompt: The user/task prompt (already rendered by legacy builder).
            use_system_causal_contract: When True, prepend shared causal contract
                and use slimmed role prompts.
            block_order: Custom ordering of system prompt blocks.
                None → uses _DEFAULT_BLOCK_ORDER.
            prompt_file_overrides: Override which .txt file to load for a block.
                Keys like "causal_contract", "role_<rolename>", "phase_preamble_critique".
                Values are filenames relative to the prompts directory.

        Returns:
            PromptBuildResult with assembled system_prompt and forwarded user_prompt.
        """
        overrides = prompt_file_overrides or {}
        order = block_order if block_order is not None else _DEFAULT_BLOCK_ORDER

        # --- Build a dict of block_name → content for all available blocks ---
        available: dict[str, str] = {}

        # causal_contract (only when enabled)
        if use_system_causal_contract:
            override_file = overrides.get("causal_contract")
            if override_file:
                available["causal_contract"] = _load_prompt_file(override_file)
            else:
                available["causal_contract"] = SYSTEM_CAUSAL_CONTRACT

        # role_system / judge_system
        if phase == "judge":
            override_file = overrides.get("judge_system")
            if override_file:
                available["judge_system"] = _load_prompt_file(override_file)
            else:
                available["judge_system"] = (
                    "You are the Judge. Synthesize the debate and produce final orders with an audited memo."
                )
            # Map role_system → judge_system so ordering with "role_system" still works
            available["role_system"] = available["judge_system"]
        else:
            override_file = overrides.get(f"role_{role}")
            if override_file:
                available["role_system"] = _load_prompt_file(override_file)
            else:
                rp = get_role_prompts(use_system_causal_contract)
                available["role_system"] = rp.get(role, rp.get(AgentRole.MACRO, ""))

        # phase_preamble (critique/revise only)
        if phase in ("critique", "revise"):
            override_key = f"phase_preamble_{phase}"
            override_file = overrides.get(override_key)
            if override_file:
                available["phase_preamble"] = _load_prompt_file(override_file)
            else:
                available["phase_preamble"] = (
                    _CRITIQUE_PREAMBLE if phase == "critique" else _REVISE_PREAMBLE
                )

        # tone (critique/revise only, when β is provided)
        tone_file = ""
        beta_bucket = ""
        if beta is not None and phase in ("critique", "revise"):
            beta_bucket = beta_to_bucket(beta)
            tone_key = (phase, beta_bucket)
            override_file = overrides.get(f"tone_{phase}_{beta_bucket}")
            if override_file:
                tone_text = _load_prompt_file(override_file)
            else:
                tone_filename = _TONE_FILES.get(tone_key, "")
                tone_text = _load_tone(tone_filename) if tone_filename else ""
                tone_file = tone_filename
            if tone_text:
                available["tone"] = tone_text
                if not tone_file and not override_file:
                    pass  # no tone file loaded
                elif override_file:
                    tone_file = override_file

        # --- Assemble in specified order, skipping unavailable blocks ---
        blocks_used = []
        parts = []
        for block_name in order:
            content = available.get(block_name)
            if content:
                parts.append(content)
                blocks_used.append(block_name)

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
