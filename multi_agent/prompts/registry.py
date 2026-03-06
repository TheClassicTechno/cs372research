"""Unified prompt registry for debate system prompt assembly.

This module provides the PromptRegistry — a layered prompt builder that
separates concerns (role identity, phase preamble, tone injection) into
composable blocks.  All phases route through ``PromptRegistry.build()``
with tone derived from ``resolve_beta()`` (PID β → tone bucket).

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
from . import ROLE_SYSTEM_PROMPTS, SYSTEM_CAUSAL_CONTRACT, get_role_prompts, load_module

# Phase preambles — keyed by phase name (kept for backward compat with old profiles).
_PHASE_PREAMBLES = {
    "critique": "Provide explicit, substantive critiques.",
    "revise": "Revise your proposal based on critiques.",
}


class PromptCompositionError(Exception):
    """Raised when prompt composition violates profile declarations."""


def _validate_composition(
    profile_blocks: list[str],
    blocks_loaded: list[str],
    phase: str,
    beta: float | None,
) -> None:
    """Assert every declared block is accounted for.

    Only 'tone' is allowed to be absent (when beta is None).
    All other entries must resolve to a non-empty file.

    Raises:
        PromptCompositionError: If a declared block is missing from the
            rendered prompt.
    """
    for entry in profile_blocks:
        if entry == "tone" and beta is None:
            continue
        if entry not in blocks_loaded:
            raise PromptCompositionError(
                f"Block {entry!r} declared in profile but missing "
                f"from rendered {phase} prompt. blocks_loaded={blocks_loaded}"
            )

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


def resolve_beta(config: dict, phase: str) -> float | None:
    """Get PID beta for tone selection.

    For critique/revise: returns ``_current_beta`` set by the PID runner.
    For propose/judge: always returns ``None`` (no tone injection).

    Args:
        config: The LangGraph state config dict (state["config"]).
        phase: Debate phase — "propose", "critique", "revise", or "judge".

    Returns:
        Beta value in [0, 1] for critique/revise, or None for propose/judge.
    """
    if phase not in ("critique", "revise"):
        return None
    return config.get("_current_beta")


# =========================================================================
# Prompt file manifest (for once-per-round logging)
# =========================================================================

# Default user-prompt template files per phase.
_DEFAULT_PHASE_TEMPLATES = {
    "propose": "phases/proposal_allocation.txt",
    "critique": "phases/critique_allocation.txt",
    "revise": "phases/revision_allocation.txt",
    "judge": "phases/judge_allocation.txt",
}

# Override key names used in prompt_file_overrides for each phase template.
_PHASE_TEMPLATE_OVERRIDE_KEYS = {
    "propose": "proposal_template",
    "critique": "critique_template",
    "revise": "revision_template",
    "judge": "judge_template",
}


def build_prompt_manifest(config: dict) -> dict:
    """Return prompt file names for all phases, without loading content.

    Mirrors the file-selection logic in ``PromptRegistry.build()`` and the
    ``build_*_prompt()`` functions but only resolves file names — no file
    I/O.  Called once per round by the runner for compact manifest logging.

    Args:
        config: The LangGraph state config dict (state["config"]).

    Returns:
        Dict with keys: role_files, tone, beta, beta_bucket, phase_templates.
    """
    overrides = config.get("prompt_file_overrides", {})
    roles = config.get("roles", [])

    manifest: dict[str, Any] = {}

    # --- Role files ---
    role_files: dict[str, str] = {}
    for role in roles:
        override = overrides.get(f"role_{role}")
        if override:
            role_files[role] = override
        else:
            role_files[role] = f"roles/{role}.txt"
    manifest["role_files"] = role_files

    # --- Tone files (critique/revise only) ---
    beta = resolve_beta(config, "critique")
    if beta is not None:
        bucket = beta_to_bucket(beta)
        manifest["beta"] = beta
        manifest["beta_bucket"] = bucket
        tone_files: dict[str, str] = {}
        for phase in ("critique", "revise"):
            override = overrides.get(f"tone_{phase}_{bucket}")
            if override:
                tone_files[phase] = override
            else:
                filename = _TONE_FILES.get((phase, bucket), "")
                tone_files[phase] = f"tone/{filename}" if filename else ""
        manifest["tone"] = tone_files

    # --- Phase templates (user prompts) ---
    phase_templates: dict[str, str] = {}
    for phase, default_file in _DEFAULT_PHASE_TEMPLATES.items():
        override_key = _PHASE_TEMPLATE_OVERRIDE_KEYS[phase]
        phase_templates[phase] = overrides.get(override_key, default_file)
    manifest["phase_templates"] = phase_templates

    return manifest


# =========================================================================
# Tone file naming convention: {phase}_{bucket}.txt
# =========================================================================
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

    def _load_block(self, block_name, role, phase, beta, overrides):
        """Unified block loader — no gates, profile is the single source of truth."""
        # Check override first
        override_file = overrides.get(block_name)
        if block_name.startswith("role_") and not override_file:
            override_file = overrides.get(f"role_{role}")

        if override_file:
            return _load_prompt_file(override_file)

        # Known block types
        if block_name == "role_system":
            rp = get_role_prompts()
            return rp.get(role, "")

        if block_name == "judge_system":
            return "You are the Judge. Synthesize the debate and produce final orders with an audited memo."

        if block_name == "causal_contract":
            return SYSTEM_CAUSAL_CONTRACT

        if block_name == "phase_preamble":
            return _PHASE_PREAMBLES.get(phase, "")

        if block_name == "tone":
            if beta is None:
                return ""  # warning logged by caller
            bucket = beta_to_bucket(beta)
            filename = _TONE_FILES.get((phase, bucket), "")
            return _load_tone(filename) if filename else ""

        # Fallback: MODULE_CATALOG
        return load_module(block_name, overrides)

    def build(
        self,
        role: str,
        phase: str,
        beta: float | None = None,
        user_prompt: str = "",
        block_order: list[str] | None = None,
        prompt_file_overrides: dict[str, str] | None = None,
    ) -> PromptBuildResult:
        """Build a modular prompt for a given role and phase.

        The profile YAML is the single source of truth: if a block is listed
        in ``block_order``, it is loaded. No code gates override the profile.

        Args:
            role: Agent role (e.g. "macro", "value", "risk").
            phase: Debate phase ("propose", "critique", "revise", "judge").
            beta: Current PID β value. None = no tone injection.
            user_prompt: The user/task prompt (already rendered by builder).
            block_order: Ordered list of system prompt blocks from profile.
            prompt_file_overrides: Override which .txt file to load for a block.

        Returns:
            PromptBuildResult with assembled system_prompt and forwarded user_prompt.
        """
        overrides = prompt_file_overrides or {}
        order = block_order or []

        parts = []
        blocks_used = []
        tone_file = ""
        beta_bucket = ""

        for block_name in order:
            content = self._load_block(block_name, role, phase, beta, overrides)
            if content:
                parts.append(content)
                blocks_used.append(block_name)
                # Track tone metadata
                if block_name == "tone" and beta is not None:
                    beta_bucket = beta_to_bucket(beta)
                    tone_file = _TONE_FILES.get((phase, beta_bucket), "")
                    tone_override = overrides.get(f"tone_{phase}_{beta_bucket}")
                    if tone_override:
                        tone_file = tone_override
            else:
                logger.warning(
                    "System block '%s' listed in profile but produced no content "
                    "(phase=%s, role=%s)", block_name, phase, role,
                )

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

    def build_from_profile(
        self,
        role: str,
        phase: str,
        profile: dict,
        beta: float | None = None,
        user_prompt: str = "",
    ) -> PromptBuildResult:
        """Build a prompt using the new agent profile format.

        The profile dict contains explicit file paths under
        ``system_prompts[phase]``. Each entry is loaded directly from
        ``multi_agent/prompts/``, except ``"tone"`` which resolves via
        the PID β → tone bucket mapping.

        Args:
            role: Agent role name.
            phase: Debate phase.
            profile: Loaded agent profile dict.
            beta: Current PID β (None = no tone).
            user_prompt: Pre-rendered user prompt.

        Returns:
            PromptBuildResult.
        """
        file_list = profile.get("system_prompts", {}).get(phase, [])

        parts = []
        blocks_used = []
        tone_file = ""
        beta_bucket = ""

        for entry in file_list:
            if entry == "tone":
                if beta is None:
                    continue
                bucket = beta_to_bucket(beta)
                beta_bucket = bucket
                filename = _TONE_FILES.get((phase, bucket), "")
                if filename:
                    content = _load_tone(filename)
                    tone_file = filename
                else:
                    content = ""
            else:
                # Load file directly from prompts directory
                content = _load_prompt_file(entry)

            if content:
                parts.append(content)
                blocks_used.append(entry)
            else:
                if entry == "tone" and beta is None:
                    continue
                logger.warning(
                    "System block '%s' declared in profile but produced no content "
                    "(phase=%s, role=%s)", entry, phase, role,
                )

        system_prompt = "\n".join(parts)

        # Composition validation
        _validate_composition(file_list, blocks_used, phase, beta)

        result = PromptBuildResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tone_file=tone_file,
            beta_bucket=beta_bucket,
            blocks_used=blocks_used,
        )

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
