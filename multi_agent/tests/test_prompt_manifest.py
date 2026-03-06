"""Tests for prompt manifest logging.

Covers:
  1. build_prompt_manifest() file-name resolution
  2. _extract_snapshot_id() identifier extraction
  3. Config propagation (log_prompt_manifest field)
"""

from __future__ import annotations

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.prompts.registry import (
    _TONE_FILES,
    beta_to_bucket,
    build_prompt_manifest,
    resolve_beta,
)
from multi_agent.graph.llm import _extract_snapshot_id


# ── fixtures ─────────────────────────────────────────────────────────────────

DEFAULT_ROLES = ["macro", "value", "risk", "technical"]


def _make_config(**overrides) -> dict:
    """Build a minimal LangGraph-style config dict."""
    cfg = {
        "roles": DEFAULT_ROLES,
        "prompt_file_overrides": {},
    }
    cfg.update(overrides)
    return cfg


# ── build_prompt_manifest ────────────────────────────────────────────────────


class TestBuildPromptManifest:
    """Verify file-name resolution without loading content."""

    def test_role_files_are_full_variants(self):
        """Role files always use roles/{role}.txt (no slim variants)."""
        config = _make_config()
        m = build_prompt_manifest(config)
        for role in DEFAULT_ROLES:
            assert m["role_files"][role] == f"roles/{role}.txt"

    def test_role_file_override(self):
        config = _make_config(
            prompt_file_overrides={"role_macro": "custom/macro_v2.txt"},
        )
        m = build_prompt_manifest(config)
        assert m["role_files"]["macro"] == "custom/macro_v2.txt"
        assert m["role_files"]["value"] == "roles/value.txt"

    def test_tone_files_adversarial(self):
        config = _make_config(_current_beta=0.8)
        m = build_prompt_manifest(config)
        assert m["beta"] == 0.8
        assert m["beta_bucket"] == "adversarial"
        assert m["tone"]["critique"] == "tone/critique_adversarial.txt"
        assert m["tone"]["revise"] == "tone/revise_adversarial.txt"

    def test_tone_files_collaborative(self):
        config = _make_config(_current_beta=0.1)
        m = build_prompt_manifest(config)
        assert m["beta_bucket"] == "collaborative"
        assert m["tone"]["critique"] == "tone/critique_collaborative.txt"
        assert m["tone"]["revise"] == "tone/revise_collaborative.txt"

    def test_tone_files_balanced(self):
        config = _make_config(_current_beta=0.5)
        m = build_prompt_manifest(config)
        assert m["beta_bucket"] == "balanced"
        assert m["tone"]["critique"] == "tone/critique_balanced.txt"

    def test_no_tone_when_no_beta(self):
        """When no _current_beta is set, no tone key in manifest."""
        config = _make_config()
        m = build_prompt_manifest(config)
        # Without _current_beta, beta resolves to None → no tone
        assert "tone" not in m or m.get("beta") is None

    def test_tone_override(self):
        config = _make_config(
            _current_beta=0.5,
            prompt_file_overrides={"tone_critique_balanced": "custom/tone.txt"},
        )
        m = build_prompt_manifest(config)
        assert m["tone"]["critique"] == "custom/tone.txt"
        # Revise uses default since no override
        assert m["tone"]["revise"] == "tone/revise_balanced.txt"

    def test_phase_templates_defaults(self):
        config = _make_config()
        m = build_prompt_manifest(config)
        assert m["phase_templates"]["propose"] == "phases/proposal_allocation.txt"
        assert m["phase_templates"]["critique"] == "phases/critique_allocation.txt"
        assert m["phase_templates"]["revise"] == "phases/revision_allocation.txt"
        assert m["phase_templates"]["judge"] == "phases/judge_allocation.txt"

    def test_phase_template_override(self):
        config = _make_config(
            prompt_file_overrides={"proposal_template": "custom/propose.txt"},
        )
        m = build_prompt_manifest(config)
        assert m["phase_templates"]["propose"] == "custom/propose.txt"
        assert m["phase_templates"]["critique"] == "phases/critique_allocation.txt"


# ── _extract_snapshot_id ─────────────────────────────────────────────────────


class TestExtractSnapshotId:
    """Verify snapshot identifier extraction from state data."""

    def test_full_id_from_observation_and_context(self):
        observation = {"universe": ["AAPL", "NVDA", "MSFT"]}
        context = "Cash: $100,000\nAllocation universe: AAPL, NVDA, MSFT\nAs-of: 2025Q1\n"
        result = _extract_snapshot_id(context, observation)
        assert result == "2025Q1 (AAPL, NVDA, MSFT)"

    def test_quarter_only(self):
        observation = {}
        context = "As-of: 2024Q4\n"
        result = _extract_snapshot_id(context, observation)
        assert result == "2024Q4"

    def test_universe_only(self):
        observation = {"universe": ["GOOG", "META"]}
        context = ""
        result = _extract_snapshot_id(context, observation)
        assert result == "(GOOG, META)"

    def test_fallback_universe_from_context(self):
        observation = {}
        context = "Allocation universe: JPM, XOM, LLY\nAs-of: 2025Q1\n"
        result = _extract_snapshot_id(context, observation)
        assert result == "2025Q1 (JPM, XOM, LLY)"

    def test_returns_na_when_nothing_found(self):
        result = _extract_snapshot_id("", {})
        assert result == "N/A"

    def test_date_format_extracted(self):
        observation = {"universe": ["AAPL"]}
        context = "As-of: 2024-12-31\n"
        result = _extract_snapshot_id(context, observation)
        assert result == "2024-12-31 (AAPL)"


# ── Config propagation ───────────────────────────────────────────────────────


class TestConfigPropagation:
    """Verify log_prompt_manifest flows through DebateConfig."""

    def test_default_is_false(self):
        cfg = DebateConfig(mock=True)
        assert cfg.log_prompt_manifest is False

    def test_can_enable(self):
        cfg = DebateConfig(mock=True, log_prompt_manifest=True)
        assert cfg.log_prompt_manifest is True

    def test_in_to_dict(self):
        cfg = DebateConfig(mock=True, log_prompt_manifest=True)
        d = cfg.to_dict()
        assert d["log_prompt_manifest"] is True

    def test_disabled_in_to_dict(self):
        cfg = DebateConfig(mock=True)
        d = cfg.to_dict()
        assert d["log_prompt_manifest"] is False
