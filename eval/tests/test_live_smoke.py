"""L5 live smoke tests: real LLM API calls.

These tests make actual API calls to verify end-to-end LLM integration.
They are gated behind the ``live`` pytest marker and skip automatically
if the OPENAI_API_KEY environment variable is not set.

Usage (must clear addopts to override the default ``-m 'not live'``):

    pytest -m live -o "addopts=" -v

Or target this file directly:

    pytest eval/tests/test_live_smoke.py -o "addopts=" -v
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_HAS_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))

pytestmark = pytest.mark.live

skip_no_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="OPENAI_API_KEY not set -- skipping live LLM test",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_llm_call(model: str, system_prompt: str, user_prompt: str) -> str:
    """Make a single LLM call via the project's _call_llm function."""
    from multi_agent.graph.llm import _call_llm

    config = {
        "model_name": model,
        "temperature": 0.2,
        "mock": False,
        "no_rate_limit": True,
        "llm_stagger_ms": 0,
        "max_concurrent_llm": 0,
    }
    return _call_llm(config, system_prompt, user_prompt)


def _parse_json_response(text: str) -> dict:
    """Attempt to parse JSON from LLM response, stripping code fences."""
    from multi_agent.graph.llm import _parse_json
    return _parse_json(text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_key
class TestCheapModelReturnsValidJSON:
    """gpt-4o-mini returns valid JSON with an 'allocation' key."""

    SYSTEM_PROMPT = (
        "You are a portfolio allocation assistant. "
        "Always respond with valid JSON containing an 'allocation' key "
        "mapping ticker symbols to float weights that sum to 1.0."
    )

    USER_PROMPT = (
        "Allocate a portfolio across AAPL, MSFT, and NVDA. "
        "Return JSON with format: {\"allocation\": {\"AAPL\": 0.33, \"MSFT\": 0.33, \"NVDA\": 0.34}}"
    )

    def test_response_is_valid_json(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        assert len(raw) > 0, "Empty response from LLM"

        parsed = _parse_json_response(raw)
        assert isinstance(parsed, dict), f"Response is not a dict: {raw[:200]}"

    def test_response_has_allocation_key(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        parsed = _parse_json_response(raw)

        assert "allocation" in parsed, (
            f"Response missing 'allocation' key. Keys: {list(parsed.keys())}. "
            f"Raw: {raw[:300]}"
        )

    def test_allocation_weights_are_floats(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        parsed = _parse_json_response(raw)
        allocation = parsed.get("allocation", {})

        assert len(allocation) > 0, "Allocation is empty"
        for ticker, weight in allocation.items():
            assert isinstance(weight, (int, float)), (
                f"Weight for {ticker} is {type(weight).__name__}, expected float"
            )

    def test_allocation_sums_to_approximately_one(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        parsed = _parse_json_response(raw)
        allocation = parsed.get("allocation", {})

        total = sum(float(v) for v in allocation.values())
        assert abs(total - 1.0) < 0.05, (
            f"Allocation sums to {total:.4f}, expected ~1.0"
        )


@skip_no_key
@pytest.mark.skip(reason="Production model test -- enable manually for full validation")
class TestProductionModelCRITFormat:
    """Production model returns CRIT-format keys (pillar_scores, diagnostics)."""

    SYSTEM_PROMPT = (
        "You are a CRIT reasoning quality evaluator. "
        "Evaluate the following investment thesis and respond with valid JSON "
        "containing these keys: 'pillar_scores' (with sub-keys: "
        "'logical_validity', 'evidential_support', 'alternative_consideration', "
        "'causal_alignment' -- each a float 0-1), and 'diagnostics' (with boolean "
        "sub-keys: 'contradictions_detected', 'unsupported_claims_detected', "
        "'ignored_critiques_detected', 'premature_certainty_detected', "
        "'causal_overreach_detected', 'conclusion_drift_detected')."
    )

    USER_PROMPT = (
        "Thesis: 'Equal-weight allocation across AAPL, MSFT, NVDA is optimal "
        "because technology sector fundamentals remain strong [L1-VIX]. "
        "Diversification across large-cap tech reduces idiosyncratic risk.' "
        "Evaluate this thesis."
    )

    def test_response_has_pillar_scores(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        parsed = _parse_json_response(raw)

        assert "pillar_scores" in parsed, (
            f"Missing 'pillar_scores'. Keys: {list(parsed.keys())}"
        )
        pillars = parsed["pillar_scores"]
        expected_pillars = [
            "logical_validity",
            "evidential_support",
            "alternative_consideration",
            "causal_alignment",
        ]
        for p in expected_pillars:
            assert p in pillars, f"Missing pillar: {p}"
            assert isinstance(pillars[p], (int, float)), (
                f"Pillar {p} should be numeric, got {type(pillars[p]).__name__}"
            )

    def test_response_has_diagnostics(self):
        raw = _single_llm_call("gpt-4o-mini", self.SYSTEM_PROMPT, self.USER_PROMPT)
        parsed = _parse_json_response(raw)

        assert "diagnostics" in parsed, (
            f"Missing 'diagnostics'. Keys: {list(parsed.keys())}"
        )
        diag = parsed["diagnostics"]
        expected_diag = [
            "contradictions_detected",
            "unsupported_claims_detected",
        ]
        for d in expected_diag:
            assert d in diag, f"Missing diagnostic: {d}"
