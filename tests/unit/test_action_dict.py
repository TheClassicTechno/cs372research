"""Tests for the _build_action_dict helper in nodes.py."""

import pytest

from multi_agent.graph.nodes import _build_action_dict


def _full_result(**overrides):
    """Minimal valid result dict with all required fields."""
    base = {
        "justification": "Default thesis.",
        "confidence": 0.7,
        "risks_or_falsifiers": ["default risk"],
        "allocation": {"AAPL": 1.0},
    }
    base.update(overrides)
    return base


def test_enriched_output():
    """Result with portfolio_rationale only → justification populated via fallback."""
    result = {
        "portfolio_rationale": "Strong macro tailwinds.",
        "position_rationale": [{"ticker": "AAPL", "rationale": "Momentum"}],
        "confidence": 0.8,
        "claims": ["claim1"],
        "risks_or_falsifiers": ["rate hike"],
        "allocation": {"AAPL": 0.6, "MSFT": 0.4},
    }
    ad = _build_action_dict(result, {"AAPL": 0.6, "MSFT": 0.4}, ["AAPL", "MSFT"], {})
    assert ad["justification"] == "Strong macro tailwinds."
    assert ad["portfolio_rationale"] == "Strong macro tailwinds."
    assert ad["position_rationale"] == [{"ticker": "AAPL", "rationale": "Momentum"}]
    assert ad["risks_or_falsifiers"] == ["rate hike"]


def test_base_output():
    """Result with justification only → used directly."""
    result = _full_result(justification="Value play.")
    ad = _build_action_dict(result, {"GOOG": 1.0}, ["GOOG"], {})
    assert ad["justification"] == "Value play."
    assert ad["portfolio_rationale"] == ""
    assert ad["position_rationale"] == []


def test_justification_priority():
    """When both justification and portfolio_rationale exist, justification wins."""
    result = _full_result(
        justification="Direct justification.",
        portfolio_rationale="Portfolio-level rationale.",
    )
    ad = _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})
    assert ad["justification"] == "Direct justification."
    assert ad["portfolio_rationale"] == "Portfolio-level rationale."


# ---------- Strict validation: crash on missing required fields ----------


def test_empty_result_crashes():
    """Empty dict → RuntimeError (no silent defaults)."""
    with pytest.raises(RuntimeError, match="empty or missing 'allocation'"):
        _build_action_dict({}, {}, [], {})


def test_missing_allocation_crashes():
    """Missing allocation in raw_alloc → RuntimeError."""
    result = _full_result()
    with pytest.raises(RuntimeError, match="empty or missing 'allocation'"):
        _build_action_dict(result, {}, ["AAPL"], {})


def test_missing_thesis_crashes():
    """No justification or portfolio_rationale → RuntimeError."""
    result = {"confidence": 0.5, "risks_or_falsifiers": ["x"], "allocation": {"AAPL": 1.0}}
    with pytest.raises(RuntimeError, match="no 'justification' or 'portfolio_rationale'"):
        _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})


def test_missing_confidence_crashes():
    """No confidence field → RuntimeError."""
    result = {"justification": "ok", "risks_or_falsifiers": ["x"], "allocation": {"AAPL": 1.0}}
    with pytest.raises(RuntimeError, match="no 'confidence'"):
        _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})


def test_missing_risks_crashes():
    """No risks_or_falsifiers → RuntimeError."""
    result = {"justification": "ok", "confidence": 0.5, "allocation": {"AAPL": 1.0}}
    with pytest.raises(RuntimeError, match="no 'risks_or_falsifiers'"):
        _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})


def test_critique_responses_extracted():
    """critique_responses is extracted (not silently dropped)."""
    cr = [{"from_agent": "macro", "target_claim": "C1", "disposition": "accept", "justification": "good point"}]
    result = _full_result(critique_responses=cr)
    ad = _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})
    assert ad["critique_responses"] == cr


def test_critique_responses_default_empty():
    """critique_responses defaults to [] when not present (optional for propose)."""
    result = _full_result()
    ad = _build_action_dict(result, {"AAPL": 1.0}, ["AAPL"], {})
    assert ad["critique_responses"] == []


def test_allocation_normalized():
    """Verify constraints flow through to normalize_allocation."""
    raw = {"AAPL": 0.9, "MSFT": 0.1}
    result = _full_result()
    config = {"allocation_constraints": {"max_weight": 0.5, "min_holdings": 2}}
    ad = _build_action_dict(result, raw, ["AAPL", "MSFT"], config)
    # max_weight=0.5 should cap AAPL
    assert ad["allocation"]["AAPL"] <= 0.5 + 1e-9
    # Both tickers should be present (min_holdings=2)
    assert "AAPL" in ad["allocation"]
    assert "MSFT" in ad["allocation"]
    # Should still sum to ~1.0
    total = sum(ad["allocation"].values())
    assert abs(total - 1.0) < 1e-9
