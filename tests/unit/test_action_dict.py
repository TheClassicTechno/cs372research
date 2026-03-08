"""Tests for the _build_action_dict helper in nodes.py."""

from multi_agent.graph.nodes import _build_action_dict


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
    result = {
        "justification": "Value play.",
        "confidence": 0.7,
        "allocation": {"GOOG": 1.0},
    }
    ad = _build_action_dict(result, {"GOOG": 1.0}, ["GOOG"], {})
    assert ad["justification"] == "Value play."
    assert ad["portfolio_rationale"] == ""
    assert ad["position_rationale"] == []
    assert ad["risks_or_falsifiers"] == []


def test_justification_priority():
    """When both justification and portfolio_rationale exist, justification wins."""
    result = {
        "justification": "Direct justification.",
        "portfolio_rationale": "Portfolio-level rationale.",
    }
    ad = _build_action_dict(result, {}, [], {})
    assert ad["justification"] == "Direct justification."
    assert ad["portfolio_rationale"] == "Portfolio-level rationale."


def test_empty_result():
    """Empty dict → all defaults."""
    ad = _build_action_dict({}, {}, [], {})
    assert ad["justification"] == ""
    assert ad["portfolio_rationale"] == ""
    assert ad["position_rationale"] == []
    assert ad["confidence"] == 0.5
    assert ad["claims"] == []
    assert ad["risks_or_falsifiers"] == []
    assert ad["allocation"] == {}


def test_allocation_normalized():
    """Verify constraints flow through to normalize_allocation."""
    raw = {"AAPL": 0.9, "MSFT": 0.1}
    config = {"allocation_constraints": {"max_weight": 0.5, "min_holdings": 2}}
    ad = _build_action_dict({}, raw, ["AAPL", "MSFT"], config)
    # max_weight=0.5 should cap AAPL
    assert ad["allocation"]["AAPL"] <= 0.5 + 1e-9
    # Both tickers should be present (min_holdings=2)
    assert "AAPL" in ad["allocation"]
    assert "MSFT" in ad["allocation"]
    # Should still sum to ~1.0
    total = sum(ad["allocation"].values())
    assert abs(total - 1.0) < 1e-9
