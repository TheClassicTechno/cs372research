"""Tests for graph nodes in allocation mode: proposals, critiques, revisions,
judge, pipeline skip, context building, and print helpers.

All tests use mock mode (no API keys needed).
"""

import json

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.graph import (
    _mock_proposal,
    _mock_judge,
    _print_allocation,
    _print_critique_summary,
    build_context_node,
    propose_node,
    judge_node,
)
from multi_agent.models import (
    MarketState,
    Observation,
    PortfolioState,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


TICKERS = ["AAPL", "MSFT", "GOOG"]


@pytest.fixture
def alloc_obs_dict():
    """Observation dict for allocation mode (3 tickers, memo context)."""
    return Observation(
        timestamp="2024-12-31",
        universe=TICKERS,
        market_state=MarketState(
            prices={"AAPL": 185.0, "MSFT": 390.0, "GOOG": 140.0}
        ),
        text_context="Q4 2024 macro analysis memo with [L1-VIX] and [AAPL-RET60] data.",
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
    ).model_dump()


@pytest.fixture
def alloc_config_dict():
    """DebateConfig dict with mock=True for allocation mode tests."""
    return DebateConfig(
        mock=True,
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
        max_rounds=1,
        trace_dir="/tmp/test_traces",
    ).to_dict()


def _make_state(obs_dict, config_dict, **overrides):
    state = {
        "observation": obs_dict,
        "config": config_dict,
        "news_digest": "",
        "data_analysis": "",
        "enriched_context": "",
        "proposals": [],
        "critiques": [],
        "revisions": [],
        "current_round": 0,
        "debate_turns": [],
        "final_action": {},
        "strongest_objection": "",
        "audited_memo": "",
        "trace": {},
    }
    state.update(overrides)
    return state


# ── mock proposal in allocation mode ─────────────────────────────────────────


class TestMockProposalAllocation:
    def test_returns_allocation_key(self, alloc_obs_dict, alloc_config_dict):
        result = _mock_proposal("macro", alloc_obs_dict, alloc_config_dict)
        assert "allocation" in result
        assert "orders" not in result

    def test_equal_weight(self, alloc_obs_dict, alloc_config_dict):
        result = _mock_proposal("macro", alloc_obs_dict, alloc_config_dict)
        eq = 1.0 / len(TICKERS)
        for t in TICKERS:
            assert result["allocation"][t] == pytest.approx(eq, abs=1e-3)

    def test_has_confidence(self, alloc_obs_dict, alloc_config_dict):
        result = _mock_proposal("macro", alloc_obs_dict, alloc_config_dict)
        assert 0.0 <= result["confidence"] <= 1.0


# ── build_context allocation mode ────────────────────────────────────────────


class TestBuildContextAllocation:
    def _build(self, obs_dict, config_dict):
        state = _make_state(obs_dict, config_dict)
        return build_context_node(state)

    def test_header_present(self, alloc_obs_dict, alloc_config_dict):
        result = self._build(alloc_obs_dict, alloc_config_dict)
        assert "## Portfolio Allocation Task" in result["enriched_context"]

    def test_contains_cash(self, alloc_obs_dict, alloc_config_dict):
        result = self._build(alloc_obs_dict, alloc_config_dict)
        assert "$100,000.00" in result["enriched_context"]

    def test_contains_universe(self, alloc_obs_dict, alloc_config_dict):
        result = self._build(alloc_obs_dict, alloc_config_dict)
        for t in TICKERS:
            assert t in result["enriched_context"]

    def test_memo_text_included(self, alloc_obs_dict, alloc_config_dict):
        result = self._build(alloc_obs_dict, alloc_config_dict)
        assert "Q4 2024 macro analysis memo" in result["enriched_context"]


# ── propose_node allocation mode ─────────────────────────────────────────────


class TestProposeNodeAllocation:
    def _run_propose(self, obs_dict, config_dict):
        state = _make_state(obs_dict, config_dict, enriched_context="Test context")
        return propose_node(state)

    def test_proposals_have_allocation(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_propose(alloc_obs_dict, alloc_config_dict)
        for p in result["proposals"]:
            assert "allocation" in p["action_dict"]
            assert "orders" not in p["action_dict"]

    def test_allocations_normalized(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_propose(alloc_obs_dict, alloc_config_dict)
        for p in result["proposals"]:
            alloc = p["action_dict"]["allocation"]
            assert sum(alloc.values()) == pytest.approx(1.0)

    def test_all_roles_propose(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_propose(alloc_obs_dict, alloc_config_dict)
        assert len(result["proposals"]) == 3  # macro, value, risk

    def test_debate_turns_recorded(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_propose(alloc_obs_dict, alloc_config_dict)
        assert len(result["debate_turns"]) == 3


# ── judge_node allocation mode ───────────────────────────────────────────────


class TestJudgeNodeAllocation:
    def _run_judge(self, obs_dict, config_dict):
        # Build minimal proposals/revisions for judge
        proposals = []
        for role in ["macro", "value", "risk"]:
            eq = 1.0 / len(TICKERS)
            proposals.append({
                "role": role,
                "action_dict": {
                    "allocation": {t: eq for t in TICKERS},
                    "justification": f"Mock {role} allocation",
                    "confidence": 0.6,
                    "claims": [],
                },
            })

        state = _make_state(
            obs_dict, config_dict,
            enriched_context="Test context",
            proposals=proposals,
            revisions=proposals,  # use proposals as revisions for simplicity
            critiques=[],
            current_round=2,
        )
        return judge_node(state)

    def test_final_action_has_allocation(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_judge(alloc_obs_dict, alloc_config_dict)
        assert "allocation" in result["final_action"]

    def test_final_normalized(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_judge(alloc_obs_dict, alloc_config_dict)
        alloc = result["final_action"]["allocation"]
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_has_justification(self, alloc_obs_dict, alloc_config_dict):
        result = self._run_judge(alloc_obs_dict, alloc_config_dict)
        assert result["final_action"]["justification"]


# ── _print_allocation ────────────────────────────────────────────────────────


class TestPrintAllocation:
    def test_valid_allocation(self, capsys):
        _print_allocation("macro", {
            "allocation": {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
            "confidence": 0.75,
        })
        out = capsys.readouterr().out
        assert "MACRO" in out
        assert "AAPL" in out
        assert "75%" in out

    def test_empty_allocation(self, capsys):
        _print_allocation("macro", {"allocation": {}, "confidence": 0.5})
        # Should not crash; may produce minimal output
        capsys.readouterr()

    def test_missing_allocation(self, capsys):
        _print_allocation("macro", {})
        out = capsys.readouterr().out
        assert out == ""  # early return, nothing printed

    def test_zero_weight_listed(self, capsys):
        _print_allocation("risk", {
            "allocation": {"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
            "confidence": 0.8,
        })
        out = capsys.readouterr().out
        assert "zero" in out.lower()
        assert "GOOG" in out


# ── _print_critique_summary ──────────────────────────────────────────────────


class TestPrintCritiqueSummary:
    def test_empty_critiques(self, capsys):
        _print_critique_summary("macro", {})
        out = capsys.readouterr().out
        assert out == ""  # early return

    def test_formats_targets(self, capsys):
        result = {
            "critiques": [
                {"target_role": "value", "objection": "Too heavy on tech."},
                {"target_role": "risk", "objection": "Ignores volatility."},
            ],
        }
        _print_critique_summary("macro", result)
        out = capsys.readouterr().out
        assert "VALUE" in out
        assert "RISK" in out
