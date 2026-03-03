"""Tests for allocation_to_decision() and _parse_action() allocation passthrough."""

from __future__ import annotations

import pytest

from agents.multi_agent_debate import allocation_to_decision
from models.decision import Decision
from models.decision import Order as SimOrder


# ── allocation_to_decision ───────────────────────────────────────────────────


class TestAllocationToDecision:
    def test_basic_conversion(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 0.5, "MSFT": 0.5},
            prices={"AAPL": 100.0, "MSFT": 200.0},
            cash=100_000.0,
        )
        # AAPL: 50000 / 100 = 500 shares
        # MSFT: 50000 / 200 = 250 shares
        orders = {o.ticker: o for o in decision.orders}
        assert orders["AAPL"].quantity == 500
        assert orders["MSFT"].quantity == 250

    def test_zero_weight_skipped(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 0.5, "MSFT": 0.0},
            prices={"AAPL": 100.0, "MSFT": 200.0},
            cash=100_000.0,
        )
        tickers = [o.ticker for o in decision.orders]
        assert "MSFT" not in tickers
        assert "AAPL" in tickers

    def test_negative_weight_skipped(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 0.5, "MSFT": -0.1},
            prices={"AAPL": 100.0, "MSFT": 200.0},
            cash=100_000.0,
        )
        tickers = [o.ticker for o in decision.orders]
        assert "MSFT" not in tickers

    def test_missing_price_skipped(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 0.5, "XYZ": 0.5},
            prices={"AAPL": 100.0},  # XYZ not in prices
            cash=100_000.0,
        )
        tickers = [o.ticker for o in decision.orders]
        assert "XYZ" not in tickers
        assert len(decision.orders) == 1

    def test_zero_price_skipped(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 1.0},
            prices={"AAPL": 0.0},
            cash=100_000.0,
        )
        assert len(decision.orders) == 0

    def test_floor_rounding(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 1.0},
            prices={"AAPL": 33.33},
            cash=100.0,
        )
        # 100 / 33.33 = 3.0003 → floor = 3
        assert decision.orders[0].quantity == 3

    def test_small_allocation_zero_shares(self):
        """Weight so small that dollar amount < price → 0 shares → skipped."""
        decision = allocation_to_decision(
            allocation={"AAPL": 0.001},
            prices={"AAPL": 200.0},
            cash=100.0,
        )
        # 0.001 * 100 = $0.10 → 0.10 / 200 = 0 shares
        assert len(decision.orders) == 0

    def test_all_orders_are_buys(self):
        decision = allocation_to_decision(
            allocation={"AAPL": 0.5, "MSFT": 0.5},
            prices={"AAPL": 100.0, "MSFT": 200.0},
            cash=100_000.0,
        )
        for order in decision.orders:
            assert order.side == "buy"

    def test_empty_allocation(self):
        decision = allocation_to_decision(
            allocation={},
            prices={"AAPL": 100.0},
            cash=100_000.0,
        )
        assert decision.orders == []


# ── _parse_action allocation passthrough ─────────────────────────────────────


class TestParseActionAllocation:
    """Test that MultiAgentRunner._parse_action preserves the allocation field."""

    def _make_runner(self):
        """Create a minimal MultiAgentRunner for testing _parse_action."""
        from multi_agent.config import AgentRole, DebateConfig
        from multi_agent.runner import MultiAgentRunner

        config = DebateConfig(
            mock=True,
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            allocation_mode=True,
            trace_dir="/tmp/test_traces",
        )
        return MultiAgentRunner(config)

    def test_allocation_passthrough(self):
        runner = self._make_runner()
        action = runner._parse_action({
            "allocation": {"AAPL": 0.6, "MSFT": 0.4},
            "justification": "test allocation",
            "confidence": 0.7,
            "claims": [],
        })
        assert action.allocation == {"AAPL": 0.6, "MSFT": 0.4}
        assert action.confidence == pytest.approx(0.7)

    def test_allocation_none_when_absent(self):
        runner = self._make_runner()
        action = runner._parse_action({
            "orders": [],
            "justification": "legacy mode",
            "confidence": 0.5,
        })
        assert action.allocation is None
