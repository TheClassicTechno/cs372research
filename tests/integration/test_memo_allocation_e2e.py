"""Integration tests for memo-based allocation mode.

Covers:
  1. DebateAgentSystem.invoke() with allocation_mode in mock
  2. MTM case detection
  3. Config propagation (YAML → AgentConfig → DebateConfig)
  4. Prompt template loading and rendering
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.multi_agent_debate import DebateAgentSystem
from models.agents import AgentInvocation
from models.case import Case, CaseData, CaseDataItem, StockData
from models.config import AgentConfig, SimulationConfig
from models.portfolio import PortfolioSnapshot
from multi_agent.prompts import (
    build_proposal_user_prompt,
    build_critique_prompt,
    build_revision_prompt,
    build_judge_prompt,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


TICKERS = ["AAPL", "MSFT", "GOOG"]


@pytest.fixture
def alloc_agent_config() -> AgentConfig:
    """AgentConfig with mock mode for allocation testing."""
    return AgentConfig(
        agent_system="multi_agent_debate",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        temperature=0.3,
        system_prompt_override="mock",
    )


@pytest.fixture
def alloc_case() -> Case:
    """A decision Case with 3 tickers and memo content."""
    return Case(
        case_data=CaseData(
            items=[CaseDataItem(kind="other", content="Q4 2024 analysis memo...")]
        ),
        stock_data={
            t: StockData(ticker=t, current_price=p, daily_bars=[])
            for t, p in [("AAPL", 185.0), ("MSFT", 390.0), ("GOOG", 140.0)]
        },
        portfolio=PortfolioSnapshot(cash=100_000.0, positions={}),
        case_id="memo/2025Q1",
    )


@pytest.fixture
def mtm_case() -> Case:
    """A mark-to-market case."""
    return Case(
        case_data=CaseData(items=[]),
        stock_data={
            t: StockData(ticker=t, current_price=p, daily_bars=[])
            for t, p in [("AAPL", 195.0), ("MSFT", 400.0), ("GOOG", 150.0)]
        },
        case_id="mtm/2025Q1",
    )


# ── adapter invoke ───────────────────────────────────────────────────────────


class TestAdapterInvokeAllocation:
    """End-to-end: AgentConfig → DebateAgentSystem → invoke → Decision."""

    def _invoke(self, config: AgentConfig, case: Case):
        agent = DebateAgentSystem(config)
        invocation = AgentInvocation(
            case=case,
            episode_id="test_ep",
            agent_id="test_agent",
            steps_remaining=0,
        )
        return asyncio.run(agent.invoke(invocation))

    def test_invoke_returns_buy_orders(self, alloc_agent_config, alloc_case):
        result = self._invoke(alloc_agent_config, alloc_case)
        assert len(result.decision.orders) > 0

    def test_all_orders_are_buys(self, alloc_agent_config, alloc_case):
        result = self._invoke(alloc_agent_config, alloc_case)
        for order in result.decision.orders:
            assert order.side == "buy"

    def test_order_quantities_positive_ints(self, alloc_agent_config, alloc_case):
        result = self._invoke(alloc_agent_config, alloc_case)
        for order in result.decision.orders:
            assert isinstance(order.quantity, int)
            assert order.quantity > 0

    def test_raw_output_has_allocation(self, alloc_agent_config, alloc_case):
        result = self._invoke(alloc_agent_config, alloc_case)
        # raw_output is the debate_action dict
        action_data = result.raw_output
        assert isinstance(action_data, dict)
        debate_action = action_data.get("debate_action", action_data)
        assert "allocation" in debate_action

    def test_allocation_sums_to_one(self, alloc_agent_config, alloc_case):
        result = self._invoke(alloc_agent_config, alloc_case)
        action_data = result.raw_output
        debate_action = action_data.get("debate_action", action_data)
        alloc = debate_action.get("allocation", {})
        if alloc:
            assert sum(alloc.values()) == pytest.approx(1.0)


# ── MTM case detection ───────────────────────────────────────────────────────


class TestMTMCaseDetection:
    def test_mtm_case_id_prefix(self, mtm_case):
        assert mtm_case.case_id.startswith("mtm/")

    def test_decision_case_id_prefix(self, alloc_case):
        assert alloc_case.case_id.startswith("memo/")

    def test_build_case_overwrites_id(self, alloc_case):
        """Verify that build_case changes case_id, confirming why we
        check template.case_id before build_case in the runner."""
        from simulation.case_loader import build_case

        rebuilt = build_case(
            alloc_case,
            PortfolioSnapshot(cash=100_000.0, positions={}),
            case_id="ep_000:0",
        )
        # case_id is now overwritten — template's case_id is gone
        assert rebuilt.case_id == "ep_000:0"
        # But the original template still has it
        assert alloc_case.case_id == "memo/2025Q1"


# ── config propagation ───────────────────────────────────────────────────────


class TestConfigPropagation:
    def test_yaml_memo_validation(self):
        """SimulationConfig validates memo constraints (tickers, invest_quarter)."""
        cfg = SimulationConfig(
            dataset_path="data-pipeline/final_snapshots",
            invest_quarter="2025Q1",
            tickers=["AAPL", "MSFT", "GOOG"],
            agent={
                "agent_system": "multi_agent_debate",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "temperature": 0.3,
            },
        )
        assert cfg.invest_quarter == "2025Q1"
        assert cfg.tickers == ["AAPL", "MSFT", "GOOG"]

    def test_debate_config_created(self, alloc_agent_config):
        """DebateAgentSystem creates a DebateConfig successfully."""
        agent = DebateAgentSystem(alloc_agent_config)
        assert agent._debate_cfg is not None
        assert agent._debate_cfg.mock is True


# ── prompt templates ─────────────────────────────────────────────────────────


class TestPromptTemplatesLoad:
    TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "multi_agent" / "prompts"

    @pytest.mark.parametrize("filename", [
        "phases/proposal_allocation.txt",
        "phases/critique_allocation.txt",
        "phases/revision_allocation.txt",
        "phases/judge_allocation.txt",
        "output_format/allocation_output_instructions.txt",
    ])
    def test_allocation_templates_exist(self, filename):
        assert (self.TEMPLATE_DIR / filename).exists(), f"Missing: {filename}"

    def test_proposal_renders(self):
        result = build_proposal_user_prompt("Test context", allocation_mode=True)
        assert len(result) > 0
        assert "allocation" in result.lower() or "portfolio" in result.lower()

    def test_critique_renders(self):
        result = build_critique_prompt(
            "macro", "Test context",
            [{"role": "value", "proposal": "equal weight"}],
            "my proposal text",
            agreeableness=0.3,
            allocation_mode=True,
        )
        assert len(result) > 0

    def test_revision_renders(self):
        result = build_revision_prompt(
            "macro", "Test context", "my proposal",
            [{"from_role": "value", "objection": "too risky"}],
            agreeableness=0.3,
            allocation_mode=True,
        )
        assert len(result) > 0

    def test_judge_renders(self):
        result = build_judge_prompt(
            "Test context",
            [{"role": "macro", "action": "{}", "confidence": 0.5}],
            "critique text",
            allocation_mode=True,
        )
        assert len(result) > 0
