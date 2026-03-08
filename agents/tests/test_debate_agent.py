"""Tests for the multi-agent debate → simulation adapter.

Covers:
    1. build_observation conversion (feature_engineering)
    2. DebateAgentSystem lifecycle (bind_tools / invoke)
    3. End-to-end with mock debate runner (allocation mode)
    4. Registry integration
    5. Layer boundary verification (adapter must NOT call broker)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agents.multi_agent_debate import DebateAgentSystem
from simulation.feature_engineering import build_observation
from models.agents import AgentInvocation, AgentInvocationResult
from models.case import Case, CaseData, CaseDataItem, ClosePricePoint, StockData
from models.config import AgentConfig
from models.portfolio import PortfolioSnapshot
from multi_agent.models import (
    Action,
    AgentTrace,
    Claim,
    Observation,
    ReasoningType,
)
from multi_agent.models import Order as DebateOrder


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_case() -> Case:
    """A realistic Case from the simulation side with multi-day bars."""
    return Case(
        case_data=CaseData(
            items=[
                CaseDataItem(kind="earnings", content="AAPL beat earnings by 5%."),
                CaseDataItem(kind="news", content="Fed signals rate cuts in Q2."),
                CaseDataItem(kind="other", content="Sector rotation ongoing."),
            ]
        ),
        stock_data={
            "AAPL": StockData(
                ticker="AAPL",
                current_price=185.50,
                daily_bars=[
                    ClosePricePoint(timestamp="2025-03-12", close=180.00),
                    ClosePricePoint(timestamp="2025-03-13", close=182.00),
                    ClosePricePoint(timestamp="2025-03-14", close=183.00),
                    ClosePricePoint(timestamp="2025-03-15", close=185.50),
                ],
            ),
            "GOOGL": StockData(
                ticker="GOOGL",
                current_price=142.30,
                daily_bars=[
                    ClosePricePoint(timestamp="2025-03-12", close=140.00),
                    ClosePricePoint(timestamp="2025-03-13", close=140.50),
                    ClosePricePoint(timestamp="2025-03-14", close=141.00),
                    ClosePricePoint(timestamp="2025-03-15", close=142.30),
                ],
            ),
        },
        portfolio=PortfolioSnapshot(
            cash=50000.0,
            positions={"AAPL": 100, "GOOGL": 0},
        ),
        case_id="ep_000:0",
        decision_point_idx=0,
        information_cutoff_timestamp="2025-03-15T10:00:00Z",
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """AgentConfig for the debate agent system."""
    return AgentConfig(
        agent_system="multi_agent_debate",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        temperature=0.3,
    )


# =============================================================================
# 1. FEATURE ENGINEERING: CASE → OBSERVATION
# =============================================================================


class TestBuildObservation:
    """Test build_observation (simulation/feature_engineering.py)."""

    def test_prices_mapped(self, sample_case: Case):
        obs = build_observation(sample_case)
        assert obs.market_state.prices["AAPL"] == 185.50
        assert obs.market_state.prices["GOOGL"] == 142.30

    def test_universe_mapped(self, sample_case: Case):
        obs = build_observation(sample_case)
        assert set(obs.universe) == {"AAPL", "GOOGL"}

    def test_portfolio_mapped(self, sample_case: Case):
        obs = build_observation(sample_case)
        assert obs.portfolio_state.cash == 50000.0
        assert obs.portfolio_state.positions["AAPL"] == 100.0
        assert obs.portfolio_state.positions["GOOGL"] == 0.0

    def test_positions_are_float(self, sample_case: Case):
        """Simulation uses int positions; debate uses float."""
        obs = build_observation(sample_case)
        for val in obs.portfolio_state.positions.values():
            assert isinstance(val, float)

    def test_text_context_concatenated(self, sample_case: Case):
        obs = build_observation(sample_case)
        assert obs.text_context is not None
        assert "[EARNINGS]" in obs.text_context
        assert "AAPL beat earnings" in obs.text_context
        assert "[NEWS]" in obs.text_context
        assert "Fed signals" in obs.text_context
        assert "[INFO]" in obs.text_context
        assert "Sector rotation" in obs.text_context

    def test_timestamp_from_case(self, sample_case: Case):
        obs = build_observation(sample_case)
        assert obs.timestamp == "2025-03-15T10:00:00Z"

    def test_timestamp_fallback_when_none(self):
        """When case has no information_cutoff_timestamp, use current time."""
        case = Case(
            case_data=CaseData(),
            stock_data={
                "AAPL": StockData(
                    ticker="AAPL", current_price=180.0, daily_bars=[]
                )
            },
            portfolio=PortfolioSnapshot(cash=10000, positions={}),
        )
        obs = build_observation(case)
        assert obs.timestamp is not None
        assert len(obs.timestamp) > 0

    def test_empty_case_data(self):
        """Case with no items → text_context is None."""
        case = Case(
            case_data=CaseData(items=[]),
            stock_data={
                "AAPL": StockData(
                    ticker="AAPL", current_price=180.0, daily_bars=[]
                )
            },
            portfolio=PortfolioSnapshot(cash=10000, positions={}),
        )
        obs = build_observation(case)
        assert obs.text_context is None

    def test_returns_computed(self, sample_case: Case):
        """Returns should be computed from daily bars (last close / first close - 1)."""
        obs = build_observation(sample_case)
        assert obs.market_state.returns is not None
        # AAPL: (185.50 - 180.00) / 180.00 ≈ 0.03056
        assert abs(obs.market_state.returns["AAPL"] - (185.50 - 180.0) / 180.0) < 1e-6
        # GOOGL: (142.30 - 140.00) / 140.00 ≈ 0.01643
        assert abs(obs.market_state.returns["GOOGL"] - (142.30 - 140.0) / 140.0) < 1e-6

    def test_volatility_computed(self, sample_case: Case):
        """Volatility should be computed as std-dev of daily log-returns."""
        obs = build_observation(sample_case)
        assert obs.market_state.volatility is not None
        assert obs.market_state.volatility["AAPL"] > 0
        assert obs.market_state.volatility["GOOGL"] > 0

    def test_returns_zero_with_single_bar(self):
        """Single bar → returns = 0, volatility = 0."""
        case = Case(
            case_data=CaseData(),
            stock_data={
                "AAPL": StockData(
                    ticker="AAPL",
                    current_price=180.0,
                    daily_bars=[ClosePricePoint(timestamp="2025-03-15", close=180.0)],
                )
            },
            portfolio=PortfolioSnapshot(cash=10000, positions={}),
        )
        obs = build_observation(case)
        assert obs.market_state.returns["AAPL"] == 0.0
        assert obs.market_state.volatility["AAPL"] == 0.0

    def test_roundtrip_is_valid_observation(self, sample_case: Case):
        """The resulting Observation should serialize and deserialize cleanly."""
        obs = build_observation(sample_case)
        d = obs.model_dump()
        restored = Observation(**d)
        assert restored.market_state.prices == obs.market_state.prices
        assert restored.portfolio_state.cash == obs.portfolio_state.cash


# =============================================================================
# 2. AGENT SYSTEM LIFECYCLE
# =============================================================================


class TestDebateAgentSystemLifecycle:
    """Test bind_tools / invoke contract."""

    def test_bind_tools_stores_tool(self, agent_config: AgentConfig):
        agent = DebateAgentSystem(agent_config)
        mock_tool = MagicMock()
        agent.bind_tools(mock_tool)
        assert agent._submit_decision_tool is mock_tool


# =============================================================================
# 3. END-TO-END WITH MOCK DEBATE
# =============================================================================


class TestDebateAgentEndToEnd:
    """Test full invoke flow with a mocked MultiAgentRunner."""

    @pytest.fixture
    def mock_debate_action(self) -> Action:
        return Action(
            orders=[],
            allocation={"AAPL": 0.6, "GOOGL": 0.4},
            justification="Mock bullish signal.",
            confidence=0.7,
            claims=[
                Claim(
                    claim_text="Test claim",
                    reasoning_type=ReasoningType.OBSERVATIONAL,
                    confidence=0.6,
                )
            ],
        )

    @pytest.fixture
    def mock_debate_trace(self) -> AgentTrace:
        return AgentTrace(
            observation_timestamp="2025-03-15T10:00:00Z",
            architecture="debate",
            what_i_saw="Test observation",
            hypothesis="Test hypothesis",
            decision="Buy AAPL",
            action=Action(
                orders=[DebateOrder(ticker="AAPL", side="buy", size=20.0)],
                justification="Mock",
                confidence=0.7,
            ),
            logged_at="2025-03-15T10:01:00Z",
        )

    def test_full_invoke_returns_decision_without_tool_call(
        self,
        agent_config: AgentConfig,
        sample_case: Case,
        mock_debate_action: Action,
        mock_debate_trace: AgentTrace,
    ):
        """Full invoke: Case → Observation → debate → Action → Decision.

        Per integration_plan.md §8: the adapter must NOT call the broker.
        It returns the Decision and the runner handles broker execution.
        """
        agent = DebateAgentSystem(agent_config)

        # Bind a mock tool (adapter stores it but should NOT call it)
        mock_tool = MagicMock()
        mock_tool_func = MagicMock()
        mock_tool.func = mock_tool_func
        agent.bind_tools(mock_tool)

        # Patch MultiAgentRunner.run to return our mock action/trace
        with patch.object(
            agent._debate_runner,
            "run",
            return_value=(mock_debate_action, mock_debate_trace),
        ):
            invocation = AgentInvocation(
                case=sample_case,
                episode_id="ep_000",
                agent_id="debate_000",
                steps_remaining=5,
            )
            result = asyncio.run(
                agent.invoke(invocation)
            )

        # Verify result — allocation produces buy orders from weights
        assert isinstance(result, AgentInvocationResult)
        assert len(result.decision.orders) == 2
        tickers = {o.ticker for o in result.decision.orders}
        assert tickers == {"AAPL", "GOOGL"}
        for order in result.decision.orders:
            assert order.side == "buy"
            assert order.quantity > 0

        # CRITICAL: adapter must NOT call the tool (§8 layer boundary)
        mock_tool_func.assert_not_called()

    def test_hold_action_empty_decision(
        self,
        agent_config: AgentConfig,
        sample_case: Case,
        mock_debate_trace: AgentTrace,
    ):
        """When debate returns empty allocation (hold), decision should be empty."""
        agent = DebateAgentSystem(agent_config)

        mock_tool = MagicMock()
        agent.bind_tools(mock_tool)

        hold_action = Action(orders=[], allocation={}, justification="Hold.", confidence=0.5)

        with patch.object(
            agent._debate_runner,
            "run",
            return_value=(hold_action, mock_debate_trace),
        ):
            invocation = AgentInvocation(
                case=sample_case,
                episode_id="ep_000",
                agent_id="debate_000",
            )
            result = asyncio.run(
                agent.invoke(invocation)
            )

        assert len(result.decision.orders) == 0

    def test_raw_output_contains_debate_info(
        self,
        agent_config: AgentConfig,
        sample_case: Case,
        mock_debate_action: Action,
        mock_debate_trace: AgentTrace,
    ):
        """raw_output should contain the debate action and trace for logging."""
        agent = DebateAgentSystem(agent_config)

        mock_tool = MagicMock()
        agent.bind_tools(mock_tool)

        with patch.object(
            agent._debate_runner,
            "run",
            return_value=(mock_debate_action, mock_debate_trace),
        ):
            invocation = AgentInvocation(
                case=sample_case,
                episode_id="ep_000",
                agent_id="debate_000",
            )
            result = asyncio.run(
                agent.invoke(invocation)
            )

        assert isinstance(result.raw_output, dict)
        assert "debate_action" in result.raw_output
        assert "debate_trace" in result.raw_output
        assert result.raw_output["debate_confidence"] == 0.7
        assert result.raw_output["debate_justification"] == "Mock bullish signal."


# =============================================================================
# 4. REGISTRY
# =============================================================================


class TestRegistry:
    """Test that the debate agent is registered correctly."""

    def test_multi_agent_debate_registered(self):
        from agents.registry import _REGISTRY, _ensure_builtins_loaded

        _ensure_builtins_loaded()
        assert "multi_agent_debate" in _REGISTRY

    def test_create_debate_agent(self, agent_config: AgentConfig):
        from agents.registry import create_agent_system

        agent = create_agent_system(agent_config)
        assert isinstance(agent, DebateAgentSystem)


# =============================================================================
# 5. LAYER BOUNDARY VERIFICATION (integration_plan.md §2)
# =============================================================================


class TestLayerBoundary:
    """Verify that the adapter respects architectural boundaries."""

    def test_adapter_does_not_import_broker(self):
        """The adapter module must NOT import from simulation.broker."""
        import agents.multi_agent_debate as mod
        import inspect

        source = inspect.getsource(mod)
        assert "from simulation.broker" not in source
        assert "import simulation.broker" not in source

    def test_adapter_does_not_import_eval(self):
        """The adapter module must NOT import from eval/."""
        import agents.multi_agent_debate as mod
        import inspect

        source = inspect.getsource(mod)
        assert "from eval" not in source
        assert "import eval" not in source
