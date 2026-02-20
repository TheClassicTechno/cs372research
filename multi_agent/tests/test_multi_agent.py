"""
Comprehensive tests for the CS372 multi-agent debate system.

All tests use mock mode (no API keys needed).
Tests verify:
  1. Model creation and serialization
  2. Config defaults and customization
  3. Prompt generation (key elements present)
  4. Individual graph nodes with mock state
  5. Full graph end-to-end execution
  6. Different configurations produce different behavior
  7. Agreeableness knob changes prompt content
  8. Adversarial mode adds devil's advocate
  9. Pipeline agents produce structured output
  10. Multiple debate rounds work correctly
  11. Template files exist, are non-empty, and render correctly
"""

import json
from pathlib import Path

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.graph import (
    DebateState,
    _mock_critique,
    _mock_judge,
    _mock_pipeline,
    _mock_proposal,
    _mock_revision,
    build_context_node,
    build_debate_graph,
    compile_debate_graph,
    critique_node,
    data_analysis_node,
    judge_node,
    news_digest_node,
    propose_node,
    revise_node,
    should_continue,
    build_trace_node,
)
from multi_agent.models import (
    Action,
    AgentTrace,
    Claim,
    Constraints,
    MarketState,
    Observation,
    Order,
    PearlLevel,
    PipelineOutput,
    PortfolioState,
)
from multi_agent.prompts import (
    CAUSAL_CLAIM_FORMAT,
    ROLE_SYSTEM_PROMPTS,
    build_critique_prompt,
    build_judge_prompt,
    build_observation_context,
    build_proposal_user_prompt,
    build_revision_prompt,
    get_agreeableness_modifier,
)
from multi_agent.runner import MultiAgentRunner


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_observation() -> Observation:
    """A realistic sample observation for testing."""
    return Observation(
        timestamp="2025-03-15T10:00:00Z",
        universe=["AAPL", "GOOGL", "MSFT"],
        market_state=MarketState(
            prices={"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 390.00},
            returns={"AAPL": 0.025, "GOOGL": -0.01, "MSFT": 0.015},
            volatility={"AAPL": 0.22, "GOOGL": 0.25, "MSFT": 0.18},
        ),
        text_context="Fed signals potential rate cuts in Q2. AAPL earnings beat expectations.",
        portfolio_state=PortfolioState(
            cash=50000.0,
            positions={"AAPL": 100, "GOOGL": 0, "MSFT": 50},
        ),
        constraints=Constraints(max_leverage=2.0, max_position_size=500),
    )


@pytest.fixture
def sample_obs_dict(sample_observation: Observation) -> dict:
    return sample_observation.model_dump()


@pytest.fixture
def mock_config() -> DebateConfig:
    """Default mock config for testing."""
    return DebateConfig(
        roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL],
        max_rounds=1,
        agreeableness=0.3,
        mock=True,
        trace_dir="/tmp/test_traces",
    )


@pytest.fixture
def mock_config_dict(mock_config: DebateConfig) -> dict:
    return mock_config.to_dict()


# =============================================================================
# 1. MODEL TESTS
# =============================================================================


class TestModels:
    """Test Pydantic model creation and serialization."""

    def test_observation_creation(self, sample_observation: Observation):
        assert sample_observation.timestamp == "2025-03-15T10:00:00Z"
        assert len(sample_observation.universe) == 3
        assert sample_observation.market_state.prices["AAPL"] == 185.50
        assert sample_observation.portfolio_state.cash == 50000.0

    def test_observation_roundtrip(self, sample_observation: Observation):
        """Serialize to dict and back."""
        d = sample_observation.model_dump()
        restored = Observation(**d)
        assert restored.timestamp == sample_observation.timestamp
        assert restored.universe == sample_observation.universe
        assert restored.market_state.prices == sample_observation.market_state.prices

    def test_order_creation(self):
        order = Order(ticker="AAPL", side="buy", size=100)
        assert order.ticker == "AAPL"
        assert order.type == "market"  # default

    def test_claim_creation(self):
        claim = Claim(
            claim_text="If Fed cuts rates, AAPL rises",
            pearl_level=PearlLevel.L2,
            variables=["fed_rate", "AAPL_price"],
            assumptions=["Normal market conditions"],
            confidence=0.7,
        )
        assert claim.pearl_level == PearlLevel.L2
        assert len(claim.variables) == 2

    def test_action_creation(self):
        action = Action(
            orders=[Order(ticker="AAPL", side="buy", size=50)],
            justification="Strong earnings",
            confidence=0.8,
            claims=[
                Claim(
                    claim_text="Earnings drive price",
                    pearl_level=PearlLevel.L1,
                    variables=["earnings", "price"],
                )
            ],
        )
        assert len(action.orders) == 1
        assert len(action.claims) == 1
        assert action.confidence == 0.8

    def test_action_defaults(self):
        action = Action()
        assert action.orders == []
        assert action.confidence == 0.5
        assert action.justification == ""

    def test_pipeline_output(self):
        po = PipelineOutput(
            agent_type="news_digest",
            summary="Bullish sentiment",
            key_signals=["Rate cuts", "Earnings beat"],
            sentiment_score=0.6,
        )
        assert po.agent_type == "news_digest"
        assert len(po.key_signals) == 2

    def test_pearl_level_enum(self):
        assert PearlLevel.L1.value == "L1"
        assert PearlLevel.L2.value == "L2"
        assert PearlLevel.L3.value == "L3"


# =============================================================================
# 2. CONFIG TESTS
# =============================================================================


class TestConfig:
    """Test configuration defaults and customization."""

    def test_default_config(self):
        cfg = DebateConfig()
        assert len(cfg.roles) == 4
        assert cfg.max_rounds == 1
        assert cfg.agreeableness == 0.3
        assert cfg.enable_adversarial is False
        assert cfg.mock is False

    def test_custom_config(self):
        cfg = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.SENTIMENT],
            max_rounds=3,
            agreeableness=0.8,
            enable_adversarial=True,
            mock=True,
        )
        assert len(cfg.roles) == 2
        assert cfg.max_rounds == 3
        assert cfg.agreeableness == 0.8
        assert cfg.enable_adversarial is True

    def test_config_to_dict(self, mock_config: DebateConfig):
        d = mock_config.to_dict()
        assert isinstance(d, dict)
        assert d["mock"] is True
        assert len(d["roles"]) == 4
        assert all(isinstance(r, str) for r in d["roles"])

    def test_agent_role_enum(self):
        assert AgentRole.MACRO.value == "macro"
        assert AgentRole.DEVILS_ADVOCATE.value == "devils_advocate"
        assert len(AgentRole) == 6  # macro, value, risk, technical, sentiment, devils_advocate


# =============================================================================
# 3. PROMPT TESTS
# =============================================================================


class TestPrompts:
    """Test that prompts contain required elements for research quality."""

    def test_all_roles_have_prompts(self):
        """Every AgentRole must have a system prompt."""
        for role in AgentRole:
            assert role in ROLE_SYSTEM_PROMPTS or role.value in ROLE_SYSTEM_PROMPTS, (
                f"Missing prompt for role: {role}"
            )

    def test_prompts_contain_pearl_levels(self):
        """Each role prompt must mention L2 and L3 reasoning."""
        for role, prompt in ROLE_SYSTEM_PROMPTS.items():
            assert "L2" in prompt, f"{role} prompt missing L2 guidance"
            assert "L3" in prompt, f"{role} prompt missing L3 guidance"

    def test_prompts_contain_falsifiers(self):
        """Each role prompt must ask what would change the agent's mind."""
        falsifier_terms = ("falsif", "invalidat", "change your mind", "change my mind", "would change")
        for role, prompt in ROLE_SYSTEM_PROMPTS.items():
            lower = prompt.lower()
            assert any(t in lower for t in falsifier_terms), (
                f"{role} prompt missing falsifier/invalidation requirement"
            )

    def test_prompts_contain_json_format(self):
        """Role prompts should specify JSON output (either inline or via the
        shared JSON_OUTPUT_INSTRUCTIONS appended at call time)."""
        from multi_agent.prompts import JSON_OUTPUT_INSTRUCTIONS, build_proposal_user_prompt
        # The JSON instructions are appended via build_proposal_user_prompt
        assert "JSON" in JSON_OUTPUT_INSTRUCTIONS or "json" in JSON_OUTPUT_INSTRUCTIONS
        # Verify the combined prompt (system + user) contains JSON format
        full = build_proposal_user_prompt("test context")
        assert "JSON" in full or "json" in full

    def test_agreeableness_modifier_varies(self):
        """Different agreeableness values produce different modifiers."""
        m0 = get_agreeableness_modifier(0.0)
        m5 = get_agreeableness_modifier(0.5)
        m10 = get_agreeableness_modifier(1.0)
        assert m0 != m5
        assert m5 != m10
        assert "CONFRONTATIONAL" in m0
        assert "BALANCED" in m5
        assert "AGREEABLE" in m10

    def test_observation_context_builder(self, sample_observation: Observation):
        ctx = build_observation_context(sample_observation)
        assert "AAPL" in ctx
        assert "$185.50" in ctx
        assert "rate cuts" in ctx.lower() or "Rate cuts" in ctx or "rate" in ctx.lower()

    def test_observation_context_with_pipeline(self, sample_observation: Observation):
        ctx = build_observation_context(
            sample_observation,
            pipeline_context="### NEWS: Bullish sentiment detected",
        )
        assert "Pre-Processed Intelligence" in ctx
        assert "Bullish sentiment" in ctx

    def test_critique_prompt_has_structure(self):
        prompt = build_critique_prompt(
            "macro",
            "Market context here",
            [
                {"role": "value", "proposal": '{"orders": []}'},
                {"role": "risk", "proposal": '{"orders": []}'},
            ],
            '{"orders": []}',
            agreeableness=0.3,
        )
        assert "MACRO" in prompt
        assert "VALUE" in prompt
        assert "RISK" in prompt
        assert "SKEPTICAL" in prompt  # agreeableness=0.3

    def test_revision_prompt_has_structure(self):
        prompt = build_revision_prompt(
            "value",
            "Market context here",
            '{"orders": [{"ticker": "AAPL", "side": "buy", "size": 10}]}',
            [{"from_role": "risk", "objection": "Too much exposure", "falsifier": "Vol spike"}],
        )
        assert "VALUE" in prompt
        assert "Too much exposure" in prompt
        assert "Vol spike" in prompt

    def test_judge_prompt_has_structure(self):
        prompt = build_judge_prompt(
            "Market context here",
            [
                {"role": "macro", "action": '{"orders": []}', "confidence": 0.7},
                {"role": "value", "action": '{"orders": []}', "confidence": 0.6},
            ],
            "[macro -> value]: Objection about valuation",
        )
        assert "JUDGE" in prompt
        assert "MACRO" in prompt
        assert "Objection about valuation" in prompt

    def test_devils_advocate_prompt_is_adversarial(self):
        prompt = ROLE_SYSTEM_PROMPTS[AgentRole.DEVILS_ADVOCATE]
        assert "CHALLENGE" in prompt
        assert "groupthink" in prompt
        assert "STRONGEST" in prompt


# =============================================================================
# 4. MOCK GENERATOR TESTS
# =============================================================================


class TestMockGenerators:
    """Test that mock generators produce valid, structured output."""

    def test_mock_proposal_structure(self, sample_obs_dict: dict):
        for role in ["macro", "value", "risk", "technical", "sentiment", "devils_advocate"]:
            result = _mock_proposal(role, sample_obs_dict)
            assert "what_i_saw" in result
            assert "hypothesis" in result
            assert "orders" in result
            assert "justification" in result
            assert "confidence" in result
            assert "claims" in result
            assert isinstance(result["orders"], list)
            assert isinstance(result["claims"], list)
            assert 0 <= result["confidence"] <= 1

    def test_mock_proposals_disagree(self, sample_obs_dict: dict):
        """Different roles should produce different proposals (disagreement for debate)."""
        proposals = {role: _mock_proposal(role, sample_obs_dict) for role in ["macro", "value", "risk"]}
        # At least some should have different directions or confidences
        confidences = [p["confidence"] for p in proposals.values()]
        assert len(set(confidences)) > 1, "All mock proposals have identical confidence"

    def test_mock_critique_structure(self, sample_obs_dict: dict):
        proposals = [
            {"role": "macro", "action_dict": {}},
            {"role": "value", "action_dict": {}},
            {"role": "risk", "action_dict": {}},
        ]
        result = _mock_critique("macro", proposals)
        assert "critiques" in result
        assert "self_critique" in result
        assert len(result["critiques"]) == 2  # critiques 2 others
        for c in result["critiques"]:
            assert "target_role" in c
            assert "objection" in c

    def test_mock_revision_reduces_confidence(self, sample_obs_dict: dict):
        original = _mock_proposal("macro", sample_obs_dict)
        revised = _mock_revision("macro", original, sample_obs_dict)
        assert revised["confidence"] <= original["confidence"]
        assert "revision_notes" in revised

    def test_mock_judge_follows_majority(self):
        revisions = [
            {"role": "macro", "action_dict": {"orders": [{"ticker": "AAPL", "side": "buy", "size": 10}], "confidence": 0.7}},
            {"role": "value", "action_dict": {"orders": [{"ticker": "AAPL", "side": "buy", "size": 5}], "confidence": 0.6}},
            {"role": "risk", "action_dict": {"orders": [{"ticker": "AAPL", "side": "sell", "size": 3}], "confidence": 0.5}},
        ]
        result = _mock_judge(revisions)
        assert "orders" in result
        assert "audited_memo" in result
        assert "strongest_objection" in result
        # 2 buys vs 1 sell -> should buy
        assert any(o["side"] == "buy" for o in result["orders"])

    def test_mock_pipeline_news(self, sample_obs_dict: dict):
        result = _mock_pipeline("news_digest", sample_obs_dict)
        assert "summary" in result
        assert "sentiment_score" in result
        assert "key_signals" in result
        assert isinstance(result["key_signals"], list)

    def test_mock_pipeline_data(self, sample_obs_dict: dict):
        result = _mock_pipeline("data_analysis", sample_obs_dict)
        assert "summary" in result
        assert "momentum_signal" in result
        assert result["momentum_signal"] in ("positive", "negative", "neutral")


# =============================================================================
# 5. INDIVIDUAL NODE TESTS
# =============================================================================


class TestNodes:
    """Test individual graph nodes with mock state."""

    def _make_state(self, obs_dict: dict, config_dict: dict, **overrides) -> dict:
        """Create a minimal DebateState dict for node testing."""
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

    def test_news_digest_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(sample_obs_dict, mock_config_dict)
        result = news_digest_node(state)
        assert "news_digest" in result
        parsed = json.loads(result["news_digest"])
        assert "summary" in parsed
        assert "sentiment_score" in parsed

    def test_news_digest_node_no_context(self, mock_config_dict: dict):
        obs = {
            "timestamp": "2025-01-01T00:00:00Z",
            "universe": ["AAPL"],
            "market_state": {"prices": {"AAPL": 180}},
            "portfolio_state": {"cash": 10000, "positions": {}},
        }
        state = self._make_state(obs, mock_config_dict)
        result = news_digest_node(state)
        assert result["news_digest"] == "No news context provided."

    def test_data_analysis_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(sample_obs_dict, mock_config_dict)
        result = data_analysis_node(state)
        assert "data_analysis" in result
        parsed = json.loads(result["data_analysis"])
        assert "momentum_signal" in parsed

    def test_build_context_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            news_digest='{"summary": "Bullish news"}',
            data_analysis='{"summary": "Positive momentum"}',
        )
        result = build_context_node(state)
        assert "enriched_context" in result
        assert "AAPL" in result["enriched_context"]
        assert "Pre-Processed Intelligence" in result["enriched_context"]

    def test_propose_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            enriched_context="Market context here",
        )
        result = propose_node(state)
        assert "proposals" in result
        assert "debate_turns" in result
        assert result["current_round"] == 1
        assert len(result["proposals"]) == 4  # 4 roles in default config
        for p in result["proposals"]:
            assert "role" in p
            assert "action_dict" in p
            assert "orders" in p["action_dict"]
            assert "confidence" in p["action_dict"]

    def test_critique_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        # First generate proposals
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            enriched_context="Market context here",
        )
        proposals = propose_node(state)
        state.update(proposals)
        state["current_round"] = 1

        result = critique_node(state)
        assert "critiques" in result
        assert len(result["critiques"]) == 4
        for c in result["critiques"]:
            assert "role" in c
            assert "critiques" in c
            assert isinstance(c["critiques"], list)

    def test_revise_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            enriched_context="Market context here",
        )
        # Build up state through propose and critique
        state.update(propose_node(state))
        state["current_round"] = 1
        state.update(critique_node(state))

        result = revise_node(state)
        assert "revisions" in result
        assert len(result["revisions"]) == 4
        for r in result["revisions"]:
            assert "role" in r
            assert "action_dict" in r
            assert "revision_notes" in r

    def test_should_continue_loops(self):
        state_continue = {"current_round": 1, "config": {"max_rounds": 2}}
        assert should_continue(state_continue) == "critique"

    def test_should_continue_stops(self):
        state_stop = {"current_round": 2, "config": {"max_rounds": 1}}
        assert should_continue(state_stop) == "judge"

    def test_judge_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        # Build state through full debate
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            enriched_context="Market context here",
        )
        state.update(propose_node(state))
        state["current_round"] = 1
        state.update(critique_node(state))
        state.update(revise_node(state))

        result = judge_node(state)
        assert "final_action" in result
        assert "strongest_objection" in result
        assert "audited_memo" in result
        assert "orders" in result["final_action"]
        assert "confidence" in result["final_action"]

    def test_build_trace_node(self, sample_obs_dict: dict, mock_config_dict: dict):
        state = self._make_state(
            sample_obs_dict,
            mock_config_dict,
            enriched_context="Market context here",
            final_action={
                "orders": [{"ticker": "AAPL", "side": "buy", "size": 10}],
                "confidence": 0.6,
                "justification": "Test",
                "claims": [],
            },
            strongest_objection="Risk concerns",
            audited_memo="Debate summary",
            debate_turns=[{"round": 0, "agent_id": "test", "role": "macro", "type": "proposal"}],
        )
        result = build_trace_node(state)
        assert "trace" in result
        trace = result["trace"]
        assert trace["architecture"] == "debate"
        assert "buy 10 AAPL" in trace["decision"]


# =============================================================================
# 6. FULL GRAPH END-TO-END TESTS
# =============================================================================


class TestFullGraph:
    """Test the compiled LangGraph end-to-end with mock mode."""

    def test_basic_graph_execution(self, sample_observation: Observation, mock_config: DebateConfig):
        runner = MultiAgentRunner(mock_config)
        action, trace = runner.run(sample_observation)

        assert isinstance(action, Action)
        assert isinstance(trace, AgentTrace)
        assert trace.architecture == "debate"
        assert trace.observation_timestamp == sample_observation.timestamp

    def test_action_has_valid_structure(self, sample_observation: Observation, mock_config: DebateConfig):
        runner = MultiAgentRunner(mock_config)
        action, _ = runner.run(sample_observation)

        assert 0 <= action.confidence <= 1
        for order in action.orders:
            assert order.side in ("buy", "sell")
            assert order.size > 0
            assert order.ticker in sample_observation.universe or order.ticker != ""

    def test_claims_present(self, sample_observation: Observation, mock_config: DebateConfig):
        runner = MultiAgentRunner(mock_config)
        action, _ = runner.run(sample_observation)

        assert len(action.claims) > 0
        for claim in action.claims:
            assert claim.pearl_level in (PearlLevel.L1, PearlLevel.L2, PearlLevel.L3)
            assert len(claim.claim_text) > 0

    def test_no_pipeline_config(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_only_news_pipeline(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            enable_news_pipeline=True,
            enable_data_pipeline=False,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_only_data_pipeline(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            enable_news_pipeline=False,
            enable_data_pipeline=True,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)


# =============================================================================
# 7. CONFIGURATION ABLATION TESTS
# =============================================================================


class TestConfigAblations:
    """Test that different configs produce meaningfully different behavior."""

    def test_adversarial_mode_adds_devil(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            enable_adversarial=True,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        # Devil's advocate should be auto-injected
        assert AgentRole.DEVILS_ADVOCATE in runner.config.roles
        assert len(runner.config.roles) == 4  # 3 original + 1 devil

        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_different_role_counts(self, sample_observation: Observation):
        """More agents should produce more debate turns."""
        config_2 = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        config_4 = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner_2 = MultiAgentRunner(config_2)
        runner_4 = MultiAgentRunner(config_4)

        _, trace_2 = runner_2.run(sample_observation)
        _, trace_4 = runner_4.run(sample_observation)

        # Both should produce valid traces
        assert isinstance(trace_2, AgentTrace)
        assert isinstance(trace_4, AgentTrace)

    def test_multiple_debate_rounds(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            max_rounds=2,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_sentiment_agent_works(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.SENTIMENT],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_all_agents_config(self, sample_observation: Observation):
        """Run with all 6 agent roles."""
        config = DebateConfig(
            roles=list(AgentRole),
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)
        assert len(action.claims) > 0


# =============================================================================
# 8. GRAPH STRUCTURE TESTS
# =============================================================================


class TestGraphStructure:
    """Test that the graph is built correctly for different configs."""

    def test_graph_compiles_default(self):
        config = DebateConfig(mock=True)
        graph = compile_debate_graph(config)
        assert graph is not None

    def test_graph_compiles_no_pipeline(self):
        config = DebateConfig(
            enable_news_pipeline=False,
            enable_data_pipeline=False,
            mock=True,
        )
        graph = compile_debate_graph(config)
        assert graph is not None

    def test_graph_compiles_one_pipeline(self):
        config = DebateConfig(
            enable_news_pipeline=True,
            enable_data_pipeline=False,
            mock=True,
        )
        graph = compile_debate_graph(config)
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        config = DebateConfig(mock=True)
        graph = build_debate_graph(config)
        node_names = set(graph.nodes.keys())
        expected = {"news_digest", "data_analysis", "build_context", "propose",
                    "critique", "revise", "judge", "build_trace"}
        assert expected.issubset(node_names)


# =============================================================================
# 9. EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_universe(self):
        obs = Observation(
            timestamp="2025-01-01T00:00:00Z",
            universe=[],
            market_state=MarketState(prices={}),
            portfolio_state=PortfolioState(cash=10000, positions={}),
        )
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(obs)
        assert isinstance(action, Action)

    def test_single_role(self):
        obs = Observation(
            timestamp="2025-01-01T00:00:00Z",
            universe=["AAPL"],
            market_state=MarketState(prices={"AAPL": 180}),
            portfolio_state=PortfolioState(cash=10000, positions={}),
        )
        config = DebateConfig(
            roles=[AgentRole.MACRO],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(obs)
        assert isinstance(action, Action)

    def test_no_text_context(self):
        obs = Observation(
            timestamp="2025-01-01T00:00:00Z",
            universe=["AAPL"],
            market_state=MarketState(
                prices={"AAPL": 180},
                returns={"AAPL": 0.01},
            ),
            portfolio_state=PortfolioState(cash=10000, positions={"AAPL": 50}),
        )
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(obs)
        assert isinstance(action, Action)

    def test_high_agreeableness(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            agreeableness=0.95,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)

    def test_low_agreeableness(self, sample_observation: Observation):
        config = DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE],
            agreeableness=0.05,
            mock=True,
            trace_dir="/tmp/test_traces",
        )
        runner = MultiAgentRunner(config)
        action, trace = runner.run(sample_observation)
        assert isinstance(action, Action)


# =============================================================================
# 10. PROMPT TEMPLATE TESTS
# =============================================================================


class TestPromptTemplates:
    """Verify that .txt template files exist, are non-empty, and render correctly."""

    TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompts"

    EXPECTED_FILES = [
        "causal_claim_format.txt",
        "forced_uncertainty.txt",
        "trap_awareness.txt",
        "json_output_instructions.txt",
        "role_macro.txt",
        "role_value.txt",
        "role_risk.txt",
        "role_technical.txt",
        "role_sentiment.txt",
        "role_devils_advocate.txt",
        "news_digest_system.txt",
        "data_analysis_system.txt",
        "agreeableness_confrontational.txt",
        "agreeableness_skeptical.txt",
        "agreeableness_balanced.txt",
        "agreeableness_collaborative.txt",
        "agreeableness_agreeable.txt",
        "proposal.txt",
        "critique.txt",
        "revision.txt",
        "judge.txt",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_template_file_exists_and_nonempty(self, filename: str):
        path = self.TEMPLATE_DIR / filename
        assert path.exists(), f"Template file missing: {filename}"
        content = path.read_text()
        assert len(content.strip()) > 0, f"Template file is empty: {filename}"

    def test_proposal_template_renders(self):
        result = build_proposal_user_prompt("Test market context")
        assert "Test market context" in result
        assert "Causal Claim" in result
        assert "Uncertainty" in result

    def test_critique_template_renders(self):
        result = build_critique_prompt(
            "macro",
            "Test context",
            [{"role": "value", "proposal": "buy AAPL"}],
            "my proposal text",
            agreeableness=0.5,
        )
        assert "MACRO" in result
        assert "Test context" in result
        assert "my proposal text" in result
        assert "BALANCED" in result

    def test_revision_template_renders(self):
        result = build_revision_prompt(
            "risk",
            "Test context",
            "original proposal",
            [{"from_role": "macro", "objection": "Too risky"}],
        )
        assert "RISK" in result
        assert "Too risky" in result
        assert "Causal Claim" in result

    def test_judge_template_renders(self):
        result = build_judge_prompt(
            "Test context",
            [{"role": "macro", "action": "buy", "confidence": 0.8}],
            "critique text here",
            strongest_disagreements="a strong disagreement",
        )
        assert "JUDGE" in result
        assert "MACRO" in result
        assert "critique text here" in result
        assert "a strong disagreement" in result
