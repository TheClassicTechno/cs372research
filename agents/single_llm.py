"""Single-LLM agent system: one LLM with the submit_decision tool.

This is the simplest agent architecture — a ReAct-style agent backed by a
single chat model that receives the case in its prompt and can call
``submit_decision`` to execute trades.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from agents.base import AgentSystem
from agents.registry import register
from models.agents import AgentInvocation, AgentInvocationResult
from models.decision import Decision
from models.config import AgentConfig

logger = logging.getLogger(__name__)

# Default system prompt — can be overridden via config.
_DEFAULT_SYSTEM_PROMPT = """\
You are a quantitative trading agent. You will be given market data, news, and \
your current portfolio. Analyse the information and decide whether to buy, sell, \
or hold.

Use the submit_decision tool to execute your decision. You may submit an empty \
orders list to hold. If your decision is rejected, read the rejection message \
carefully, revise your orders, and resubmit.

Guidelines:
- Only trade tickers present in the case data.
- Ensure you have sufficient cash for buys and sufficient shares for sells.
- You may submit multiple orders in a single decision (portfolio rebalancing).
- Reason step-by-step before submitting.
"""


def _create_llm(config: AgentConfig):
    """Instantiate the appropriate LangChain chat model from config."""
    provider = config.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.llm_model,
            temperature=config.temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported: 'openai', 'anthropic'."
        )


@register("single_llm")
class SingleLLMAgent(AgentSystem):
    """ReAct agent using a single LLM with the ``submit_decision`` tool."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._llm = _create_llm(config)
        self._tool: BaseTool | None = None
        self._system_prompt = config.system_prompt_override or _DEFAULT_SYSTEM_PROMPT

    def bind_tools(self, submit_decision_fn: Callable[..., Any]) -> None:
        """Accept a pre-built ``StructuredTool`` as *submit_decision_fn*.

        In practice the runner passes the tool object produced by
        ``make_submit_decision_tool``; we store it for the next ``invoke``.
        """
        # The runner passes the StructuredTool directly.
        self._tool = submit_decision_fn  # type: ignore[assignment]

    async def invoke(self, invocation: AgentInvocation) -> AgentInvocationResult:
        """Run the ReAct agent for a single decision point."""
        if self._tool is None:
            raise RuntimeError("bind_tools() must be called before invoke().")

        # Build the ReAct agent graph with the current tool.
        agent_executor = create_react_agent(
            self._llm,
            tools=[self._tool],
        )

        # Format the case payload for the prompt.
        case_payload = json.dumps(invocation.case.for_agent(), indent=2)

        steps_info = ""
        if invocation.steps_remaining is not None:
            steps_info = f"\n\nDecision steps remaining after this one: {invocation.steps_remaining}"

        human_content = (
            f"Here is the current market case and your portfolio:\n\n"
            f"```json\n{case_payload}\n```"
            f"{steps_info}\n\n"
            f"Analyse the data and submit your trading decision."
        )

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=human_content),
        ]

        # Invoke the agent asynchronously.
        result = await agent_executor.ainvoke(
            {"messages": messages},
            config={"recursion_limit": self.config.max_retries * 2 + 5},
        )

        # Extract the decision from the tool's stored state.
        decision = self._extract_decision()

        # Serialize the full message trace for logging.
        raw_output = _serialize_messages(result.get("messages", []))

        logger.info(
            "Agent %s decision for case %s: %d order(s)",
            invocation.agent_id,
            invocation.case.case_id,
            len(decision.orders),
        )

        return AgentInvocationResult(decision=decision, raw_output=raw_output)

    def _extract_decision(self) -> Decision:
        """Extract the last submitted decision from the tool's state.

        If the agent never called submit_decision, return hold (empty orders).
        """
        tool = self._tool
        if tool is None:
            return Decision(orders=[])

        # The tool function stores _last_decision on itself.
        func = tool.func  # type: ignore[union-attr]
        last_decision = getattr(func, "_last_decision", None)
        if last_decision is None:
            logger.warning("Agent did not call submit_decision — defaulting to hold.")
            return Decision(orders=[])

        return last_decision


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _serialize_messages(messages: list) -> str:
    """Convert LangChain message objects into a human-readable trace string."""
    parts: list[str] = []
    for msg in messages:
        role = msg.type.upper()  # "system", "human", "ai", "tool"
        content = getattr(msg, "content", "") or ""

        # AI messages may also carry tool_calls.
        tool_calls = getattr(msg, "tool_calls", None)

        header = f"--- {role} ---"
        parts.append(header)

        if content:
            parts.append(content)

        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                parts.append(f"[tool_call: {name}({json.dumps(args, indent=2)})]")

    return "\n".join(parts)
