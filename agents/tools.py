"""LangChain tool factories for the agent–broker interface.

The ``make_submit_decision_tool`` function creates a fresh tool instance that
closes over the current broker state and case, so the agent's tool call routes
directly to ``Broker.execute_decision()``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from models.case import Case
from models.decision import Decision, DecisionResult, Order, SubmitDecisionInput
from simulation.broker import Broker


# ------------------------------------------------------------------
# Tool factory
# ------------------------------------------------------------------

def make_submit_decision_tool(
    broker: Broker,
    case: Case,
    agent_id: str,
) -> StructuredTool:
    """Create a ``submit_decision`` tool bound to *broker* and *case*.

    Each decision point should get a **fresh** tool so the closure captures the
    correct broker state and case reference.

    Returns a ``StructuredTool`` ready to be bound to a LangChain agent/LLM.
    """

    def _submit_decision(orders: list[dict[str, Any]]) -> str:
        """Validate and execute the submitted orders, returning JSON result."""
        parsed_orders = [Order(**o) for o in orders]
        decision = Decision(orders=parsed_orders)
        result: DecisionResult = broker.execute_decision(decision, case, agent_id)

        # Store the last decision on the tool for extraction after invoke.
        _submit_decision._last_decision = decision  # type: ignore[attr-defined]
        _submit_decision._last_result = result  # type: ignore[attr-defined]

        return result.model_dump_json()

    # Initialise sentinel attributes.
    _submit_decision._last_decision = None  # type: ignore[attr-defined]
    _submit_decision._last_result = None  # type: ignore[attr-defined]

    return StructuredTool.from_function(
        func=_submit_decision,
        name="submit_decision",
        description=(
            "Submit a trading decision consisting of buy/sell orders. "
            "Each order specifies a ticker, side ('buy' or 'sell'), and "
            "quantity (positive integer). An empty orders list means hold. "
            "The tool returns a JSON result indicating whether the decision "
            "was 'accepted' (with executed trades) or 'rejected' (with a "
            "message explaining why). If rejected you may revise and resubmit."
        ),
        args_schema=SubmitDecisionInput,
    )
