"""Generalized intervention framework for multi-agent debate.

Detects undesirable debate dynamics (premature convergence, reasoning
collapse) and triggers corrective retries with prompt nudges.
"""

from .engine import InterventionEngine
from .rules import build_intervention_engine, RULE_REGISTRY
from .types import InterventionContext, InterventionResult, InterventionRule

__all__ = [
    "InterventionContext",
    "InterventionEngine",
    "InterventionResult",
    "InterventionRule",
    "RULE_REGISTRY",
    "build_intervention_engine",
]
