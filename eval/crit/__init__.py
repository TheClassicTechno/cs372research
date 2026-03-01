"""CRIT — Blind reasoning quality auditor for multi-agent debate.

Public API:
    CritScorer       — Main scorer class (takes an LLM function, runs audit)
    RoundCritResult  — Aggregated result for one round (per-agent scores + ρ̄)
    CritResult       — Single-agent audit result (ρ_i)
    PillarScores     — Four-pillar score container
    Diagnostics      — Binary failure-mode flags
    Explanations     — Per-pillar textual explanations
"""

from eval.crit.scorer import CritScorer
from eval.crit.schema import (
    CritResult,
    Diagnostics,
    Explanations,
    PillarScores,
    RoundCritResult,
    aggregate_agent_scores,
)

__all__ = [
    "CritScorer",
    "RoundCritResult",
    "CritResult",
    "PillarScores",
    "Diagnostics",
    "Explanations",
    "aggregate_agent_scores",
]
