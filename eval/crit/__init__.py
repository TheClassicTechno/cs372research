"""CRIT — Blind reasoning quality auditor for multi-agent debate.

Public API:
    CritScorer    — Main scorer class (takes an LLM function, runs audit)
    CritResult    — Complete audit result (pillar scores + rho_bar + diagnostics)
    PillarScores  — Four-pillar score container
    Diagnostics   — Binary failure-mode flags
    Explanations  — Per-pillar textual explanations
"""

from eval.crit.scorer import CritScorer
from eval.crit.schema import CritResult, Diagnostics, Explanations, PillarScores

__all__ = ["CritScorer", "CritResult", "PillarScores", "Diagnostics", "Explanations"]
