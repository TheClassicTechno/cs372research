"""
Data models and validation for CRIT reasoning audit results.

CRIT evaluates four pillars of reasoning quality:
    1. Internal Consistency — Are the agent's claims logically compatible?
    2. Evidence Support — Are claims backed by cited evidence?
    3. Trace Alignment — Does the final decision follow from the reasoning?
    4. Causal Integrity — Are causal claims properly scoped (Pearl levels)?

Each pillar produces a score in [0, 1] and a binary diagnostic flag.
The composite score rho_bar (mean of four pillars) feeds into the PID
controller as the quality signal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PillarScores(BaseModel):
    """Four-pillar reasoning quality scores. All in [0, 1]."""

    internal_consistency: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    trace_alignment: float = Field(ge=0.0, le=1.0)
    causal_integrity: float = Field(ge=0.0, le=1.0)


class Diagnostics(BaseModel):
    """Binary flags for specific reasoning failure modes."""

    contradictions_detected: bool
    unsupported_claims_detected: bool
    conclusion_drift_detected: bool
    causal_overreach_detected: bool


class Explanations(BaseModel):
    """Per-pillar textual explanations of the score."""

    internal_consistency: str
    evidence_support: str
    trace_alignment: str
    causal_integrity: str


class CritResult(BaseModel):
    """Complete CRIT audit result for one debate round.

    rho_bar is the mean of the four pillar scores and serves as the
    quality signal that feeds into the PID controller.
    """

    pillar_scores: PillarScores
    diagnostics: Diagnostics
    explanations: Explanations
    rho_bar: float = Field(ge=0.0, le=1.0)


def validate_raw_response(raw: dict) -> CritResult:
    """Parse and validate raw LLM JSON into CritResult.

    Computes rho_bar as the mean of the four pillar scores.

    Raises:
        KeyError: If required top-level keys are missing.
        ValueError/ValidationError: If field values are out of range
            or have incorrect types.
    """
    pillar_scores = PillarScores(**raw["pillar_scores"])
    diagnostics = Diagnostics(**raw["diagnostics"])
    explanations = Explanations(**raw["explanations"])
    rho_bar = (
        pillar_scores.internal_consistency
        + pillar_scores.evidence_support
        + pillar_scores.trace_alignment
        + pillar_scores.causal_integrity
    ) / 4.0
    return CritResult(
        pillar_scores=pillar_scores,
        diagnostics=diagnostics,
        explanations=explanations,
        rho_bar=rho_bar,
    )
