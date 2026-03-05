"""
Data models and validation for CRIT reasoning audit results.

CRIT evaluates four pillars of reasoning quality:
    1. Logical Validity — Does the reasoning logically support the conclusion?
    2. Evidential Support — Are claims backed by cited evidence?
    3. Alternative Consideration — Does the agent consider competing explanations?
    4. Causal Alignment — Are causal claims justified by the evidence?

Each pillar produces a score in [0, 1] and binary diagnostic flags detect
specific failure modes.

Per the RAudit paper (Section 3.3), CRIT scores each agent individually
(ρ_i), then averages into ρ̄ = 1/n Σ ρ_i. The composite ρ̄ feeds into
the PID controller as the quality signal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PillarScores(BaseModel):
    """Four-pillar reasoning quality scores. All in [0, 1]."""

    logical_validity: float = Field(ge=0.0, le=1.0)
    evidential_support: float = Field(ge=0.0, le=1.0)
    alternative_consideration: float = Field(ge=0.0, le=1.0)
    causal_alignment: float = Field(ge=0.0, le=1.0)


class Diagnostics(BaseModel):
    """Binary flags for specific reasoning failure modes."""

    contradictions_detected: bool
    unsupported_claims_detected: bool
    ignored_critiques_detected: bool
    premature_certainty_detected: bool
    causal_overreach_detected: bool
    conclusion_drift_detected: bool


class Explanations(BaseModel):
    """Per-pillar textual explanations of the score."""

    logical_validity: str
    evidential_support: str
    alternative_consideration: str
    causal_alignment: str


class CritResult(BaseModel):
    """CRIT audit result for a single agent (ρ_i).

    rho_bar is the mean of the four pillar scores for this agent.
    """

    pillar_scores: PillarScores
    diagnostics: Diagnostics
    explanations: Explanations
    rho_bar: float = Field(ge=0.0, le=1.0)


class RoundCritResult(BaseModel):
    """Aggregated CRIT result for one debate round (all agents).

    Per the RAudit paper (Section 3.3, Algorithm 1 lines 7-8):
        - Each agent i gets scored individually → ρ_i (CritResult)
        - ρ̄ = 1/n Σ_i ρ_i feeds into the PID controller

    agent_scores maps role name → per-agent CritResult.
    rho_bar is the mean of all per-agent rho_bars.
    """

    agent_scores: dict[str, CritResult]
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
        pillar_scores.logical_validity
        + pillar_scores.evidential_support
        + pillar_scores.alternative_consideration
        + pillar_scores.causal_alignment
    ) / 4.0
    return CritResult(
        pillar_scores=pillar_scores,
        diagnostics=diagnostics,
        explanations=explanations,
        rho_bar=rho_bar,
    )


def aggregate_agent_scores(agent_scores: dict[str, CritResult]) -> RoundCritResult:
    """Aggregate per-agent CritResults into a RoundCritResult.

    Computes ρ̄ = 1/n Σ_i ρ_i as per RAudit Eq. in Algorithm 1 line 8.

    Args:
        agent_scores: Mapping of role name → CritResult for that agent.

    Returns:
        RoundCritResult with per-agent scores and aggregated rho_bar.

    Raises:
        ValueError: If agent_scores is empty.
    """
    if not agent_scores:
        raise ValueError("agent_scores must not be empty")
    rho_bar = sum(cr.rho_bar for cr in agent_scores.values()) / len(agent_scores)
    return RoundCritResult(agent_scores=agent_scores, rho_bar=rho_bar)
