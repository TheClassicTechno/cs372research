from typing import Dict, Any
from ..config import CritConfig
from ..types import ReasonEvaluation, ReasonScore


def validate_reason(
    normalized_transcript: Dict[str, Any],
    omega: str,
    reason_text: str,
    direction: str,
    config: CritConfig
) -> ReasonEvaluation:
    score = ReasonScore(gamma=0.0, theta=0.0, weight=0.0)

    return ReasonEvaluation(
        reason_id="TODO",
        reason_text=reason_text,
        direction=direction,
        evidence_text="TODO",
        evidence_type="B",
        score=score,
        scoring_trace=None
    )
