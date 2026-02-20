from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ReasonScore:
    gamma: float
    theta: float
    weight: float


@dataclass
class ReasonEvaluation:
    reason_id: str
    reason_text: str
    direction: str  # supports_omega | attacks_omega
    evidence_text: str
    evidence_type: str  # A | B | C | D
    score: ReasonScore
    scoring_trace: Optional[Dict[str, Any]] = None


@dataclass
class CritResult:
    document_id: str
    omega: str
    reasons: List[ReasonEvaluation]
    counter_reasons: List[ReasonEvaluation]
    gamma_total: float
    justification: str
