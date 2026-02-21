from typing import Dict, Any, List
from ..config import CritConfig
from ..types import ReasonEvaluation


def generate_counter_reasons(
    normalized_transcript: Dict[str, Any],
    omega: str,
    validated_reasons: List[ReasonEvaluation],
    config: CritConfig
) -> List[str]:
    return []
