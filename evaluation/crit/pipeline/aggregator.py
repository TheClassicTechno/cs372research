from typing import List
from ..types import ReasonEvaluation


def aggregate_scores(
    reasons: List[ReasonEvaluation],
    counter_reasons: List[ReasonEvaluation]
) -> float:
    all_items = reasons + counter_reasons
    if not all_items:
        return 0.0
    total = sum(r.score.weight for r in all_items)
    return total / len(all_items)
