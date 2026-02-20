from typing import Dict, Any, List

from .config import CritConfig
from .types import CritResult, ReasonEvaluation
from .pipeline.transcript_normalizer import normalize_transcript
from .pipeline.conclusion_extractor import extract_conclusion
from .pipeline.reason_extractor import extract_reasons
from .pipeline.reason_validator import validate_reason
from .pipeline.counter_reason_generator import generate_counter_reasons
from .pipeline.aggregator import aggregate_scores
from .pipeline.justification_generator import generate_justification


class CritEvaluator:

    def __init__(self, config: CritConfig):
        self.config = config

    def evaluate(self, transcript: Dict[str, Any]) -> CritResult:
        normalized = normalize_transcript(transcript)

        omega = extract_conclusion(normalized, self.config)

        reasons = extract_reasons(normalized, omega, self.config)

        validated_reasons: List[ReasonEvaluation] = [
            validate_reason(normalized, omega, r, "supports_omega", self.config)
            for r in reasons
        ]

        counter_reasons = generate_counter_reasons(
            normalized,
            omega,
            validated_reasons,
            self.config
        )

        validated_counter: List[ReasonEvaluation] = [
            validate_reason(normalized, omega, r, "attacks_omega", self.config)
            for r in counter_reasons
        ]

        gamma_total = aggregate_scores(validated_reasons, validated_counter)

        justification = generate_justification(
            omega,
            validated_reasons,
            validated_counter,
            gamma_total,
            self.config
        )

        return CritResult(
            document_id="TODO_document_id",
            omega=omega,
            reasons=validated_reasons,
            counter_reasons=validated_counter,
            gamma_total=gamma_total,
            justification=justification
        )
