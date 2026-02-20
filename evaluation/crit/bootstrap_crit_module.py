import os
from pathlib import Path

BASE_DIR = Path("eval/crit")
OVERWRITE = False  # Set to True if you want to overwrite existing files


def write_file(path: Path, content: str):
    if path.exists() and not OVERWRITE:
        print(f"Skipping existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n")
    print(f"Created: {path}")


FILES = {

# =========================
# Top-level
# =========================

"__init__.py": """
from .config import CritConfig
from .evaluator import CritEvaluator

__all__ = ["CritConfig", "CritEvaluator"]
""",

"config.py": """
from dataclasses import dataclass


@dataclass
class CritConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    max_reasons: int = 6
    max_counter_reasons: int = 4

    enable_recursion: bool = False
    max_recursion_depth: int = 1

    socratic_mode: str = "weakest_link"

    trace_llm_calls: bool = True
""",

"types.py": """
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
""",

"evaluator.py": """
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
""",

# =========================
# LLM Layer
# =========================

"llm/__init__.py": "",

"llm/client.py": """
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...
""",

"llm/tracing.py": """
from typing import Dict, Any


def build_trace_entry(
    model_name: str,
    prompt_id: str,
    raw_response: str
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "prompt_id": prompt_id,
        "raw_response": raw_response,
    }
""",

# =========================
# Pipeline
# =========================

"pipeline/__init__.py": "",

"pipeline/transcript_normalizer.py": """
from typing import Dict, Any


def normalize_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    return transcript
""",

"pipeline/conclusion_extractor.py": """
from typing import Dict, Any
from ..config import CritConfig


def extract_conclusion(
    normalized_transcript: Dict[str, Any],
    config: CritConfig
) -> str:
    return "TODO_OMEGA"
""",

"pipeline/reason_extractor.py": """
from typing import Dict, Any, List
from ..config import CritConfig


def extract_reasons(
    normalized_transcript: Dict[str, Any],
    omega: str,
    config: CritConfig
) -> List[str]:
    return []
""",

"pipeline/reason_validator.py": """
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
""",

"pipeline/counter_reason_generator.py": """
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
""",

"pipeline/aggregator.py": """
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
""",

"pipeline/justification_generator.py": """
from typing import List
from ..config import CritConfig
from ..types import ReasonEvaluation


def generate_justification(
    omega: str,
    reasons: List[ReasonEvaluation],
    counter_reasons: List[ReasonEvaluation],
    gamma_total: float,
    config: CritConfig
) -> str:
    return "TODO_JUSTIFICATION"
""",

# =========================
# Adapters
# =========================

"adapters/__init__.py": "",

"adapters/debate_trace_adapter.py": """
from typing import Dict, Any


def adapt_debate_trace(raw_trace: Dict[str, Any]) -> Dict[str, Any]:
    return raw_trace
""",

"adapters/judge_memo_adapter.py": """
from typing import Dict, Any


def adapt_judge_memo(raw_memo: Dict[str, Any]) -> Dict[str, Any]:
    return raw_memo
""",

# =========================
# Utils
# =========================

"utils/__init__.py": "",

"utils/id_generation.py": """
import hashlib


def generate_reason_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
""",

"utils/scoring.py": """
def compute_weight(gamma: float, theta: float) -> float:
    return gamma * theta
""",

"utils/json_io.py": """
import json
from typing import Any


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)
"""
}

# Create files
for relative_path, content in FILES.items():
    full_path = BASE_DIR / relative_path
    write_file(full_path, content)

print("\\nCRIT module fully bootstrapped.")