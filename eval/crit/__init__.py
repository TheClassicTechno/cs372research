"""CRIT post-hoc evaluator — transcript parsing and artifact building.

Provides shared infrastructure for CRIT-based evaluation:
  - TranscriptParser: extracts canonical traces from multi_agent debate transcripts.
  - CanonicalTrace: normalised per-turn data structure for evaluation.
  - build_eval_artifact: builds eval.schema.json v1.2.0 artifacts from scored results.

For the RAudit four-pillar scorer, see eval.crit_raudit.
"""

from eval.crit.transcript_parser import TranscriptParser, CanonicalTrace
from eval.crit.artifact_builder import build_eval_artifact

__all__ = [
    "TranscriptParser",
    "CanonicalTrace",
    "build_eval_artifact",
]
