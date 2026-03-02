"""
Unified post-hoc evaluation pipeline for debate traces.

Orchestrates CRIT, RCA consistency, portfolio divergence, and financial
metrics into a single ``EvalArtifact`` conforming to eval.schema.json.

Runs automatically at the end of every simulation (across all episodes),
and can also be invoked standalone on saved trace files.

Usage (CLI):
    python -m eval.pipeline <trace_file.json> [--out eval.json]

Usage (programmatic):
    from eval.pipeline import EvalPipeline

    pipeline = EvalPipeline(llm_fn=my_llm_fn)
    artifact = pipeline.evaluate_trace(
        trace_data=loaded_trace_dict,
        episode_log=episode_log,
        initial_cash=100_000,
    )
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2.0"
EVALUATOR_VERSION = "pipeline_v1"


# =========================================================================
# Result models (aligned with eval.schema.json)
# =========================================================================


class CritSummary(BaseModel):
    """Aggregate CRIT scores across debate turns."""

    rho_bar_mean: float | None = None
    pillar_means: dict[str, float] | None = None
    threshold_pass: bool | None = None
    notes: str | None = None

    model_config = {"extra": "allow"}


class RcaSummary(BaseModel):
    """Aggregate RCA consistency results."""

    trace_consistency_rate: float | None = None
    sycophantic_rate: float | None = None
    stubborn_rate: float | None = None
    verdict: Literal["pass", "fail", "mixed"] | None = None
    notes: str | None = None

    model_config = {"extra": "allow"}


class DivergenceSummary(BaseModel):
    """Aggregate portfolio divergence results."""

    js_divergence_by_round: dict[int, float] | None = None
    active_share_by_round: dict[int, float] | None = None
    notes: str | None = None

    model_config = {"extra": "allow"}


class FinancialSummary(BaseModel):
    """Financial performance summary from eval.financial."""

    total_return_pct: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    max_drawdown_pct: float | None = None
    calmar_ratio: float | None = None
    notes: str | None = None

    model_config = {"extra": "allow"}


class RunSummary(BaseModel):
    """Aggregated evaluation results for the run."""

    crit_summary: CritSummary | None = None
    rca_summary: RcaSummary | None = None
    divergence_summary: DivergenceSummary | None = None
    financial_summary: FinancialSummary | None = None

    model_config = {"extra": "allow"}


class TurnEval(BaseModel):
    """Per-turn evaluation output."""

    turn_id: str
    turn_type: str | None = None
    role: str | None = None
    rca_verdict: str | None = None
    rca_confidence: float | None = None

    model_config = {"extra": "allow"}


class EvalMetadata(BaseModel):
    """Metadata about the evaluator versions used."""

    evaluator_version: str = EVALUATOR_VERSION
    crit_version: str | None = None
    rca_version: str | None = None
    notes: str | None = None

    model_config = {"extra": "allow"}


class EvalArtifact(BaseModel):
    """Top-level evaluation artifact conforming to eval.schema.json v1.2.0."""

    schema_version: str = SCHEMA_VERSION
    debate_id: str
    run_id: str
    evaluation_mode: Literal["posthoc", "in_loop"] = "posthoc"
    evaluated_at: str
    eval_metadata: EvalMetadata = Field(default_factory=EvalMetadata)
    run_summary: RunSummary = Field(default_factory=RunSummary)
    turn_evaluations: list[TurnEval] | None = None
    control_trace: list[dict] | None = None

    model_config = {"extra": "allow"}


# =========================================================================
# Pipeline
# =========================================================================


class EvalPipeline:
    """Unified post-hoc evaluation pipeline.

    Orchestrates CRIT, RCA consistency, portfolio divergence, and
    financial metrics.  Each evaluator can be toggled on/off.

    CRIT requires an ``llm_fn``; if not provided, CRIT is skipped.
    Consistency requires an LLM (uses ChatOpenAI internally); if
    ``run_consistency`` is True but no API key is available, it will
    log a warning and skip.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str] | None = None,
        run_crit: bool = True,
        run_consistency: bool = True,
        run_divergence: bool = True,
        run_financials: bool = True,
    ) -> None:
        self._llm_fn = llm_fn
        self._run_crit = run_crit and llm_fn is not None
        self._run_consistency = run_consistency
        self._run_divergence = run_divergence
        self._run_financials = run_financials

    def evaluate_trace(
        self,
        trace_data: dict,
        episode_log: Any | None = None,
        initial_cash: float = 100_000,
        debate_id: str = "",
        run_id: str = "",
    ) -> EvalArtifact:
        """Run all enabled evaluators on a single debate trace.

        Args:
            trace_data: Loaded trace JSON dict (must have ``debate_turns``
                and optionally ``trace`` keys).
            episode_log: ``EpisodeLog`` for financial metrics.
            initial_cash: Starting cash for financial metrics.
            debate_id: Identifier for the debate.
            run_id: Identifier for the run.

        Returns:
            ``EvalArtifact`` with all evaluation results.
        """
        now = datetime.now(timezone.utc).isoformat()

        crit_summary = self._run_crit_eval(trace_data) if self._run_crit else None
        rca_summary, turn_evals = self._run_rca_eval(trace_data) if self._run_consistency else (None, None)
        div_summary = self._run_divergence_eval(trace_data) if self._run_divergence else None
        fin_summary = self._run_financial_eval(episode_log, initial_cash) if self._run_financials else None

        return EvalArtifact(
            debate_id=debate_id or trace_data.get("trace", {}).get("observation_timestamp", "unknown"),
            run_id=run_id or "unknown",
            evaluated_at=now,
            eval_metadata=EvalMetadata(
                crit_version="crit_v1" if self._run_crit else None,
                rca_version="rca_v1" if self._run_consistency else None,
            ),
            run_summary=RunSummary(
                crit_summary=crit_summary,
                rca_summary=rca_summary,
                divergence_summary=div_summary,
                financial_summary=fin_summary,
            ),
            turn_evaluations=turn_evals,
        )

    # -----------------------------------------------------------------
    # Individual evaluators
    # -----------------------------------------------------------------

    def _run_crit_eval(self, trace_data: dict) -> CritSummary | None:
        """Run CRIT reasoning audit on debate turns."""
        try:
            from eval.crit import CritScorer

            debate_turns = trace_data.get("debate_turns", [])
            if not debate_turns:
                return None

            scorer = CritScorer(llm_fn=self._llm_fn)  # type: ignore[arg-type]

            case_data = ""
            trace = trace_data.get("trace", {})
            if isinstance(trace, dict):
                case_data = trace.get("what_i_saw", "")

            decisions = [
                t for t in debate_turns
                if t.get("type") in ("proposal", "revision", "judge_decision")
            ]

            result = scorer.score(
                case_data=case_data,
                agent_traces=debate_turns,
                decisions=decisions,
            )

            pillar_means: dict[str, float] = {}
            if result.agent_scores:
                ic_vals, es_vals, ta_vals, ci_vals = [], [], [], []
                for cr in result.agent_scores.values():
                    ps = cr.pillar_scores
                    ic_vals.append(ps.internal_consistency)
                    es_vals.append(ps.evidence_support)
                    ta_vals.append(ps.trace_alignment)
                    ci_vals.append(ps.causal_integrity)
                pillar_means = {
                    "internal_consistency": sum(ic_vals) / len(ic_vals),
                    "evidence_support": sum(es_vals) / len(es_vals),
                    "trace_alignment": sum(ta_vals) / len(ta_vals),
                    "causal_integrity": sum(ci_vals) / len(ci_vals),
                }

            return CritSummary(
                rho_bar_mean=result.rho_bar,
                pillar_means=pillar_means or None,
                threshold_pass=result.rho_bar >= 0.8 if result.rho_bar is not None else None,
            )
        except Exception as exc:
            logger.warning("CRIT evaluation failed: %s", exc)
            return CritSummary(notes=f"CRIT failed: {exc}")

    def _run_rca_eval(
        self, trace_data: dict
    ) -> tuple[RcaSummary | None, list[TurnEval] | None]:
        """Run RCA consistency checks on debate turns."""
        try:
            from eval.consistency import analyze_trace

            result = analyze_trace(trace_data)
            if result is None:
                return None, None

            turn_evals = [
                TurnEval(
                    turn_id=f"turn_{tr.turn_index}",
                    turn_type=tr.turn_type,
                    role=tr.role,
                    rca_verdict=tr.verdict,
                    rca_confidence=tr.confidence,
                )
                for tr in result.turn_results
            ]

            rate = result.consistency_rate
            if rate >= 0.8:
                verdict: Literal["pass", "fail", "mixed"] = "pass"
            elif rate <= 0.3:
                verdict = "fail"
            else:
                verdict = "mixed"

            return RcaSummary(
                trace_consistency_rate=rate,
                sycophantic_rate=result.sycophantic_rate,
                stubborn_rate=result.stubborn_rate,
                verdict=verdict,
            ), turn_evals

        except Exception as exc:
            logger.warning("RCA evaluation failed: %s", exc)
            return RcaSummary(notes=f"RCA failed: {exc}"), None

    def _run_divergence_eval(self, trace_data: dict) -> DivergenceSummary | None:
        """Compute portfolio divergence metrics from debate turns."""
        try:
            from eval.divergence import analyze_divergence

            result = analyze_divergence(trace_data)

            if not result.js_divergence_by_round and not result.active_share_by_round:
                return None

            return DivergenceSummary(
                js_divergence_by_round=result.js_divergence_by_round or None,
                active_share_by_round=result.active_share_by_round or None,
            )
        except Exception as exc:
            logger.warning("Divergence evaluation failed: %s", exc)
            return DivergenceSummary(notes=f"Divergence failed: {exc}")

    def _run_financial_eval(
        self, episode_log: Any | None, initial_cash: float
    ) -> FinancialSummary | None:
        """Compute financial performance metrics from an EpisodeLog."""
        if episode_log is None:
            return None
        try:
            from eval.financial import compute_financial_metrics

            metrics = compute_financial_metrics(episode_log, initial_cash)
            return FinancialSummary(
                total_return_pct=metrics.total_return_pct,
                sharpe_ratio=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                max_drawdown=metrics.max_drawdown,
                max_drawdown_pct=metrics.max_drawdown_pct,
                calmar_ratio=metrics.calmar_ratio,
            )
        except Exception as exc:
            logger.warning("Financial evaluation failed: %s", exc)
            return FinancialSummary(notes=f"Financial eval failed: {exc}")


# =========================================================================
# CLI entry point
# =========================================================================


def _build_llm_fn() -> Callable[[str, str], str]:
    """Build an LLM function using the same backend as the debate agents."""
    from multi_agent.graph import _call_llm

    config = {"model_name": "gpt-4o-mini", "temperature": 0.3}
    return lambda sys_prompt, usr_prompt: _call_llm(config, sys_prompt, usr_prompt)


def main() -> None:
    """Run the evaluation pipeline on a saved trace file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the unified post-hoc evaluation pipeline on a debate trace file.",
    )
    parser.add_argument("trace_file", help="Path to the debate trace JSON file.")
    parser.add_argument("--out", "-o", default=None, help="Output path for eval artifact JSON (default: stdout).")
    parser.add_argument("--no-crit", action="store_true", help="Skip CRIT evaluation.")
    parser.add_argument("--no-consistency", action="store_true", help="Skip RCA consistency evaluation.")
    parser.add_argument("--no-divergence", action="store_true", help="Skip divergence evaluation.")
    parser.add_argument("--no-financials", action="store_true", help="Skip financial metrics.")
    args = parser.parse_args()

    try:
        with open(args.trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {args.trace_file}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {args.trace_file}: {exc}")
        sys.exit(1)

    needs_llm = not args.no_crit
    llm_fn = _build_llm_fn() if needs_llm else None

    pipeline = EvalPipeline(
        llm_fn=llm_fn,
        run_crit=not args.no_crit,
        run_consistency=not args.no_consistency,
        run_divergence=not args.no_divergence,
        run_financials=not args.no_financials,
    )

    artifact = pipeline.evaluate_trace(
        trace_data=trace_data,
        debate_id=trace_data.get("trace", {}).get("observation_timestamp", "unknown"),
        run_id="cli",
    )

    output = json.dumps(artifact.model_dump(), indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Eval artifact written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
