"""Build eval_v1.2 JSON artifacts from CRIT evaluation results.

=============================================================================
PURPOSE
=============================================================================

This module constructs eval artifacts conforming to eval.schema.json (v1.2.0).
An eval artifact is the structured output of evaluating a complete debate —
it contains both debate-level summary scores and per-turn breakdowns.

This is a shared utility used by both the RAudit CRIT scorer and any future
evaluation methods. It is decoupled from the scorer itself so that different
scoring approaches can produce artifacts in the same format.

=============================================================================
MULTI-TURN AGGREGATION LOGIC
=============================================================================

For a debate with N evaluated turns (proposals + revisions + judge decision):

  gamma_mean = (1/N) * sum(gamma_i for each turn i)
      The arithmetic mean of all per-turn gamma (reasonableness) scores.
      Represents the average reasoning quality across the debate.

  theta_mean = (1/N) * sum(theta_i for each turn i)
      The arithmetic mean of all per-turn theta (structural confidence) scores.
      Represents the average calibration quality across the debate.

  Per-turn pass criteria (conjunction — both must be met):
      gamma_i >= GAMMA_THRESHOLD (default 0.8, from RAudit rho*)
      AND
      theta_i >= THETA_THRESHOLD (default 0.5)

  Debate-level threshold_pass:
      True ONLY if ALL individual turns pass.
      This is deliberately strict — one weak turn in an otherwise strong
      debate means the debate as a whole is unreliable.

  overall_verdict (three-valued):
      "pass"  — all turns pass both thresholds (consistently strong reasoning)
      "fail"  — no turns pass either threshold (consistently weak reasoning)
      "mixed" — some turns pass, some fail (inconsistent quality)

  The "mixed" verdict is particularly informative for debugging: it means
  the system CAN produce good reasoning but doesn't do so reliably. This
  often indicates sensitivity to topic complexity, agent role, or round
  number (e.g., proposals are strong but revisions degrade).

=============================================================================
SCHEMA COMPATIBILITY
=============================================================================

The eval schema (v1.2.0) reserves slots for multiple evaluation methods:
  - crit:   CRIT scores (gamma, theta, threshold_pass, notes)
  - rca:    Root Cause Analysis (not yet implemented)
  - t3:     T3 evaluation (not yet implemented)
  - pid:    PID controller metrics (not yet implemented)
  - raudit: RAudit-specific extended fields (not yet implemented)

This builder only fills the "crit" slot. Other slots are set to None.
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Any

# ---------------------------------------------------------------------------
# Thresholds — from RAudit paper §4.2
# ---------------------------------------------------------------------------
# GAMMA_THRESHOLD (rho*): minimum reasonableness score for a turn to pass.
# 0.8 means the four-pillar average must be at least 8/10 normalised.
# This is deliberately high — adequate reasoning (6-7/10) is not sufficient.
GAMMA_THRESHOLD = 0.8

# THETA_THRESHOLD: minimum structural confidence for a turn to pass.
# Lower than gamma because calibration quality is harder to assess.
THETA_THRESHOLD = 0.5

# Version strings for provenance tracking in the artifact
CRIT_VERSION = "crit_v1"
EVALUATOR_VERSION = "eval_crit_v1"
SCHEMA_VERSION = "1.2.0"


def build_eval_artifact(
    debate_id: str,
    run_id: str,
    turn_results: list[tuple[str, Any]],
    evaluation_mode: str = "posthoc",
    experiment_label: str | None = None,
    notes: str | None = None,
) -> dict:
    """Build a complete eval artifact from per-turn CRIT results.

    This is a generic artifact builder that works with any result object
    that has .gamma, .theta, and .notes attributes (duck typing). Both
    RAuditCRITResult and any future result types are compatible.

    Parameters:
        debate_id: Stable debate identifier from the transcript. Used to
            link eval artifacts back to the debate they evaluate.
            Example: "debate_macro_value_risk_technical"

        run_id: Run identifier from the transcript. Distinguishes different
            runs of the same debate configuration.
            Example: "run_2026-02-19T20:26:08"

        turn_results: List of (turn_id, result) tuples, one per evaluated
            turn. The result object must have .gamma (float), .theta (float),
            and .notes (str) attributes. Turn order should match the debate
            transcript order.

        evaluation_mode: "posthoc" (evaluate after debate completes) or
            "in_loop" (evaluate during debate, with potential feedback).
            Currently only posthoc is implemented.

        experiment_label: Optional label for A/B experiment tracking.

        notes: Optional free-text notes for the artifact.

    Returns:
        Dict conforming to eval.schema.json v1.2.0.
    """
    if not turn_results:
        return _build_empty_artifact(debate_id, run_id, evaluation_mode)

    # ---------------------------------------------------------------------------
    # Per-turn evaluation: apply thresholds to each turn independently
    # ---------------------------------------------------------------------------
    turn_evaluations = []
    gammas = []       # Collect gamma values for debate-level aggregation
    thetas = []       # Collect theta values for debate-level aggregation
    turn_passes = []  # Track per-turn pass/fail for verdict computation

    for turn_id, result in turn_results:
        # A turn passes only if BOTH gamma >= 0.8 AND theta >= 0.5.
        # This conjunction ensures both reasoning quality and calibration
        # quality meet minimum standards.
        turn_pass = (
            result.gamma >= GAMMA_THRESHOLD and result.theta >= THETA_THRESHOLD
        )
        turn_passes.append(turn_pass)
        gammas.append(result.gamma)
        thetas.append(result.theta)

        turn_evaluations.append({
            "turn_id": turn_id,
            "crit": {
                "gamma_mean": round(result.gamma, 4),
                "theta_mean": round(result.theta, 4),
                "threshold_pass": turn_pass,
                "notes": result.notes or None,
            },
            # Reserved slots for other evaluation methods (not yet implemented)
            "rca": None,
            "t3": None,
            "pid": None,
            "raudit": None,
        })

    # ---------------------------------------------------------------------------
    # Debate-level aggregation
    # ---------------------------------------------------------------------------
    gamma_mean = sum(gammas) / len(gammas)
    theta_mean = sum(thetas) / len(thetas)

    # Verdict: strict conjunction across all turns
    all_pass = all(turn_passes)
    none_pass = not any(turn_passes)

    if all_pass:
        overall_verdict = "pass"    # Every turn meets both thresholds
    elif none_pass:
        overall_verdict = "fail"    # No turn meets thresholds
    else:
        overall_verdict = "mixed"   # Inconsistent quality across turns

    # ---------------------------------------------------------------------------
    # Build the artifact dict
    # ---------------------------------------------------------------------------
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "debate_id": debate_id,
        "run_id": run_id,
        "evaluation_mode": evaluation_mode,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "eval_metadata": {
            "evaluator_version": EVALUATOR_VERSION,
            "crit_version": CRIT_VERSION,
            "rca_version": None,
            "t3_version": None,
            "pid_version": None,
            "raudit_version": None,
            "notes": notes,
        },
        "run_summary": {
            "overall_verdict": overall_verdict,
            "crit_summary": {
                "gamma_mean": round(gamma_mean, 4),
                "theta_mean": round(theta_mean, 4),
                "threshold_pass": all_pass,
                "notes": notes,
            },
        },
        "turn_evaluations": turn_evaluations,
    }

    # Optional experiment config for A/B testing
    if experiment_label:
        artifact["experiment_config"] = {
            "label": experiment_label,
            "category": None,
            "interventions": {
                "crit_in_loop": evaluation_mode == "in_loop",
                "rca_in_loop": False,
            },
            "control": None,
            "extra_dimensions": None,
            "notes": None,
        }

    return artifact


def _build_empty_artifact(
    debate_id: str, run_id: str, evaluation_mode: str
) -> dict:
    """Build an artifact when no turns were evaluated.

    This can happen if the transcript parser found no evaluable turns
    (e.g., a trace with only critiques and no proposals/revisions/judge).
    An empty artifact always has verdict "fail" since we can't assess
    reasoning quality without any reasoning to evaluate.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "debate_id": debate_id,
        "run_id": run_id,
        "evaluation_mode": evaluation_mode,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "eval_metadata": {
            "evaluator_version": EVALUATOR_VERSION,
            "crit_version": CRIT_VERSION,
            "rca_version": None,
            "t3_version": None,
            "pid_version": None,
            "raudit_version": None,
            "notes": "No turns evaluated.",
        },
        "run_summary": {
            "overall_verdict": "fail",
            "crit_summary": {
                "gamma_mean": None,
                "theta_mean": None,
                "threshold_pass": False,
                "notes": "No turns evaluated.",
            },
        },
        "turn_evaluations": None,
    }
