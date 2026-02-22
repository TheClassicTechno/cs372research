#!/usr/bin/env python
"""CLI entry point for RAudit CRIT post-hoc evaluation.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

This is the main entry point for evaluating a multi-agent debate trace using
the RAudit CRIT protocol. It orchestrates the full pipeline:

  1. Load a debate trace JSON (output of the LangGraph multi-agent system)
  2. Parse it into canonical CRIT traces (one per evaluable turn)
  3. Score each trace using the RAudit four-pillar blind evaluator
  4. Aggregate scores into a debate-level eval artifact (JSON)

The pipeline is "post-hoc" — it evaluates the debate AFTER it completes,
with no influence on the debate itself. (Future work: "in-loop" evaluation
where CRIT scores feed back into the debate agents mid-conversation.)

=============================================================================
USAGE EXAMPLES
=============================================================================

  # Basic: evaluate a trace and print results to stdout
  python eval/run_crit.py --input demo_debate_traces/debate_langgraph_2026-02-19_08-26-08pm.json

  # Save the eval artifact next to the input file (as *_crit_eval.json)
  python eval/run_crit.py --input trace.json --save

  # Save to a specific output path
  python eval/run_crit.py --input trace.json --output eval_result.json

  # Use a different evaluator model
  python eval/run_crit.py --input trace.json --model gpt-4o --provider openai

  # Use Anthropic as the evaluator
  python eval/run_crit.py --input trace.json --provider anthropic --model claude-sonnet-4-5-20250929

  # Override debate/run IDs (useful for experiment tracking)
  python eval/run_crit.py --input trace.json --debate-id exp_001 --run-id run_001

  # Verbose logging (shows LLM request/response details)
  python eval/run_crit.py --input trace.json -v

=============================================================================
REQUIRED ENVIRONMENT VARIABLES
=============================================================================

  OPENAI_API_KEY      — required if --provider openai (default)
  ANTHROPIC_API_KEY   — required if --provider anthropic

=============================================================================
INPUT FORMAT
=============================================================================

Multi_agent debate trace JSON with top-level keys:
  "trace"        — AgentTrace (overall decision summary, market context)
  "debate_turns" — flat list of debate turns (proposals, critiques, revisions, judge)
  "config"       — debate configuration (roles, max_rounds, model, etc.)

See eval/crit/transcript_parser.py for full format documentation.

=============================================================================
OUTPUT FORMAT
=============================================================================

Eval artifact JSON conforming to eval.schema.json v1.2.0. Key fields:
  run_summary.overall_verdict  — "pass", "fail", or "mixed"
  run_summary.crit_summary     — gamma_mean, theta_mean, threshold_pass
  turn_evaluations[]           — per-turn gamma, theta, pass/fail, notes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports. This lets us run the script
# directly (python eval/run_crit.py) without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# RAuditCRITScorer: the blind four-pillar evaluator (1 LLM call per turn)
# build_raudit_eval_artifact: aggregates per-turn scores into debate-level artifact
from eval.crit_raudit import RAuditCRITScorer, build_raudit_eval_artifact

# TranscriptParser: transforms raw debate trace JSON into canonical CRIT traces
from eval.crit.transcript_parser import TranscriptParser

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAudit CRIT post-hoc evaluation on a multi_agent debate trace."
    )
    parser.add_argument(
        "--input", required=True, help="Path to multi_agent debate trace JSON."
    )
    parser.add_argument(
        "--output", default=None, help="Path to write eval artifact JSON. Defaults to stdout."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Write eval artifact next to input file as <input>_crit_eval.json."
    )
    parser.add_argument(
        "--debate-id", default=None,
        help="Debate identifier for the artifact. Defaults to input filename stem."
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier for the artifact. Defaults to trace.logged_at or input filename."
    )
    parser.add_argument(
        "--provider", default="openai", help="LLM provider (default: openai)."
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def validate_trace(data: dict) -> None:
    """Basic structural validation of the input multi_agent trace.

    Checks that the JSON has the required keys for the transcript parser.
    This catches common mistakes like passing a raw LLM response instead
    of a full debate trace file.
    """
    if "debate_turns" not in data:
        raise ValueError(
            "Input JSON missing 'debate_turns' key. "
            "Expected a multi_agent debate trace with: trace, debate_turns, config."
        )

    turns = data.get("debate_turns")
    if turns is None or (isinstance(turns, list) and len(turns) == 0):
        raise ValueError("Input trace has no debate_turns.")

    if "trace" not in data:
        logger.warning("Input JSON missing top-level 'trace' key. Context will be empty.")


def derive_debate_id(data: dict, input_path: Path) -> str:
    """Derive a debate_id from the trace or filename.

    Tries to build a meaningful ID from the debate config (e.g.,
    "debate_macro_value_risk_technical"). Falls back to the filename.
    """
    trace = data.get("trace", {})
    config = data.get("config", {})

    # Try to build from config roles (e.g., ["macro", "value", "risk"])
    roles = config.get("roles", [])
    if roles:
        return f"debate_{'_'.join(roles)}"

    # Fall back to filename
    return input_path.stem


def derive_run_id(data: dict, input_path: Path) -> str:
    """Derive a run_id from the trace or filename.

    Uses the trace's logged_at timestamp if available (gives a unique,
    sortable run ID). Falls back to the filename.
    """
    trace = data.get("trace", {})
    logged_at = trace.get("logged_at", "")
    if logged_at:
        return f"run_{logged_at}"
    return f"run_{input_path.stem}"


async def run_crit(args: argparse.Namespace) -> dict:
    """Execute the full RAudit CRIT evaluation pipeline.

    This is the core async function that:
      1. Loads and validates the debate trace JSON
      2. Parses it into canonical CRIT traces via TranscriptParser
      3. Scores each trace using RAuditCRITScorer (1 LLM call per turn)
      4. Aggregates into a debate-level eval artifact

    Each turn is scored sequentially (not in parallel) to avoid rate limit
    issues and to make progress output readable. For large evaluations,
    consider batching with asyncio.gather and rate limiting.
    """
    # --- Step 1: Load and validate the debate trace ---
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    validate_trace(trace_data)

    config = trace_data.get("config", {})
    print(f"Loaded trace: {input_path.name}")
    print(f"  Roles: {config.get('roles', 'N/A')}")
    print(f"  Max rounds: {config.get('max_rounds', 'N/A')}")
    print(f"  Model: {config.get('model_name', 'N/A')}")

    # Derive stable IDs for the eval artifact. These identify which debate
    # and which run this evaluation corresponds to, enabling tracking across
    # experiments.
    debate_id = args.debate_id or derive_debate_id(trace_data, input_path)
    run_id = args.run_id or derive_run_id(trace_data, input_path)

    # --- Step 2: Parse the trace into canonical CRIT traces ---
    # The TranscriptParser extracts proposals, revisions, and judge decisions
    # (skipping critiques) and normalises them into CanonicalTrace objects.
    parser = TranscriptParser(trace_data)
    traces = parser.extract_traces()
    print(f"Extracted {len(traces)} canonical traces from debate.")

    if not traces:
        print("Warning: No evaluable traces found.", file=sys.stderr)
        return build_raudit_eval_artifact(
            debate_id=debate_id,
            run_id=run_id,
            turn_results=[],
        )

    # --- Step 3: Score each trace using the RAudit four-pillar evaluator ---
    # The scorer sends each canonical trace to an evaluator LLM that is
    # BLIND to ground truth. It scores the four pillars (logical validity,
    # evidential support, alternative consideration, causal alignment) and
    # returns gamma (reasonableness), theta (structural confidence), and
    # per-pillar justifications.
    scorer = RAuditCRITScorer(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    turn_results = []
    for i, trace in enumerate(traces):
        print(f"  Scoring turn {i + 1}/{len(traces)}: {trace.turn_id} "
              f"({trace.speaker_id})...", end="", flush=True)

        # Each score_async() call makes one LLM request containing the full
        # reasoning trace and the RAudit rubric. The evaluator returns a JSON
        # object with four pillar scores, theta, and notes.
        result = await scorer.score_async(
            claim=trace.claim,
            reasons=trace.reasons,
            counterarguments=trace.counterarguments,
            assumptions=trace.assumptions,
            final_decision=trace.final_decision,
            context=trace.context,
        )

        # Print per-turn scores for immediate feedback.
        # gamma (γ) = reasonableness (mean of 4 pillars), theta (θ) = calibration.
        print(f" γ={result.gamma:.2f}, θ={result.theta:.2f}")
        turn_results.append((trace.turn_id, result))

    # --- Step 4: Aggregate into a debate-level eval artifact ---
    # The artifact builder computes debate-level gamma_mean, theta_mean,
    # applies thresholds, and determines the overall verdict.
    artifact = build_raudit_eval_artifact(
        debate_id=debate_id,
        run_id=run_id,
        turn_results=turn_results,
    )

    return artifact


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Run the async pipeline synchronously from the CLI entry point
    artifact = asyncio.run(run_crit(args))

    # Output the eval artifact JSON
    output_json = json.dumps(artifact, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"\nEval artifact written to: {output_path}")
    elif args.save:
        input_path = Path(args.input)
        output_path = input_path.with_name(
            input_path.stem + "_crit_eval.json"
        )
        output_path.write_text(output_json, encoding="utf-8")
        print(f"\nEval artifact written to: {output_path}")
    else:
        print("\n--- Eval Artifact ---")
        print(output_json)

    # Print a summary for quick visual inspection
    summary = artifact.get("run_summary", {})
    crit = summary.get("crit_summary", {})
    print(f"\nVerdict: {summary.get('overall_verdict', 'N/A')}")
    print(f"  gamma_mean: {crit.get('gamma_mean', 'N/A')}")
    print(f"  theta_mean: {crit.get('theta_mean', 'N/A')}")
    print(f"  threshold_pass: {crit.get('threshold_pass', 'N/A')}")


if __name__ == "__main__":
    main()
