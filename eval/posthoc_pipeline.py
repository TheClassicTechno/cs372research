"""
Post-hoc evaluation pipeline for multi-agent debate simulations.

Reads episode logs and trace files produced by run_simulation.py, runs
CRIT (blind reasoning quality) and ConsistencyJudge (trace-output
consistency) evaluations, and outputs an eval.json artifact conforming
to eval/schemas/eval.schema.json.

Usage:
    python -m eval.posthoc_pipeline results/debate_2agent_real
    python -m eval.posthoc_pipeline results/debate_2agent_real --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from eval.consistency import (
    ConsistencyJudge,
    DebateConsistencyVerdict,
    ProposalConsistencyVerdict,
)
from eval.crit.scorer import CritScorer

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trace_file(traces_dir: Path, timestamp_hint: str) -> Path | None:
    """Find the trace file closest to the given timestamp hint.

    The trace files are named like:
        debate_langgraph_2026-03-01_04-24-37am.json

    We match by finding the trace whose name contains a timestamp
    closest to the episode decision timestamp.
    """
    if not traces_dir.exists():
        return None
    candidates = sorted(traces_dir.glob("debate_langgraph_*.json"))
    if not candidates:
        return None
    # Simple heuristic: return all candidates and let the caller pick
    return candidates


def _load_trace_for_decision(
    traces_dir: Path,
    decision_timestamp: str,
    used_traces: set[str],
) -> dict | None:
    """Load the trace file for a specific decision point.

    Matches by finding the trace file with the closest timestamp
    that hasn't already been used.
    """
    candidates = _find_trace_file(traces_dir, decision_timestamp)
    if not candidates:
        return None

    for candidate in candidates:
        if str(candidate) not in used_traces:
            used_traces.add(str(candidate))
            with open(candidate) as f:
                return json.load(f)
    return None


def _build_case_context(agent_output: dict) -> str:
    """Extract the case context string from agent_output's debate_trace."""
    trace = agent_output.get("debate_trace", {})
    return trace.get("what_i_saw", "")


def _is_consistent(verdict: Any) -> bool:
    """Check if a consistency verdict is CONSISTENT."""
    if isinstance(verdict, str):
        return verdict == "CONSISTENT"
    return verdict in (
        DebateConsistencyVerdict.CONSISTENT,
        ProposalConsistencyVerdict.CONSISTENT,
    )


# ---------------------------------------------------------------------------
# CRIT evaluation
# ---------------------------------------------------------------------------

def run_crit_on_trace(
    scorer: CritScorer,
    trace_data: dict,
    case_context: str,
) -> dict[str, Any]:
    """Run CRIT scorer on all agents in a trace file.

    Returns dict with per-agent scores and aggregated rho_bar.
    """
    turns = trace_data.get("debate_turns", [])
    if not turns:
        return {"error": "no debate turns", "rho_bar": None}

    # Extract agent traces and decisions (last revision or proposal per role)
    agent_traces = []
    decisions = []
    latest_by_role: dict[str, dict] = {}

    for turn in turns:
        role = turn.get("role", "unknown")
        turn_type = turn.get("type", "unknown")
        content = turn.get("content", {})

        agent_traces.append({
            "role": role,
            "type": turn_type,
            "content": content,
        })

        # Track the latest proposal/revision per role as their "decision"
        if turn_type in ("proposal", "revision") and isinstance(content, dict):
            latest_by_role[role] = {"role": role, **content}

    decisions = list(latest_by_role.values())

    try:
        result = scorer.score(case_context, agent_traces, decisions)
        return {
            "rho_bar": result.rho_bar,
            "agent_scores": {
                role: {
                    "rho_bar": cr.rho_bar,
                    "pillar_scores": cr.pillar_scores.model_dump(),
                    "diagnostics": cr.diagnostics.model_dump(),
                }
                for role, cr in result.agent_scores.items()
            },
        }
    except Exception as e:
        logger.warning("CRIT scoring failed: %s", e)
        return {"error": str(e), "rho_bar": None}


# ---------------------------------------------------------------------------
# Consistency (RCA) evaluation
# ---------------------------------------------------------------------------

def run_consistency_on_trace(
    judge: ConsistencyJudge,
    trace_data: dict,
) -> dict[str, Any]:
    """Run ConsistencyJudge on all turns in a trace file.

    Returns per-turn verdicts and an aggregate consistency rate.
    """
    turns = trace_data.get("debate_turns", [])
    if not turns:
        return {"error": "no debate turns", "consistency_rate": None}

    import copy

    results = []
    consistent_count = 0
    total_checks = 0

    for i, turn in enumerate(turns):
        turn_type = turn.get("type", "unknown")
        role = turn.get("role", "unknown")

        try:
            if turn_type == "proposal":
                result = judge.check_proposal(turn)
                results.append({
                    "turn": i,
                    "type": turn_type,
                    "role": role,
                    "verdict": result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict),
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                })
                total_checks += 1
                if _is_consistent(result.verdict):
                    consistent_count += 1

            elif turn_type == "critique":
                content = turn.get("content", {})
                input_params = turn.get("input_params", {})
                critiques = content.get("critiques", [])
                proposals = input_params.get("all_proposals_for_critique", [])

                critiques_by_role = {
                    c.get("target_role", "").upper(): c for c in critiques
                }
                proposals_by_role = {
                    p.get("role", "").upper(): p for p in proposals
                }

                for target_role, crit in critiques_by_role.items():
                    if target_role not in proposals_by_role:
                        continue
                    turn_for_judge = copy.deepcopy(turn)
                    turn_for_judge["content"]["critiques"] = [crit]
                    turn_for_judge["input_params"]["all_proposals_for_critique"] = [
                        proposals_by_role[target_role]
                    ]
                    result = judge.check_critique(turn_for_judge)
                    results.append({
                        "turn": i,
                        "type": turn_type,
                        "role": role,
                        "target": target_role,
                        "verdict": result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict),
                        "confidence": result.confidence,
                        "explanation": result.explanation,
                    })
                    total_checks += 1
                    if _is_consistent(result.verdict):
                        consistent_count += 1

            elif turn_type == "revision":
                result = judge.check_revision(turn)
                results.append({
                    "turn": i,
                    "type": turn_type,
                    "role": role,
                    "verdict": result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict),
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                })
                total_checks += 1
                if _is_consistent(result.verdict):
                    consistent_count += 1

            elif turn_type == "judge_decision":
                result = judge.check_judge_decision(turn)
                results.append({
                    "turn": i,
                    "type": turn_type,
                    "role": role,
                    "verdict": result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict),
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                })
                total_checks += 1
                if _is_consistent(result.verdict):
                    consistent_count += 1

        except Exception as e:
            logger.warning("Consistency check failed on turn %d (%s/%s): %s", i, role, turn_type, e)
            results.append({
                "turn": i,
                "type": turn_type,
                "role": role,
                "error": str(e),
            })

    consistency_rate = consistent_count / total_checks if total_checks > 0 else None
    return {
        "consistency_rate": consistency_rate,
        "total_checks": total_checks,
        "consistent_count": consistent_count,
        "turn_results": results,
    }


# ---------------------------------------------------------------------------
# Causal ladder (T3) analysis
# ---------------------------------------------------------------------------

def analyze_causal_claims(trace_data: dict) -> dict[str, Any]:
    """Analyze the Pearl causal ladder levels in agent claims.

    Checks whether agents use L1 (association), L2 (intervention), or
    L3 (counterfactual) reasoning, and whether the claims match.
    """
    turns = trace_data.get("debate_turns", [])
    levels_seen: set[str] = set()
    claim_count = 0

    for turn in turns:
        content = turn.get("content", {})
        if not isinstance(content, dict):
            continue
        claims = content.get("claims", [])
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if isinstance(claim, dict):
                level = claim.get("pearl_level", "")
                if level:
                    levels_seen.add(level)
                    claim_count += 1

    # Determine highest detected rung
    detected = "L1"
    if "L3" in levels_seen:
        detected = "L3"
    elif "L2" in levels_seen:
        detected = "L2"

    return {
        "detected_rung": detected,
        "levels_seen": sorted(levels_seen),
        "claim_count": claim_count,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def evaluate_episode(
    episode_dir: Path,
    traces_dir: Path,
    scorer: CritScorer,
    judge: ConsistencyJudge,
) -> dict[str, Any]:
    """Evaluate a single episode's decision points."""
    log_path = episode_dir / "episode_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"No episode_log.json in {episode_dir}")

    with open(log_path) as f:
        episode_log = json.load(f)

    decision_evals = []
    crit_rho_bars = []
    consistency_rates = []
    all_t3_levels: set[str] = set()
    used_traces: set[str] = set()

    for dp in episode_log.get("decision_point_logs", []):
        case_id = dp.get("case_id", "unknown")
        timestamp = dp.get("timestamp", "")
        agent_output = dp.get("agent_output", {})

        # Try to find the corresponding trace file
        trace_data = _load_trace_for_decision(traces_dir, timestamp, used_traces)

        dp_eval: dict[str, Any] = {"case_id": case_id}

        if trace_data:
            # Run CRIT
            case_context = _build_case_context(agent_output)
            crit_result = run_crit_on_trace(scorer, trace_data, case_context)
            dp_eval["crit"] = crit_result
            if crit_result.get("rho_bar") is not None:
                crit_rho_bars.append(crit_result["rho_bar"])

            # Run consistency
            consistency_result = run_consistency_on_trace(judge, trace_data)
            dp_eval["consistency"] = {
                "consistency_rate": consistency_result["consistency_rate"],
                "total_checks": consistency_result["total_checks"],
                "consistent_count": consistency_result["consistent_count"],
            }
            dp_eval["consistency_details"] = consistency_result["turn_results"]
            if consistency_result["consistency_rate"] is not None:
                consistency_rates.append(consistency_result["consistency_rate"])

            # Analyze causal claims
            t3_result = analyze_causal_claims(trace_data)
            dp_eval["t3"] = t3_result
            all_t3_levels.update(t3_result["levels_seen"])
        else:
            dp_eval["error"] = "no matching trace file found"

        decision_evals.append(dp_eval)

    return {
        "episode_id": episode_log.get("episode_id", "unknown"),
        "decision_evals": decision_evals,
        "crit_rho_bars": crit_rho_bars,
        "consistency_rates": consistency_rates,
        "t3_levels": sorted(all_t3_levels),
    }


def build_eval_artifact(
    run_dir: Path,
    episode_results: list[dict],
    config: dict | None,
) -> dict[str, Any]:
    """Build the final eval.json artifact conforming to eval.schema.json."""
    # Aggregate CRIT scores
    all_rho_bars = []
    all_consistency = []
    all_t3: set[str] = set()
    all_decision_evals = []

    for ep in episode_results:
        all_rho_bars.extend(ep["crit_rho_bars"])
        all_consistency.extend(ep["consistency_rates"])
        all_t3.update(ep["t3_levels"])
        all_decision_evals.extend(ep["decision_evals"])

    mean_rho = sum(all_rho_bars) / len(all_rho_bars) if all_rho_bars else None
    mean_consistency = (
        sum(all_consistency) / len(all_consistency) if all_consistency else None
    )

    # Determine T3 detected rung
    detected_rung = "L1"
    if "L3" in all_t3:
        detected_rung = "L3"
    elif "L2" in all_t3:
        detected_rung = "L2"

    # Determine overall verdict
    crit_pass = mean_rho is not None and mean_rho >= 0.7
    rca_pass = mean_consistency is not None and mean_consistency >= 0.8
    if crit_pass and rca_pass:
        overall = "pass"
    elif not crit_pass and not rca_pass:
        overall = "fail"
    else:
        overall = "mixed"

    # Build config label
    agent_cfg = config.get("agent", {}) if config else {}
    roles = agent_cfg.get("debate_roles", [])
    label = f"{len(roles) or '?'}agent_debate"
    if agent_cfg.get("system_prompt_override") == "mock":
        label += "_mock"

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "debate_id": f"{run_dir.name}",
        "run_id": run_dir.name,
        "evaluation_mode": "posthoc",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": {
            "label": label,
            "category": "core_ablation",
            "interventions": {
                "crit_in_loop": agent_cfg.get("pid_enabled", False),
                "rca_in_loop": False,
            },
            "control": None,
            "extra_dimensions": {
                "num_agents": len(roles) if roles else None,
                "max_rounds": agent_cfg.get("max_rounds", 1),
                "model": agent_cfg.get("llm_model", "unknown"),
            },
            "notes": None,
        },
        "eval_metadata": {
            "evaluator_version": "posthoc_v1",
            "crit_version": "crit_v2",
            "rca_version": "rca_v1",
            "t3_version": "t3_v1",
            "pid_version": None,
            "raudit_version": None,
            "notes": f"Evaluated {len(episode_results)} episode(s), "
                     f"{len(all_decision_evals)} decision point(s).",
        },
        "run_summary": {
            "overall_verdict": overall,
            "crit_summary": {
                "gamma_mean": mean_rho,
                "theta_mean": mean_rho,  # same for now (single-round aggregate)
                "threshold_pass": crit_pass,
                "notes": f"Averaged over {len(all_rho_bars)} decision point(s).",
            },
            "rca_summary": {
                "trace_consistency_rate": mean_consistency,
                "verdict": "pass" if rca_pass else ("fail" if mean_consistency is not None else None),
                "notes": f"Averaged over {len(all_consistency)} decision point(s).",
            },
            "t3_summary": {
                "required_rung": "L2",
                "detected_rung": detected_rung,
                "trap_detected": None,
                "pass": detected_rung in ("L2", "L3"),
                "notes": f"Levels seen: {sorted(all_t3)}",
            },
            "pid_summary": None,
            "raudit_summary": None,
        },
        "turn_evaluations": all_decision_evals,
        "control_trace": None,
    }

    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation pipeline for debate simulations."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to simulation results directory (e.g. results/debate_2agent_real)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for CRIT and ConsistencyJudge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for eval.json (default: <run_dir>/eval.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist.")
        sys.exit(1)

    # Load config
    config_path = run_dir / "config.yaml"
    config = None
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Set up evaluators
    llm = ChatOpenAI(model=args.model, temperature=0.0)

    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return resp.content

    scorer = CritScorer(llm_fn=llm_fn)
    judge = ConsistencyJudge(model_name=args.model, temperature=0.0)

    # Find traces directory
    traces_dir = Path("traces")

    # Find episode directories
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        print(f"Error: no episodes/ directory in {run_dir}")
        sys.exit(1)

    episode_dirs = sorted(episodes_dir.iterdir())
    if not episode_dirs:
        print(f"Error: no episodes found in {episodes_dir}")
        sys.exit(1)

    logger.info("Evaluating %d episode(s) from %s", len(episode_dirs), run_dir)

    # Evaluate each episode
    episode_results = []
    for ep_dir in episode_dirs:
        if not ep_dir.is_dir():
            continue
        logger.info("Evaluating episode: %s", ep_dir.name)
        result = evaluate_episode(ep_dir, traces_dir, scorer, judge)
        episode_results.append(result)

    # Build and write eval artifact
    artifact = build_eval_artifact(run_dir, episode_results, config)

    output_path = args.output or (run_dir / "eval.json")
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    logger.info("Eval artifact written to %s", output_path)

    # Print summary
    summary = artifact["run_summary"]
    print(f"\n{'='*60}")
    print(f"Post-hoc Evaluation Complete: {run_dir.name}")
    print(f"{'='*60}")
    print(f"Overall verdict: {summary['overall_verdict']}")
    if summary["crit_summary"]["gamma_mean"] is not None:
        print(f"CRIT rho_bar:    {summary['crit_summary']['gamma_mean']:.3f} "
              f"({'PASS' if summary['crit_summary']['threshold_pass'] else 'FAIL'})")
    if summary["rca_summary"]["trace_consistency_rate"] is not None:
        print(f"Consistency:     {summary['rca_summary']['trace_consistency_rate']:.3f} "
              f"({summary['rca_summary']['verdict']})")
    print(f"Causal ladder:   {summary['t3_summary']['detected_rung']} "
          f"({'PASS' if summary['t3_summary']['pass'] else 'FAIL'})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
