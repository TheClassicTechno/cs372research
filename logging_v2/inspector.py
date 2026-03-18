"""CLI inspector for logging_v2 event logs.

Usage:
    python -m logging_v2.inspector <events.jsonl>
    python -m logging_v2.inspector <events.jsonl> --summary
    python -m logging_v2.inspector <events.jsonl> --interventions
    python -m logging_v2.inspector <events.jsonl> --pre-crit
    python -m logging_v2.inspector <events.jsonl> --diff-snapshots
    python -m logging_v2.inspector <events.jsonl> --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .loader import (
    count_events_by_type,
    filter_events,
    get_decision_boundaries,
    get_errors,
    get_intervention_timeline,
    get_llm_calls,
    get_pre_intervention_crit,
    get_run_metadata,
    get_state_snapshots,
    load_event_log,
    validate_event_ordering,
)


def _format_crit(crit_scores: dict[str, Any]) -> str:
    """Format CRIT scores for display."""
    lines = []
    for role, data in sorted(crit_scores.items()):
        pillars = data.get("pillars", {})
        rho_i = data.get("rho_i", "N/A")
        lv = pillars.get("logical_validity", "N/A")
        es = pillars.get("evidential_support", "N/A")
        ac = pillars.get("alternative_consideration", "N/A")
        ca = pillars.get("causal_alignment", "N/A")
        lines.append(
            f"  {role:>12}: rho_i={rho_i:.3f}  "
            f"LV={lv:.3f}  ES={es:.3f}  AC={ac:.3f}  CA={ca:.3f}"
            if isinstance(rho_i, (int, float)) else
            f"  {role:>12}: {data}"
        )
    return "\n".join(lines)


def _format_allocations(allocations: dict[str, dict[str, float]]) -> str:
    """Format allocations for display."""
    lines = []
    for role, alloc in sorted(allocations.items()):
        weights = "  ".join(f"{t}={w:.2%}" for t, w in sorted(alloc.items()))
        lines.append(f"  {role:>12}: {weights}")
    return "\n".join(lines)


def cmd_summary(events: list[dict[str, Any]]) -> None:
    """Print event log summary."""
    counts = count_events_by_type(events)
    meta = get_run_metadata(events)

    print("=" * 60)
    print("EVENT LOG SUMMARY")
    print("=" * 60)

    if meta:
        print(f"  Experiment:  {meta.get('experiment', 'N/A')}")
        print(f"  Run ID:      {meta.get('run_id', 'N/A')}")
        print(f"  Debate ID:   {meta.get('debate_id', 'N/A')}")
        print(f"  Start Time:  {meta.get('start_time', 'N/A')}")
        print(f"  Git Commit:  {meta.get('git_commit_hash', 'N/A')}")
        roles = meta.get("roles", [])
        universe = meta.get("ticker_universe", [])
        print(f"  Roles:       {', '.join(roles)}")
        print(f"  Universe:    {', '.join(universe)}")
        print(f"  Max Rounds:  {meta.get('max_rounds', 'N/A')}")

    print()
    print("Event counts:")
    for etype, count in sorted(counts.items()):
        print(f"  {etype:>25}: {count}")
    print(f"  {'TOTAL':>25}: {len(events)}")

    # LLM call breakdown
    llm_calls = get_llm_calls(events)
    if llm_calls:
        print()
        print("LLM calls by phase:")
        by_phase: dict[str, int] = {}
        for c in llm_calls:
            p = c.get("phase", "unknown")
            by_phase[p] = by_phase.get(p, 0) + 1
        for phase, count in sorted(by_phase.items()):
            print(f"  {phase:>15}: {count}")

        print()
        print("LLM calls by context:")
        by_ctx: dict[str, int] = {}
        for c in llm_calls:
            ctx = c.get("call_context", "unknown")
            by_ctx[ctx] = by_ctx.get(ctx, 0) + 1
        for ctx, count in sorted(by_ctx.items()):
            print(f"  {ctx:>25}: {count}")

    # Errors
    errors = get_errors(events)
    if errors:
        print()
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  [{err.get('error_type')}] {err.get('message', '')[:100]}")

    print()


def cmd_interventions(events: list[dict[str, Any]]) -> None:
    """Print intervention timeline."""
    timeline = get_intervention_timeline(events)

    print("=" * 60)
    print("INTERVENTION TIMELINE")
    print("=" * 60)

    if not timeline:
        print("  No intervention events found.")
        return

    for evt in timeline:
        clock = evt.get("logical_clock", "?")
        etype = evt.get("event_type", "?")
        print(f"\n  [clock={clock}] {etype}")

        if etype == "decision_boundary":
            bt = evt.get("boundary_type", "?")
            decision = evt.get("decision", "?")
            print(f"    boundary_type: {bt}")
            print(f"    decision: {decision}")
            inputs = evt.get("inputs", {})
            if "rho_bar" in inputs:
                print(f"    rho_bar: {inputs['rho_bar']}")
            if "agent_crit_scores" in inputs:
                print("    agent_crit_scores:")
                for role, scores in inputs["agent_crit_scores"].items():
                    pillars = scores.get("pillars", scores)
                    ca = pillars.get("causal_alignment", pillars.get("CA", "?"))
                    print(f"      {role}: CA={ca}")
            outputs = evt.get("outputs", {})
            if "results" in outputs:
                for r in outputs["results"]:
                    print(f"    → {r.get('rule', '?')}: "
                          f"action={r.get('action', '?')} "
                          f"targets={r.get('target_roles', [])}")

        elif etype == "state_snapshot":
            st = evt.get("snapshot_type", "?")
            print(f"    snapshot_type: {st}")
            if evt.get("crit_scores"):
                print("    crit_scores:")
                print(_format_crit(evt["crit_scores"]))

        elif etype == "llm_call":
            print(f"    phase: {evt.get('phase', '?')}")
            print(f"    role: {evt.get('role', '?')}")
            print(f"    call_context: {evt.get('call_context', '?')}")
            print(f"    call_index: {evt.get('call_index', '?')}")
            print(f"    latency_ms: {evt.get('latency_ms', '?'):.0f}")

    print()


def cmd_pre_crit(events: list[dict[str, Any]]) -> None:
    """Extract and display pre-intervention CRIT scores."""
    print("=" * 60)
    print("PRE-INTERVENTION CRIT SCORES")
    print("=" * 60)

    pre_crit = get_pre_intervention_crit(events)
    if pre_crit is None:
        print("  No pre-intervention CRIT snapshot found.")
        print("  (No intervention was fired in this debate.)")
        return

    print(f"\n  Event ID:      {pre_crit.get('event_id', '?')}")
    print(f"  Logical Clock: {pre_crit.get('logical_clock', '?')}")
    print(f"  Round:         {pre_crit.get('round_num', '?')}")
    print(f"  Snapshot Type: {pre_crit.get('snapshot_type', '?')}")
    print(f"  rho_bar:       {pre_crit.get('rho_bar', 'N/A')}")

    if pre_crit.get("crit_scores"):
        print("\n  CRIT Scores (BEFORE intervention):")
        print(_format_crit(pre_crit["crit_scores"]))

    if pre_crit.get("allocations"):
        print("\n  Allocations (BEFORE intervention):")
        print(_format_allocations(pre_crit["allocations"]))

    # Also show post-intervention for comparison
    post_crit = None
    for evt in events:
        if (evt.get("event_type") == "state_snapshot"
                and evt.get("snapshot_type") == "post_crit_retry"):
            post_crit = evt
            break

    if post_crit:
        print(f"\n  --- POST-INTERVENTION COMPARISON ---")
        print(f"  Logical Clock: {post_crit.get('logical_clock', '?')}")
        if post_crit.get("crit_scores"):
            print("\n  CRIT Scores (AFTER intervention retry):")
            print(_format_crit(post_crit["crit_scores"]))
        if post_crit.get("crit_diff"):
            print("\n  CRIT Diff (post - pre):")
            for role, diff in post_crit["crit_diff"].items():
                parts = []
                if "rho_i_delta" in diff:
                    parts.append(f"rho_i: {diff['rho_i_delta']:+.3f}")
                for pillar, delta in diff.get("pillars", {}).items():
                    parts.append(f"{pillar}: {delta:+.3f}")
                print(f"    {role}: {', '.join(parts)}")

    print()


def cmd_diff_snapshots(events: list[dict[str, Any]]) -> None:
    """Show diffs between consecutive state snapshots."""
    snapshots = get_state_snapshots(events)

    print("=" * 60)
    print("STATE SNAPSHOT DIFFS")
    print("=" * 60)

    if not snapshots:
        print("  No state snapshots found.")
        return

    for snap in snapshots:
        clock = snap.get("logical_clock", "?")
        stype = snap.get("snapshot_type", "?")
        print(f"\n  [clock={clock}] {stype}")

        if snap.get("allocation_diff"):
            print("    Allocation changes:")
            for role, diffs in snap["allocation_diff"].items():
                changes = "  ".join(
                    f"{t}={d:+.4f}" for t, d in sorted(diffs.items())
                )
                print(f"      {role}: {changes}")

        if snap.get("crit_diff"):
            print("    CRIT changes:")
            for role, diff in snap["crit_diff"].items():
                parts = []
                if "rho_i_delta" in diff:
                    parts.append(f"rho_i: {diff['rho_i_delta']:+.3f}")
                for pillar, delta in diff.get("pillars", {}).items():
                    parts.append(f"{pillar}: {delta:+.3f}")
                print(f"      {role}: {', '.join(parts)}")

        if snap.get("rho_bar") is not None:
            print(f"    rho_bar: {snap['rho_bar']:.4f}")
        if snap.get("js_divergence") is not None:
            print(f"    JS divergence: {snap['js_divergence']:.4f}")

    print()


def cmd_validate(events: list[dict[str, Any]]) -> None:
    """Validate event log integrity."""
    print("=" * 60)
    print("EVENT LOG VALIDATION")
    print("=" * 60)

    # Check ordering
    ordering_errors = validate_event_ordering(events)
    if ordering_errors:
        print(f"\n  FAIL: {len(ordering_errors)} ordering errors:")
        for err in ordering_errors[:10]:
            print(f"    {err}")
    else:
        print("  PASS: Logical clock ordering is strictly monotonic")

    # Check run_metadata is first
    if events and events[0].get("event_type") == "run_metadata":
        print("  PASS: run_metadata is the first event")
    elif events:
        print(f"  FAIL: First event is '{events[0].get('event_type')}', "
              f"expected 'run_metadata'")
    else:
        print("  FAIL: Event log is empty")

    # Check schema version consistency
    versions = {e.get("schema_version") for e in events}
    if len(versions) == 1:
        print(f"  PASS: Consistent schema_version: {versions.pop()}")
    else:
        print(f"  WARN: Multiple schema versions: {versions}")

    # Check debate_id consistency
    debate_ids = {e.get("debate_id") for e in events if "debate_id" in e}
    if len(debate_ids) == 1:
        print(f"  PASS: Consistent debate_id: {debate_ids.pop()[:16]}...")
    elif len(debate_ids) > 1:
        print(f"  WARN: Multiple debate_ids: {len(debate_ids)}")

    # Check all LLM calls have hashes
    llm_calls = get_llm_calls(events)
    missing_hashes = [
        c for c in llm_calls
        if not c.get("system_prompt_hash") or not c.get("user_prompt_hash")
    ]
    if missing_hashes:
        print(f"  WARN: {len(missing_hashes)} LLM calls missing prompt hashes")
    else:
        print(f"  PASS: All {len(llm_calls)} LLM calls have prompt hashes")

    # Check for errors
    errors = get_errors(events)
    if errors:
        print(f"  INFO: {len(errors)} error events recorded")
    else:
        print("  PASS: No error events")

    # Intervention consistency
    boundaries = get_decision_boundaries(events)
    eval_count = sum(1 for b in boundaries if b.get("boundary_type") == "intervention_eval")
    fire_count = sum(1 for b in boundaries if b.get("boundary_type") == "intervention_fire")
    skip_count = sum(1 for b in boundaries if b.get("boundary_type") == "intervention_skip")
    print(f"\n  Interventions: {eval_count} evals, {fire_count} fires, {skip_count} skips")

    # Check pre-intervention CRIT capture
    pre_crit = get_pre_intervention_crit(events)
    if fire_count > 0:
        if pre_crit:
            print("  PASS: Pre-intervention CRIT snapshot captured")
        else:
            print("  FAIL: Intervention fired but no pre-intervention CRIT snapshot")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect logging_v2 event logs",
        prog="python -m logging_v2.inspector",
    )
    parser.add_argument("events_file", type=Path, help="Path to events.jsonl")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Print event log summary (default)")
    parser.add_argument("--interventions", action="store_true",
                        help="Show intervention timeline")
    parser.add_argument("--pre-crit", action="store_true",
                        help="Extract pre-intervention CRIT scores")
    parser.add_argument("--diff-snapshots", action="store_true",
                        help="Show diffs between state snapshots")
    parser.add_argument("--validate", action="store_true",
                        help="Validate event log integrity")
    parser.add_argument("--all", action="store_true",
                        help="Run all inspection modes")

    args = parser.parse_args()

    if not args.events_file.exists():
        print(f"Error: File not found: {args.events_file}", file=sys.stderr)
        sys.exit(1)

    events = load_event_log(args.events_file)

    if args.all:
        cmd_summary(events)
        cmd_interventions(events)
        cmd_pre_crit(events)
        cmd_diff_snapshots(events)
        cmd_validate(events)
    elif args.interventions:
        cmd_interventions(events)
    elif args.pre_crit:
        cmd_pre_crit(events)
    elif args.diff_snapshots:
        cmd_diff_snapshots(events)
    elif args.validate:
        cmd_validate(events)
    else:
        cmd_summary(events)


if __name__ == "__main__":
    main()
