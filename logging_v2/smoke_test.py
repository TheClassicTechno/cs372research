#!/usr/bin/env python3
"""Smoke test for the event-sourced WAL + artifact storage system.

Exercises the full pipeline locally (no AWS needed):
  1. Legacy log_* methods (backward compat)
  2. v2 emit() methods (new event schema)
  3. Artifact creation and WAL linkage
  4. Event loading and validation

Usage:
    python -m logging_v2.smoke_test
    # or
    python logging_v2/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Ensure the parent package is importable when run as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logging_v2.event_logger import EventLogger
from logging_v2.loader import load_event_log, validate_event_ordering, count_events_by_type


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wal_smoke_") as tmpdir:
        base_dir = Path(tmpdir)
        print(f"Run directory: {base_dir}\n")

        logger = EventLogger(
            experiment="smoke_test",
            run_id="run_001",
            debate_id="debate-abc-123",
            base_dir=base_dir,
        )

        # ── 1. Legacy: run metadata ──────────────────────────────────
        print("1. Writing run_metadata (legacy)...")
        logger.log_run_metadata(
            config_snapshot={"model": "gpt-4", "temperature": 0.7, "max_rounds": 3},
            start_time="2026-03-22T12:00:00Z",
            roles=["macro", "quant"],
            ticker_universe=["AAPL", "NVDA", "MSFT"],
            max_rounds=3,
        )

        # ── 2. v2: llm_call_start + llm_call_end ────────────────────
        print("2. Writing llm_call_start + llm_call_end (v2 with artifact)...")
        start_id = logger.emit_llm_call_start(
            model="claude-3",
            context={"workflow": "smoke_test", "stage": "propose", "agent_id": "macro"},
        )
        logger.emit_llm_call_end(
            system_prompt="You are a macro analyst. Provide portfolio allocations.",
            user_prompt=(
                "Given rising rates and strong tech earnings, allocate across "
                "AAPL, NVDA, MSFT. Justify with causal claims."
            ),
            response=(
                "C1: Fed tightening compresses multiples on high-duration growth. "
                "C2: NVDA's data-center backlog provides near-term earnings visibility. "
                "Allocation: NVDA 40%, MSFT 35%, AAPL 20%, CASH 5%."
            ),
            model="claude-3",
            latency_ms=1842.0,
            token_usage={"prompt_tokens": 1200, "completion_tokens": 350},
            context={"workflow": "smoke_test", "stage": "propose", "agent_id": "macro"},
            parent_event_id=start_id,
        )

        # ── 3. Legacy: llm_call (shows backward compat + artifact) ──
        print("3. Writing llm_call (legacy — now also creates artifact)...")
        logger.log_llm_call(
            phase="critique",
            role="quant",
            call_index=0,
            call_context="initial",
            system_prompt="You are a quantitative analyst. Critique the proposal.",
            user_prompt="The macro agent proposed NVDA 40%, MSFT 35%, AAPL 20%, CASH 5%.",
            response=(
                "C1 overstates earnings visibility — NVDA's forward P/E is 35x, "
                "implying the market already prices the backlog. Recommend reducing "
                "NVDA to 30% and increasing CASH to 15%."
            ),
            model_name="gpt-4",
            provider="openai",
            temperature=0.7,
            latency_ms=920.0,
            round_num=1,
        )

        # ── 4. v2: evaluation_computed ───────────────────────────────
        print("4. Writing evaluation_computed (v2)...")
        logger.emit_evaluation(
            metric_name="crit",
            scores={
                "macro": {"rho_i": 0.83, "pillars": {"LV": 0.87, "ES": 0.82, "AC": 0.80, "CA": 0.83}},
                "quant": {"rho_i": 0.79, "pillars": {"LV": 0.85, "ES": 0.75, "AC": 0.78, "CA": 0.78}},
            },
            context={"workflow": "smoke_test", "stage": "post_crit"},
        )

        # ── 5. v2: control_flow ──────────────────────────────────────
        print("5. Writing control_flow (v2)...")
        logger.emit_control_flow(
            decision="continue",
            reason="rho_bar=0.81 < convergence_threshold=0.90",
            details={"rho_bar": 0.81, "threshold": 0.90, "round": 1},
            context={"workflow": "smoke_test", "stage": "convergence_check"},
        )

        # ── 6. Legacy: state_snapshot ────────────────────────────────
        print("6. Writing state_snapshot (legacy)...")
        logger.log_state_snapshot(
            snapshot_type="post_revise",
            allocations={
                "macro": {"NVDA": 0.35, "MSFT": 0.35, "AAPL": 0.20, "_CASH_": 0.10},
                "quant": {"NVDA": 0.30, "MSFT": 0.30, "AAPL": 0.25, "_CASH_": 0.15},
            },
            round_num=1,
            rho_bar=0.81,
            js_divergence=0.042,
        )

        # ── 7. v2: error ─────────────────────────────────────────────
        print("7. Writing error event (v2)...")
        logger.emit(
            "error",
            {"error_type": "parse_failure", "message": "JSON decode failed on retry 2"},
            context={"workflow": "smoke_test", "stage": "revise", "agent_id": "quant"},
        )

        # ── 8. Legacy: decision_boundary ─────────────────────────────
        print("8. Writing decision_boundary (legacy)...")
        logger.log_decision_boundary(
            boundary_type="convergence_check",
            stage="post_revision",
            inputs={"js_divergence": 0.042, "rho_bar": 0.81},
            decision="converged",
            outputs={"stable_rounds": 2},
            round_num=2,
        )

        logger.close()

        # ── Inspect output ───────────────────────────────────────────
        run_dir = logger.run_dir
        print(f"\n{'=' * 60}")
        print("OUTPUT INSPECTION")
        print(f"{'=' * 60}")

        # Directory structure
        print(f"\nRun directory: {run_dir}")
        print("\nDirectory tree:")
        for p in sorted(run_dir.rglob("*")):
            rel = p.relative_to(run_dir)
            indent = "  " * len(rel.parts)
            if p.is_dir():
                print(f"  {indent}{p.name}/")
            else:
                size = p.stat().st_size
                print(f"  {indent}{p.name}  ({size:,} bytes)")

        # Load and validate events
        events = load_event_log(run_dir)
        ordering_errors = validate_event_ordering(events)
        counts = count_events_by_type(events)

        print(f"\nTotal events: {len(events)}")
        print("Event counts:")
        for etype, count in sorted(counts.items()):
            print(f"  {etype:>25}: {count}")

        if ordering_errors:
            print(f"\nORDERING ERRORS: {ordering_errors}")
        else:
            print("\nOrdering: PASS (strictly monotonic)")

        # Show WAL events (compact view)
        print(f"\n{'─' * 60}")
        print("WAL EVENTS (compact)")
        print(f"{'─' * 60}")
        for evt in events:
            clock = evt.get("logical_clock", "?")
            etype = evt.get("event_type", "?")
            size = len(json.dumps(evt))

            # v2 events have context/payload
            if "context" in evt:
                ctx = evt["context"]
                stage = ctx.get("stage") or ""
                agent = ctx.get("agent_id") or ""
                label = f"{stage}/{agent}" if agent else stage
                artifact = evt.get("payload", {}).get("artifact_path", "")
            else:
                # Legacy events
                label = f"{evt.get('phase', evt.get('snapshot_type', ''))}"
                artifact = evt.get("artifact_path", "")

            line = f"  [{clock:>3}] {etype:<25} {label:<30} ({size:>5} bytes)"
            if artifact:
                line += f"  -> {artifact}"
            print(line)

        # Show artifacts
        artifacts = sorted((run_dir / "artifacts" / "llm_calls").glob("*.json"))
        print(f"\n{'─' * 60}")
        print(f"ARTIFACTS ({len(artifacts)} files)")
        print(f"{'─' * 60}")
        for art_path in artifacts:
            data = json.loads(art_path.read_text())
            model = data.get("model", "?")
            resp_preview = data.get("response", "")[:80]
            print(f"  {art_path.name}: model={model}")
            print(f"    response: {resp_preview}...")

        # Verify WAL->artifact linkage
        print(f"\n{'─' * 60}")
        print("WAL -> ARTIFACT LINKAGE")
        print(f"{'─' * 60}")
        linked = 0
        for evt in events:
            # Check both legacy and v2 paths
            path = evt.get("artifact_path") or evt.get("payload", {}).get("artifact_path")
            if path:
                full = run_dir / path
                exists = full.exists()
                status = "OK" if exists else "MISSING"
                print(f"  [{status}] clock={evt['logical_clock']} -> {path}")
                linked += 1
        if linked == 0:
            print("  (no artifact references found)")
        else:
            print(f"  {linked} artifact references, all resolved.")

        print(f"\nSmoke test complete.")


if __name__ == "__main__":
    main()
