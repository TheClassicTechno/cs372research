#!/usr/bin/env python3
"""
Intervention firing and revision-limit compliance for ablations 7, 8, 10
=======================================================================
camera_ready_plan.md items 3.1 and 3.1c. Reads the shared run package
(``CS372_210_debate_runs/logging/runs``), not the repo's empty ``logging/runs``.

For each ablation x condition: how many runs logged at least one intervention
(``rounds/*/interventions/*.json``).

For every agent that received an intervention nudge: how many tickers its
stored revision changed relative to its original proposal, and the largest
single-ticker move. ``round_state.json`` stores the post-retry revision; the
comparison with non-retried agents (median tickers changed) is the check.
The revision limit (``max_changed_tickers``, ``max_change_pct``) is a prompt
instruction appended by ``multi_agent/runner.py:_enrich_nudge_with_proposal``;
nothing in the code enforces it.

Usage:
    python3 analysis/intervention_audit.py
"""

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "CS372_210_debate_runs" / "logging" / "runs"
OUT_PATH = REPO_ROOT / "analysis" / "rq1" / "intervention_audit.json"

CHANGE_EPS = 0.005  # weight moves below half a point count as unchanged
MAX_TICKERS, MAX_MOVE = 2, 0.05  # revision_limits in all A7/A8/A10 configs


def _moves(state, role):
    p = state["proposals"][role]["allocation"]
    q = state["revisions"][role]["allocation"]
    d = [abs(q.get(k, 0.0) - p.get(k, 0.0)) for k in set(p) | set(q)]
    return sum(x > CHANGE_EPS for x in d), max(d)


def main():
    fired, total = defaultdict(int), defaultdict(int)
    retried, untouched = defaultdict(list), defaultdict(list)
    for run in sorted(RUNS.glob("vskarich_ablation_*/run_*")):
        abl = run.parent.name.rsplit("_", 1)[1]
        cond = "baseline" if (run / "final" / "baseline_no_intervention.yaml").exists() else "treatment"
        total[f"{abl}:{cond}"] += 1
        if glob.glob(str(run / "rounds" / "*" / "interventions" / "*.json")):
            fired[f"{abl}:{cond}"] += 1
        for rd in sorted(run.glob("rounds/round_*")):
            state = json.loads((rd / "round_state.json").read_text())
            events = [json.loads(f.read_text()) for f in sorted(rd.glob("interventions/*.json"))]
            nudged = set()
            for e in events:
                nt = e.get("nudge_text")
                nudged |= set(nt) if isinstance(nt, dict) else set(state["proposals"])
            for role in state["proposals"]:
                (retried if role in nudged else untouched)[f"{abl}:{cond}"].append(_moves(state, role))

    out = {"firing": {}, "compliance": {}}
    for k in sorted(total):
        out["firing"][k] = {"runs_with_intervention": fired[k], "runs": total[k]}
        print(f"A{k:<14} fired in {fired[k]}/{total[k]} runs")
    for k in sorted(set(retried) | set(untouched)):
        r, u = retried.get(k, []), untouched.get(k, [])
        rec = {
            "retried_agent_rounds": len(r),
            "retried_within_limit": sum(n <= MAX_TICKERS and m <= MAX_MOVE + 1e-9 for n, m in r),
            "retried_median_tickers_changed": statistics.median(n for n, _ in r) if r else None,
            "untouched_agent_rounds": len(u),
            "untouched_median_tickers_changed": statistics.median(n for n, _ in u) if u else None,
        }
        out["compliance"][k] = rec
        print(f"A{k:<14} {rec}")
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
