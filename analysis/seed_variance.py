#!/usr/bin/env python3
"""
Run-to-Run Variance Probe
=========================
Quantifies how much of the between-condition difference reported in Table 2
could be produced by LLM sampling noise alone.

Design
------
  Independent variable : repeated sampling only. Configuration
                         (single_agent_slim = Table 2 column A), scenario, and
                         evaluation window are all held fixed; the only thing
                         that differs between repeats is the sampling draw at
                         temperature 0.3.
  Dependent variables  : total return %, annualized Sharpe.
  Unit of replication  : one (scenario, repeat) run.

Because the configuration is Table 2's column A, the within-scenario spread
measured here is directly comparable to that table's between-column effects.

Interpretation is symmetric and stated in advance:
  - If within-scenario SD is comparable to or larger than the between-condition
    differences, those marginal differences are not separable from sampling
    noise, and the corresponding claims must be weakened.
  - If within-scenario SD is small relative to them, the paired-scenario design
    is adequately powered against run-to-run noise.

No new claim is attached to the point estimates themselves; this measures a
noise floor, not a treatment effect.

Usage
-----
    python3 analysis/seed_variance.py
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results" / "seed_variance"
REPORT = REPO / "analysis" / "seed_variance_report.txt"

# Between-condition effects from Table 2, measured against column A
# (single agent, basic) over the same 35 paired scenarios. Produced by
# analysis/verify_paper_tables.py; reproduced here only for comparison.
TABLE2_EFFECTS = {
    "total_return_pct": [
        ("B  3 Roles Mean, Basic",    -0.12),
        ("C  3 Roles Mean, Causal",   +0.34),
        ("D  3 Roles Mean, Enriched", -0.21),
    ],
    "annualized_sharpe": [
        ("B  3 Roles Mean, Basic",    -0.01),
        ("C  3 Roles Mean, Causal",   +0.07),
        ("D  3 Roles Mean, Enriched", +0.04),
    ],
}

METRICS = [
    ("total_return_pct", "Total return %", 2),
    ("annualized_sharpe", "Annualized Sharpe", 3),
]


def load() -> dict[str, dict[str, list[float]]]:
    """Return {scenario: {metric: [value per repeat]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    if not RUNS.is_dir():
        raise SystemExit(f"no runs found at {RUNS}; run analysis/seed_variance_runs.sh first")
    for summary in sorted(RUNS.rglob("summary.json")):
        # results/seed_variance/<scenario>__rep<N>/<config>/summary.json
        stamp = summary.parent.parent.name
        m = re.match(r"(.+?)__rep(\d+)$", stamp)
        if not m:
            continue
        scenario = m.group(1)
        data = json.loads(summary.read_text())
        eps = data.get("episode_summaries") or []
        if not eps:
            # Fail loudly rather than silently dropping a run from the variance
            # estimate, which would bias it downward.
            raise RuntimeError(f"{summary} has no episode_summaries")
        daily = eps[0].get("daily_metrics") or {}
        bucket = out.setdefault(scenario, {k: [] for k, _, _ in METRICS})
        for key, _label, _p in METRICS:
            if key in daily and daily[key] is not None:
                bucket[key].append(float(daily[key]))
    return out


def main() -> int:
    data = load()
    lines: list[str] = [
        "RUN-TO-RUN VARIANCE PROBE",
        "=" * 74,
        "Config held fixed : single_agent_slim (Table 2, column A)",
        "Varying           : LLM sampling draw only (temperature 0.3)",
        f"Scenarios         : {len(data)}",
        "",
    ]

    pooled: dict[str, list[float]] = {k: [] for k, _, _ in METRICS}

    for key, label, prec in METRICS:
        lines.append(f"--- {label} ---")
        lines.append(f"  {'scenario':<40} {'n':>2} {'mean':>9} {'sd':>8} {'range':>9}")
        for scenario in sorted(data):
            vals = data[scenario][key]
            if len(vals) < 2:
                lines.append(f"  {scenario:<40} {len(vals):>2}   (insufficient repeats)")
                continue
            sd = st.stdev(vals)
            pooled[key].append(sd)
            lines.append(
                f"  {scenario:<40} {len(vals):>2} {st.mean(vals):>9.{prec}f} "
                f"{sd:>8.{prec}f} {max(vals) - min(vals):>9.{prec}f}"
            )
        if pooled[key]:
            # Pool variances rather than averaging SDs.
            sd_run = math.sqrt(st.mean([s * s for s in pooled[key]]))
            # A paired difference between two single runs at the same scenario
            # has SD ~ sqrt(2) * sd_run; Table 2 averages that over 35 paired
            # scenarios, so the SEM of its mean difference is that over sqrt(35).
            sd_paired = math.sqrt(2.0) * sd_run
            sem_35 = sd_paired / math.sqrt(35.0)
            lines.append("")
            lines.append(f"  pooled within-scenario SD, one run   : {sd_run:.{prec}f}")
            lines.append(f"  implied SD of a 1-vs-1 paired diff    : {sd_paired:.{prec}f}")
            lines.append(f"  implied SEM of that diff over n=35    : {sem_35:.{prec}f}")
            lines.append("")
            lines.append("  Table 2 between-condition effects vs column A:")
            for name, eff in TABLE2_EFFECTS[key]:
                lines.append(
                    f"    {name:<30} {eff:+.{prec}f}   "
                    f"|effect| vs single-run SD = {abs(eff)/sd_run:.2f}x   "
                    f"vs n=35 SEM = {abs(eff)/sem_35:.2f}x")
            lines.append("")
            lines.append("  Reading: the first ratio says whether a SINGLE run at ONE scenario")
            lines.append("  could show the effect by chance; the second says whether the paired")
            lines.append("  35-scenario design can resolve it. Only the second bears on Table 2.")
        lines.append("")

    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
