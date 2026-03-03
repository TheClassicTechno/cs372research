"""Output writing, validation, and suite summary for ablation runs.

Produces:
  - summary.csv (one row per run, all metrics)
  - aggregated.csv (one row per config when replicates > 1, with _mean/_std)
  - config.json (full sweep definition + CLI args)
  - suite_summary.txt (printed and saved statistics)
  - errors.log (any failed runs)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from scripts.ablation.metrics import aggregate_replicates

logger = logging.getLogger(__name__)


def validate_run(result: dict) -> list[str]:
    """Validate a single run result. Returns list of warning messages."""
    warnings: list[str] = []
    run_id = result.get("run_id", "?")

    if result.get("status") != "completed":
        return warnings  # Non-completed runs skip metric validation

    # Quadrant percentages should sum to ~1.0
    quad_sum = (
        (result.get("quadrant_stuck_pct") or 0)
        + (result.get("quadrant_chaotic_pct") or 0)
        + (result.get("quadrant_converged_pct") or 0)
        + (result.get("quadrant_healthy_pct") or 0)
    )
    if abs(quad_sum - 1.0) > 0.05:
        warnings.append(
            f"[{run_id}] Quadrant pcts sum to {quad_sum:.3f} (expected ~1.0)"
        )

    # Check for NaN in required stability columns
    required_numeric = [
        "final_rho_bar", "mean_rho_bar", "final_beta", "beta_range",
        "steady_state_error", "beta_overshoot", "rho_variance", "mean_JS",
    ]
    for col in required_numeric:
        val = result.get(col)
        if val is not None and isinstance(val, float) and (val != val):  # NaN check
            warnings.append(f"[{run_id}] NaN in required column: {col}")

    return warnings


def write_results(
    results: list[dict],
    output_dir: Path,
    cli_args: dict | None = None,
) -> None:
    """Write all output files to the output directory.

    Args:
        results: List of per-run result dicts.
        output_dir: Directory to write output files.
        cli_args: CLI arguments dict for config.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- summary.csv (all individual runs) ----
    _write_csv(results, output_dir / "summary.csv")

    # ---- aggregated.csv (grouped by config when replicates > 1) ----
    _write_aggregated(results, output_dir / "aggregated.csv")

    # ---- config.json ----
    config_data = {
        "cli_args": cli_args or {},
        "num_runs": len(results),
        "num_completed": sum(1 for r in results if r.get("status") == "completed"),
        "num_skipped": sum(1 for r in results if "skipped" in str(r.get("status", ""))),
        "num_failed": sum(1 for r in results if "error" in str(r.get("status", ""))),
    }
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2, default=str))

    # ---- Validate + errors.log ----
    all_warnings: list[str] = []
    for r in results:
        all_warnings.extend(validate_run(r))

    errors_path = output_dir / "errors.log"
    error_lines: list[str] = []
    for r in results:
        status = r.get("status", "")
        if status != "completed" and "skipped" not in status:
            error_lines.append(f"[{r.get('run_id', '?')}] {status}")
    error_lines.extend(all_warnings)
    errors_path.write_text("\n".join(error_lines) if error_lines else "No errors.\n")

    # ---- suite_summary.txt ----
    summary_text = build_suite_summary(results)
    (output_dir / "suite_summary.txt").write_text(summary_text)

    logger.info("Results written to %s", output_dir)


def _write_csv(results: list[dict], path: Path) -> None:
    """Write results to CSV, dynamically determining columns from data."""
    if not results:
        path.write_text("")
        return

    # Collect all keys across all results for the header
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _write_aggregated(results: list[dict], path: Path) -> None:
    """Write replicate-aggregated results to CSV."""
    # Group by (group, param, value, scenario) — strip replicate
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r.get("status") != "completed":
            continue
        key = f"{r.get('group')}|{r.get('param')}|{r.get('value')}|{r.get('scenario')}"
        groups.setdefault(key, []).append(r)

    agg_results = []
    for _, group_results in sorted(groups.items()):
        agg = aggregate_replicates(group_results)
        if agg:
            agg_results.append(agg)

    _write_csv(agg_results, path)


def build_suite_summary(results: list[dict]) -> str:
    """Build the suite-level summary string."""
    total = len(results)
    completed = [r for r in results if r.get("status") == "completed"]
    skipped = [r for r in results if "skipped" in str(r.get("status", ""))]
    failed = [r for r in results
              if r.get("status") not in ("completed",) and "skipped" not in str(r.get("status", ""))]

    n_completed = len(completed)
    n_skipped = len(skipped)
    n_failed = len(failed)

    lines = [
        "=" * 60,
        "  ABLATION SUITE SUMMARY",
        "=" * 60,
        f"  Total runs:     {total}",
        f"    Completed:    {n_completed}",
        f"    Skipped:      {n_skipped} (unstable)",
        f"    Failed:       {n_failed}",
        "",
    ]

    if completed:
        # Oscillation stats
        beta_osc = sum(1 for r in completed if r.get("beta_oscillation_flag"))
        rho_osc = sum(1 for r in completed if r.get("rho_oscillation_flag"))
        lines.append(f"  β oscillatory:  {beta_osc} ({_pct(beta_osc, n_completed)})")
        lines.append(f"  ρ oscillatory:  {rho_osc} ({_pct(rho_osc, n_completed)})")

        # Stability labels
        ctrl_stable = sum(1 for r in completed if r.get("control_stable"))
        behav_stable = sum(1 for r in completed if r.get("behavioral_stable"))
        lines.append(f"  Control stable: {ctrl_stable} ({_pct(ctrl_stable, n_completed)})")
        lines.append(f"  Behavioral stable: {behav_stable} ({_pct(behav_stable, n_completed)})")

        # Convergence
        conv_window = sum(1 for r in completed if r.get("convergence_window_met"))
        lines.append(f"  Converged (window): {conv_window} ({_pct(conv_window, n_completed)})")

        # Mean metrics
        sse_vals = [r["steady_state_error"] for r in completed
                    if r.get("steady_state_error") is not None]
        overshoot_vals = [r["beta_overshoot"] for r in completed
                         if r.get("beta_overshoot") is not None]
        kappa_vals = [r["empirical_kappa"] for r in completed
                      if r.get("empirical_kappa") is not None]
        paranoia_vals = [r["paranoia_rate"] for r in completed
                         if r.get("paranoia_rate") is not None]
        net_vals = [r["net_effect"] for r in completed
                    if r.get("net_effect") is not None]

        lines.append("")
        if sse_vals:
            lines.append(f"  Mean steady_state_error: {sum(sse_vals)/len(sse_vals):.4f}")
        if overshoot_vals:
            lines.append(f"  Mean beta_overshoot: {sum(overshoot_vals)/len(overshoot_vals):.4f}")
        if kappa_vals:
            lines.append(f"  Mean empirical_kappa: {sum(kappa_vals)/len(kappa_vals):.4f}")
        if paranoia_vals:
            lines.append(f"  Mean paranoia_rate: {sum(paranoia_vals)/len(paranoia_vals):.4f}")
        if net_vals:
            lines.append(f"  Mean net_effect: {sum(net_vals)/len(net_vals):.4f}")

        # Stochastic regime breakdown
        stoch_runs = [r for r in completed if r.get("stochastic_regime")]
        determ_runs = [r for r in completed if not r.get("stochastic_regime")]
        if stoch_runs and determ_runs:
            lines.append("")
            lines.append("  --- By Temperature Regime ---")
            s_stable = sum(1 for r in stoch_runs if r.get("behavioral_stable"))
            d_stable = sum(1 for r in determ_runs if r.get("behavioral_stable"))
            lines.append(f"  Stochastic (T≥0.7): {len(stoch_runs)} runs, "
                         f"{_pct(s_stable, len(stoch_runs))} behavioral stable")
            lines.append(f"  Deterministic (T<0.7): {len(determ_runs)} runs, "
                         f"{_pct(d_stable, len(determ_runs))} behavioral stable")

        # Rho_star reachability
        rho_star_reached = sum(
            1 for r in completed
            if r.get("final_rho_bar") is not None
            and r.get("rho_star") is not None
            and r["final_rho_bar"] >= r["rho_star"]
        )
        lines.append("")
        lines.append(f"  ρ* reached in ≥1 run: {rho_star_reached}/{n_completed}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _pct(n: int, total: int) -> str:
    """Format as percentage string."""
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"
