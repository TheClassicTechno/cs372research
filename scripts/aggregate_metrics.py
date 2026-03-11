#!/usr/bin/env python3
"""Aggregate debate metrics across all runs, grouped by experiment.

Usage:
    python logging/aggregate_metrics.py                    # write summary JSON
    python logging/aggregate_metrics.py --pretty           # also pretty-print to stdout
"""

import argparse
import json
import statistics
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "logging" / "runs"
OUTPUT_PATH = RUNS_DIR / "ablation_summary.json"

# Pillar abbreviation -> canonical name
PILLAR_MAP = {
    "LV": "logical_validity",
    "IC": "logical_validity",
    "ES": "evidential_support",
    "AC": "alternative_consideration",
    "TA": "alternative_consideration",
    "CA": "causal_alignment",
    "CI": "causal_alignment",
}

CANONICAL_PILLARS = [
    "logical_validity",
    "evidential_support",
    "alternative_consideration",
    "causal_alignment",
]


def _stats(values: list[float], full: bool = False) -> dict:
    """Return mean/stdev (and optionally min/max) for a list of floats."""
    values = [v for v in values if v is not None]
    if not values:
        return {}
    out: dict = {
        "mean": round(statistics.mean(values), 4),
        "stdev": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }
    if full:
        out["min"] = round(min(values), 4)
        out["max"] = round(max(values), 4)
    return out


def _safe_load(path: Path) -> dict | None:
    """Load JSON or return None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _collect_run(run_dir: Path) -> dict | None:
    """Extract per-round metrics from a single run directory."""
    manifest = _safe_load(run_dir / "manifest.json")
    if manifest is None:
        return None

    actual_rounds = manifest.get("actual_rounds") or 0
    if actual_rounds == 0:
        return None

    rounds_data = []
    for r in range(1, actual_rounds + 1):
        rdir = run_dir / "rounds" / f"round_{r:03d}" / "metrics"
        crit = _safe_load(rdir / "crit_scores.json")
        js = _safe_load(rdir / "js_divergence.json")
        eo = _safe_load(rdir / "evidence_overlap.json")
        pid = _safe_load(rdir / "pid_state.json")
        rounds_data.append({
            "round": r,
            "crit": crit,
            "js": js,
            "eo": eo,
            "pid": pid,
        })

    # Derive scenario and agent_config labels
    config_paths = manifest.get("config_paths") or []
    if len(config_paths) >= 2:
        scenario = Path(config_paths[1]).stem
    else:
        scenario = manifest.get("invest_quarter", "unknown")
    if len(config_paths) >= 1:
        agent_config = Path(config_paths[0]).stem
    else:
        agent_config = "default"

    return {
        "manifest": manifest,
        "rounds": rounds_data,
        "actual_rounds": actual_rounds,
        "scenario": scenario,
        "agent_config": agent_config,
    }


def _normalize_pillars(pillar_scores: dict) -> dict:
    """Map abbreviation keys to canonical pillar names."""
    out = {}
    for abbr, val in pillar_scores.items():
        canonical = PILLAR_MAP.get(abbr)
        if canonical is not None:
            out[canonical] = val
    return out


def _group_runs(runs: list[dict], key: str) -> dict[str, list[dict]]:
    """Group runs by a top-level key (e.g. 'scenario', 'agent_config')."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run[key], []).append(run)
    return groups


def aggregate_experiment(runs: list[dict], breakdowns: bool = True) -> dict:
    """Aggregate metrics across all runs in an experiment."""
    if not runs:
        return {}

    model = runs[0]["manifest"].get("model_name", "unknown")

    # -- per-round collectors --
    rho_by_round: dict[int, list[float]] = {}
    rho_by_agent_round: dict[str, dict[int, list[float]]] = {}
    pillar_by_round: dict[str, dict[int, list[float]]] = {}
    pillar_by_agent: dict[str, dict[str, dict[int, list[float]]]] = {}
    js_by_round: dict[int, list[float]] = {}
    eo_by_round: dict[int, list[float]] = {}

    # -- final-round collectors --
    beta_finals: list[float] = []
    quadrant_counts: dict[str, int] = {}
    tone_counts: dict[str, int] = {}
    total_final = 0

    # -- collapse counters (final round) --
    js_final_vals: list[float] = []
    rho_final_vals: list[float] = []

    for run in runs:
        actual = run["actual_rounds"]
        for rd in run["rounds"]:
            r = rd["round"]
            crit = rd["crit"]
            js_data = rd["js"]
            eo_data = rd["eo"]
            pid_data = rd["pid"]

            # rho_bar
            if crit and "rho_bar" in crit:
                rho_by_round.setdefault(r, []).append(crit["rho_bar"])

            # agent scores + pillars
            if crit and "agent_scores" in crit:
                for role, scores in crit["agent_scores"].items():
                    if "rho_i" in scores:
                        rho_by_agent_round.setdefault(role, {}).setdefault(r, []).append(scores["rho_i"])
                    if "pillar_scores" in scores:
                        normed = _normalize_pillars(scores["pillar_scores"])
                        for pname, pval in normed.items():
                            pillar_by_round.setdefault(pname, {}).setdefault(r, []).append(pval)
                            pillar_by_agent.setdefault(role, {}).setdefault(pname, {}).setdefault(r, []).append(pval)

            # JS divergence
            if js_data and "js_divergence" in js_data:
                js_by_round.setdefault(r, []).append(js_data["js_divergence"])

            # evidence overlap
            if eo_data and "mean_overlap" in eo_data:
                eo_by_round.setdefault(r, []).append(eo_data["mean_overlap"])

            # PID (final round only)
            if r == actual and pid_data:
                if "beta_new" in pid_data:
                    beta_finals.append(pid_data["beta_new"])
                q = pid_data.get("quadrant")
                if q:
                    quadrant_counts[q] = quadrant_counts.get(q, 0) + 1
                t = pid_data.get("tone_bucket")
                if t:
                    tone_counts[t] = tone_counts.get(t, 0) + 1
                total_final += 1

            # Collapse tracking (final round)
            if r == actual:
                if js_data and "js_divergence" in js_data:
                    js_final_vals.append(js_data["js_divergence"])
                if crit and "rho_bar" in crit:
                    rho_final_vals.append(crit["rho_bar"])

    # -- build rho section --
    all_rounds_sorted = sorted(rho_by_round.keys())
    final_round_num = max(all_rounds_sorted) if all_rounds_sorted else None

    rho_section = {
        "per_round": {str(r): _stats(rho_by_round[r], full=True) for r in all_rounds_sorted},
        "final_round": _stats(rho_by_round.get(final_round_num, []), full=True) if final_round_num else {},
        "all_rounds": _stats([v for r in all_rounds_sorted for v in rho_by_round[r]]),
    }

    # -- rho per agent --
    rho_per_agent = {}
    for role, by_round in rho_by_agent_round.items():
        rsorted = sorted(by_round.keys())
        fr = max(rsorted) if rsorted else None
        rho_per_agent[role] = {
            "per_round": {str(r): _stats(by_round[r]) for r in rsorted},
            "final_round": _stats(by_round.get(fr, [])) if fr else {},
            "all_rounds": _stats([v for r in rsorted for v in by_round[r]]),
        }

    # -- pillars --
    pillars_section = {}
    for pname in CANONICAL_PILLARS:
        by_round = pillar_by_round.get(pname, {})
        rsorted = sorted(by_round.keys())
        fr = max(rsorted) if rsorted else None
        pillars_section[pname] = {
            "per_round": {str(r): _stats(by_round[r]) for r in rsorted},
            "final_round": _stats(by_round.get(fr, [])) if fr else {},
            "all_rounds": _stats([v for r in rsorted for v in by_round[r]]),
        }

    # -- pillars per agent --
    pillars_per_agent = {}
    for role, pillar_dict in pillar_by_agent.items():
        pillars_per_agent[role] = {}
        for pname in CANONICAL_PILLARS:
            by_round = pillar_dict.get(pname, {})
            rsorted = sorted(by_round.keys())
            fr = max(rsorted) if rsorted else None
            pillars_per_agent[role][pname] = {
                "final_round": _stats(by_round.get(fr, [])) if fr else {},
            }

    # -- JS divergence --
    js_rounds_sorted = sorted(js_by_round.keys())
    js_fr = max(js_rounds_sorted) if js_rounds_sorted else None

    # trajectory: mean change from first to last round per run
    deltas = []
    for run in runs:
        js_vals = []
        for rd in run["rounds"]:
            if rd["js"] and "js_divergence" in rd["js"]:
                js_vals.append(rd["js"]["js_divergence"])
        if len(js_vals) >= 2:
            deltas.append(js_vals[-1] - js_vals[0])

    js_section = {
        "per_round": {str(r): _stats(js_by_round[r]) for r in js_rounds_sorted},
        "final_round": _stats(js_by_round.get(js_fr, []), full=True) if js_fr else {},
        "trajectory": {
            "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
            "pct_decreased": round(100.0 * sum(1 for d in deltas if d < 0) / len(deltas), 1) if deltas else None,
        },
    }

    # -- evidence overlap --
    eo_rounds_sorted = sorted(eo_by_round.keys())
    eo_fr = max(eo_rounds_sorted) if eo_rounds_sorted else None
    eo_section = {
        "per_round": {str(r): _stats(eo_by_round[r]) for r in eo_rounds_sorted},
        "final_round": _stats(eo_by_round.get(eo_fr, []), full=True) if eo_fr else {},
    }

    # -- PID --
    quad_dist = {k: round(v / total_final, 4) for k, v in quadrant_counts.items()} if total_final else {}
    tone_dist = {k: round(v / total_final, 4) for k, v in tone_counts.items()} if total_final else {}
    pid_section = {
        "beta_final": _stats(beta_finals, full=True),
        "quadrant_distribution": quad_dist,
        "tone_distribution": tone_dist,
    }

    # -- Collapse --
    n_final = len(js_final_vals)
    js_lt_005 = sum(1 for v in js_final_vals if v < 0.05)
    js_lt_007 = sum(1 for v in js_final_vals if v < 0.07)
    js_lt_010 = sum(1 for v in js_final_vals if v < 0.10)
    # high_rho_low_js: rho >= 0.8 AND js < 0.10 in final round
    paired = list(zip(rho_final_vals, js_final_vals))
    high_rho_low_js = sum(1 for rho, js in paired if rho >= 0.8 and js < 0.10)

    collapse_section = {
        "js_lt_005": js_lt_005,
        "js_lt_007": js_lt_007,
        "js_lt_010": js_lt_010,
        "high_rho_low_js": high_rho_low_js,
        "pct_high_rho_low_js": round(100.0 * high_rho_low_js / n_final, 1) if n_final else 0,
    }

    result = {
        "run_count": len(runs),
        "model": model,
        "rho": rho_section,
        "rho_per_agent": rho_per_agent,
        "pillars": pillars_section,
        "pillars_per_agent": pillars_per_agent,
        "js_divergence": js_section,
        "evidence_overlap": eo_section,
        "pid": pid_section,
        "collapse": collapse_section,
    }

    if breakdowns:
        result["per_scenario"] = {
            name: aggregate_experiment(group, breakdowns=False)
            for name, group in sorted(_group_runs(runs, "scenario").items())
        }
        result["per_agent_config"] = {
            name: aggregate_experiment(group, breakdowns=False)
            for name, group in sorted(_group_runs(runs, "agent_config").items())
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Aggregate debate metrics across experiments")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print summary to stdout")
    args = parser.parse_args()

    experiments: dict[str, list[dict]] = {}

    for exp_dir in sorted(RUNS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        # Skip the test fixture directory
        if exp_name == "test":
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            run_data = _collect_run(run_dir)
            if run_data:
                experiments.setdefault(exp_name, []).append(run_data)

    summary = {
        exp_name: aggregate_experiment(runs)
        for exp_name, runs in experiments.items()
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({len(summary)} experiments)")

    if args.pretty:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
