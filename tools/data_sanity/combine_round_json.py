"""Combine all JSON files in each round directory into a single list.

Accepts either:
  - A single round directory   (e.g. .../round_001)
  - An experiment directory     (e.g. .../vskarich_ablation_6)

When given an experiment directory, iterates every run_*/rounds/round_*
and produces an all_round_data.json in each, plus a raw_run_fingerprint.json
at the experiment level with organized per-round data keyed by run name.
"""

import json
import sys
from pathlib import Path


def combine_round_json(round_path: Path) -> list[dict]:
    """Combine JSON files in a round dir. Returns the collected objects."""
    all_data = []
    for f in sorted(round_path.glob("*.json")):
        if f.name == "all_round_data.json":
            continue
        with open(f, encoding="utf-8") as fh:
            obj = json.load(fh)
        obj["_source_file"] = f.name
        all_data.append(obj)

    # Include intervention files if the subdirectory exists
    interventions_dir = round_path / "interventions"
    if interventions_dir.is_dir():
        for f in sorted(interventions_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                obj = json.load(fh)
            obj["_source_file"] = f"interventions/{f.name}"
            all_data.append(obj)

    if not all_data:
        return all_data

    out_path = round_path / "all_round_data.json"
    out_path.write_text(json.dumps(all_data, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {len(all_data)} objects → {out_path}")
    return all_data


def organize_round(raw: list[dict]) -> dict:
    """Transform a flat list of round JSON objects into a structured dict."""
    organized = {
        "round": None,
        "agents": [],
        "tickers": [],
        "crit": {},
        "phases": {},
        "allocations": {
            "proposals": {},
            "revisions": {},
        },
        "metrics": {},
        "interventions": [],
    }

    for item in raw:
        # CRIT scores
        if item.get("phase") == "critique":
            agent = item["agent"]
            organized["round"] = item["round"]
            organized["crit"][agent] = {
                "rho": item["rho"],
                "pillars": item["pillars"],
                "reasoning": item["reasoning"],
            }

        # Phase telemetry
        elif "phase" in item and item["phase"] != "critique":
            phase = item["phase"]
            organized["round"] = item["round"]
            organized["agents"] = item.get("agents", organized["agents"])
            organized["tickers"] = item.get("tickers", organized["tickers"])
            organized["phases"][phase] = {
                "timestamp": item.get("timestamp"),
                "allocations": item.get("allocations"),
                "vectors": item.get("vectors"),
                "allocation_sums": item.get("allocation_sums"),
                "nonzero_positions": item.get("nonzero_positions"),
                "vector_norms": item.get("vector_norms"),
                "js_divergence": item.get("js_divergence"),
                "evidence_overlap": item.get("evidence_overlap"),
            }

        # Round state summary
        if "proposals" in item and "revisions" in item:
            organized["metrics"] = item.get("metrics", {})
            organized["allocations"]["proposals"] = {
                agent: v["allocation"]
                for agent, v in item["proposals"].items()
            }
            organized["allocations"]["revisions"] = {
                agent: v["allocation"]
                for agent, v in item["revisions"].items()
            }

        # Interventions
        if item.get("type") == "intervention":
            organized["interventions"].append({
                "stage": item["stage"],
                "rule": item["rule"],
                "action": item["action"],
                "severity": item["severity"],
                "retry": item["retry"],
                "metrics": item["metrics"],
                "nudge_text": item["nudge_text"],
            })

    # Derive metrics from phase telemetry if round_state.json was absent
    if not organized["metrics"] and organized["phases"]:
        # Final phase: highest retry if present, otherwise revision
        retry_keys = sorted(k for k in organized["phases"] if k.startswith("retry_"))
        final_key = retry_keys[-1] if retry_keys else "revision"
        final_phase = organized["phases"].get(final_key, {})
        if final_phase.get("js_divergence") is not None:
            organized["metrics"] = {
                "js_divergence": final_phase["js_divergence"],
                "evidence_overlap": final_phase.get("evidence_overlap", 0.0),
            }

    # Derive allocations from phase telemetry if round_state.json was absent
    if not organized["allocations"]["proposals"] and "propose" in organized["phases"]:
        propose_allocs = organized["phases"]["propose"].get("allocations")
        if propose_allocs:
            organized["allocations"]["proposals"] = propose_allocs
    if not organized["allocations"]["revisions"] and organized["phases"]:
        retry_keys = sorted(k for k in organized["phases"] if k.startswith("retry_"))
        final_key = retry_keys[-1] if retry_keys else "revision"
        final_allocs = organized["phases"].get(final_key, {}).get("allocations")
        if final_allocs:
            organized["allocations"]["revisions"] = final_allocs

    return organized


def combine_experiment(experiment_path: Path) -> None:
    round_dirs = sorted(experiment_path.glob("run_*/rounds/round_*"))
    if not round_dirs:
        print(f"No round directories found under {experiment_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(round_dirs)} round(s) across {experiment_path.name}")

    # Collect organized round data per run
    run_data: dict[str, list[dict]] = {}
    for round_dir in round_dirs:
        run_name = round_dir.parent.parent.name  # run_*/rounds/round_* → run_*
        objects = combine_round_json(round_dir)
        if objects:
            organized = organize_round(objects)
            run_data.setdefault(run_name, []).append(organized)

    # Write per-run fingerprint files
    for run_name, rounds in run_data.items():
        run_path = experiment_path / run_name / "raw_run_fingerprint.json"
        run_path.write_text(json.dumps(rounds, indent=2, default=str), encoding="utf-8")
        print(f"  {run_name}: {len(rounds)} rounds → {run_path}")

    # Write experiment-level fingerprint
    out_path = experiment_path / "raw_run_fingerprint.json"
    out_path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")
    total = sum(len(v) for v in run_data.values())
    print(f"Wrote {total} organized rounds across {len(run_data)} runs → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <experiment_dir | round_dir>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"Error: {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Detect: if it contains run_* subdirs, treat as experiment; otherwise as single round
    if any(target.glob("run_*")):
        combine_experiment(target)
    else:
        combine_round_json(target)
