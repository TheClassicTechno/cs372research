import os
import json
import yaml
import subprocess
import re
import argparse
import csv
import math
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

def get_file_hash(path):
    """Calculate SHA-256 hash of file content."""
    if not Path(path).exists():
        return "MISSING"
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]

def load_scenario_list(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("scenarios", []), data.get("scale", "custom")

def get_config_name(config_path):
    return Path(config_path).stem

def get_tracking_file(config_path):
    c_name = get_config_name(config_path)
    return f"results_tracking_{c_name}.csv"

def get_processed_scenarios(tracking_file, current_config_hash):
    processed = {}
    if os.path.exists(tracking_file):
        with open(tracking_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only skip if the config hash matches and there was no error
                if not row.get("error") and row.get("config_hash") == current_config_hash:
                    processed[row["scenario"]] = {
                        "results_dir": row["results_dir"],
                        "log_dir": row.get("log_dir", "N/A"),
                        "scenario_hash": row.get("scenario_hash", "N/A"),
                        "duration_seconds": float(row.get("duration_seconds", 0.0)) if row.get("duration_seconds") else None,
                        "timestamp": row.get("timestamp", "N/A")
                    }
    return processed

def run_simulation(config_path, scenario_path, output_root, verbose=False, is_parallel=False):
    # Derive a subdirectory for this specific run
    s_name = Path(scenario_path).stem
    c_name = Path(config_path).stem

    # If running in parallel, put each scenario in its own unique subfolder to prevent race conditions
    # on directory creation inside run_simulation.py
    if is_parallel:
        actual_output_root = output_root / c_name / s_name
    else:
        actual_output_root = output_root / c_name

    cmd = [
        "python3", "run_simulation.py",
        "--agents", config_path,
        "--scenario", scenario_path,
        "--no-display",
        "--output-dir", str(actual_output_root),
        "--log-level", "WARNING"
    ]

    results_dir = "N/A"
    log_dir = "N/A"
    error = None

    try:
        if verbose and not is_parallel:
            # Capture stdout so we can parse RESULTS_DIR, but also print it live
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                stripped = line.strip()
                if "RESULTS_DIR: " in stripped:
                    results_dir = stripped.split("RESULTS_DIR: ")[1]
                elif "[Logged] " in stripped:
                    log_dir = stripped.split("[Logged] ")[1]
            process.wait()
            if process.returncode != 0:
                error = f"Exit code {process.returncode}"
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                line = line.strip()
                if "RESULTS_DIR: " in line:
                    results_dir = line.split("RESULTS_DIR: ")[1]
                elif "[Logged] " in line:
                    log_dir = line.split("[Logged] ")[1]
                elif "Error" in line or "Exception" in line:
                    if verbose:
                        print(f"[{s_name}] {line}", flush=True)

            process.wait()
            if process.returncode != 0:
                error = f"Exit code {process.returncode}"
    except Exception as e:
        error = str(e)

    return results_dir, log_dir, error

def flatten_dict(d, prefix="", excluded_fields=None):
    """Recursively flatten a dict, skipping excluded fields."""
    if excluded_fields is None:
        excluded_fields = set()
    
    flat = {}
    if not isinstance(d, dict):
        return flat
        
    for k, v in d.items():
        if k in excluded_fields:
            continue
            
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_dict(v, prefix=f"{key}_", excluded_fields=excluded_fields))
        elif isinstance(v, (int, float, bool)) and v is not None:
            # We want to skip bool if it's not useful, but total_trades is int.
            # Simulation doesn't really have booleans in metrics anyway.
            flat[key] = float(v)
            
    return flat

def aggregate_metrics(results_map):
    excluded_fields = {"episode_id", "initial_cash", "final_positions", "final_prices", "position_values"}
    all_metrics = []
    initial_cash = None
    
    for scenario, paths in results_map.items():
        res_dir = paths["results_dir"]
        summary_path = Path(res_dir) / "summary.json"
        if not summary_path.exists():
            continue
            
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            
        eps = summary_data.get("episode_summaries", [])
        if not eps:
            continue
            
        # Use first episode summary
        ep = eps[0]
        if initial_cash is None:
            initial_cash = ep.get("initial_cash")
            
        flat_metrics = flatten_dict(ep, excluded_fields=excluded_fields)
        
        # Add run duration if available in the tracking map
        if "duration_seconds" in paths:
            flat_metrics["duration_seconds"] = paths["duration_seconds"]
            
        all_metrics.append(flat_metrics)
        
    if not all_metrics:
        return {}, None
        
    # Get the union of all keys across all scenarios
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
        
    stats = {}
    for k in sorted(list(all_keys)):
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        if not vals:
            continue
            
        mean = sum(vals) / len(vals)
        
        # Standard Error of Mean (SEM)
        if len(vals) > 1:
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std_dev = math.sqrt(variance)
            sem = std_dev / math.sqrt(len(vals))
        else:
            sem = 0.0
            
        stats[k] = {
            "mean": mean,
            "sem": sem,
            "count": len(vals)
        }
            
    return stats, initial_cash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True, help="Path to top_scenarios_*.json")
    parser.add_argument("--config", required=True, help="Path to debate agents YAML config")
    parser.add_argument("--output-dir", default="results/scenario_runs", help="Root dir for results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print subprocess output to screen")
    parser.add_argument("--parallel", type=int, default=1, help="Number of scenarios to run concurrently")
    args = parser.parse_args()

    scenarios, scale = load_scenario_list(args.scenarios)
    config_name = get_config_name(args.config)
    scenario_list_name = Path(args.scenarios).stem
    
    current_config_hash = get_file_hash(args.config)
    tracking_file = get_tracking_file(args.config)
    processed = get_processed_scenarios(tracking_file, current_config_hash)
    
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Running config '{args.config}' (hash: {current_config_hash})")
    print(f"Against {len(scenarios)} scenarios from {args.scenarios}")
    print(f"Tracking file: {tracking_file}")
    if args.parallel > 1:
        print(f"Running {args.parallel} scenarios in parallel")
    
    csv_lock = threading.Lock()
    
    # Filter scenarios to run
    scenarios_to_run = []
    for scenario in scenarios:
        s_hash = get_file_hash(scenario)
        if scenario in processed:
            if processed[scenario]["scenario_hash"] == s_hash:
                continue
            else:
                print(f"  Scenario {scenario} content changed (hash {processed[scenario]['scenario_hash']} -> {s_hash}). Re-running.")
        scenarios_to_run.append((scenario, s_hash))
        
    if args.limit:
        scenarios_to_run = scenarios_to_run[:args.limit]
        
    print(f"\n{len(scenarios_to_run)} scenarios left to run.")

    def process_scenario(idx, total, scenario, s_hash):
        print(f"[{idx}/{total}] Processing {scenario}...")
        t0 = time.monotonic()
        res_dir, l_dir, err = run_simulation(args.config, scenario, output_root, verbose=args.verbose, is_parallel=(args.parallel > 1))
        duration = time.monotonic() - t0
        now_ts = datetime.now(timezone.utc).isoformat()
        
        with csv_lock:
            # Log to tracking file
            fieldnames = ["scenario", "scenario_hash", "results_dir", "log_dir", "config_hash", "duration_seconds", "timestamp", "error"]
            file_exists = os.path.exists(tracking_file)
            with open(tracking_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "scenario": scenario,
                    "scenario_hash": s_hash,
                    "results_dir": res_dir, 
                    "log_dir": l_dir,
                    "config_hash": current_config_hash,
                    "duration_seconds": round(duration, 2),
                    "timestamp": now_ts,
                    "error": err
                })
                
            if not err:
                processed[scenario] = {
                    "results_dir": res_dir, 
                    "log_dir": l_dir,
                    "scenario_hash": s_hash,
                    "duration_seconds": duration,
                    "timestamp": now_ts
                }
            else:
                print(f"  FAILED: {err}")
                
        return scenario, not bool(err)

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(process_scenario, idx+1, len(scenarios_to_run), scenario, s_hash) 
                for idx, (scenario, s_hash) in enumerate(scenarios_to_run)
            ]
            for future in as_completed(futures):
                scenario, success = future.result()
                if success:
                    print(f"  Finished {Path(scenario).stem}")
    else:
        for idx, (scenario, s_hash) in enumerate(scenarios_to_run):
            process_scenario(idx+1, len(scenarios_to_run), scenario, s_hash)

    # Aggregation
    stats, initial_cash = aggregate_metrics(processed)
    if not stats:
        print("No summary data found to aggregate.")
        return

    # Structured output
    results_payload = {
        "config_path": args.config,
        "scenario_list_path": args.scenarios,
        "tracking_file_path": str(Path(tracking_file).resolve()),
        "initial_cash": initial_cash,
        "scale": scale,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "num_scenarios": len(processed),
        "metrics": stats
    }
    
    final_results_json = output_root / f"aggregated_{config_name}_{scenario_list_name}.json"
    with open(final_results_json, "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 85)
    print(f"AGGREGATE PERFORMANCE (N={len(processed)} scenarios)")
    print("-" * 85)
    print(f"{'Metric':<35} | {'Mean':>12} | {'SEM':>12}")
    print("-" * 85)
    
    sorted_keys = sorted(stats.keys(), key=lambda x: (x.startswith("daily_metrics"), x))
    for k in sorted_keys:
        m = stats[k]
        display_name = k.replace("daily_metrics_", "[D] ")
        mean_str = f"{m['mean']:>12.4f}"
        sem_str = f"{m['sem']:>12.4f}"
        
        # Only use percentage formatting for actual percentages/returns
        if "pct" in k or "return" in k:
            mean_str = f"{m['mean']:>11.2f}%"
            sem_str = f"{m['sem']:>11.2f}%"
            
        print(f"{display_name:<35} | {mean_str} | {sem_str}")
    
    print("-" * 85)
    print(f"Structured results saved to: {final_results_json}")
    print("=" * 85)

if __name__ == "__main__":
    main()
