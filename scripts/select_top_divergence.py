import csv
import json
import math
from pathlib import Path

CSV_FILE = "baseline_sweep_results.csv"
OUTPUT_DIR = Path("config/scenarios/top_divergence")

K_MAP = {
    3: "xs",
    10: "s",
    20: "m",
    35: "l",
    60: "xl"
}

def load_results():
    runs = []
    if not Path(CSV_FILE).exists():
        print(f"Error: {CSV_FILE} not found.")
        return []
    
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error") or row.get("js_divergence") == "N/A":
                continue
            try:
                row["js_divergence"] = float(row["js_divergence"])
                runs.append(row)
            except ValueError:
                continue
    return runs

def select_top_k(runs, k):
    unique_quarters = sorted(list(set(r["invest_quarter"] for r in runs)))
    num_quarters = len(unique_quarters)
    if num_quarters == 0:
        return []
    
    # Per-quarter limit: round(k/total_quarters) + 1
    per_quarter_limit = round(k / num_quarters) + 1
    
    # Sort by JS Divergence descending
    sorted_runs = sorted(runs, key=lambda x: x["js_divergence"], reverse=True)
    
    selected = []
    quarter_counts = {q: 0 for q in unique_quarters}
    
    for run in sorted_runs:
        if len(selected) >= k:
            break
            
        q = run["invest_quarter"]
        if quarter_counts[q] < per_quarter_limit:
            selected.append(run)
            quarter_counts[q] += 1
            
    return selected

def main():
    runs = load_results()
    if not runs:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    unique_quarters_in_runs = sorted(list(set(r["invest_quarter"] for r in runs)))
    num_quarters = len(unique_quarters_in_runs)
    
    print(f"Processing {len(runs)} valid runs across {num_quarters} quarters.")
    print("-" * 80)
    print(f"{'Scale':<5} | {'k':<4} | {'Limit':<6} | {'Min JSD':<10} | {'Mean Q':<8} | {'Actual Count'}")
    print("-" * 80)

    for k, suffix in sorted(K_MAP.items()):
        selected = select_top_k(runs, k)
        
        if not selected:
            print(f"{suffix:<5} | {k:<4} | N/A    | N/A        | N/A      | 0")
            continue
            
        min_jsd = min(r["js_divergence"] for r in selected)
        limit = round(k / num_quarters) + 1
        
        # Calculate mean quarters per result (among quarters actually included)
        quarter_distribution = defaultdict(int)
        for r in selected:
            quarter_distribution[r["invest_quarter"]] += 1
        
        num_quarters_included = len(quarter_distribution)
        mean_q = len(selected) / num_quarters_included if num_quarters_included > 0 else 0
        
        print(f"{suffix:<5} | {k:<4} | {limit:<6} | {min_jsd:<10.4f} | {mean_q:<8.2f} | {len(selected)}")
        
        # Save scenario paths
        output_file = OUTPUT_DIR / f"top_scenarios_{suffix}.json"
        scenario_paths = [r["scenario_file"] for r in selected]
        with open(output_file, "w") as f:
            json.dump({
                "k": k,
                "scale": suffix,
                "min_jsd": min_jsd,
                "mean_q": mean_q,
                "scenarios": scenario_paths
            }, f, indent=2)

    print("-" * 80)
    print(f"Selected scenarios saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    from collections import defaultdict
    main()
