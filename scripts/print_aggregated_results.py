import json
import argparse
from pathlib import Path

def format_value(key, mean, sem):
    """Format a metric with mean ± SEM and appropriate units."""
    is_pct = "pct" in key or "return" in key
    
    if is_pct:
        return f"{mean:>8.2f}% ± {sem:>6.2f}%"
    elif "duration" in key or "trading_days" in key or "total_trades" in key:
        return f"{mean:>8.1f}  ± {sem:>6.1f} "
    else:
        return f"{mean:>8.4f} ± {sem:>6.4f}"

def print_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metrics = data.get("metrics", {})
    scale = data.get("scale", "N/A").upper()
    n = data.get("num_scenarios", 0)
    
    print("\n" + "="*70)
    print(f" AGGREGATED PERFORMANCE REPORT — Scale: {scale} (N={n})")
    print(f" Config: {data.get('config_path')}")
    print("="*70)
    print(f"{'Metric':<35} | {'Mean ± SEM':^25}")
    print(f"{'-'*35}-|-{'-'*25}")

    # Sort: put daily metrics together
    sorted_keys = sorted(metrics.keys(), key=lambda x: (x.startswith("daily_metrics"), x))
    
    for k in sorted_keys:
        m = metrics[k]
        
        # Clean up labels for display
        label = k.replace("daily_metrics_", "[D] ").replace("_", " ").title()
        label = label.replace("Pct", "%").replace("Spy", "SPY").replace("Js", "JS")
        
        val_str = format_value(k, m["mean"], m["sem"])
        print(f"{label:<35} | {val_str}")

    print("="*70)
    print(f" As of: {data.get('as_of')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to aggregated results JSON file")
    args = parser.parse_args()
    
    if Path(args.json_file).exists():
        print_results(args.json_file)
    else:
        print(f"Error: File not found: {args.json_file}")
