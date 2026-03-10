import argparse
import csv
import json
from pathlib import Path

def flatten_dict(d, prefix="", excluded_fields=None):
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
            flat[key] = float(v)
            
    return flat

def load_scenario_metrics(csv_path):
    results = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("error") and row.get("results_dir"):
                    summary_path = Path(row["results_dir"]) / "summary.json"
                    if summary_path.exists():
                        with open(summary_path, 'r') as sf:
                            data = json.load(sf)
                            if data.get("episode_summaries"):
                                ep = data["episode_summaries"][0]
                                excluded = {"episode_id", "initial_cash", "final_positions", "final_prices", "position_values"}
                                results[row["scenario"]] = flatten_dict(ep, excluded_fields=excluded)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
    return results

def format_val(key, val):
    if val is None:
        return f"{'N/A':>10}"
    if "pct" in key or "return" in key:
        return f"{val:>9.2f}%"
    elif "trades" in key or "days" in key:
        return f"{int(val):>10}"
    else:
        return f"{val:>10.4f}"

def format_diff(key, val1, val2):
    if val1 is None or val2 is None:
        return f"{'N/A':>10}"
    diff = val2 - val1
    if "pct" in key or "return" in key:
        return f"{diff:>+9.2f}%"
    elif "trades" in key or "days" in key:
        return f"{int(diff):>+10}"
    else:
        return f"{diff:>+10.4f}"

def main():
    parser = argparse.ArgumentParser(description="Compare per-scenario metrics between two config runs.")
    parser.add_argument("csv1", help="Path to first CSV tracking file (Baseline)")
    parser.add_argument("csv2", help="Path to second CSV tracking file (Comparison)")
    parser.add_argument("--metrics", nargs="+", 
                        default=["daily_metrics_annualized_sharpe", "return_pct", "return_pct_with_cash_interest", "daily_metrics_total_return_pct","daily_metrics_max_drawdown_pct", "total_trades"],
                        help="List of flattened metric keys to compare")
    args = parser.parse_args()

    metrics1 = load_scenario_metrics(args.csv1)
    metrics2 = load_scenario_metrics(args.csv2)

    name1 = Path(args.csv1).stem.replace("results_tracking_", "")
    name2 = Path(args.csv2).stem.replace("results_tracking_", "")

    # Get intersection of scenarios
    all_scenarios = sorted(list(set(metrics1.keys()).union(set(metrics2.keys()))))
    
    if not all_scenarios:
        print("No valid scenarios found in either CSV.")
        return

    print(f"\nComparing {len(all_scenarios)} scenarios:")
    print(f" [A] {name1}")
    print(f" [B] {name2}")
    
    for metric in args.metrics:
        print("\n" + "="*85)
        display_metric = metric.replace("daily_metrics_", "[D] ").replace("_", " ").title()
        display_metric = display_metric.replace("Pct", "%").replace("Spy", "SPY")
        print(f" METRIC: {display_metric}")
        print("="*85)
        
        c1_head = "[A]"
        c2_head = "[B]"
        print(f"{'Scenario':<35} | {c1_head:>12} | {c2_head:>12} | {'Diff (B-A)':>12}")
        print(f"{'-'*35}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}")
        
        for sc in all_scenarios:
            s_name = Path(sc).stem
            if len(s_name) > 34:
                s_name = s_name[:31] + "..."
                
            m1 = metrics1.get(sc, {})
            m2 = metrics2.get(sc, {})
            
            v1 = m1.get(metric)
            v2 = m2.get(metric)
            
            s_v1 = format_val(metric, v1)
            s_v2 = format_val(metric, v2)
            s_diff = format_diff(metric, v1, v2)
            
            print(f"{s_name:<35} | {s_v1:>12} | {s_v2:>12} | {s_diff:>12}")
            
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
