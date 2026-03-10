import csv
import json
import sys
from pathlib import Path

csv_file = "results_tracking_debate_slim_no_macro.csv"

try:
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"{'Scenario / Dir':<60} | {'Return':<10} | {'Return+Int':<10} | {'Daily':<10} | {'Diff'}")
        print("-" * 110)
        
        for row in reader:
            res_dir = Path(row.get('results_dir', ''))
            scenario = row.get('scenario', '').split('/')[-1]
            summary_path = res_dir / "summary.json"
            
            if not summary_path.exists():
                print(f"{str(res_dir):<60} | MISSING summary.json")
                continue
                
            try:
                with open(summary_path, 'r') as sf:
                    summary = json.load(sf)
                    
                ep_summaries = summary.get("episode_summaries", [])
                if not ep_summaries:
                    continue
                ep_000 = ep_summaries[0]
                
                ret_pct = float(ep_000.get("return_pct", 0.0))
                ret_int = float(ep_000.get("return_pct_with_cash_interest", ret_pct))
                daily_metrics = ep_000.get("daily_metrics", {})
                daily_ret = float(daily_metrics.get("total_return_pct", 0.0))
                
                diff = abs(ret_int - daily_ret)
                
                if diff > 0.1: # Only print significant discrepancies
                    print(f"{scenario:<30} {res_dir.name:<29} | {ret_pct:>10.4f} | {ret_int:>10.4f} | {daily_ret:>10.4f} | {diff:.4f}")
                
            except Exception as e:
                print(f"{str(res_dir):<60} | ERROR: {e}")
                
except Exception as e:
    print("Error reading tracking CSV:", e)
