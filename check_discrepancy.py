import csv
import sys

csv_file = "results_tracking_debate_slim_no_macro.csv"

try:
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print("Headers:", reader.fieldnames)
        print(f"{'Run':<40} | {'Return Pct':<12} | {'Daily Return Pct':<18} | {'Diff'}")
        print("-" * 80)
        for row in reader:
            run_name = row.get('run_id', row.get('Run', row.get('run_dir', 'unknown')))
            
            try:
                ret_pct = float(row.get('return_pct', 0))
            except:
                ret_pct = 0.0
                
            try:
                daily_ret_pct = float(row.get('daily_metrics_total_return_pct', 0))
            except:
                daily_ret_pct = 0.0
                
            diff = abs(ret_pct - daily_ret_pct)
            if diff > 0.1: # lower threshold
                print(f"{run_name:<40} | {ret_pct:>12.4f} | {daily_ret_pct:>18.4f} | {diff:.4f}")
except Exception as e:
    print("Error:", e)
