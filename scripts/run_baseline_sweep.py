import os
import json
import yaml
import subprocess
import re
import argparse
import csv
from pathlib import Path
from collections import defaultdict

# Fixed output filename
CSV_FILE = "baseline_sweep_results.csv"

def next_quarter(quarter_str):
    """Convert YEAR_QX to YEARQX+1 (e.g., 2022_Q4 -> 2023Q1)"""
    year, q = quarter_str.split("_")
    year = int(year)
    q_num = int(q[1])
    if q_num == 4:
        return f"{year+1}Q1"
    else:
        return f"{year}Q{q_num+1}"

def get_processed_runs():
    """Read existing CSV to find which (quarter, tickers) pairs are already done."""
    processed = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("invest_quarter")
                t = row.get("tickers")
                if q and t and not row.get("error"):
                    processed.add((q, t))
    return processed

def append_result_to_csv(result):
    """Append a single result row to the CSV."""
    file_exists = os.path.exists(CSV_FILE)
    fieldnames = ["invest_quarter", "num_tickers", "js_divergence", "results_dir", "log_dir", "scenario_file", "tickers", "error"]
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def run_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of quarters to run")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    args = parser.parse_args()

    # Change to project root if needed
    os.chdir(Path(__file__).resolve().parent.parent)
    
    data_dir = Path("data-pipeline/quarterly_asset_details/data")
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found.", flush=True)
        return

    # Map each data quarter to the set of available tickers
    quarter_to_tickers = defaultdict(set)
    for ticker_dir in data_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            for json_file in ticker_dir.glob("*.json"):
                quarter = json_file.stem  # e.g., "2022_Q4"
                quarter_to_tickers[quarter].add(ticker)

    sorted_quarters = sorted(quarter_to_tickers.keys())
    
    # Resume logic
    processed_runs = set() if args.force else get_processed_runs()
    if processed_runs:
        print(f"Resuming sweep. Already processed {len(processed_runs)} unique (quarter, tickers) runs.", flush=True)

    results = []
    
    # Create temp dir for generated scenarios
    scenario_dir = Path("config/scenarios/baseline_sweep")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    agent_config = "config/debate/baselines/debate_slim_no_macro.yaml"

    limit_msg = f" (LIMIT {args.limit})" if args.limit else ""
    print(f"Found {len(sorted_quarters)} data quarters. Starting sweep{limit_msg}...", flush=True)
    print("=" * 80, flush=True)

    run_count = 0
    for i in range(len(sorted_quarters)):
        if args.limit and run_count >= args.limit:
            break
            
        q_t = sorted_quarters[i]
        
        # The agents see q_t data and invest for the next quarter.
        invest_quarter = next_quarter(q_t)
        
        # To satisfy snapshot_builder's pre-flight, we need data for BOTH 
        # the prior quarter (q_t) AND the invest quarter (inv_q).
        inv_q_data_format = f"{invest_quarter[:4]}_Q{invest_quarter[5]}"
        
        if inv_q_data_format not in quarter_to_tickers:
            print(f"Skipping {q_t} -> {invest_quarter}: No data for invest quarter {inv_q_data_format} to compute exit prices.", flush=True)
            continue
            
        # Use tickers available in both quarters to ensure P&L can be computed
        common_tickers = sorted(list(quarter_to_tickers[q_t].intersection(quarter_to_tickers[inv_q_data_format])))
        tickers_str = ",".join(common_tickers)
        
        if not common_tickers:
            print(f"Skipping {q_t} -> {invest_quarter}: No common tickers found.", flush=True)
            continue

        # Skip if already processed
        if (invest_quarter, tickers_str) in processed_runs:
            print(f"Skipping {invest_quarter} with {len(common_tickers)} tickers (already in {CSV_FILE})", flush=True)
            continue

        # Generate scenario YAML
        scenario_file = scenario_dir / f"scenario_{invest_quarter}.yaml"
        scenario_data = {
            "invest_quarter": invest_quarter,
            "tickers": common_tickers,
            "use_cash_virtual_ticker": True,
            "output_dir": "results/baseline_sweep",
            "allocation_constraints": {
                "fully_invested": True,
                "max_tickers": len(common_tickers)
            }
        }
        
        with open(scenario_file, "w") as f:
            yaml.dump(scenario_data, f, default_flow_style=False)
            
        print(f"\n[QUARTER] {invest_quarter} | Tickers: {len(common_tickers)}", flush=True)
        print(f"  Scenario: {scenario_file}", flush=True)
        
        # Run simulation
        cmd = [
            "python3", "run_simulation.py",
            "--agents", agent_config,
            "--scenario", str(scenario_file),
            "--no-display",
            "--logging-mode", "standard",
            "--log-level", "WARNING"
        ]
        
        current_result = {
            "invest_quarter": invest_quarter,
            "num_tickers": len(common_tickers),
            "tickers": ",".join(common_tickers),
            "scenario_file": str(scenario_file),
            "js_divergence": "N/A",
            "results_dir": "N/A",
            "log_dir": "N/A",
            "error": None
        }

        try:
            # Capture output to extract JS divergence and results_dir
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in process.stdout:
                line = line.strip()
                if not line: continue
                
                # Print output lines in real-time for progress visibility
                if "JS Divergence" in line:
                    print(f"  {line}", flush=True)
                    match = re.search(r"JS Divergence\): ([\d.]+) bits", line)
                    if match:
                        current_result["js_divergence"] = match.group(1)
                elif "Output: " in line:
                    print(f"  {line}", flush=True)
                    # Extract path like 'results/baseline_sweep/debate_slim_no_macro_001'
                    match = re.search(r"Output: (.*)", line)
                    if match:
                        current_result["results_dir"] = match.group(1)
                elif "[Logged]" in line:
                    print(f"  {line}", flush=True)
                    match = re.search(r"\[Logged\] (.*)", line)
                    if match:
                        current_result["log_dir"] = match.group(1)
                elif "Error" in line or "Exception" in line or "WARNING" in line:
                    if "No invest-quarter snapshot" not in line:
                        print(f"  {line}", flush=True)

            process.wait()
            
            if process.returncode != 0:
                print(f"  Simulation failed with return code {process.returncode}", flush=True)
                current_result["error"] = f"Exit code {process.returncode}"
            
        except Exception as e:
            print(f"  Execution error: {e}", flush=True)
            current_result["error"] = str(e)

        # Save result incrementally
        append_result_to_csv(current_result)
        results.append(current_result)

    # Print Final Summary Table
    if not results:
        print("\nNo new quarters were processed.", flush=True)
        return

    print("\n" + "=" * 60, flush=True)
    print("BASELINE SWEEP SUMMARY (NEW RUNS)")
    print("=" * 60, flush=True)
    print(f"{'Invest Q':<10} | {'Tickers':<8} | {'JS Div (bits)':<15} | {'Log Dir'}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        q = r.get("invest_quarter", "N/A")
        t = r.get("num_tickers", "N/A")
        js = r.get("js_divergence", "N/A")
        err = r.get("error")
        if err:
            print(f"{q:<10} | {t:<8} | {('ERROR: ' + err):<15}", flush=True)
        else:
            log = Path(r.get("log_dir", "")).name
            print(f"{q:<10} | {t:<8} | {js:<15} | {log}", flush=True)
    print("=" * 60, flush=True)
    print(f"Full results updated in {CSV_FILE}", flush=True)

if __name__ == "__main__":
    run_sweep()
