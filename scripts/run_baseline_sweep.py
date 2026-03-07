import os
import json
import yaml
import subprocess
import re
import argparse
import csv
import random
import hashlib
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

def get_sector_tickers(sector_name):
    """Load tickers for a specific sector from supported_tickers.yaml"""
    supported_path = Path("data-pipeline/supported_tickers.yaml")
    if not supported_path.exists():
        return set()
    with open(supported_path, "r") as f:
        data = yaml.safe_load(f)
    return {t["symbol"] for t in data.get("supported_tickers", []) if t.get("sector") == sector_name}

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_processed_runs():
    """Read existing CSV to find which ticker combinations are already done per quarter."""
    # Mapping of invest_quarter -> list of ticker sets tried
    processed_sets = defaultdict(list)
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("invest_quarter")
                t_str = row.get("tickers")
                if q and t_str and not row.get("error"):
                    processed_sets[q].append(set(t_str.split(",")))
    return processed_sets

def append_result_to_csv(result):
    """Append a single result row to the CSV."""
    file_exists = os.path.exists(CSV_FILE)
    fieldnames = ["invest_quarter", "strategy", "num_tickers", "js_divergence", "results_dir", "log_dir", "scenario_file", "tickers", "error"]
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def run_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of quarters to run")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsets")
    args = parser.parse_args()

    # Change to project root if needed
    os.chdir(Path(__file__).resolve().parent.parent)
    
    data_dir = Path("data-pipeline/quarterly_asset_details/data")
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found.", flush=True)
        return

    random.seed(args.seed)

    tech_tickers_global = get_sector_tickers("Technology")
    financial_tickers_global = get_sector_tickers("Financials")
    print(f"Identified {len(tech_tickers_global)} tech tickers: {', '.join(sorted(list(tech_tickers_global)))}", flush=True)
    print(f"Identified {len(financial_tickers_global)} financial tickers: {', '.join(sorted(list(financial_tickers_global)))}", flush=True)

    # Map each data quarter to the set of available tickers
    quarter_to_tickers = defaultdict(set)
    for ticker_dir in data_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            for json_file in ticker_dir.glob("*.json"):
                quarter = json_file.stem  # e.g., "2022_Q4"
                quarter_to_tickers[quarter].add(ticker)

    sorted_quarters = sorted(quarter_to_tickers.keys())
    
    # Mapping of invest_quarter -> list of ticker sets tried
    processed_runs = defaultdict(list) if args.force else get_processed_runs()
    num_processed = sum(len(v) for v in processed_runs.values())
    if num_processed:
        print(f"Resuming sweep. Already processed {num_processed} unique ticker combinations across {len(processed_runs)} quarters.", flush=True)

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
        invest_quarter = next_quarter(q_t)
        inv_q_data_format = f"{invest_quarter[:4]}_Q{invest_quarter[5]}"
        
        if inv_q_data_format not in quarter_to_tickers:
            continue
            
        common_tickers_all = sorted(list(quarter_to_tickers[q_t].intersection(quarter_to_tickers[inv_q_data_format])))
        if not common_tickers_all:
            continue

        # Build list of strategies to run for this quarter
        # Format: (strategy_name, tickers_list)
        strategies = []
        
        # 1. Standard Sector Strategies
        strategies.append(("all", common_tickers_all))
        
        tech_tickers = sorted(list(set(common_tickers_all).intersection(tech_tickers_global)))
        if tech_tickers:
            strategies.append(("tech", tech_tickers))
            
        fin_tickers = sorted(list(set(common_tickers_all).intersection(financial_tickers_global)))
        if fin_tickers:
            strategies.append(("financials", fin_tickers))

        # 2. Fixed Random Subsets (3, 7, 12) - 2 runs each
        # for size in [3, 7, 12]:
        #     if len(common_tickers_all) > size:
        #         for _ in range(2):
        #             subset = sorted(random.sample(common_tickers_all, size))
        #             strategies.append((f"rand{size}", subset))

        # 3. New Random Range Strategy (size 3-15) - 2 runs
        for _ in range(2):
            # Try multiple times to find a subset that satisfies the similarity constraint
            for attempt in range(10):
                size = random.randint(3, min(15, len(common_tickers_all)))
                subset = sorted(random.sample(common_tickers_all, size))
                subset_set = set(subset)
                
                # Check similarity against everything we've built so far for this quarter
                # AND what was already in processed_runs
                is_too_similar = False
                existing_for_q = processed_runs[invest_quarter] + [set(s[1]) for s in strategies]
                
                for existing_set in existing_for_q:
                    if jaccard_similarity(subset_set, existing_set) >= 0.4:
                        is_too_similar = True
                        break
                
                if not is_too_similar:
                    strategies.append((f"rand{size}", subset))
                    break

        for strategy_name, tickers in strategies:
            if not tickers:
                continue

            tickers_set = set(tickers)
            tickers_str = ",".join(tickers)
            
            # Check for EXACT match or Jaccard similarity >= 0.6
            skip = False
            for existing_set in processed_runs[invest_quarter]:
                if tickers_set == existing_set:
                    skip = True
                    break
                
                similarity = jaccard_similarity(tickers_set, existing_set)
                if similarity >= 0.4:
                    # If it's a "rand" strategy, we silently skip to pick another one or move on.
                    # Sector strategies are always checked for exact match only.
                    if strategy_name.startswith("rand"):
                        skip = True
                        break
            
            if skip:
                continue

            if args.limit and run_count >= args.limit:
                break

            # Unique directory suffix using a hash of the tickers to prevent collisions
            ticker_hash = hashlib.md5(tickers_str.encode()).hexdigest()[:8]
            dir_suffix = f"{strategy_name}_{ticker_hash}"

            # Generate scenario YAML
            scenario_file = scenario_dir / f"scenario_{invest_quarter}_{dir_suffix}.yaml"
            
            if scenario_file.exists() and not args.force:
                # Still add to processed so we don't try to run it again this session
                processed_runs[invest_quarter].append(tickers_set)
                continue

            run_count += 1

            scenario_data = {
                "invest_quarter": invest_quarter,
                "tickers": tickers,
                "use_cash_virtual_ticker": True,
                "output_dir": f"results/baseline_sweep/{strategy_name}/{ticker_hash}",
                "allocation_constraints": {
                    "fully_invested": True,
                    "max_tickers": len(tickers)
                }
            }
            
            with open(scenario_file, "w") as f:
                yaml.dump(scenario_data, f, default_flow_style=False)
                
            print(f"\n[QUARTER] {invest_quarter} | Strategy: {strategy_name} | Tickers: {len(tickers)} (hash: {ticker_hash})", flush=True)
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
                "strategy": strategy_name,
                "num_tickers": len(tickers),
                "tickers": tickers_str,
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
                    
                    if "JS Divergence" in line:
                        print(f"  {line}", flush=True)
                        match = re.search(r"JS Divergence\): ([\d.]+) bits", line)
                        if match:
                            current_result["js_divergence"] = match.group(1)
                    elif "RESULTS_DIR: " in line:
                        print(f"  {line}", flush=True)
                        match = re.search(r"RESULTS_DIR: (.*)", line)
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
        print("\nNo new runs were processed.", flush=True)
        return

    print("\n" + "=" * 80, flush=True)
    print("BASELINE SWEEP SUMMARY (NEW RUNS)")
    print("=" * 80, flush=True)
    print(f"{'Invest Q':<10} | {'Strategy':<10} | {'Tickers':<8} | {'JS Div (bits)':<15} | {'Log Dir'}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        q = r.get("invest_quarter", "N/A")
        s = r.get("strategy", "all")
        t = r.get("num_tickers", "N/A")
        js = r.get("js_divergence", "N/A")
        err = r.get("error")
        if err:
            print(f"{q:<10} | {s:<10} | {t:<8} | {('ERROR: ' + err):<15}", flush=True)
        else:
            log = Path(r.get("log_dir", "")).name
            print(f"{q:<10} | {s:<10} | {t:<8} | {js:<15} | {log}", flush=True)
    print("=" * 80, flush=True)
    print(f"Full results updated in {CSV_FILE}", flush=True)

if __name__ == "__main__":
    run_sweep()
