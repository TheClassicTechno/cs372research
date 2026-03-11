import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

# Add the scripts directory to sys.path so we can import from it
sys.path.append(str(Path(__file__).resolve().parent))
from compute_mean_revisions import get_or_compute_mean_revisions

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

def load_scenario_metrics(csv_path, use_mean_revisions=False):
    results = {}
    order = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = row.get("scenario")
                if not scenario:
                    continue
                order.append(scenario)
                if not row.get("error") and row.get("results_dir"):
                    results_dir = Path(row["results_dir"])
                    summary_path = results_dir / "summary.json"
                    
                    metrics = {}
                    if use_mean_revisions:
                        rev_metrics = get_or_compute_mean_revisions(results_dir)
                        if rev_metrics:
                            metrics = flatten_dict({"daily_metrics": rev_metrics})
                    else:
                        if summary_path.exists():
                            with open(summary_path, 'r') as sf:
                                data = json.load(sf)
                                if data.get("episode_summaries"):
                                    ep = data["episode_summaries"][0]
                                    excluded = {"episode_id", "initial_cash", "final_positions", "final_prices", "position_values", "mean_revisions_metrics"}
                                    metrics = flatten_dict(ep, excluded_fields=excluded)
                                    
                    if metrics:
                        # Add duration from CSV if available
                        if row.get("duration_seconds"):
                            try:
                                metrics["duration_seconds"] = float(row["duration_seconds"])
                            except ValueError:
                                pass
                                
                        results[scenario] = metrics
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
    return results, order

def format_val(key, val, precision=4):
    if val is None:
        return f"{'N/A':>10}"
    if "pct" in key or "return" in key:
        return f"{val:>9.2f}%"
    elif "trades" in key or "days" in key:
        return f"{int(val):>10}"
    else:
        fmt = f"{{:>{10}.{precision}f}}"
        return fmt.format(val)

def format_diff(key, val1, val2, precision=4):
    if val1 is None or val2 is None:
        return f"{'N/A':>10}"
    diff = val2 - val1
    if "pct" in key or "return" in key:
        return f"{diff:>+9.2f}%"
    elif "trades" in key or "days" in key:
        return f"{int(diff):>+10}"
    else:
        fmt = f"{{:>+{10}.{precision}f}}"
        return fmt.format(diff)

def main():
    parser = argparse.ArgumentParser(description="Compare per-scenario metrics between two config runs.")
    parser.add_argument("csv1", help="Path to first CSV tracking file (Baseline)")
    parser.add_argument("csv2", help="Path to second CSV tracking file (Comparison)")
    parser.add_argument("--metrics", nargs="+", default=[],
                        help="List of flattened metric keys to compare. If empty, all overlapping metrics are used.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only compare the first N scenarios from the baseline CSV.")
    parser.add_argument("--use-mean-revisions-a", action="store_true",
                        help="Use the mean of the agent revisions instead of the judge output for side A.")
    parser.add_argument("--use-mean-revisions-b", action="store_true",
                        help="Use the mean of the agent revisions instead of the judge output for side B.")
    args = parser.parse_args()

    metrics1, order1 = load_scenario_metrics(args.csv1, use_mean_revisions=args.use_mean_revisions_a)
    metrics2, _ = load_scenario_metrics(args.csv2, use_mean_revisions=args.use_mean_revisions_b)

    name1 = Path(args.csv1).stem.replace("results_tracking_", "")
    if args.use_mean_revisions_a:
        name1 += " (Mean Rev)"
        
    name2 = Path(args.csv2).stem.replace("results_tracking_", "")
    if args.use_mean_revisions_b:
        name2 += " (Mean Rev)"

    # Get intersection of scenarios, preserving order of csv1
    all_scenarios = [sc for sc in order1 if sc in metrics1 and sc in metrics2]
    
    if args.limit is not None:
        all_scenarios = all_scenarios[:args.limit]
    
    if not all_scenarios:
        print("No overlapping scenarios found in both CSVs.")
        return

    # Discover all metrics if none provided
    if not args.metrics:
        all_keys = set()
        for sc in all_scenarios:
            all_keys.update(metrics1[sc].keys())
            all_keys.update(metrics2[sc].keys())
        
        # Sort them nicely: top level first, then daily_metrics
        args.metrics = sorted(list(all_keys), key=lambda x: (x.startswith("daily_metrics"), x))

    print(f"\nComparing {len(all_scenarios)} scenarios (Order from: {args.csv1}):")
    print(f" [A] {name1}")
    print(f" [B] {name2}")
    
    summary_data = []
    
    for metric in args.metrics:
        vals1 = []
        vals2 = []
        
        for sc in all_scenarios:
            v1 = metrics1.get(sc, {}).get(metric)
            v2 = metrics2.get(sc, {}).get(metric)
            if v1 is not None and v2 is not None:
                vals1.append(v1)
                vals2.append(v2)
        
        if not vals1:
            continue

        display_metric = metric.replace("daily_metrics_", "[D] ").replace("_", " ").title()
        display_metric = display_metric.replace("Pct", "%").replace("Spy", "SPY").replace("Js", "JS")
        
        print("\n" + "="*85)
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
            
        # Aggregate statistics and paired t-test
        if len(vals1) > 1:
            v1_arr = np.array(vals1)
            v2_arr = np.array(vals2)
            diffs = v2_arr - v1_arr
            
            t_stat, p_val = stats.ttest_rel(v2_arr, v1_arr)
            mean_diff = np.mean(diffs)
            se_diff = stats.sem(diffs)
            
            mean1 = np.mean(v1_arr)
            sem1 = stats.sem(v1_arr)
            mean2 = np.mean(v2_arr)
            sem2 = stats.sem(v2_arr)
            
            # 95% Confidence Interval
            ci = stats.t.interval(0.95, len(diffs)-1, loc=mean_diff, scale=se_diff)
            
            summary_data.append({
                "metric": metric,
                "display": display_metric,
                "mean1": mean1,
                "sem1": sem1,
                "mean2": mean2,
                "sem2": sem2,
                "diff": mean_diff,
                "ci": ci,
                "pval": p_val,
                "n": len(diffs)
            })
            
            print(f"{'-'*35}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}")
            print(f"{'MEAN':<35} | {format_val(metric, mean1):>12} | {format_val(metric, mean2):>12} | {format_diff(metric, 0, mean_diff):>12}")
            print(f"{'STD DEV':<35} | {format_val(metric, np.std(v1_arr, ddof=1)):>12} | {format_val(metric, np.std(v2_arr, ddof=1)):>12} | {format_val(metric, np.std(diffs, ddof=1)):>12}")
            print(f"{'-'*85}")
            print(f" PAIRED T-TEST (B vs A):")
            print(f"  n:           {len(diffs)}")
            print(f"  t-statistic: {t_stat:>+10.4f}")
            print(f"  p-value:     {p_val:>10.4f}")
            
            # Format CI
            if "pct" in metric or "return" in metric:
                ci_str = f"({ci[0]:+6.2f}%, {ci[1]:+6.2f}%)"
            else:
                ci_str = f"({ci[0]:+10.4f}, {ci[1]:+10.4f})"
            print(f"  95% CI diff: {ci_str:>10}")
            
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            color = "\033[92m" if p_val < 0.05 else "\033[91m"
            reset = "\033[0m"
            print(f"  Result:      {color}{sig}{reset} (alpha=0.05)")

    # Print overall summary table
    if summary_data:
        print("\n" + "="*175)
        print(f" OVERALL SUMMARY (Paired Comparison, N_total={len(all_scenarios)})")
        print("="*175)
        
        c1_head = f"{name1} (A)"
        c2_head = f"{name2} (B)"
        
        print(f"{'Metric':<36} | {'n':^5} | {c1_head:^32} | {c2_head:^32} | {'95% CI (Diff B - A)':^33} | {'P-Value':^15}")
        print(f"{'-'*36}-|-{'-'*5}-|-{'-'*32}-|-{'-'*32}-|-{'-'*33}-|-{'-'*15}")
        
        for s in summary_data:
            m = s["metric"]
            
            def fmt_stat(mean, sem):
                is_pct = "pct" in m or "return" in m
                if is_pct:
                    return f"{mean:>8.2f}% ± {sem:>6.2f}%"
                elif "trades" in m or "days" in m:
                    return f"{mean:>8.1f}  ± {sem:>6.1f} "
                else:
                    return f"{mean:>8.4f} ± {sem:>6.4f}"

            val1 = fmt_stat(s["mean1"], s["sem1"])
            val2 = fmt_stat(s["mean2"], s["sem2"])
            
            ci = s["ci"]
            is_pct = "pct" in m or "return" in m
            if is_pct:
                ci_str = f"[{ci[0]:>+7.2f}%, {ci[1]:>+7.2f}%]"
            elif "trades" in m or "days" in m:
                ci_str = f"[{ci[0]:>+7.1f} , {ci[1]:>+7.1f} ]"
            else:
                ci_str = f"[{ci[0]:>+7.4f}, {ci[1]:>+7.4f}]"
            
            pval = s["pval"]
            if pval < 0.001:
                pval_str = f"{'< 0.001':^15}"
            else:
                pval_str = f"{pval:^15.3f}"
            
            if pval < 0.05:
                color = "\033[92m"
                reset = "\033[0m"
                pval_str = f"{color}{pval_str.strip():^15}{reset}"
                
            print(f"{s['display']:<36} | {s['n']:^5} | {val1:^32} | {val2:^32} | {ci_str:^33} | {pval_str}")
            
        print("="*175 + "\n")
        
        import pandas as pd
        
        df_data = []
        for s in summary_data:
            m = s["metric"]
            is_pct = "pct" in m or "return" in m
            is_trade = "trades" in m or "days" in m
            
            def fmt_stat_latex(mean, sem):
                if is_pct:
                    return f"{mean:.2f}\\% \\pm {sem:.2f}\\%"
                elif is_trade:
                    return f"{mean:.1f} \\pm {sem:.1f}"
                else:
                    return f"{mean:.4f} \\pm {sem:.4f}"
            
            val1 = fmt_stat_latex(s["mean1"], s["sem1"])
            val2 = fmt_stat_latex(s["mean2"], s["sem2"])
            
            ci = s["ci"]
            if is_pct:
                ci_str = f"[{ci[0]:+.2f}\\%, {ci[1]:+.2f}\\%]"
            elif is_trade:
                ci_str = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
            else:
                ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                
            pval = s["pval"]
            if pval < 0.001:
                pval_str = "$< 0.001$"
            else:
                pval_str = f"{pval:.3f}"
                
            if pval < 0.05:
                pval_str = f"\\textbf{{{pval_str}}}"
                
            df_data.append({
                "Metric": s["display"].replace("%", "\\%").replace("_", "\\_"),
                "n": s["n"],
                f"{name1.replace('_', '\\_')} (A)": val1,
                f"{name2.replace('_', '\\_')} (B)": val2,
                "95\\% CI (Diff B - A)": ci_str,
                "P-Value": pval_str
            })
            
        df = pd.DataFrame(df_data)
        print("\n" + "="*85)
        print(" LaTeX SUMMARY TABLE")
        print("="*85)
        # to_latex requires escape=False because we manually added LaTeX commands like \textbf and \pm
        print(df.to_latex(index=False, escape=False))

    print("\n" + "="*85 + "\n")

if __name__ == "__main__":
    main()
