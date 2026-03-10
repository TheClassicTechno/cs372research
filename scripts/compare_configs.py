import argparse
import csv
import json
import math
from pathlib import Path

try:
    from scipy.stats import ttest_ind_from_stats, t
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

def aggregate_metrics(csv_path):
    results_map = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("error"):
                    results_map[row["scenario"]] = row
    except Exception as e:
        print(f"Warning: could not read {csv_path}: {e}")
        return {}, 0
                
    excluded_fields = {"episode_id", "initial_cash", "final_positions", "final_prices", "position_values"}
    all_metrics = []
    
    for scenario, paths in results_map.items():
        res_dir = paths.get("results_dir")
        if not res_dir:
            continue
        summary_path = Path(res_dir) / "summary.json"
        if not summary_path.exists():
            continue
            
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
                
            eps = summary_data.get("episode_summaries", [])
            if not eps:
                continue
                
            ep = eps[0]
            flat_metrics = flatten_dict(ep, excluded_fields=excluded_fields)
            if "duration_seconds" in paths and paths["duration_seconds"]:
                try:
                    flat_metrics["duration_seconds"] = float(paths["duration_seconds"])
                except:
                    pass
                    
            all_metrics.append(flat_metrics)
        except Exception:
            continue
        
    if not all_metrics:
        return {}, 0
        
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
        
    stats = {}
    for k in sorted(list(all_keys)):
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        if not vals:
            continue
            
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std_dev = math.sqrt(variance)
            sem = std_dev / math.sqrt(len(vals))
        else:
            variance = 0.0
            sem = 0.0
            
        stats[k] = {
            "mean": mean,
            "variance": variance,
            "std_dev": math.sqrt(variance) if len(vals) > 1 else 0.0,
            "sem": sem,
            "count": len(vals)
        }
            
    return stats, len(all_metrics)

def format_value(key, mean, sem):
    is_pct = "pct" in key or "return" in key
    if is_pct:
        return f"{mean:>8.2f}% ± {sem:>6.2f}%"
    elif "duration" in key or "trading_days" in key or "total_trades" in key:
        return f"{mean:>8.1f}  ± {sem:>6.1f} "
    else:
        return f"{mean:>8.4f} ± {sem:>6.4f}"

def format_ci(key, diff, ci_margin):
    is_pct = "pct" in key or "return" in key
    lower = diff - ci_margin
    upper = diff + ci_margin
    if is_pct:
        return f"[{lower:>+7.2f}%, {upper:>+7.2f}%]"
    elif "duration" in key or "trading_days" in key or "total_trades" in key:
        return f"[{lower:>+7.1f} , {upper:>+7.1f} ]"
    else:
        return f"[{lower:>+7.4f}, {upper:>+7.4f}]"

def main():
    parser = argparse.ArgumentParser(description="Compare two configs side-by-side using their run_scenario_list.py CSV outputs.")
    parser.add_argument("csv1", help="Path to first CSV tracking file (Baseline)")
    parser.add_argument("csv2", help="Path to second CSV tracking file (Treatment)")
    args = parser.parse_args()

    stats1, count1 = aggregate_metrics(args.csv1)
    stats2, count2 = aggregate_metrics(args.csv2)

    name1 = Path(args.csv1).stem.replace("results_tracking_", "")
    name2 = Path(args.csv2).stem.replace("results_tracking_", "")

    print("\n" + "="*165)
    print(f" SIDE-BY-SIDE CONFIGURATION COMPARISON")
    print("="*165)
    
    c1_header = f"{name1} (N={count1})"
    if len(c1_header) > 35:
        c1_header = c1_header[:32] + "...)"
    c2_header = f"{name2} (N={count2})"
    if len(c2_header) > 35:
        c2_header = c2_header[:32] + "...)"
        
    print(f"{'Metric':<36} | {c1_header:^32} | {c2_header:^32} | {'95% CI (Diff B - A)':^33} | {'P-Value':^15}")
    print(f"{'-'*36}-|-{'-'*32}-|-{'-'*32}-|-{'-'*33}-|-{'-'*15}")

    all_keys = set(stats1.keys()).union(set(stats2.keys()))
    sorted_keys = sorted(list(all_keys), key=lambda x: (x.startswith("daily_metrics"), x))

    for k in sorted_keys:
        label = k.replace("daily_metrics_", "[D] ").replace("_", " ").title()
        label = label.replace("Pct", "%").replace("Spy", "SPY").replace("Js", "JS")
        
        if len(label) > 35:
            label = label[:32] + "..."

        if k in stats1:
            val1 = format_value(k, stats1[k]["mean"], stats1[k]["sem"])
        else:
            val1 = f"{'N/A':^32}"
            
        if k in stats2:
            val2 = format_value(k, stats2[k]["mean"], stats2[k]["sem"])
        else:
            val2 = f"{'N/A':^32}"
            
        # Calculate 95% CI for the difference (B - A) and P-value
        ci_str = f"{'N/A':^33}"
        pval_str = f"{'N/A':^15}"
        if k in stats1 and k in stats2 and stats1[k]["count"] > 1 and stats2[k]["count"] > 1:
            mean1 = stats1[k]["mean"]
            var1 = stats1[k]["variance"]
            std1 = stats1[k]["std_dev"]
            n1 = stats1[k]["count"]
            
            mean2 = stats2[k]["mean"]
            var2 = stats2[k]["variance"]
            std2 = stats2[k]["std_dev"]
            n2 = stats2[k]["count"]
            
            diff = mean2 - mean1
            se_diff = math.sqrt((var1/n1) + (var2/n2))
            
            if se_diff > 0:
                # Welch-Satterthwaite degrees of freedom
                df_num = ((var1/n1) + (var2/n2))**2
                df_den = ((var1/n1)**2 / (n1-1)) + ((var2/n2)**2 / (n2-1))
                df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)
                
                if HAS_SCIPY:
                    # More precise t-critical value using scipy
                    t_crit = t.ppf(0.975, df)
                    # Compute two-tailed p-value
                    ttest_result = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
                    pval = ttest_result.pvalue
                else:
                    # Approx t-critical value for 95% CI
                    t_crit = 1.96 if df > 30 else 2.1 
                    pval = None # Can't accurately compute pval without scipy
                
                margin = t_crit * se_diff
                ci_str = f"{format_ci(k, diff, margin):^33}"
                
                if pval is not None and not math.isnan(pval):
                    if pval < 0.001:
                        pval_str = f"{'< 0.001':^15}"
                    elif pval < 0.05:
                        pval_str = f"{pval:^15.3f}"
                    else:
                        pval_str = f"{pval:^15.3f}"
                    
                    if pval < 0.05:
                        pval_str = pval_str.replace(pval_str.strip(), f"*{pval_str.strip()}*")
            else:
                ci_str = f"{'[+0.00, +0.00]':^33}"
                pval_str = f"{'1.000':^15}"

        print(f"{label:<36} | {val1:^32} | {val2:^32} | {ci_str} | {pval_str}")

    print("="*165 + "\n")

if __name__ == "__main__":
    main()
