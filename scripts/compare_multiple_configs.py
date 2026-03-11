import argparse
import csv
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

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

def format_val(key, mean, sem, precision=2, latex=False):
    if mean is None:
        return "N/A"
    
    # Choose separator
    sep = " \\pm " if latex else " ± "
    
    if "pct" in key or "return" in key or "cash" in key:
        if latex:
            return f"{mean:.2f}\\%{sep}{sem:.2f}\\%"
        return f"{mean:.2f}%{sep}{sem:.2f}%"
    else:
        fmt = f"{{:.{precision}f}}"
        return f"{fmt.format(mean)}{sep}{fmt.format(sem)}"

def get_superscript_letter(letter, bold=False):
    # Mapping for superscript letters (A-Z)
    # Note: Terminal support for these varies. 
    # For now, we'll just use ^Letter and optional ANSI bolding for terminal.
    # In LaTeX, we'll use ^{...}
    if bold:
        return f"^\033[1m{letter}\033[0m"
    else:
        return f"^{letter}"

def render_pdf(tex_path):
    try:
        print(f"Rendering PDF from {tex_path}...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error rendering PDF:\n{result.stdout}\n{result.stderr}")
        else:
            pdf_path = tex_path.with_suffix(".pdf")
            print(f"PDF successfully rendered: {pdf_path}")
            
            # Clean up auxiliary files
            for ext in [".aux", ".log", ".out"]:
                aux_file = tex_path.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()
    except Exception as e:
        print(f"Failed to run pdflatex: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple config runs side-by-side with statistical significance.")
    parser.add_argument("csvs", nargs="+", help="Paths to CSV tracking files to compare")
    parser.add_argument("--names", nargs="+", help="Custom names for the configurations")
    parser.add_argument("--metrics", nargs="+", default=[],
                        help="List of flattened metric keys to compare. If empty, all overlapping metrics are used.")
    parser.add_argument("--alpha-weak", type=float, default=0.15, help="P-value threshold for weak significance (non-bold)")
    parser.add_argument("--alpha-strong", type=float, default=0.01, help="P-value threshold for strong significance (bold)")
    parser.add_argument("--use-mean-revisions", action="store_true", help="Use mean revisions instead of judge output")
    parser.add_argument("--latex", action="store_true", help="Output a LaTeX table instead of terminal table")
    parser.add_argument("--output-tex", type=str, help="Path to save the generated LaTeX file")
    parser.add_argument("--output-pdf", action="store_true", help="Automatically render the LaTeX to a PDF (requires --output-tex)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N scenarios")
    args = parser.parse_args()

    if not args.csvs:
        print("At least one CSV file is required.")
        return

    # Load all metrics
    all_config_metrics = []
    config_names = []
    config_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, csv_path in enumerate(args.csvs):
        metrics, order = load_scenario_metrics(csv_path, use_mean_revisions=args.use_mean_revisions)
        all_config_metrics.append(metrics)
        
        if args.names and i < len(args.names):
            name = args.names[i]
        else:
            name = Path(csv_path).stem.replace("results_tracking_", "").replace("_", " ")
        config_names.append(name)

    # Find common scenarios across ALL configs
    common_scenarios = set(all_config_metrics[0].keys())
    for metrics in all_config_metrics[1:]:
        common_scenarios = common_scenarios.intersection(set(metrics.keys()))
    
    # Sort scenarios by the order in the first CSV
    _, first_order = load_scenario_metrics(args.csvs[0], use_mean_revisions=args.use_mean_revisions)
    ordered_scenarios = [sc for sc in first_order if sc in common_scenarios]
    
    if args.limit:
        ordered_scenarios = ordered_scenarios[:args.limit]
        
    if not ordered_scenarios:
        print("No common scenarios found across all configurations.")
        return

    # Discover metrics
    if not args.metrics:
        all_keys = set()
        for sc in ordered_scenarios:
            for metrics in all_config_metrics:
                all_keys.update(metrics[sc].keys())
        all_keys -= set(["book_value", "calmar_ratio", "duration_seconds", "max_drawdown", "return_pct", "return_pct_with_cash_interest", "daily_metrics_trading_days", "daily_metrics_spy_return_pct", "max_drawdown_pct"])
        args.metrics = sorted(list(all_keys), key=lambda x: (x.startswith("daily_metrics"), x))

    # Perform comparisons
    results_table = []
    num_configs = len(args.csvs)
    sig_summary = [[{'strong': 0, 'weak': 0} for _ in range(num_configs)] for _ in range(num_configs)]
    
    metric_priorities = [
        "daily_metrics_excess_return_pct",
        "daily_metrics_annualized_sharpe",
        "daily_metrics_annualized_volatility",
        "daily_metrics_max_drawdown_pct",
        "daily_metrics_return_pct",
        "daily_metrics_annualized_sortino",
        "daily_metrics_calmar_ratio",
        "total_trades",
        "final_cash",
    ]
    args.metrics = sorted(args.metrics, key=lambda x: metric_priorities.index(x) if x in metric_priorities else 100)

    for metric in args.metrics:
        row = {"Metric": metric}
        display_metric = metric.replace("daily_metrics_", "").replace("_", " ").title()
        display_metric = display_metric.replace("Annualized", "").replace("Pct", "%").replace("Spy", "SPY").replace("Js", "JS")
        display_metric = display_metric.replace("Final Cash", "Uninvested Cash")
        display_metric = display_metric.replace("Total Trades", "Num Positions")
        display_metric = display_metric.replace("Excess Return %", "Excess Return % (v SP500)")
        row["Display Metric"] = display_metric
        
        # Get values for each config
        config_vals = []
        for i in range(len(args.csvs)):
            vals = []
            for sc in ordered_scenarios:
                v = all_config_metrics[i].get(sc, {}).get(metric)
                if v is not None:
                    vals.append(v)
            config_vals.append(vals)
            
        # Compute means, SEM, and significance
        for i in range(len(args.csvs)):
            mean_val = np.mean(config_vals[i]) if config_vals[i] else None
            sem_val = stats.sem(config_vals[i]) if config_vals[i] and len(config_vals[i]) > 1 else 0.0
            if display_metric == "Uninvested Cash":
                mean_val /= 1000
                sem_val /= 1000
            formatted_val = format_val(metric, mean_val, sem_val, latex=args.latex or args.output_tex)
            
            # Compare with all other configs
            sigs = []
            for j in range(len(args.csvs)):
                if i == j:
                    continue
                
                # We need paired values for ttest_rel
                pair_vals_i = []
                pair_vals_j = []
                for sc in ordered_scenarios:
                    vi = all_config_metrics[i].get(sc, {}).get(metric)
                    vj = all_config_metrics[j].get(sc, {}).get(metric)
                    if vi is not None and vj is not None:
                        pair_vals_i.append(vi)
                        pair_vals_j.append(vj)

                if len(pair_vals_i) > 1:
                    t_stat, p_val = stats.ttest_rel(pair_vals_i, pair_vals_j)
                    
                    if p_val < args.alpha_strong:
                        sigs.append((config_letters[j], True)) # Strong
                        sig_summary[i][j]['strong'] += 1
                    elif p_val < args.alpha_weak:
                        sigs.append((config_letters[j], False)) # Weak
                        sig_summary[i][j]['weak'] += 1
            
            # Add superscripts
            if args.latex or args.output_tex:
                if sigs:
                    content = "".join([f"\\mathbf{{{l}}}" if b else f"{l}" for l, b in sorted(sigs)])
                    formatted_val = f"{{{formatted_val}}}^{{{content}}}"
            else:
                superscripts = "".join([get_superscript_letter(l, b) for l, b in sorted(sigs)])
                formatted_val = f"{formatted_val}{superscripts}"
                
            row[config_letters[i]] = formatted_val
            
        results_table.append(row)

    # Print Header Information
    print(f"\nComparing {len(ordered_scenarios)} common scenarios.")
    for i in range(len(args.csvs)):
        print(f" [{config_letters[i]}] {config_names[i]}")
    print(f" Significance levels: Weak < {args.alpha_weak}, Strong < {args.alpha_strong}")
    print()

    if args.latex or args.output_tex:
        df_data = []
        for r in results_table:
            row_data = {"Metric": r["Display Metric"].replace("%", "\\%").replace("_", "\\_")}
            for i in range(len(args.csvs)):
                renames = {
                    "single agent slim": "Single Agent\nBasic",
                    "vote slim no macro": "3 Roles\nMean, Basic",
                    "vote causal no macro causal out fix scenario": "3 Roles\nMean, Causal",
                    "vote no macro enriched": "3 Roles\nMean, Enriched",
                    "debate 1 round no macro causal out": "3 Roles\nDebate, Causal",
                    "debate 1 round no macro causal out rca revise": "3 Roles\nDebate, Causal + RCA",
                }
                # Split Name by \n
                config_names[i] = renames.get(config_names[i], config_names[i])
                parts = config_names[i].split("\n")
                # First line is (A) FirstWord
                if parts:
                    first_line = f"({config_letters[i]}) {parts[0]}"
                    other_lines = " \\\\ ".join(parts[1:])
                    header_content = f"{first_line} \\\\ {other_lines}" if other_lines else first_line
                else:
                    header_content = f"({config_letters[i]})"
                
                header_col = f"\\shortstack{{{header_content}}}"
                row_data[header_col] = f"${r[config_letters[i]]}$"
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        latex_table = df.to_latex(index=False, escape=False)
        
        if args.latex:
            print(latex_table)
            
        if args.output_tex:
            tex_path = Path(args.output_tex)
            full_latex = f"""\\documentclass[10pt]{{article}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage{{booktabs}}
        \\usepackage{{amsmath}}
        \\usepackage{{geometry}}
        \\geometry{{a4paper, margin=1in, landscape}}

        \\begin{{document}}
        \\section*{{Configuration Comparison (N={len(ordered_scenarios)})}}
        \\begin{{table*}}[h]
        \\centering
        {latex_table}
        \\caption{{N = {len(ordered_scenarios)} scenarios. $\pm$ SEM. Superscripts indicate stat.\\ sig.\\ in paired t-test (Weak: $p < {args.alpha_weak}$, \\textbf{{Strong}}: $p < {args.alpha_strong}$).}}
        \\end{{table*}}
        \\end{{document}}
        """
            with open(tex_path, 'w') as f:
                f.write(full_latex)
            print(f"LaTeX file saved to {tex_path}")

            
            if args.output_pdf:
                render_pdf(tex_path)
                
    else:
        # Simple terminal table
        col_width = max(25, *(len(r[config_letters[i]]) for r in results_table for i in range(len(args.csvs)))) + 2
        header = f"{'Metric':<40} | " + " | ".join([f"({config_letters[i]}) {config_names[i]:^{col_width-4}}" for i in range(len(args.csvs))])
        print(header)
        print("-" * len(header))
        for r in results_table:
            row_str = f"{r['Display Metric']:<40} | " + " | ".join([f"{r[config_letters[i]]:>{col_width}}" for i in range(len(args.csvs))])
            print(row_str)

    print("\nSignificance Summary:")
    total_strong = 0
    total_weak = 0
    for i in range(num_configs):
        for j in range(i + 1, num_configs):
            total_strong += sig_summary[i][j]['strong']
            total_weak += sig_summary[i][j]['weak']
    print(f"Total across all pairs: {total_strong} strong, {total_weak} weak")


if __name__ == "__main__":
    main()
