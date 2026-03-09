import os
import json
import yaml
import argparse
import re
from pathlib import Path
from typing import Any

import sys
# Add project root to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.case import ClosePricePoint
from models.log import EpisodeLog
from eval.evidence import parse_memo_evidence
from eval.financial import compute_daily_financial_metrics

def _parse_invest_quarter(invest_quarter: str) -> tuple[int, str]:
    year = int(invest_quarter[:4])
    q = invest_quarter[4:]
    return year, q

def _prior_quarter(year: int, q: str) -> tuple[int, str]:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(q)
    if idx == 0:
        return year - 1, "Q4"
    return year, labels[idx - 1]

def repair_summary(run_dir: Path, force: bool = False):
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "simulation_log.json"
    
    if not summary_path.exists() or not log_path.exists():
        return False

    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    needs_repair = False
    for ep_summary in summary.get("episode_summaries", []):
        # Fix zero prices if they exist
        final_prices = ep_summary.get("final_prices", {})
        has_zero_prices = any(v == 0.0 for k, v in final_prices.items() if k != "_CASH_")
        
        if "daily_metrics" not in ep_summary or has_zero_prices or force:
            needs_repair = True
            break
    
    if not needs_repair:
        return False

    print(f"Repairing {summary_path}...")

    # Load simulation log to get config and episode data
    with open(log_path, "r") as f:
        log_data = json.load(f)
    
    config_dict = log_data.get("config", {})
    invest_quarter = config_dict.get("invest_quarter")
    tickers = config_dict.get("tickers", [])
    initial_cash = config_dict.get("broker", {}).get("initial_cash", 100000.0)
    
    if not invest_quarter:
        print(f"  Missing invest_quarter in log config, skipping.")
        return False

    # Try to extract risk-free rate from the correct prior-quarter memo
    risk_free_rate = log_data.get("config", {}).get("risk_free_rate", 0.0)
    year, q_label = _parse_invest_quarter(invest_quarter)
    p_year, p_q = _prior_quarter(year, q_label)
    memo_path = Path(f"data-pipeline/final_snapshots/memo_data/memo_{p_year}_{p_q}.txt")
    
    if memo_path.exists():
        with open(memo_path, "r") as f:
            memo_text = f.read()
        evidence = parse_memo_evidence(memo_text)
        if "L1-FF" in evidence:
            match = re.search(r"([\d\.]+)", evidence["L1-FF"])
            if match:
                risk_free_rate = float(match.group(1)) / 100.0

    # Load exit prices from rebuilt snapshots to fix zero final_prices
    snapshot_path = Path("data-pipeline/final_snapshots/json_data") / f"snapshot_{year}_{q_label}.json"
    snapshot_prices = {}
    if snapshot_path.exists():
        with open(snapshot_path, "r") as f:
            snap = json.load(f)
        for t in tickers:
            td = snap.get("ticker_data", {}).get(t, {})
            af = td.get("asset_features", {})
            if af and af.get("close") is not None:
                snapshot_prices[t] = af["close"]
    
    # Load daily prices (logic borrowed from AsyncSimulationRunner)
    daily_dir = Path("data-pipeline/daily_prices/data")
    
    daily_prices = {}
    for t in tickers:
        p = daily_dir / t / f"{year}_{q_label}.json"
        if p.exists():
            with open(p, "r") as f:
                doc = json.load(f)
            bars = [
                ClosePricePoint(timestamp=d["date"], close=d["close"])
                for d in doc.get("daily_close", [])
            ]
            if bars:
                daily_prices[t] = bars
    
    spy_daily = None
    spy_path = daily_dir / "SPY" / f"{year}_{q_label}.json"
    if spy_path.exists():
        with open(spy_path, "r") as f:
            doc = json.load(f)
        spy_daily = [
            ClosePricePoint(timestamp=d["date"], close=d["close"])
            for d in doc.get("daily_close", [])
        ]

    # Process each episode
    episode_logs_data = log_data.get("episode_logs", [])
    for ep_summary in summary.get("episode_summaries", []):
        ep_id = ep_summary.get("episode_id")
        
        # 1. Fix final_prices, book_value, return_pct if zero prices found
        final_prices = ep_summary.get("final_prices", {})
        has_zero_prices = any(v == 0.0 for k, v in final_prices.items() if k != "_CASH_")
        
        if has_zero_prices and snapshot_prices:
            for t in final_prices:
                if final_prices[t] == 0.0 and t in snapshot_prices:
                    final_prices[t] = snapshot_prices[t]
            
            # Recalculate book value
            final_pos = ep_summary.get("final_positions", {})
            book_val = ep_summary.get("final_cash", 0.0) + sum(
                qty * final_prices.get(t, 0.0) for t, qty in final_pos.items()
            )
            ep_summary["book_value"] = book_val
            
            # Recalculate return_pct_with_cash_interest using the extracted rate
            n_days = len(daily_prices.get(next(iter(daily_prices)), [])) if daily_prices else 63
            quarterly_rate = (1 + risk_free_rate)**(n_days / 252.0) - 1.0
            interest_wealth = ep_summary.get("final_cash", 0.0) * quarterly_rate
            total_wealth = book_val + interest_wealth
            
            ep_summary["return_pct"] = ((book_val - initial_cash) / initial_cash) * 100
            ep_summary["return_pct_with_cash_interest"] = ((total_wealth - initial_cash) / initial_cash) * 100
            ep_summary["final_prices"] = final_prices
            print(f"  Fixed zero prices and recalculated book_value for {ep_id}")

        # 2. Fix daily_metrics
        if "daily_metrics" not in ep_summary or force:
            ep_log_dict = next((e for e in episode_logs_data if e.get("episode_id") == ep_id), None)
            if ep_log_dict:
                # We can just get positions and cash from the summary directly
                positions = ep_summary.get("final_positions", {})
                cash = ep_summary.get("final_cash", initial_cash)

                daily_fin = compute_daily_financial_metrics(
                    positions=positions, cash=cash, initial_value=initial_cash,
                    daily_prices=daily_prices, risk_free_rate=risk_free_rate, spy_daily=spy_daily,
                )
                if daily_fin:
                    ep_summary["daily_metrics"] = {
                        "trading_days": daily_fin.trading_days,
                        "total_return_pct": daily_fin.total_return_pct,
                        "annualized_sharpe": daily_fin.annualized_sharpe,
                        "annualized_sortino": daily_fin.annualized_sortino,
                        "annualized_volatility": daily_fin.annualized_volatility,
                        "max_drawdown_pct": daily_fin.max_drawdown_pct,
                        "calmar_ratio": daily_fin.calmar_ratio,
                        "spy_return_pct": daily_fin.spy_return_pct,
                        "excess_return_pct": daily_fin.excess_return_pct,
                    }
                    print(f"  Fixed daily_metrics for {ep_id} (using RF rate: {risk_free_rate*100:.2f}%)")
                else:
                    # Fallback to update return_pct_with_cash_interest even if daily fails
                    pass

        # If force, recalculate return_pct_with_cash_interest just in case
        if force:
            book_val = ep_summary.get("book_value", initial_cash)
            n_days = len(daily_prices.get(next(iter(daily_prices)), [])) if daily_prices else 63
            quarterly_rate = (1 + risk_free_rate)**(n_days / 252.0) - 1.0
            interest_wealth = ep_summary.get("final_cash", 0.0) * quarterly_rate
            total_wealth = book_val + interest_wealth
            ep_summary["return_pct_with_cash_interest"] = ((total_wealth - initial_cash) / initial_cash) * 100

    # Write back
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Repair missing or incorrect metrics in summary.json files.")
    parser.add_argument("--results-dir", default="results", help="Directory to search for summary.json files.")
    parser.add_argument("--force", action="store_true", help="Force repair even if daily_metrics exists.")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results directory {results_root} not found.")
        return

    # Find all directories containing summary.json
    count = 0
    for summary_file in results_root.rglob("summary.json"):
        if repair_summary(summary_file.parent, force=args.force):
            count += 1
    
    print(f"\nRepaired {count} summary.json files.")

if __name__ == "__main__":
    main()
