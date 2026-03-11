import json
import os
from pathlib import Path
import glob
import re

import sys
# Ensure we can import from eval and models
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.financial import compute_daily_financial_metrics
from models.case import ClosePricePoint
from eval.evidence import parse_memo_evidence

_TRACE_INDEX = None

def _build_trace_index():
    global _TRACE_INDEX
    if _TRACE_INDEX is not None:
        return _TRACE_INDEX
    
    _TRACE_INDEX = {}
    traces = glob.glob("traces/*.json")
    for t in traces:
        try:
            with open(t, 'r') as f:
                data = json.load(f)
            if "trace" in data and "logged_at" in data["trace"]:
                _TRACE_INDEX[data["trace"]["logged_at"]] = t
        except Exception:
            pass
    return _TRACE_INDEX

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

def get_or_compute_mean_revisions(run_dir: str | Path) -> dict:
    """
    Returns the mean_revisions_metrics for a given scenario run directory.
    If it doesn't exist in summary.json, it computes it from the trace logs and saves it.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "simulation_log.json"
    
    if not summary_path.exists() or not log_path.exists():
        return {}

    with open(summary_path, "r") as f:
        summary = json.load(f)
        
    ep = summary.get("episode_summaries", [])[0]
    if "mean_revisions_metrics" in ep:
        return ep["mean_revisions_metrics"]
        
    # Need to compute it
    print(f"  Computing mean revisions for {run_dir.name}...")
    with open(log_path, "r") as f:
        log_data = json.load(f)
        
    ep_log = log_data.get("episode_logs", [])[0]
    dp_log = ep_log.get("decision_point_logs", [])[0]
    agent_output = dp_log.get("agent_output", {})
    debate_trace = agent_output.get("debate_trace", {})
    logged_at = debate_trace.get("logged_at")
    
    if not logged_at:
        print(f"    Warning: No logged_at found in simulation_log.json")
        return {}
        
    index = _build_trace_index()
    trace_path = index.get(logged_at)
    if not trace_path:
        print(f"    Warning: Could not find trace file for logged_at={logged_at}")
        return {}
        
    with open(trace_path, "r") as f:
        full_trace = json.load(f)
        
    debate_turns = full_trace.get("debate_turns", [])
    if not debate_turns:
        print(f"    Warning: No debate_turns found in trace file {trace_path}")
        return {}
        
    # Find last proposal/revision for each role
    final_allocations = {}
    for turn in debate_turns:
        t_type = turn.get("type")
        if t_type in ("proposal", "revision"):
            role = turn.get("role")
            alloc = turn.get("content", {}).get("allocation")
            if alloc:
                final_allocations[role] = alloc
                
    if not final_allocations:
        print(f"    Warning: No agent allocations found in trace.")
        return {}
        
    # Compute mean allocation
    mean_alloc = {}
    num_agents = len(final_allocations)
    for alloc in final_allocations.values():
        for t, w in alloc.items():
            mean_alloc[t] = mean_alloc.get(t, 0.0) + w / num_agents
            
    # Normalize mean allocation to 1.0 just in case
    total_w = sum(mean_alloc.values())
    if total_w > 0:
        for t in mean_alloc:
            mean_alloc[t] /= total_w

    # Load market data
    config_dict = log_data.get("config", {})
    invest_quarter = config_dict.get("invest_quarter")
    tickers = config_dict.get("tickers", [])
    initial_cash = config_dict.get("broker", {}).get("initial_cash", 100000.0)
    
    year, q_label = _parse_invest_quarter(invest_quarter)
    p_year, p_q = _prior_quarter(year, q_label)
    
    # Extract risk-free rate
    risk_free_rate = config_dict.get("risk_free_rate", 0.0)
    memo_path = Path(f"data-pipeline/final_snapshots/memo_data/memo_{p_year}_{p_q}.txt")
    if memo_path.exists():
        with open(memo_path, "r") as f:
            memo_text = f.read()
        evidence = parse_memo_evidence(memo_text)
        if "L1-FF" in evidence:
            match = re.search(r"([\d\.]+)", evidence["L1-FF"])
            if match:
                risk_free_rate = float(match.group(1)) / 100.0

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
        
    initial_prices = debate_trace.get("initial_market_state", {}).get("prices", {})
    if not initial_prices:
        # Fallback to daily_prices first day
        for t, bars in daily_prices.items():
            if bars:
                initial_prices[t] = bars[0].close
                
    # Calculate positions and cash
    positions = {}
    cash = 0.0
    for t, weight in mean_alloc.items():
        if t == "_CASH_":
            cash += weight * initial_cash
        else:
            p = initial_prices.get(t)
            if p and p > 0:
                shares = (weight * initial_cash) / p
                positions[t] = shares
                
    # Compute metrics
    daily_fin = compute_daily_financial_metrics(
        positions=positions, cash=cash, initial_value=initial_cash,
        daily_prices=daily_prices, risk_free_rate=risk_free_rate, spy_daily=spy_daily,
    )
    
    if not daily_fin:
        print(f"    Warning: Failed to compute daily metrics")
        return {}
        
    metrics = {
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
    
    # Save back to summary.json
    ep["mean_revisions_metrics"] = metrics
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    return metrics
