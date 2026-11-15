#!/usr/bin/env python3
"""
Agent-level CRIT vs. each agent's OWN portfolio outcome (camera-ready, Reviewer SnBG / Task 5)
============================================================================================

Question. RQ1 correlates the group score rho_bar with the outcome of the portfolio the LLM judge
(or mean) produced. If CRIT is a valid verifier but the judge's aggregation breaks the link, then
each agent's own score rho_i should track the outcome of that agent's OWN final allocation.

Prediction stated before running. If agent-level r (pooled or within-scenario) has a
scenario-clustered 95% CI excluding 0 in the positive direction, the RQ1 null reflects aggregation
rather than verifier invalidity, and the paper must say so. If the CIs include 0, aggregation does
not explain the null.

Data (all in CS372_210_debate_runs/, the checksummed 210-run package):
  rho_i       final crit_round in final/pid_crit_all_rounds.json -> crit.agents[role].rho_i
              (the same file and round that produce rho_bar in analysis/rq1/per_run_results.csv)
  allocation  the agent's last revision: rounds/round_001/revisions_retry_NNN/<role>/portfolio.json
              if any retry exists for that role, else rounds/round_001/revisions/<role>/portfolio.json
  outcome     buy-and-hold backtest over the invest quarter, computed exactly as
              tools/dashboard/run_scanner.py::_compute_run_financial_metrics computes the stored
              judge-portfolio metrics (same daily prices and integer shares), with the risk-free
              rate read from the run's own shared_context/memo.txt.

Sanity check (fails loudly): running the same backtest on final/final_portfolio.json must reproduce
the stored _dashboard/financial_metrics.json Sharpe and return for all 210 runs.

Units of analysis: 490 agent-runs (A7: 70x2, A8: 70x3, A10: 70x2), 35 scenario clusters.
Statistics: pooled r; within-scenario r (demean by scenario); within-run r (demean by run, so
agents are compared with their own co-agents on the same run). 95% CIs: scenario-cluster bootstrap,
10,000 draws, seed 20261030 (as in rq1_clustered_stats.py).

Usage (repo root):
    uv run --no-project --with numpy --with pandas --with pyyaml python analysis/agent_level_crit.py
Writes analysis/rq1/agent_level_results.csv and analysis/rq1/agent_level_stats.json.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eval.evidence import parse_memo_evidence  # noqa: E402
from eval.financial import compute_daily_financial_metrics  # noqa: E402
from models.case import ClosePricePoint  # noqa: E402

PKG = REPO / "CS372_210_debate_runs"
RUNS_CSV = REPO / "analysis" / "rq1" / "per_run_results.csv"
FRESH = REPO / "analysis" / "task1_camera_ready" / "results" / "per_run_results_fresh.csv"
OUT_CSV = REPO / "analysis" / "rq1" / "agent_level_results.csv"
OUT_JSON = REPO / "analysis" / "rq1" / "agent_level_stats.json"
DAILY = REPO / "data-pipeline" / "daily_prices" / "data"
SEED = 20261030
N_BOOT = 10_000


def backtest(run_dir: Path, weights: dict[str, float]) -> dict[str, float]:
    """Mirror of run_scanner._compute_run_financial_metrics (judge mode, integer shares)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    universe = manifest["ticker_universe"]
    scen = deb = None
    for yf in (run_dir / "final").glob("*.yaml"):
        c = yaml.safe_load(yf.read_text())
        if isinstance(c, dict) and "invest_quarter" in c:
            scen = c
        elif isinstance(c, dict) and "broker" in c:
            deb = c
    assert scen is not None and deb is not None, run_dir
    iq = str(scen["invest_quarter"])
    year, q = int(iq[:4]), iq[4:]
    initial_cash = deb.get("broker", {}).get("initial_cash", 100_000.0)

    prices = {}
    for t in universe:
        if t == "_CASH_":
            continue
        p = DAILY / t / f"{year}_{q}.json"
        if p.exists():
            bars = [ClosePricePoint(timestamp=d["date"], close=d["close"])
                    for d in json.loads(p.read_text()).get("daily_close", [])]
            if bars:
                prices[t] = bars
    spy_p = DAILY / "SPY" / f"{year}_{q}.json"
    spy = [ClosePricePoint(timestamp=d["date"], close=d["close"])
           for d in json.loads(spy_p.read_text())["daily_close"]] if spy_p.exists() else None

    # Risk-free rate: Fed Funds [L1-FF] from the memo this run actually received. (The scanner reads
    # data-pipeline/final_snapshots/memo_data/, where some quarters' memos are missing locally and it
    # silently falls back to 0; the run's own copy is the faithful source.)
    ev = parse_memo_evidence((run_dir / "shared_context" / "memo.txt").read_text())
    m = re.search(r"([\d.]+)", ev["L1-FF"])
    rf = float(m.group(1)) / 100.0

    p0 = {t: b[0].close for t, b in prices.items()}
    cash = weights.get("_CASH_", 0.0) * initial_cash
    pos = {}
    for t, w in weights.items():
        if t == "_CASH_" or w <= 0 or not p0.get(t):
            continue
        sh = int(w * initial_cash / p0[t])
        if sh > 0:
            pos[t] = sh
    assert pos, f"no positions for {run_dir}"
    with redirect_stdout(io.StringIO()):  # library prints an interest line per call
        d = compute_daily_financial_metrics(positions=pos, cash=cash, initial_value=initial_cash,
                                            daily_prices=prices, spy_daily=spy, risk_free_rate=rf)
    return {"sharpe": d.annualized_sharpe or 0.0, "total_return": d.total_return_pct}


def final_allocation(run_dir: Path, role: str) -> dict[str, float]:
    rd = run_dir / "rounds" / "round_001"
    retries = sorted(rd.glob(f"revisions_retry_*/{role}/portfolio.json"))
    path = retries[-1] if retries else rd / "revisions" / role / "portfolio.json"
    a = json.loads(path.read_text())
    assert isinstance(a, dict) and abs(sum(a.values()) - 1.0) < 0.02, (path, sum(a.values()))
    return a


def _r(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _demean(v, labels):
    return v - pd.Series(v).groupby(labels).transform("mean").to_numpy()


def stats(df: pd.DataFrame, score: str, outcome: str) -> dict:
    scen = df.scenario_id.to_numpy()
    runs = df.run_id.to_numpy()
    x = df[score].to_numpy(float)
    y = df[outcome].to_numpy(float)
    out = {}
    variants = {
        "pooled": lambda idx, sl, rl: _r(x[idx], y[idx]),
        "within_scenario": lambda idx, sl, rl: _r(_demean(x[idx], sl), _demean(y[idx], sl)),
        "within_run": lambda idx, sl, rl: _r(_demean(x[idx], rl), _demean(y[idx], rl)),
    }
    uniq = np.unique(scen)
    members = {s: np.flatnonzero(scen == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(N_BOOT)]
    for name, f in variants.items():
        point = f(np.arange(len(x)), scen, runs)
        boot = []
        for d in draws:
            idx = np.concatenate([members[s] for s in d])
            # relabel resampled clusters so duplicated scenarios/runs stay distinct clusters
            sl = np.concatenate([np.full(len(members[s]), k) for k, s in enumerate(d)])
            rl = np.array([f"{k}:{r}" for k, s in enumerate(d) for r in runs[members[s]]])
            boot.append(f(idx, sl, rl))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out[name] = {"r": point, "ci95": [float(lo), float(hi)], "n": int(len(x)),
                     "n_clusters": int(len(uniq))}
    return out


def main() -> int:
    runs = pd.read_csv(RUNS_CSV)
    assert len(runs) == 210
    fresh = pd.read_csv(FRESH).set_index("path")  # Alanood's fresh gpt-5 re-scoring (Task 1)
    assert len(fresh) == 210

    rows, max_dev = [], {"sharpe": 0.0, "total_return": 0.0}
    for r in runs.itertuples():
        rd = PKG / r.run_path
        # --- sanity: reproduce stored judge-portfolio metrics ---
        judge = json.loads((rd / "final" / "final_portfolio.json").read_text())
        judge = judge.get("allocation", judge)
        stored = json.loads((rd / "_dashboard" / "financial_metrics.json").read_text())
        got = backtest(rd, judge)
        for k, sk in (("sharpe", "daily_metrics_annualized_sharpe"),
                      ("total_return", "daily_metrics_total_return_pct")):
            max_dev[k] = max(max_dev[k], abs(got[k] - stored[sk]))
        # --- agents ---
        crit = [c for c in json.loads((rd / "final" / "pid_crit_all_rounds.json").read_text())
                if c.get("type") == "crit_round"][-1]["crit"]
        assert abs(crit["rho_bar"] - r.rho_bar) < 1e-9, rd
        for role, a in crit["agents"].items():
            o = backtest(rd, final_allocation(rd, role))
            row = {"run_id": r.run_id, "run_path": r.run_path, "scenario_id": r.scenario_id,
                   "ablation_id": r.ablation_id, "condition": r.condition, "role": role,
                   "rho_i": a["rho_i"], "own_sharpe": o["sharpe"],
                   "own_total_return": o["total_return"]}
            if fresh is not None:
                row["rho_i_fresh"] = fresh.loc[r.run_path, f"rho_{role}"]
            rows.append(row)

    assert max_dev["sharpe"] < 1e-6 and max_dev["total_return"] < 1e-6, \
        f"backtest does not reproduce stored judge metrics: {max_dev}"
    df = pd.DataFrame(rows)
    assert len(df) == 490, len(df)
    if "rho_i_fresh" in df:
        assert df.rho_i_fresh.notna().all()
    df.to_csv(OUT_CSV, index=False)

    res = {"config": {"seed": SEED, "n_boot": N_BOOT, "cluster": "scenario_id",
                      "judge_reproduction_max_abs_dev": max_dev, "n_agent_runs": len(df)}}
    for score in ["rho_i"] + (["rho_i_fresh"] if "rho_i_fresh" in df else []):
        for outcome in ["own_sharpe", "own_total_return"]:
            res[f"{score}:{outcome}"] = stats(df, score, outcome)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"judge backtest reproduction, max |dev|: {max_dev}")
    for k, v in res.items():
        if k == "config":
            continue
        for name, s in v.items():
            print(f"{k:28s} {name:16s} r={s['r']:+.3f} 95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] n={s['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
