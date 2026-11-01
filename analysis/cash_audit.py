#!/usr/bin/env python3
"""
Where does the "uninvested cash" in Tables 2 and 4 come from?
=============================================================
camera_ready_plan.md item 3.2. Prompts say "fully invested, no cash reserve",
but the tables report 7.5-17% uninvested cash.

Mechanism (code, not data):
  * Every baseline_sweep scenario sets ``use_cash_virtual_ticker: true``, so a
    ``_CASH_`` ticker is appended to the allocation universe
    (``simulation/runner.py``) and agents are told it "earns the risk-free rate"
    (``multi_agent/graph/nodes.py``). Weights summing to 1 may include it.
  * ``agents/multi_agent_debate.py:allocation_to_decision`` skips ``_CASH_``,
    skips tickers whose entry price is missing (0.0), and buys
    ``int(weight * cash / price)`` shares. All three leave cash at the broker.

The per-agent allocations are not stored for these runs (log dirs absent), so
the split is bounded, not measured exactly:
  rounding  <= sum of entry prices of tickers with a valid price
               (each ticker leaves less than one share of cash)
  zero-price tickers: their (unknown) weight also becomes cash; counted.
  _CASH_ / deliberate cash >= final_cash - rounding bound   (when no
               zero-price ticker is in the universe)

Self-check: initial_cash - sum(shares * entry_price) is compared with the
stored ``final_cash``. Most runs match to the cent; a few differ by tens of
dollars because the daily price files were re-downloaded after the runs. The
gap is recorded per run and the script stops if any gap exceeds 1% of cash.

Usage:
    python3 analysis/cash_audit.py
"""

import csv
import json
import statistics
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = REPO_ROOT / "data-pipeline" / "daily_prices" / "data"
OUT_PATH = REPO_ROOT / "analysis" / "cash_audit.json"

# Same tracking CSVs as analysis/verify_paper_tables.py.
CONFIGS = [
    ("T2-A Single Agent, Basic", "results_tracking_single_agent_slim.csv"),
    ("T2-B 3 Roles Mean, Basic", "results_tracking_vote_slim_no_macro.csv"),
    ("T2-C / T4-A 3 Roles Mean, Causal", "results_tracking_vote_causal_no_macro_causal_out_fix_scenario.csv"),
    ("T2-D 3 Roles Mean, Enriched", "results_tracking_vote_no_macro_enriched.csv"),
    ("T4-B 3 Roles Debate, Causal", "results_tracking_debate_1_round_no_macro_causal_out.csv"),
    ("T4-C Debate, Causal + RCA", "results_tracking_debate_1_round_no_macro_causal_out_rca_revise.csv"),
]


def prior_quarter(q):
    y, n = int(q[:4]), int(q[5])
    return (y - 1, 4) if n == 1 else (y, n - 1)


_px_cache = {}


def entry_prices(invest_quarter, tickers):
    """Entry price = last daily close of the prior quarter, rounded to cents.

    This is the value the snapshot builder stores as ``asset_features.close``
    (checked: AAPL 2021Q4 173.759 -> 173.76). The local snapshot JSONs cannot be
    used: they are git-ignored and were regenerated on 2026-08-29 for the
    seed-variance scenarios only, so most tickers are missing from them.
    A missing file gives None, which the broker would have seen as price 0.0.
    """
    y, n = prior_quarter(invest_quarter)
    out = {}
    for t in tickers:
        key = (t, y, n)
        if key not in _px_cache:
            f = DAILY_DIR / t / f"{y}_Q{n}.json"
            bars = json.loads(f.read_text())["daily_close"] if f.exists() else []
            _px_cache[key] = round(float(bars[-1]["close"]), 2) if bars else None
        out[t] = _px_cache[key]
    return out


def audit_run(scenario_yaml, results_dir):
    sc = yaml.safe_load((REPO_ROOT / scenario_yaml).read_text())
    ep = json.loads((REPO_ROOT / results_dir / "summary.json").read_text())["episode_summaries"][0]
    universe = [t for t in sc["tickers"] if t != "_CASH_"]
    px = entry_prices(sc["invest_quarter"], universe)
    priced = {t: float(px[t]) for t in universe if px.get(t) not in (None, 0, 0.0)}
    zero_price = sorted(set(universe) - set(priced))
    if any(t not in priced for t in ep["final_positions"]):
        return None  # entry prices unavailable (2021Q4 scenarios need 2021Q3 data)
    invested = sum(q * priced[t] for t, q in ep["final_positions"].items())
    implied = ep["initial_cash"] - invested
    gap = abs(implied - ep["final_cash"])
    if gap > 0.01 * ep["initial_cash"]:
        raise AssertionError(f"{results_dir}: entry prices do not reproduce final_cash "
                             f"({implied:.2f} vs {ep['final_cash']:.2f})")
    init = ep["initial_cash"]
    rounding_bound = min(sum(priced.values()), ep["final_cash"])
    return {
        "cash_pct": 100 * ep["final_cash"] / init,
        "rounding_bound_pct": 100 * rounding_bound / init,
        "deliberate_lower_bound_pct": 100 * max(ep["final_cash"] - rounding_bound, 0) / init,
        "n_zero_price": len(zero_price),
        "zero_price": zero_price,
        "n_universe": len(universe),
        "reproduction_gap_pct": 100 * gap / init,
    }


def main():
    out = {}
    for label, csv_name in CONFIGS:
        runs, excluded = [], []
        with open(REPO_ROOT / csv_name) as fh:
            for row in csv.DictReader(fh):
                if row.get("error") or not row.get("results_dir") or row["results_dir"] == "N/A":
                    continue
                r = audit_run(row["scenario"], row["results_dir"])
                if r is None:
                    excluded.append(Path(row["scenario"]).stem)
                else:
                    runs.append(r)
        clean = [r for r in runs if r["n_zero_price"] == 0]
        rec = {
            "n_runs": len(runs),
            "excluded_no_entry_prices": excluded,
            "mean_cash_pct": statistics.mean(r["cash_pct"] for r in runs),
            "mean_rounding_bound_pct": statistics.mean(r["rounding_bound_pct"] for r in runs),
            "runs_with_zero_price_ticker": len(runs) - len(clean),
            "zero_price_tickers": sorted({t for r in runs for t in r["zero_price"]}),
            "max_reproduction_gap_pct": max(r["reproduction_gap_pct"] for r in runs),
            "runs_exact_to_cent": sum(r["reproduction_gap_pct"] < 1e-5 for r in runs),
            "clean_runs": len(clean),
            "clean_mean_cash_pct": statistics.mean(r["cash_pct"] for r in clean) if clean else None,
            "clean_mean_deliberate_lower_bound_pct":
                statistics.mean(r["deliberate_lower_bound_pct"] for r in clean) if clean else None,
            "clean_runs_cash_over_rounding_bound":
                sum(r["deliberate_lower_bound_pct"] > 0 for r in clean),
        }
        out[label] = rec
        print(f"{label:<34} n={rec['n_runs']:>2} (excl {len(excluded)})  cash={rec['mean_cash_pct']:5.2f}%  "
              f"rounding<={rec['mean_rounding_bound_pct']:5.2f}%  zero-price runs={rec['runs_with_zero_price_ticker']:>2} "
              f"{rec['zero_price_tickers']}  clean: cash={rec['clean_mean_cash_pct']:5.2f}% "
              f"deliberate>={rec['clean_mean_deliberate_lower_bound_pct']:5.2f}% "
              f"({rec['clean_runs_cash_over_rounding_bound']}/{rec['clean_runs']} runs above rounding bound)  "
              f"exact={rec['runs_exact_to_cent']}/{rec['n_runs']} maxgap={rec['max_reproduction_gap_pct']:.3f}%")
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
