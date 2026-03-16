#!/usr/bin/env python3
"""Compare debate portfolio returns against an equal-weight benchmark.

For each completed run across ablations 7, 8, 10 (and 1), computes the
equal-weight portfolio return over the same invest_quarter and ticker
universe, then reports how often the debate system beats 1/N.

Usage:
    python analysis/vs_equal_weight.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "logging" / "runs"
PRICE_DIR = REPO_ROOT / "data-pipeline" / "daily_prices" / "data"

ABLATIONS = [
    ("Ablation 1",  "vskarich_ablation_1"),
    ("Ablation 7",  "vskarich_ablation_7"),
    ("Ablation 8",  "vskarich_ablation_8"),
    ("Ablation 10", "vskarich_ablation_10"),
]


def quarter_to_price_key(q: str) -> str:
    """Convert '2022Q4' to '2022_Q4'."""
    return q[:4] + "_" + q[4:]


def date_to_quarter(d: str) -> str:
    """Convert '2022-09-30' to '2022Q4' (invest_quarter = start of quarter)."""
    year, month, _ = d.split("-")
    month = int(month)
    if month <= 3:
        return f"{year}Q1"
    elif month <= 6:
        return f"{year}Q2"
    elif month <= 9:
        return f"{year}Q3"
    else:
        return f"{year}Q4"


def load_quarter_return(ticker: str, quarter: str) -> float | None:
    """Load total return for ticker over quarter from daily price data."""
    price_key = quarter_to_price_key(quarter)
    price_file = PRICE_DIR / ticker / f"{price_key}.json"
    if not price_file.exists():
        return None
    data = json.loads(price_file.read_text())
    closes = data.get("daily_close", [])
    if len(closes) < 2:
        return None
    first_close = closes[0]["close"]
    last_close = closes[-1]["close"]
    return (last_close - first_close) / first_close * 100.0


def compute_ew_return(tickers: list[str], quarter: str) -> float | None:
    """Compute equal-weight portfolio return for tickers over quarter."""
    returns = []
    for t in tickers:
        r = load_quarter_return(t, quarter)
        if r is not None:
            returns.append(r)
    if not returns:
        return None
    return np.mean(returns)


def compute_ew_sharpe(tickers: list[str], quarter: str) -> float | None:
    """Compute equal-weight portfolio Sharpe (daily, annualized) for quarter."""
    # Load daily closes for all tickers
    price_key = quarter_to_price_key(quarter)
    all_closes = {}
    for t in tickers:
        price_file = PRICE_DIR / t / f"{price_key}.json"
        if not price_file.exists():
            continue
        data = json.loads(price_file.read_text())
        closes = data.get("daily_close", [])
        if len(closes) < 2:
            continue
        all_closes[t] = [c["close"] for c in closes]

    if not all_closes:
        return None

    # Align lengths
    min_len = min(len(v) for v in all_closes.values())
    for t in all_closes:
        all_closes[t] = all_closes[t][:min_len]

    # Compute daily portfolio returns (equal weight)
    n_tickers = len(all_closes)
    daily_port_returns = []
    for i in range(1, min_len):
        day_ret = 0.0
        for t in all_closes:
            day_ret += (all_closes[t][i] / all_closes[t][i - 1] - 1.0) / n_tickers
        daily_port_returns.append(day_ret)

    if len(daily_port_returns) < 5:
        return None

    arr = np.array(daily_port_returns)
    mean_r = np.mean(arr)
    std_r = np.std(arr, ddof=1)
    if std_r == 0:
        return None
    return float((mean_r / std_r) * np.sqrt(252))


def load_runs(experiment: str) -> list[dict]:
    exp_dir = RUNS_DIR / experiment
    if not exp_dir.exists():
        return []

    runs = []
    for run_dir in sorted(exp_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        fin_path = run_dir / "_dashboard" / "financial_metrics.json"
        if not manifest_path.exists() or not fin_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        fin = json.loads(fin_path.read_text())

        config_paths = manifest.get("config_paths", [])
        debate_config = config_paths[0] if config_paths else ""
        is_baseline = "baseline" in Path(debate_config).stem.lower()

        # Get scenario config to extract tickers and invest_quarter
        scenario_path = Path(config_paths[1]) if len(config_paths) > 1 else None
        if scenario_path is None or not scenario_path.exists():
            continue

        with open(scenario_path) as f:
            scenario = yaml.safe_load(f)

        tickers = scenario.get("tickers", [])
        invest_quarter = scenario.get("invest_quarter", "")

        # manifest stores invest_quarter as date string
        if not invest_quarter and manifest.get("invest_quarter"):
            invest_quarter = date_to_quarter(manifest["invest_quarter"])

        if not tickers or not invest_quarter:
            continue

        runs.append({
            "condition": "baseline" if is_baseline else "treatment",
            "scenario": Path(config_paths[1]).stem,
            "tickers": tickers,
            "quarter": invest_quarter,
            "sharpe": fin["daily_metrics_annualized_sharpe"],
            "total_return": fin["daily_metrics_total_return_pct"],
        })

    return runs


def main() -> None:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    w("=" * 90)
    w("  DEBATE PORTFOLIOS vs EQUAL-WEIGHT (1/N) BENCHMARK")
    w("=" * 90)
    w()

    # Cache EW returns/sharpes per (quarter, ticker_tuple)
    ew_cache_ret: dict[tuple, float] = {}
    ew_cache_sharpe: dict[tuple, float] = {}

    all_results = []

    for label, experiment in ABLATIONS:
        runs = load_runs(experiment)
        if not runs:
            w(f"  {label}: no data found")
            w()
            continue

        results = []
        for r in runs:
            key = (r["quarter"], tuple(sorted(r["tickers"])))
            if key not in ew_cache_ret:
                ew_cache_ret[key] = compute_ew_return(list(key[1]), r["quarter"])
                ew_cache_sharpe[key] = compute_ew_sharpe(list(key[1]), r["quarter"])

            ew_ret = ew_cache_ret[key]
            ew_sharpe = ew_cache_sharpe[key]
            if ew_ret is None:
                continue

            results.append({
                "condition": r["condition"],
                "scenario": r["scenario"],
                "quarter": r["quarter"],
                "port_return": r["total_return"],
                "ew_return": ew_ret,
                "excess_vs_ew": r["total_return"] - ew_ret,
                "port_sharpe": r["sharpe"],
                "ew_sharpe": ew_sharpe,
                "sharpe_diff": r["sharpe"] - (ew_sharpe or 0),
            })

        if not results:
            continue

        all_results.extend([(label, r) for r in results])

        port_rets = np.array([r["port_return"] for r in results])
        ew_rets = np.array([r["ew_return"] for r in results])
        excess = port_rets - ew_rets
        port_sharpes = np.array([r["port_sharpe"] for r in results])
        ew_sharpes = np.array([r["ew_sharpe"] for r in results if r["ew_sharpe"] is not None])

        beats_ew_ret = np.sum(excess > 0)
        n = len(results)

        t_ret, p_ret = stats.ttest_1samp(excess, 0)

        w(f"  {label}  (n={n})")
        w(f"  {'-' * 80}")
        w(f"  {'':30} {'Debate':>12} {'Equal-Wt':>12} {'Diff':>10} {'p':>8}")
        w(f"  {'Mean Return %':<30} {np.mean(port_rets):>+12.3f} {np.mean(ew_rets):>+12.3f} {np.mean(excess):>+10.3f} {p_ret:>8.4f}")

        # Sharpe comparison
        sharpe_diffs = np.array([r["sharpe_diff"] for r in results if r["ew_sharpe"] is not None])
        if len(sharpe_diffs) > 2:
            t_sh, p_sh = stats.ttest_1samp(sharpe_diffs, 0)
            port_sh_mean = np.mean([r["port_sharpe"] for r in results if r["ew_sharpe"] is not None])
            ew_sh_mean = np.mean([r["ew_sharpe"] for r in results if r["ew_sharpe"] is not None])
            w(f"  {'Mean Sharpe':<30} {port_sh_mean:>+12.3f} {ew_sh_mean:>+12.3f} {np.mean(sharpe_diffs):>+10.3f} {p_sh:>8.4f}")

        w()
        w(f"  Beats EW on return:  {beats_ew_ret}/{n} ({beats_ew_ret/n*100:.0f}%)")

        # Binomial test: is win rate significantly different from 50%?
        binom_p = stats.binom_test(beats_ew_ret, n, 0.5) if hasattr(stats, 'binom_test') else stats.binomtest(beats_ew_ret, n, 0.5).pvalue
        w(f"  Binomial test (win rate vs 50%): p={binom_p:.4f}")

        # By condition
        for cond in ["baseline", "treatment"]:
            cond_results = [r for r in results if r["condition"] == cond]
            if not cond_results:
                continue
            cond_excess = np.array([r["excess_vs_ew"] for r in cond_results])
            cond_beats = np.sum(cond_excess > 0)
            cond_n = len(cond_results)
            t_c, p_c = stats.ttest_1samp(cond_excess, 0)
            w(f"    {cond:>12}: excess={np.mean(cond_excess):+.3f}%, beats={cond_beats}/{cond_n} ({cond_beats/cond_n*100:.0f}%), p={p_c:.4f}")

        w()

    # Pooled summary
    if all_results:
        w("=" * 90)
        w("  POOLED ACROSS ALL ABLATIONS")
        w("=" * 90)
        all_excess = np.array([r["excess_vs_ew"] for _, r in all_results])
        all_beats = np.sum(all_excess > 0)
        total_n = len(all_results)
        t_all, p_all = stats.ttest_1samp(all_excess, 0)
        w(f"  n={total_n}")
        w(f"  Mean excess vs EW: {np.mean(all_excess):+.3f}%  (t={t_all:+.3f}, p={p_all:.4f})")
        w(f"  Beats EW: {all_beats}/{total_n} ({all_beats/total_n*100:.0f}%)")
        binom_all = stats.binomtest(all_beats, total_n, 0.5).pvalue
        w(f"  Binomial test: p={binom_all:.4f}")
        w()

        # Sharpe pooled
        all_sharpe_diffs = np.array([r["sharpe_diff"] for _, r in all_results if r["ew_sharpe"] is not None])
        if len(all_sharpe_diffs) > 2:
            t_sh_all, p_sh_all = stats.ttest_1samp(all_sharpe_diffs, 0)
            w(f"  Mean Sharpe diff vs EW: {np.mean(all_sharpe_diffs):+.3f}  (t={t_sh_all:+.3f}, p={p_sh_all:.4f})")
        w()

    report = "\n".join(lines)
    out_path = REPO_ROOT / "analysis" / "vs_equal_weight_report.txt"
    out_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
