#!/usr/bin/env python3
"""
Scenario-aware test of "debate beats 1/N only 33% of the time"
==============================================================
camera_ready_plan.md item T2g. The paper's binomial p<0.001 treats 455 runs as
independent, but runs reuse scenarios. Ablation 1 (245 of the 455) has no
surviving run directories, so only the 210 A7/A8/A10 runs in the shared
package can be re-tested.

Uses the unchanged equal-weight computation in ``vs_equal_weight.py`` and a
copy of its loader that reads each run's own scenario file (the manifests hold
the co-author's absolute paths), pointed at ``CS372_210_debate_runs/logging/runs``.
Checks that the per-ablation win counts reproduce the stored report, then:
  * scenario-level test: average each scenario's 6 runs, then a t-test and a
    sign test over the 35 scenario means (one observation per scenario);
  * scenario-cluster bootstrap (10,000, fixed seed) CIs for the win rate and
    the mean return / Sharpe gap.

Usage:
    uv run --with scipy --with pyyaml python analysis/vs_equal_weight_clustered.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_equal_weight as orig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "analysis" / "rq1" / "vs_equal_weight_clustered.json"
SEED = 20261031
N_BOOT = 10_000
# Win counts in analysis/vs_equal_weight_report.txt (A7, A8, A10; n=70 each).
REPORT_WINS = {"vskarich_ablation_7": 21, "vskarich_ablation_8": 24, "vskarich_ablation_10": 28}


def load_runs(exp):
    """Same fields as ``vs_equal_weight.load_runs``, but the manifests hold the
    co-author's absolute paths, so the scenario is read from the run's own
    ``final/<scenario>.yaml`` copy and checked against the repo's copy."""
    import yaml
    runs = []
    for run_dir in sorted((orig.RUNS_DIR / exp).iterdir()):
        man = json.loads((run_dir / "manifest.json").read_text())
        fin = json.loads((run_dir / "_dashboard" / "financial_metrics.json").read_text())
        dbg, scn = (Path(p) for p in man["config_paths"][:2])
        local = yaml.safe_load((run_dir / "final" / scn.name).read_text())
        repo = yaml.safe_load((REPO_ROOT / "config" / "scenarios" / "baseline_sweep" / scn.name).read_text())
        assert (local["tickers"], local["invest_quarter"]) == (repo["tickers"], repo["invest_quarter"]), run_dir
        runs.append({
            "condition": "baseline" if "baseline" in dbg.stem.lower() else "treatment",
            "scenario": scn.stem,
            "tickers": local["tickers"],
            "quarter": local["invest_quarter"],
            "sharpe": fin["daily_metrics_annualized_sharpe"],
            "total_return": fin["daily_metrics_total_return_pct"],
        })
    return runs


def load():
    orig.RUNS_DIR = REPO_ROOT / "CS372_210_debate_runs" / "logging" / "runs"
    rows = []
    for exp, want in REPORT_WINS.items():
        runs = load_runs(exp)
        assert len(runs) == 70, (exp, len(runs))
        wins = 0
        for r in runs:
            ew_ret = orig.compute_ew_return(r["tickers"], r["quarter"])
            ew_sh = orig.compute_ew_sharpe(r["tickers"], r["quarter"])
            if ew_ret is None or ew_sh is None:
                raise ValueError(f"no equal-weight benchmark for {exp} {r['scenario']}")
            wins += r["total_return"] > ew_ret
            rows.append({"exp": exp, "scenario": r["scenario"],
                         "ret_gap": r["total_return"] - ew_ret, "sharpe_gap": r["sharpe"] - ew_sh,
                         "win": float(r["total_return"] > ew_ret)})
        assert wins == want, f"{exp}: {wins} wins, report says {want}"
    return rows


def main():
    rows = load()
    by_sc = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    assert len(by_sc) == 35 and all(len(v) == 6 for v in by_sc.values())
    keys = sorted(by_sc)
    sc = {k: np.array([[r["win"], r["ret_gap"], r["sharpe_gap"]] for r in by_sc[k]]) for k in keys}
    means = np.array([sc[k].mean(axis=0) for k in keys])  # 35 x 3

    out = {"n_runs": len(rows), "n_scenarios": len(keys), "seed": SEED, "n_boot": N_BOOT,
           "win_rate": float(np.mean([r["win"] for r in rows])),
           "naive_binomial_p": float(stats.binomtest(int(sum(r["win"] for r in rows)), len(rows)).pvalue)}
    for j, name in enumerate(["win_rate", "ret_gap", "sharpe_gap"]):
        t = stats.ttest_1samp(means[:, j] - (0.5 if name == "win_rate" else 0.0), 0.0)
        out[f"scenario_ttest_{name}"] = {"t": float(t.statistic), "p": float(t.pvalue)}
    below = int((means[:, 0] < 0.5).sum())
    above = int((means[:, 0] > 0.5).sum())
    out["scenario_sign_test_win_rate"] = {
        "scenarios_below_half": below, "scenarios_above_half": above, "ties": 35 - below - above,
        "p": float(stats.binomtest(below, below + above).pvalue)}

    rng = np.random.default_rng(SEED)
    stacked = [sc[k] for k in keys]
    boots = np.empty((N_BOOT, 3))
    for b in range(N_BOOT):
        pick = rng.integers(0, 35, size=35)
        boots[b] = np.concatenate([stacked[i] for i in pick]).mean(axis=0)
    for j, name in enumerate(["win_rate", "ret_gap", "sharpe_gap"]):
        out[f"cluster_ci95_{name}"] = np.quantile(boots[:, j], [0.025, 0.975]).tolist()
    out["mean_ret_gap"] = float(np.mean([r["ret_gap"] for r in rows]))
    out["mean_sharpe_gap"] = float(np.mean([r["sharpe_gap"] for r in rows]))

    print(json.dumps(out, indent=2))
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
