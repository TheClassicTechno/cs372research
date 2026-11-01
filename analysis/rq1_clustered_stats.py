#!/usr/bin/env python3
"""
Dependence-aware uncertainty for the RQ1 CRIT-outcome correlations
==================================================================
camera_ready_plan.md items T2a-T2d. Input: ``analysis/rq1/per_run_results.csv``
(built and validated by ``build_per_run_table.py``).

The 210 runs are 35 scenarios x 3 ablations x 2 conditions, so rows sharing a
scenario share realized prices and are not independent. For each statistic we
report the point estimate and a cluster bootstrap that resamples whole
scenarios (all 6 runs, and therefore all 3 baseline/treatment pairs, travel
together). Sensitivity: resampling whole calendar quarters instead (12
clusters, so those intervals are themselves unstable).

Statistics
  pooled   Pearson r over all 210 runs (Table 1 rows 1-2)
  within   r after subtracting each scenario's mean from rho_bar and outcome
  fe       r of residuals after scenario + (ablation x condition) fixed effects
  paired   r of treatment-minus-baseline deltas, 105 pairs (Table 1 rows 3-4)

Equivalence: the margin delta = 0.20 was chosen *after* the point estimates
were known (they were already in the submitted paper). "90% cluster-bootstrap
percentile CI inside (-delta, +delta)" is a screen, not a calibrated
cluster-TOST. We also report the largest |r| each 90% CI still admits.

Usage:
    uv run --with scipy python analysis/rq1_clustered_stats.py [--selftest]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = REPO_ROOT / "analysis" / "rq1" / "per_run_results.csv"
OUT_PATH = REPO_ROOT / "analysis" / "rq1" / "rq1_clustered_stats.json"

SEED = 20261030
N_BOOT = 10_000
DELTA = 0.20
OUTCOMES = ["sharpe", "total_return"]


def _r(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else np.nan


def _demean(v, labels):
    n = np.bincount(labels)
    return v - (np.bincount(labels, v) / n)[labels]


def _resid_two_way(v, a, b, iters=200, tol=1e-12):
    """Residual of v on additive fixed effects for labels a and b (alternating projections)."""
    r = v - v.mean()
    for _ in range(iters):
        new = _demean(_demean(r, a), b)
        if np.max(np.abs(new - r)) < tol:
            return new
        r = new
    return r


# Each statistic takes (x, y, cluster_labels, cell_labels) for the rows in the sample.
def stat_pooled(x, y, c, cell):
    return _r(x, y)


def stat_within(x, y, c, cell):
    return _r(_demean(x, c), _demean(y, c))


def stat_fe(x, y, c, cell):
    return _r(_resid_two_way(x, c, cell), _resid_two_way(y, c, cell))


def stat_paired(dx, dy, c, cell):
    return _r(dx, dy)


STATS = {"pooled": stat_pooled, "within": stat_within, "fe": stat_fe, "paired": stat_paired}


def paired_frame(df):
    w = df.pivot(index=["scenario_id", "quarter", "ablation_id"], columns="condition",
                 values=["rho_bar"] + OUTCOMES)
    out = w.index.to_frame(index=False)
    for m in ["rho_bar"] + OUTCOMES:
        out[m] = (w[(m, "treatment")] - w[(m, "baseline")]).to_numpy()
    out["condition"] = "delta"
    assert len(out) == 105 and not out.isna().any().any()
    return out


def cluster_bootstrap(d, y, cluster_col, fn, rng, n_boot=N_BOOT):
    """Percentile CI from resampling whole clusters; duplicated clusters stay distinct."""
    x_all, y_all = d.rho_bar.to_numpy(float), d[y].to_numpy(float)
    cell_all = pd.factorize(d.ablation_id.astype(str) + "_" + d.condition)[0]
    codes, keys = pd.factorize(d[cluster_col])
    members = [np.flatnonzero(codes == k) for k in range(len(keys))]
    est = fn(x_all, y_all, codes, cell_all)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([members[k] for k in pick])
        lab = np.repeat(np.arange(len(pick)), [len(members[k]) for k in pick])
        draws[b] = fn(x_all[idx], y_all[idx], lab, cell_all[idx])
    if not np.isfinite(draws).all():
        raise ValueError("undefined bootstrap replicate")
    return {
        "r": est,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "ci90": np.quantile(draws, [0.05, 0.95]).tolist(),
        "boot_sd": float(draws.std(ddof=1)),
        "n_clusters": int(len(keys)),
        "n_obs": int(len(d)),
    }


def naive_fisher_ci(r, n, z=1.959964):
    zr, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(zr - z * se)), float(np.tanh(zr + z * se))]


def run(df, rng):
    pairs = paired_frame(df)
    out = {"config": {"seed": SEED, "n_boot": N_BOOT, "delta": DELTA,
                      "delta_chosen_post_hoc": True,
                      "ci_method": "cluster bootstrap, percentile",
                      "cluster_primary": "scenario_id", "cluster_sensitivity": "quarter",
                      "input": str(IN_PATH.relative_to(REPO_ROOT)), "n_runs": len(df)},
           "results": {}}
    for y in OUTCOMES:
        for name, fn in STATS.items():
            d = pairs if name == "paired" else df
            res = cluster_bootstrap(d, y, "scenario_id", fn, rng)
            res["quarter_cluster"] = cluster_bootstrap(d, y, "quarter", fn, rng)
            res["naive_iid_ci95"] = naive_fisher_ci(res["r"], len(d))
            lo, hi = res["ci90"]
            res["equiv_screen_inside_delta"] = bool(-DELTA < lo and hi < DELTA)
            res["max_abs_r_in_ci90"] = float(max(abs(lo), abs(hi)))
            out["results"][f"{name}:{y}"] = res
            q = res["quarter_cluster"]["ci95"]
            print(f"{name + ':' + y:<20} r={res['r']:+.3f}  "
                  f"scen95=[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}]  "
                  f"scen90=[{lo:+.3f},{hi:+.3f}]  qtr95=[{q[0]:+.3f},{q[1]:+.3f}]  "
                  f"iid95=[{res['naive_iid_ci95'][0]:+.3f},{res['naive_iid_ci95'][1]:+.3f}]  "
                  f"in±{DELTA}: {res['equiv_screen_inside_delta']}")
    return out


def _synthetic(rng, rho_w=0.5, between=3.0):
    """35 scenarios x 6 runs; strong opposite-sign scenario effect, within-scenario r = rho_w."""
    mu = np.repeat(rng.normal(size=35), 6)
    e = rng.multivariate_normal([0, 0], [[1, rho_w], [rho_w, 1]], size=210)
    return pd.DataFrame({
        "scenario_id": np.repeat([f"s{i}" for i in range(35)], 6),
        "quarter": np.repeat([f"q{i % 12}" for i in range(35)], 6),
        "ablation_id": np.tile([7, 7, 8, 8, 10, 10], 35),
        "condition": np.tile(["baseline", "treatment"], 105),
        "rho_bar": between * mu + e[:, 0],
        "sharpe": -between * mu + e[:, 1],
        "total_return": rng.normal(size=210),
    })


def selftest():
    """Checks unbiasedness over many draws, then bootstrap coverage on a small batch."""
    rng = np.random.default_rng(0)
    est = {k: [] for k in STATS}
    pooled = []
    for _ in range(200):
        d = _synthetic(rng)
        c = pd.factorize(d.scenario_id)[0]
        cell = pd.factorize(d.ablation_id.astype(str) + d.condition)[0]
        x, y = d.rho_bar.to_numpy(), d.sharpe.to_numpy()
        pooled.append(stat_pooled(x, y, c, cell))
        for k in ("within", "fe"):
            est[k].append(STATS[k](x, y, c, cell))
        p = paired_frame(d)
        est["paired"].append(stat_paired(p.rho_bar.to_numpy(), p.sharpe.to_numpy(), None, None))
    print(f"mean pooled={np.mean(pooled):+.3f} (expect ~-0.85: scenario effect dominates)")
    assert np.mean(pooled) < -0.8
    for k in ("within", "fe", "paired"):
        print(f"mean {k}={np.mean(est[k]):+.3f} (expect ~+0.50)")
        assert abs(np.mean(est[k]) - 0.5) < 0.03, k
    covered = 0
    for _ in range(40):
        d = _synthetic(rng)
        res = cluster_bootstrap(d, "sharpe", "scenario_id", stat_within, rng, n_boot=500)
        covered += res["ci95"][0] < 0.5 < res["ci95"][1]
    print(f"95% CI coverage of within r: {covered}/40")
    assert covered >= 34
    print("selftest OK")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    df = pd.read_csv(IN_PATH)
    assert len(df) == 210 and df.scenario_id.nunique() == 35
    out = run(df, np.random.default_rng(SEED))
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
