#!/usr/bin/env python3
"""
Per-run CRIT rho_bar vs Sharpe for the 210-run RQ1 corpus (Appendix figure
fig:crit-scatter).

Reads only verified artifacts:
  analysis/rq1/per_run_results.csv     one row per run (built by build_per_run_table.py)
  analysis/rq1/rq1_clustered_stats.json  r and scenario-cluster CIs (Table tab:rq1_clustered)

Left panel: pooled. Right panel: within scenario (rho_bar and Sharpe each minus
their scenario mean). The r printed on each panel is read from the JSON, and the
script fails if r recomputed from the CSV does not match it. Rendering only.

Usage:
    uv run --with matplotlib --with pandas --with numpy python analysis/make_crit_scatter_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "analysis" / "rq1" / "per_run_results.csv"
STATS = REPO / "analysis" / "rq1" / "rq1_clustered_stats.json"
OUT = REPO / "crit_vs_sharpe.pdf"


def _r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def main() -> int:
    df = pd.read_csv(RUNS)
    assert len(df) == 210 and df.scenario_id.nunique() == 35, "unexpected corpus size"
    stats = json.loads(STATS.read_text())["results"]

    x = df.rho_bar.to_numpy(float)
    y = df.sharpe.to_numpy(float)
    xw = x - df.groupby("scenario_id").rho_bar.transform("mean").to_numpy()
    yw = y - df.groupby("scenario_id").sharpe.transform("mean").to_numpy()
    assert abs(_r(x, y) - stats["pooled:sharpe"]["r"]) < 1e-9
    assert abs(_r(xw, yw) - stats["within:sharpe"]["r"]) < 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    panels = [
        (axes[0], x, y, "pooled:sharpe", "Pooled",
         r"CRIT $\bar\rho$", "Sharpe ratio"),
        (axes[1], xw, yw, "within:sharpe", "Within scenario",
         r"CRIT $\bar\rho$ minus scenario mean", "Sharpe minus scenario mean"),
    ]
    for ax, xs, ys, key, title, xl, yl in panels:
        ax.scatter(xs, ys, s=12, color="#4d4d4d", alpha=0.6, edgecolors="none", zorder=2)
        slope, icept = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min(), xs.max(), 2)
        ax.plot(gx, icept + slope * gx, color="#1a1a1a", lw=1.4, zorder=3)
        s = stats[key]
        lo, hi = s["ci95"]
        ax.text(0.03, 0.97, f"r = {s['r']:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6))
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
