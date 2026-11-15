#!/usr/bin/env python3
"""
Poster figures (Stanford palette) for the Verify-Agents workshop poster.

Rendering only. Every plotted number comes from artifacts the paper already uses,
and the script asserts that it reproduces the published values.

  fig_crit_vs_sharpe.pdf
      Per-run CRIT rho_bar vs annualized Sharpe, 210 runs (A7/A8/A10 x baseline/treatment
      x 35 scenarios). Left: all runs. Right: within scenario (each variable minus its
      scenario mean). r and scenario-cluster 95% CIs are read from
      analysis/rq1/rq1_clustered_stats.json; r is recomputed here as a check.

  fig_effect_vs_noise.pdf
      Paired differences of three 3-agent independent ensembles vs the single-agent
      baseline over the same 35 scenarios (analysis/verify_paper_tables.csv, table
      tab:prompts, paired_vs=A, scenario-bootstrap 95% CIs). Noise bands are the paired
      SEM over n=35 implied by re-running the single-agent config 6x on 5 scenarios
      (results/seed_variance), computed per metric: the drawdown panel uses a drawdown
      estimate, not the return estimate. The return band must reproduce the published
      0.31 pp (analysis/seed_variance_report.txt).

Usage (repo root):
    uv run --no-project --with matplotlib --with pandas --with numpy \
        python neurips_poster_template_for_agentic_financial_one/make_poster_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "analysis"))
from make_effect_figure import load_effects, load_noise  # noqa: E402

RUNS = REPO / "analysis" / "rq1" / "per_run_results.csv"
STATS = REPO / "analysis" / "rq1" / "rq1_clustered_stats.json"
SEEDS = REPO / "results" / "seed_variance"

# Stanford palette (same values as poster.tex)
CARDINAL = "#8C1515"
CHARCOAL = "#2E2D29"
GRAY = "#767674"
BAND = "#E9E6E1"   # neutral band, distinct from both mark colours
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 22,
    "text.color": CHARCOAL,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": CHARCOAL,
    "xtick.color": CHARCOAL,
    "ytick.color": CHARCOAL,
    "axes.linewidth": 1.0,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
})


def _r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def crit_vs_sharpe() -> None:
    df = pd.read_csv(RUNS)
    assert len(df) == 210 and df.scenario_id.nunique() == 35, "unexpected corpus size"
    assert df.groupby(["ablation_id", "condition"]).size().eq(35).all()
    stats = json.loads(STATS.read_text())["results"]

    x = df.rho_bar.to_numpy(float)
    y = df.sharpe.to_numpy(float)
    xw = x - df.groupby("scenario_id").rho_bar.transform("mean").to_numpy()
    yw = y - df.groupby("scenario_id").sharpe.transform("mean").to_numpy()
    assert abs(_r(x, y) - stats["pooled:sharpe"]["r"]) < 1e-9
    assert abs(_r(xw, yw) - stats["within:sharpe"]["r"]) < 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    panels = [
        (axes[0], x, y, "pooled:sharpe", "Across all 210 runs",
         r"CRIT score $\bar\rho$", "Sharpe ratio"),
        (axes[1], xw, yw, "within:sharpe", "Within the same market scenario",
         r"$\bar\rho$ minus scenario mean", "Sharpe minus mean"),
    ]
    for ax, xs, ys, key, title, xl, yl in panels:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#ECECEC", lw=0.8)
        ax.set_axisbelow(True)
        ax.scatter(xs, ys, s=30, color=GRAY, alpha=0.55, edgecolors=WHITE,
                   linewidths=0.5, zorder=3, label="one run")
        slope, icept = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min(), xs.max(), 2)
        ax.plot(gx, icept + slope * gx, color=CARDINAL, lw=3, zorder=4, label="linear fit")
        s = stats[key]
        lo, hi = s["ci95"]
        ax.text(0.03, 0.97, f"r = {s['r']:+.2f}\n95% CI [{lo:+.2f}, {hi:+.2f}]",
                transform=ax.transAxes, va="top", ha="left", fontsize=20, color=CHARCOAL,
                bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=CARDINAL, lw=1.4))
        ax.set_title(title, fontsize=22, color=CHARCOAL, loc="left", pad=8,
                     fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.tight_layout(w_pad=2.5)
    out = HERE / "fig_crit_vs_sharpe.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"pooled r={_r(x, y):+.4f}  within r={_r(xw, yw):+.4f}  -> {out.name}")


def seed_noise_floor(metric: str) -> float:
    """Paired SEM over n=35 implied by the pooled within-scenario SD of repeat runs."""
    by = defaultdict(list)
    for f in sorted(glob.glob(str(SEEDS / "*__rep*" / "*" / "summary.json"))):
        sc = re.match(r".*/(.+?)__rep\d+/", f).group(1)
        dm = json.load(open(f))["episode_summaries"][0]["daily_metrics"]
        by[sc].append(float(dm[metric]))
    assert len(by) == 5 and all(len(v) == 6 for v in by.values()), "expected 5 x 6 repeats"
    ss = sum(sum((a - np.mean(v)) ** 2 for a in v) for v in by.values())
    dof = sum(len(v) - 1 for v in by.values())
    return math.sqrt(ss / dof) * math.sqrt(2) / math.sqrt(35)


def effect_vs_noise() -> None:
    eff = load_effects()
    noise_ret = seed_noise_floor("total_return_pct")
    noise_dd = seed_noise_floor("max_drawdown_pct")
    assert abs(round(noise_ret, 2) - load_noise()) < 1e-9, (noise_ret, load_noise())
    # SPY is identical within a scenario, so excess-return and total-return diffs agree.
    for c in "BCD":
        assert abs(eff["Excess Return % (v SPY)"][c][0] - eff["Total Return %"][c][0]) < 1e-9

    labels = {"B": "Basic", "C": "Causal", "D": "Enriched"}
    panels = [("Total Return %", "Return difference (pp)", noise_ret),
              ("Max Drawdown %", "Max drawdown difference (pp)", noise_dd)]
    order = ["B", "C", "D"]
    ypos = list(range(len(order)))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, (metric, xlabel, noise) in zip(axes, panels):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvspan(-noise, noise, color=BAND, zorder=0)
        ax.axvline(0, color=CHARCOAL, lw=1.2, zorder=1)
        for y, col in zip(ypos, order):
            mean, lo, hi = eff[metric][col]
            outside = lo > noise or hi < -noise
            c = CARDINAL if outside else GRAY
            ax.plot([lo, hi], [y, y], color=c, lw=3.2, zorder=2, solid_capstyle="butt")
            if outside:
                ax.plot([mean], [y], "D", ms=12, color=c, zorder=3,
                        markeredgecolor=WHITE, markeredgewidth=1.5)
            else:
                ax.plot([mean], [y], "o", ms=11, color=WHITE, zorder=3,
                        markeredgecolor=c, markeredgewidth=2.5)
        ax.text(0.0, 1.03, f"band: $\\pm${noise:.2f} pp rerun noise",
                transform=ax.transAxes, fontsize=19, color=CHARCOAL)
        ax.set_yticks(ypos)
        ax.set_yticklabels([labels[c] for c in order])
        ax.set_xlabel(xlabel)
    fig.tight_layout(w_pad=2.0)
    out = HERE / "fig_effect_vs_noise.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"noise floor return {noise_ret:.3f} pp, drawdown {noise_dd:.3f} pp; "
          f"D drawdown {eff['Max Drawdown %']['D']} -> {out.name}")


if __name__ == "__main__":
    crit_vs_sharpe()
    effect_vs_noise()
