#!/usr/bin/env python3
"""
Effect sizes against the measured sampling-noise floor.

Reads only verified artifacts:
  analysis/verify_paper_tables.csv   paired differences vs column A + bootstrap CIs
  analysis/seed_variance_report.txt  run-to-run noise floor

Question the figure answers in five seconds: which of the differences in
Table 2 are larger than the noise produced by re-running the same
configuration on the same scenario?

Nothing is computed here that is not already in those two files; this is a
rendering step, not an analysis step.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "analysis" / "verify_paper_tables.csv"
NOISE = REPO / "analysis" / "seed_variance_report.txt"
OUT = REPO / "effect_vs_noise.pdf"

LABELS = {
    "B": "3 roles, mean\nbasic",
    "C": "3 roles, mean\ncausal",
    "D": "3 roles, mean\nenriched",
}
PANELS = [
    ("Excess Return % (v SPY)", "Excess return vs SPY (pp)", "total_return_pct"),
    ("Max Drawdown %", "Max drawdown (pp)", None),
]


def load_effects() -> dict[str, dict[str, tuple[float, float, float]]]:
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    with open(AUDIT) as fh:
        for row in csv.DictReader(fh):
            if row["table"] != "tab:prompts" or row.get("paired_vs") != "A":
                continue
            out.setdefault(row["metric"], {})[row["column"]] = (
                float(row["diff_mean"]), float(row["diff_ci_low"]), float(row["diff_ci_high"]))
    return out


def load_noise() -> float:
    """Return the implied paired SEM over n=35 for total return, in pp."""
    txt = NOISE.read_text()
    m = re.search(r"implied SEM of that diff over n=35\s*:\s*([0-9.]+)", txt)
    if not m:
        raise SystemExit("could not read the noise floor from seed_variance_report.txt")
    return float(m.group(1))


def main() -> int:
    eff = load_effects()
    noise = load_noise()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    order = ["B", "C", "D"]
    ypos = list(range(len(order)))[::-1]

    for ax, (metric, xlabel, _) in zip(axes, PANELS):
        data = eff[metric]
        # Noise band: +/- the paired SEM implied by re-running one config.
        ax.axvspan(-noise, noise, color="0.88", zorder=0)
        ax.axvline(0, color="0.45", lw=0.8, zorder=1)
        for y, col in zip(ypos, order):
            mean, lo, hi = data[col]
            outside = lo > noise or hi < -noise
            colour = "#1a1a1a" if outside else "#8c8c8c"
            ax.plot([lo, hi], [y, y], color=colour, lw=1.6, zorder=2,
                    solid_capstyle="butt")
            ax.plot([mean], [y], "o", ms=5, color=colour, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABELS[c] for c in order], fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].text(0.02, 1.06, f"shaded: $\\pm$ sampling-noise floor ({noise:.2f} pp)",
                 transform=axes[0].transAxes, fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"noise floor (paired SEM, n=35): {noise} pp")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
