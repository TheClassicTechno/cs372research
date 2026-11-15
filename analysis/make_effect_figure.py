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
# (metric in verify_paper_tables.csv, x-label, section in seed_variance_report.txt).
# Each panel's noise band uses its own metric's run-to-run floor. Excess-return and
# total-return differences are identical here (SPY is the same within a scenario), so the
# total-return floor applies to the return panel.
PANELS = [
    ("Excess Return % (v SPY)", "Return difference vs single agent (pp)", "Total return %"),
    ("Max Drawdown %", "Max-drawdown difference vs single agent (pp)", "Max drawdown %"),
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


def load_noise(section: str = "Total return %") -> float:
    """Return the implied paired SEM over n=35 for one metric section, in pp."""
    txt = NOISE.read_text()
    m = re.search(r"--- " + re.escape(section) + r" ---(.*?)(?:\n--- |\Z)", txt, re.S)
    if not m:
        raise SystemExit(f"section '{section}' not found in seed_variance_report.txt")
    v = re.search(r"implied SEM of that diff over n=35\s*:\s*([0-9.]+)", m.group(1))
    if not v:
        raise SystemExit(f"could not read the noise floor for '{section}'")
    return float(v.group(1))


def main() -> int:
    eff = load_effects()
    noise_by_panel = [load_noise(sec) for _, _, sec in PANELS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    order = ["B", "C", "D"]
    ypos = list(range(len(order)))[::-1]

    for ax, (metric, xlabel, _), noise in zip(axes, PANELS, noise_by_panel):
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

        ax.text(0.02, 1.06, f"shaded: $\\pm${noise:.2f} pp run-to-run noise floor",
                transform=ax.transAxes, fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"noise floors (paired SEM, n=35): return {noise_by_panel[0]} pp, "
          f"drawdown {noise_by_panel[1]} pp")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
