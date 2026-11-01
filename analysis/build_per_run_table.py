#!/usr/bin/env python3
"""
Per-run table for the RQ1 (CRIT vs. financial outcome) corpus
==============================================================
Builds one row per run for ablations 7, 8, 10 using the *unchanged* original
loader in ``crit_financial_correlation.py``, pointed at the shared run package
(``CS372_210_debate_runs/logging/runs``) instead of the repo's empty
``logging/runs``.

Validates the corpus structure (camera_ready_plan.md, item T1) and checks that
the pooled correlations reproduce Table 1 / Appendix C.1 before any new
statistics are computed. Fails loudly on any deviation.

Usage:
    uv run --with scipy python analysis/build_per_run_table.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crit_financial_correlation as orig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "CS372_210_debate_runs"
OUT_PATH = REPO_ROOT / "analysis" / "rq1" / "per_run_results.csv"

# Table 1 / Appendix C.1 values (Pearson r), reported to 4 dp.
PAPER_R = {
    ("rho_bar", "sharpe"): 0.0735,
    ("rho_bar", "total_return"): 0.0265,
    ("delta_rho_bar", "delta_sharpe"): -0.0157,
    ("delta_rho_bar", "delta_total_return"): 0.0114,
}


def load():
    orig.RUNS_ROOT = PACKAGE_ROOT / "logging" / "runs"
    df = orig.load_all_runs()
    df = df.rename(columns={"ablation": "ablation_id", "scenario": "scenario_id"})
    df["run_path"] = [
        f"logging/runs/vskarich_ablation_{a}/{r}" for a, r in zip(df.ablation_id, df.run_id)
    ]
    df["crit_source"] = df.run_path + "/final/pid_crit_all_rounds.json"
    df["quarter"] = df.scenario_id.map(lambda s: re.match(r"scenario_(\d{4}Q\d)", s).group(1))
    return df


def paired(df):
    wide = df.pivot(
        index=["scenario_id", "ablation_id"],
        columns="condition",
        values=["rho_bar", "sharpe", "total_return"],
    )
    out = wide.index.to_frame(index=False)
    for m in ["rho_bar", "sharpe", "total_return"]:
        out[f"delta_{m}"] = (wide[(m, "treatment")] - wide[(m, "baseline")]).values
    return out


def validate(df):
    assert len(df) == 210, len(df)
    assert df.run_path.is_unique
    assert (df.groupby("ablation_id").size() == 70).all()
    assert (df.groupby(["ablation_id", "condition"]).size() == 35).all()
    assert not df[["rho_bar", "sharpe", "total_return"]].isna().any().any()
    assert np.isfinite(df[["rho_bar", "sharpe", "total_return"]].to_numpy()).all()
    # Every (scenario, ablation) appears exactly once per condition.
    assert (df.groupby(["scenario_id", "ablation_id", "condition"]).size() == 1).all()
    sets = {a: set(g.scenario_id) for a, g in df.groupby("ablation_id")}
    assert all(s == sets[7] for s in sets.values()), "ablations use different scenario sets"
    assert len(sets[7]) == 35

    # Cross-check against the package manifest written by the co-author.
    manifest = json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text())
    man = {r["path"]: r for r in manifest["runs"]}
    assert set(man) == set(df.run_path), "run membership differs from PACKAGE_MANIFEST.json"
    for row in df.itertuples():
        m = man[row.run_path]
        assert (m["ablation"], m["condition"], m["scenario"]) == (
            row.ablation_id, row.condition, row.scenario_id)
        for k in ("rho_bar", "sharpe", "total_return"):
            assert np.isclose(m[k], getattr(row, k), rtol=0, atol=1e-12), (row.run_path, k)


def check_paper(df, pairs):
    got = {
        ("rho_bar", "sharpe"): np.corrcoef(df.rho_bar, df.sharpe)[0, 1],
        ("rho_bar", "total_return"): np.corrcoef(df.rho_bar, df.total_return)[0, 1],
        ("delta_rho_bar", "delta_sharpe"): np.corrcoef(pairs.delta_rho_bar, pairs.delta_sharpe)[0, 1],
        ("delta_rho_bar", "delta_total_return"): np.corrcoef(
            pairs.delta_rho_bar, pairs.delta_total_return)[0, 1],
    }
    for key, want in PAPER_R.items():
        print(f"  {key[0]:>14} vs {key[1]:<19} r={got[key]:+.4f}  paper={want:+.4f}")
        assert round(got[key], 4) == want, f"does not reproduce paper: {key}"


def main():
    df = load()
    validate(df)
    pairs = paired(df)
    assert len(pairs) == 105 and not pairs.isna().any().any()
    print(f"Validated {len(df)} runs, {df.scenario_id.nunique()} scenarios, "
          f"{df.quarter.nunique()} quarters, {len(pairs)} pairs")
    check_paper(df, pairs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "run_path", "scenario_id", "quarter", "ablation_id", "condition",
            "rho_bar"] + sorted(c for c in df.columns if c.startswith("rho_") and c != "rho_bar") + [
            "sharpe", "total_return", "crit_source"]
    df[cols].sort_values(["ablation_id", "condition", "scenario_id"]).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
