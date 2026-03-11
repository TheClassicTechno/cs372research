"""Reconstruct round_state.json from portfolio.json files in the logging directory.

Reads proposals/*/portfolio.json, revisions/*/portfolio.json (and retry variants),
computes JS divergence, and writes round_state.json for rounds that are missing it.

Usage:
    python reconstruct_round_state.py <experiment_dir>
"""

import json
import math
import sys
from pathlib import Path


def kl_divergence(p: list[float], q: list[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi, 2)
    return s


def js_divergence(distributions: list[list[float]]) -> float:
    k = len(distributions)
    n = len(distributions[0])
    m = [0.0] * n
    for d in distributions:
        for i in range(n):
            m[i] += d[i] / k
    js = 0.0
    for d in distributions:
        js += kl_divergence(d, m)
    return js / k


def load_portfolios(phase_dir: Path) -> dict[str, dict[str, float]]:
    """Load portfolio.json for each agent in a phase directory."""
    result = {}
    if not phase_dir.is_dir():
        return result
    for agent_dir in sorted(phase_dir.iterdir()):
        pf = agent_dir / "portfolio.json"
        if pf.exists():
            result[agent_dir.name] = json.loads(pf.read_text(encoding="utf-8"))
    return result


def portfolios_to_vectors(portfolios: dict[str, dict[str, float]]) -> tuple[list[str], dict[str, list[float]]]:
    """Convert portfolio dicts to aligned vectors over sorted ticker union."""
    tickers = sorted(set().union(*(p.keys() for p in portfolios.values())))
    vectors = {
        role: [alloc.get(t, 0.0) for t in tickers]
        for role, alloc in portfolios.items()
    }
    return tickers, vectors


def compute_js(portfolios: dict[str, dict[str, float]]) -> float:
    if len(portfolios) < 2:
        return 0.0
    _, vectors = portfolios_to_vectors(portfolios)
    return js_divergence(list(vectors.values()))


def reconstruct_round(round_dir: Path) -> bool:
    """Reconstruct round_state.json for a single round directory. Returns True if written."""
    out_path = round_dir / "round_state.json"
    if out_path.exists():
        return False  # already has one

    # Load proposals
    proposals = load_portfolios(round_dir / "proposals")
    if not proposals:
        return False  # no data

    # Find final revision phase: highest retry, else revision
    revision_dirs = sorted(round_dir.glob("revisions_retry_*"))
    if revision_dirs:
        final_rev_dir = revision_dirs[-1]
    else:
        final_rev_dir = round_dir / "revisions"

    revisions = load_portfolios(final_rev_dir)
    if not revisions:
        return False

    # Compute JS divergence on final revisions
    final_js = compute_js(revisions)

    # Read evidence overlap from existing metrics if available
    evidence_overlap = 0.0
    # Check metrics dir for the matching phase
    if revision_dirs:
        retry_num = final_rev_dir.name.replace("revisions_retry_", "")
        ov_file = round_dir / "metrics" / f"evidence_overlap_retry_{retry_num}.json"
    else:
        ov_file = round_dir / "metrics" / "evidence_overlap.json"

    if ov_file.exists():
        ov_data = json.loads(ov_file.read_text(encoding="utf-8"))
        evidence_overlap = ov_data.get("evidence_overlap", 0.0)

    # Read CRIT data if available
    crit_file = round_dir / "metrics" / "crit_scores.json"
    crit_data = {}
    if crit_file.exists():
        raw_crit = json.loads(crit_file.read_text(encoding="utf-8"))
        if "rho_bar" in raw_crit:
            crit_data["rho_bar"] = raw_crit["rho_bar"]
        for key, val in raw_crit.items():
            if isinstance(val, dict) and "rho_i" in val:
                crit_data[key] = val

    # Build round_state
    round_num_str = round_dir.name.replace("round_", "")
    try:
        round_num = int(round_num_str)
    except ValueError:
        round_num = 0

    # Build proposal/revision summaries (match round_state.json schema)
    proposals_summary = {
        role: {"allocation": alloc, "confidence": 0.5}
        for role, alloc in proposals.items()
    }
    revisions_summary = {
        role: {"allocation": alloc, "confidence": 0.5}
        for role, alloc in revisions.items()
    }

    # Try to get confidence from response files
    for phase_name, phase_dir, summary in [
        ("proposals", round_dir / "proposals", proposals_summary),
        ("revisions", final_rev_dir, revisions_summary),
    ]:
        for role in summary:
            resp_file = phase_dir / role / "response.txt"
            if resp_file.exists():
                try:
                    text = resp_file.read_text(encoding="utf-8")
                    # Try to parse JSON from response
                    import re
                    match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
                    if match:
                        summary[role]["confidence"] = float(match.group(1))
                except Exception:
                    pass

    data = {
        "round": round_num,
        "beta": None,
        "proposals": proposals_summary,
        "revisions": revisions_summary,
        "metrics": {
            "js_divergence": round(final_js, 4),
            "evidence_overlap": round(evidence_overlap, 4),
        },
        "crit": crit_data,
        "pid": {},
    }

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def reconstruct_experiment(experiment_path: Path) -> None:
    round_dirs = sorted(experiment_path.glob("run_*/rounds/round_*"))
    if not round_dirs:
        print(f"No round directories found under {experiment_path}", file=sys.stderr)
        sys.exit(1)

    written = 0
    skipped = 0
    for rd in round_dirs:
        if reconstruct_round(rd):
            run_name = rd.parent.parent.name
            print(f"  Wrote {run_name}/{rd.name}/round_state.json")
            written += 1
        else:
            skipped += 1

    print(f"\nReconstructed {written} round_state.json files ({skipped} already existed or had no data)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <experiment_dir>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"Error: {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    reconstruct_experiment(target)
