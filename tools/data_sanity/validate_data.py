import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

EPS = 1e-6
ROUND_SUM_EPS = 1e-6
VECTOR_EPS = 1e-9
METRIC_EPS = 1e-6


# =========================================================
# Jensen-Shannon Divergence
# =========================================================

def kl_divergence(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * math.log(pi / qi, 2)
    return s


def js_divergence(distributions: List[List[float]]) -> float:
    """
    Multi-distribution Jensen-Shannon divergence.
    Assumes all distributions are valid probability vectors of same length.
    """
    if not distributions:
        raise ValueError("No distributions provided to js_divergence().")

    k = len(distributions)
    n = len(distributions[0])

    for d in distributions:
        if len(d) != n:
            raise ValueError("Distributions have inconsistent lengths.")

    m = [0.0] * n
    for d in distributions:
        for i in range(n):
            m[i] += d[i] / k

    js = 0.0
    for d in distributions:
        js += kl_divergence(d, m)

    return js / k


# =========================================================
# Helpers
# =========================================================

def approx_equal(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def get_phase_keys(phases: Dict[str, Any]) -> List[str]:
    return sorted(phases.keys())


def get_retry_keys(phases: Dict[str, Any]) -> List[str]:
    return sorted([k for k in phases if k.startswith("retry_")])


def get_final_phase(phases: Dict[str, Any]) -> str:
    """
    Final phase = highest retry if present, otherwise 'revision'.
    """
    retry_keys = get_retry_keys(phases)
    if retry_keys:
        return retry_keys[-1]
    return "revision"


def extract_vectors_from_phase(phase: Dict[str, Any], agent_order: List[str]) -> List[List[float]]:
    vectors = phase["vectors"]
    return [vectors[a] for a in agent_order]


def reconstruct_vector_from_alloc(alloc: Dict[str, float], tickers: List[str]) -> List[float]:
    return [float(alloc.get(t, 0.0)) for t in tickers]


def count_nonzero_positions(alloc: Dict[str, float], eps: float = VECTOR_EPS) -> int:
    return sum(1 for v in alloc.values() if abs(float(v)) > eps)


def vector_sum(v: List[float]) -> float:
    return float(sum(v))


def check_probability_vector(v: List[float]) -> Tuple[bool, str]:
    if any(x < -VECTOR_EPS for x in v):
        return False, "vector contains negative values"
    s = vector_sum(v)
    if not approx_equal(s, 1.0, ROUND_SUM_EPS):
        return False, f"vector sum != 1 (sum={s:.12f})"
    return True, ""


def allocations_equal(a1: Dict[str, Dict[str, float]], a2: Dict[str, Dict[str, float]], eps: float = METRIC_EPS) -> Tuple[bool, str]:
    agents1 = set(a1.keys())
    agents2 = set(a2.keys())
    if agents1 != agents2:
        return False, f"agent sets differ: {sorted(agents1)} vs {sorted(agents2)}"

    for agent in sorted(agents1):
        t1 = set(a1[agent].keys())
        t2 = set(a2[agent].keys())
        if t1 != t2:
            return False, f"{agent}: ticker sets differ"

        for ticker in sorted(t1):
            v1 = float(a1[agent][ticker])
            v2 = float(a2[agent][ticker])
            if not approx_equal(v1, v2, eps):
                return False, f"{agent}:{ticker} differs ({v1:.12f} vs {v2:.12f})"

    return True, ""


# =========================================================
# Validation primitives
# =========================================================

def validate_top_level_structure(round_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_keys = ["round", "agents", "tickers", "phases", "allocations", "metrics"]
    for k in required_keys:
        if k not in round_data:
            errors.append(f"Round ? missing top-level key: {k}")
    if errors:
        return errors

    r = round_data["round"]

    if not isinstance(round_data["agents"], list) or not round_data["agents"]:
        errors.append(f"Round {r}: agents must be a non-empty list")

    if not isinstance(round_data["tickers"], list) or not round_data["tickers"]:
        errors.append(f"Round {r}: tickers must be a non-empty list")

    if "propose" not in round_data["phases"]:
        errors.append(f"Round {r}: missing phase 'propose'")

    if "revision" not in round_data["phases"]:
        errors.append(f"Round {r}: missing phase 'revision'")

    return errors


def validate_phase_structure(round_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    r = round_data["round"]
    agents = round_data["agents"]
    tickers = round_data["tickers"]

    for phase_name, phase in round_data["phases"].items():
        for req in ["allocations", "vectors", "js_divergence", "evidence_overlap"]:
            if req not in phase:
                errors.append(f"Round {r} phase {phase_name}: missing key '{req}'")
                continue

        allocs = phase.get("allocations", {})
        vectors = phase.get("vectors", {})

        if set(allocs.keys()) != set(agents):
            errors.append(
                f"Round {r} phase {phase_name}: allocation agent keys mismatch "
                f"{sorted(allocs.keys())} vs expected {sorted(agents)}"
            )

        if set(vectors.keys()) != set(agents):
            errors.append(
                f"Round {r} phase {phase_name}: vector agent keys mismatch "
                f"{sorted(vectors.keys())} vs expected {sorted(agents)}"
            )

        for agent in agents:
            if agent not in allocs:
                continue
            alloc = allocs[agent]
            if set(alloc.keys()) != set(tickers):
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    "allocation ticker set does not match canonical tickers"
                )

            for ticker, val in alloc.items():
                try:
                    fval = float(val)
                except Exception:
                    errors.append(
                        f"Round {r} phase {phase_name} agent {agent} ticker {ticker}: "
                        f"allocation not numeric ({val!r})"
                    )
                    continue
                if fval < -VECTOR_EPS:
                    errors.append(
                        f"Round {r} phase {phase_name} agent {agent} ticker {ticker}: "
                        f"negative allocation {fval}"
                    )

    return errors


def validate_vector_alignment(round_data: Dict[str, Any]) -> List[str]:
    """
    Ensures vectors match allocations under the canonical ticker order.
    This is the critical ticker-order integrity check.
    """
    errors: List[str] = []
    r = round_data["round"]
    tickers = round_data["tickers"]

    for phase_name, phase in round_data["phases"].items():
        allocations = phase["allocations"]
        vectors = phase["vectors"]

        for agent, alloc in allocations.items():
            reconstructed = reconstruct_vector_from_alloc(alloc, tickers)
            stored = vectors.get(agent)

            if stored is None:
                errors.append(f"Round {r} phase {phase_name} agent {agent}: missing vector")
                continue

            if len(stored) != len(tickers):
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    f"vector length {len(stored)} != ticker_count {len(tickers)}"
                )
                continue

            for i, (a, b) in enumerate(zip(reconstructed, stored)):
                if not approx_equal(a, float(b), VECTOR_EPS):
                    errors.append(
                        f"Round {r} phase {phase_name} agent {agent}: "
                        f"vector misalignment at index {i} ticker={tickers[i]} "
                        f"alloc={a:.12f} vector={float(b):.12f}"
                    )

    return errors


def validate_allocation_mass(round_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    r = round_data["round"]

    for phase_name, phase in round_data["phases"].items():
        allocs = phase["allocations"]
        vectors = phase["vectors"]
        allocation_sums = phase.get("allocation_sums", {})
        vector_norms = phase.get("vector_norms", {})
        nonzero_positions = phase.get("nonzero_positions", {})

        for agent, alloc in allocs.items():
            alloc_sum = sum(float(v) for v in alloc.values())
            if not approx_equal(alloc_sum, 1.0, ROUND_SUM_EPS):
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    f"allocation sum != 1 (sum={alloc_sum:.12f})"
                )

            if agent in allocation_sums and not approx_equal(float(allocation_sums[agent]), alloc_sum, METRIC_EPS):
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    f"allocation_sums field mismatch "
                    f"(logged={float(allocation_sums[agent]):.12f}, actual={alloc_sum:.12f})"
                )

            vec = [float(x) for x in vectors[agent]]
            ok, msg = check_probability_vector(vec)
            if not ok:
                errors.append(f"Round {r} phase {phase_name} agent {agent}: {msg}")

            vec_sum = vector_sum(vec)
            if agent in vector_norms and not approx_equal(float(vector_norms[agent]), vec_sum, METRIC_EPS):
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    f"vector_norms field mismatch "
                    f"(logged={float(vector_norms[agent]):.12f}, actual={vec_sum:.12f})"
                )

            nz = count_nonzero_positions(alloc)
            if agent in nonzero_positions and int(nonzero_positions[agent]) != nz:
                errors.append(
                    f"Round {r} phase {phase_name} agent {agent}: "
                    f"nonzero_positions mismatch (logged={nonzero_positions[agent]}, actual={nz})"
                )

    return errors


def validate_js_and_overlap(round_data: Dict[str, Any]) -> Tuple[List[str], float, float]:
    errors: List[str] = []
    r = round_data["round"]
    phases = round_data["phases"]
    agents = round_data["agents"]

    # proposal JS
    proposal_vectors = extract_vectors_from_phase(phases["propose"], agents)
    js_recomputed = js_divergence(proposal_vectors)
    js_logged = float(phases["propose"]["js_divergence"])

    if not approx_equal(js_recomputed, js_logged, METRIC_EPS):
        errors.append(
            f"Round {r} proposal JS mismatch: "
            f"computed={js_recomputed:.12f} logged={js_logged:.12f}"
        )

    if float(phases["propose"]["evidence_overlap"]) < -METRIC_EPS:
        errors.append(f"Round {r} propose evidence_overlap is negative")

    # final phase JS
    final_key = get_final_phase(phases)
    final_vectors = extract_vectors_from_phase(phases[final_key], agents)
    final_js_recomputed = js_divergence(final_vectors)
    final_js_logged = float(phases[final_key]["js_divergence"])

    if not approx_equal(final_js_recomputed, final_js_logged, METRIC_EPS):
        errors.append(
            f"Round {r} {final_key} JS mismatch: "
            f"computed={final_js_recomputed:.12f} logged={final_js_logged:.12f}"
        )

    if float(phases[final_key]["evidence_overlap"]) < -METRIC_EPS:
        errors.append(f"Round {r} {final_key} evidence_overlap is negative")

    # final metrics consistency
    metrics_js = float(round_data["metrics"]["js_divergence"])
    metrics_ov = float(round_data["metrics"]["evidence_overlap"])

    if not approx_equal(metrics_js, final_js_logged, METRIC_EPS):
        errors.append(
            f"Round {r}: metrics.js_divergence inconsistent with final phase "
            f"({final_key}) logged={metrics_js:.12f} final={final_js_logged:.12f}"
        )

    final_ov_logged = float(phases[final_key]["evidence_overlap"])
    if not approx_equal(metrics_ov, final_ov_logged, METRIC_EPS):
        errors.append(
            f"Round {r}: metrics.evidence_overlap inconsistent with final phase "
            f"({final_key}) logged={metrics_ov:.12f} final={final_ov_logged:.12f}"
        )

    return errors, final_js_logged, js_logged


def validate_allocation_snapshots(round_data: Dict[str, Any]) -> List[str]:
    """
    Ensures top-level allocations.proposals/revisions match the expected phase snapshots.
    proposals should match phases.propose.allocations
    revisions should match final phase allocations (retry if present else revision)
    """
    errors: List[str] = []
    r = round_data["round"]
    phases = round_data["phases"]
    allocs = round_data["allocations"]

    # proposals
    if "proposals" in allocs and "propose" in phases:
        ok, msg = allocations_equal(allocs["proposals"], phases["propose"]["allocations"])
        if not ok:
            errors.append(f"Round {r}: allocations.proposals mismatch propose phase: {msg}")

    # revisions should match final phase
    final_key = get_final_phase(phases)
    if "revisions" in allocs and final_key in phases:
        ok, msg = allocations_equal(allocs["revisions"], phases[final_key]["allocations"])
        if not ok:
            errors.append(f"Round {r}: allocations.revisions mismatch final phase {final_key}: {msg}")

    return errors


def validate_crit(round_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    r = round_data["round"]
    crit = safe_get(round_data, "crit", {})
    metrics = safe_get(round_data, "metrics", {})
    rho_i = safe_get(metrics, "rho_i", {})

    for agent in round_data["agents"]:
        if agent not in crit:
            errors.append(f"Round {r}: missing crit entry for agent {agent}")
            continue

        c = crit[agent]
        if "rho" not in c or "pillars" not in c or "reasoning" not in c:
            errors.append(f"Round {r}: crit entry for {agent} missing rho/pillars/reasoning")
            continue

        # Check metrics.rho_i consistency
        if agent in rho_i and not approx_equal(float(rho_i[agent]), float(c["rho"]), METRIC_EPS):
            errors.append(
                f"Round {r}: crit rho mismatch for {agent} "
                f"(crit={float(c['rho']):.12f}, metrics.rho_i={float(rho_i[agent]):.12f})"
            )

        pillars = c["pillars"]
        for pk in ["logical_validity", "evidential_support", "alternative_consideration", "causal_alignment"]:
            if pk not in pillars:
                errors.append(f"Round {r}: agent {agent} missing pillar {pk}")
                continue
            val = float(pillars[pk])
            if not (0.0 - METRIC_EPS <= val <= 1.0 + METRIC_EPS):
                errors.append(
                    f"Round {r}: agent {agent} pillar {pk} out of range: {val}"
                )

        reasoning = c["reasoning"]
        for rk in ["logical_validity", "evidential_support", "alternative_consideration", "causal_alignment"]:
            if rk not in reasoning:
                errors.append(f"Round {r}: agent {agent} missing reasoning key {rk}")

    # rho_bar consistency
    if crit and rho_i:
        rho_vals = [float(v) for v in rho_i.values()]
        if rho_vals:
            mean_rho = sum(rho_vals) / len(rho_vals)
            logged_rho_bar = float(metrics["rho_bar"])
            if not approx_equal(mean_rho, logged_rho_bar, 1e-4):
                errors.append(
                    f"Round {r}: rho_bar inconsistent with mean(rho_i) "
                    f"(mean={mean_rho:.12f}, logged={logged_rho_bar:.12f})"
                )

    return errors


def validate_interventions(round_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    r = round_data["round"]
    phases = round_data["phases"]
    interventions = safe_get(round_data, "interventions", [])

    proposal_js = float(phases["propose"]["js_divergence"])
    revision_js = float(phases["revision"]["js_divergence"])
    final_key = get_final_phase(phases)
    final_js = float(phases[final_key]["js_divergence"])

    retry_keys = get_retry_keys(phases)
    has_retry = len(retry_keys) > 0

    if has_retry and not interventions:
        errors.append(f"Round {r}: retry phases exist but interventions list is empty")

    for idx, iv in enumerate(interventions):
        tag = f"Round {r} intervention[{idx}]"

        if "metrics" not in iv:
            errors.append(f"{tag}: missing metrics block")
            continue

        m = iv["metrics"]

        # Check logged proposal/revision metrics against actual proposal/revision phase
        if "js_proposal" in m and not approx_equal(float(m["js_proposal"]), proposal_js, METRIC_EPS):
            errors.append(
                f"{tag}: js_proposal mismatch "
                f"(logged={float(m['js_proposal']):.12f}, actual={proposal_js:.12f})"
            )

        if "js_revision" in m and not approx_equal(float(m["js_revision"]), revision_js, METRIC_EPS):
            errors.append(
                f"{tag}: js_revision mismatch "
                f"(logged={float(m['js_revision']):.12f}, actual={revision_js:.12f})"
            )

        if "collapse_ratio" in m and "js_proposal" in m and float(m["js_proposal"]) > 0:
            expected = float(m["js_revision"]) / float(m["js_proposal"])
            if not approx_equal(expected, float(m["collapse_ratio"]), 1e-4):
                errors.append(
                    f"{tag}: collapse_ratio mismatch "
                    f"(expected={expected:.12f}, logged={float(m['collapse_ratio']):.12f})"
                )

        action = safe_get(iv, "action", "")
        retry_num = int(safe_get(iv, "retry", 0))
        expected_retry_key = f"retry_{retry_num:03d}" if retry_num > 0 else None

        if action == "retry_revision":
            if expected_retry_key and expected_retry_key not in phases:
                errors.append(
                    f"{tag}: action=retry_revision but phase {expected_retry_key} not found"
                )

        # nudge_text basic checks for targeted agent names
        nudge_text = safe_get(iv, "nudge_text", {})
        if isinstance(nudge_text, dict):
            missing_agents = sorted(set(round_data["agents"]) - set(nudge_text.keys()))
            if missing_agents:
                errors.append(
                    f"{tag}: nudge_text missing agent keys {missing_agents}"
                )

    # final metrics should match final retry if intervention retry exists
    if has_retry:
        if not approx_equal(float(round_data["metrics"]["js_divergence"]), final_js, METRIC_EPS):
            errors.append(
                f"Round {r}: final metrics.js_divergence does not match final retry JS"
            )

    return errors


def validate_round(round_data: Dict[str, Any]) -> Tuple[List[str], float, float, Dict[str, Any]]:
    errors: List[str] = []

    errors.extend(validate_top_level_structure(round_data))
    if errors:
        return errors, float("nan"), float("nan"), {}

    errors.extend(validate_phase_structure(round_data))
    errors.extend(validate_vector_alignment(round_data))
    errors.extend(validate_allocation_mass(round_data))
    js_errors, final_js, proposal_js = validate_js_and_overlap(round_data)
    errors.extend(js_errors)
    errors.extend(validate_allocation_snapshots(round_data))
    errors.extend(validate_crit(round_data))
    errors.extend(validate_interventions(round_data))

    final_key = get_final_phase(round_data["phases"])
    final_phase = round_data["phases"][final_key]

    summary = {
        "round": round_data["round"],
        "proposal_phase": "propose",
        "final_phase": final_key,
        "proposal_js": proposal_js,
        "final_js": final_js,
        "proposal_evidence_overlap": float(round_data["phases"]["propose"]["evidence_overlap"]),
        "final_evidence_overlap": float(final_phase["evidence_overlap"]),
        "intervention_count": len(round_data.get("interventions", [])),
    }

    return errors, final_js, proposal_js, summary


def validate_js_chain(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_errors: List[str] = []
    round_summaries: List[Dict[str, Any]] = []
    previous_final_js = None
    previous_final_allocs = None
    previous_round_num = None

    # sort rounds defensively
    sorted_rounds = sorted(rounds, key=lambda r: r["round"])

    # round numbering continuity
    expected = None
    for r in sorted_rounds:
        rn = r["round"]
        if expected is None:
            expected = rn
        else:
            expected += 1
            if rn != expected:
                all_errors.append(
                    f"Round numbering discontinuity: expected round {expected}, found round {rn}"
                )
                expected = rn

    for r in sorted_rounds:
        errors, final_js, proposal_js, summary = validate_round(r)
        all_errors.extend(errors)
        round_summaries.append(summary)

        # cross-round JS continuity:
        # final revision/retry JS of round r-1 must match proposal JS of round r
        if previous_final_js is not None:
            if not approx_equal(previous_final_js, proposal_js, METRIC_EPS):
                all_errors.append(
                    f"JS continuity violation between rounds {previous_round_num} → {r['round']}: "
                    f"previous_final_js={previous_final_js:.12f} "
                    f"current_proposal_js={proposal_js:.12f}"
                )

        # cross-round allocation continuity:
        # final allocations of round r-1 should equal proposal allocations of round r
        if previous_final_allocs is not None:
            current_propose_allocs = r["phases"]["propose"]["allocations"]
            ok, msg = allocations_equal(previous_final_allocs, current_propose_allocs, METRIC_EPS)
            if not ok:
                all_errors.append(
                    f"Allocation continuity violation between rounds {previous_round_num} → {r['round']}: {msg}"
                )

        final_key = get_final_phase(r["phases"])
        previous_final_js = final_js
        previous_final_allocs = r["phases"][final_key]["allocations"]
        previous_round_num = r["round"]

    return {
        "ok": len(all_errors) == 0,
        "error_count": len(all_errors),
        "errors": all_errors,
        "round_summaries": round_summaries,
    }


# =========================================================
# CLI / file handling
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def validate_file(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = validate_js_chain(data)

    if report["ok"]:
        print("✓ Debate integrity validation passed")
    else:
        print("✗ Debate integrity validation failed")
        print(f"  Errors: {report['error_count']}")
        for e in report["errors"]:
            print("  -", e)

    print("\nRound summaries:")
    for rs in report["round_summaries"]:
        print(
            f"  Round {rs['round']}: "
            f"proposal_js={rs['proposal_js']:.6f}, "
            f"final_phase={rs['final_phase']}, "
            f"final_js={rs['final_js']:.6f}, "
            f"interventions={rs['intervention_count']}"
        )

    return report


def validate_experiment(experiment_path: str) -> None:
    """Find and validate every run's raw_run_fingerprint.json under an experiment dir."""
    exp = Path(experiment_path)
    fingerprints = sorted(exp.glob("run_*/raw_run_fingerprint.json"))
    if not fingerprints:
        print(f"No raw_run_fingerprint.json files found under {exp}", file=sys.stderr)
        sys.exit(1)

    total_pass = 0
    total_fail = 0
    total_skip = 0
    failed_runs: List[str] = []

    for fp in fingerprints:
        run_name = fp.parent.name
        data = load_json(str(fp))

        # Skip rounds with no telemetry or missing required phases
        valid_rounds = [
            r for r in data
            if r.get("round") is not None
            and r.get("phases", {}).get("propose") is not None
            and r.get("phases", {}).get("revision") is not None
        ]
        if not valid_rounds:
            print(f"\n  {run_name}: SKIPPED (no telemetry data or incomplete rounds)")
            total_skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"{'='*60}")
        try:
            report = validate_file(valid_rounds)
        except Exception as e:
            print(f"  ERROR: {e}")
            total_fail += 1
            failed_runs.append(f"{run_name} (exception: {e})")
            continue

        if report["ok"]:
            total_pass += 1
        else:
            total_fail += 1
            failed_runs.append(run_name)

        # Write per-run report alongside fingerprint
        report_path = fp.parent / "validation_report.json"
        write_json(str(report_path), report)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {total_pass} passed, {total_fail} failed, {total_skip} skipped out of {len(fingerprints)} runs")
    if failed_runs:
        print(f"  Failed runs:")
        for r in failed_runs:
            print(f"    - {r}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: find experiment dir relative to this script's location
        script_dir = Path(__file__).resolve().parent
        experiment_root = script_dir.parent.parent / "logging" / "runs" / "vskarich_ablation_6"
        if experiment_root.is_dir():
            validate_experiment(str(experiment_root))
            sys.exit(0)
        print(
            "Usage:\n"
            "  python validate_data.py <experiment_dir | input.json> [output_report.json]\n"
        )
        sys.exit(1)

    target = Path(sys.argv[1])

    # If target is a directory, treat as experiment
    if target.is_dir():
        validate_experiment(str(target))
    else:
        # Single file mode (original behavior)
        output_path = sys.argv[2] if len(sys.argv) >= 3 else None
        data = load_json(str(target))
        report = validate_file(data)
        if output_path is not None:
            write_json(output_path, report)
            print(f"\nValidation report written to {output_path}")