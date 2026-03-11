"""Write-only telemetry for debate round metrics.

Captures full computational context (allocation vectors, CRIT pillar scores)
at the moment metrics are produced, without mutating debate state.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_floats(obj, ndigits=4):
    """Recursively round floats in nested dicts/lists for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _write_json(path: Path, data: dict) -> None:
    """Atomic write: temp file then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Public write functions
# ---------------------------------------------------------------------------

def write_allocation_metrics(
    run_dir: Path,
    round_num: int,
    phase: str,
    decisions: list[dict],
    js_divergence: float,
    evidence_overlap: float,
) -> None:
    """Write full allocation vectors and divergence context for a debate phase."""
    try:
        allocations = {
            d["role"]: d["action_dict"]["allocation"]
            for d in decisions
        }
        tickers = sorted(set().union(*(a.keys() for a in allocations.values())))
        vectors = {
            role: [alloc.get(t, 0.0) for t in tickers]
            for role, alloc in allocations.items()
        }
        allocation_sums = {
            role: sum(alloc.values()) for role, alloc in allocations.items()
        }
        nonzero_positions = {
            role: sum(1 for v in alloc.values() if v > 0)
            for role, alloc in allocations.items()
        }
        vector_norms = {role: sum(vec) for role, vec in vectors.items()}

        data = _round_floats({
            "round": round_num,
            "phase": phase,
            "phase_iteration": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": [d["role"] for d in decisions],
            "agent_count": len(decisions),
            "tickers": tickers,
            "ticker_count": len(tickers),
            "allocations": allocations,
            "vectors": vectors,
            "allocation_sums": allocation_sums,
            "nonzero_positions": nonzero_positions,
            "vector_norms": vector_norms,
            "js_divergence": js_divergence,
            "evidence_overlap": evidence_overlap,
        })

        path = run_dir / "rounds" / f"round_{round_num:03d}" / f"metrics_{phase}.json"
        _write_json(path, data)
    except Exception:
        logger.debug("telemetry: failed to write allocation metrics", exc_info=True)


def write_round_state(
    run_dir: Path,
    round_num: int,
    beta: float | None,
    propose_decisions: list[dict],
    revision_decisions: list[dict],
    js_divergence: float,
    evidence_overlap: float,
    crit_data: dict | None = None,
    pid_data: dict | None = None,
) -> None:
    """Write round_state.json — compact round snapshot with proposals, revisions, and metrics."""
    try:
        proposals_summary = {}
        for d in propose_decisions:
            role = d["role"]
            proposals_summary[role] = {
                "allocation": d["action_dict"]["allocation"],
                "confidence": d["action_dict"]["confidence"],
            }

        revisions_summary = {}
        for d in revision_decisions:
            role = d["role"]
            revisions_summary[role] = {
                "allocation": d["action_dict"]["allocation"],
                "confidence": d["action_dict"]["confidence"],
            }

        data = _round_floats({
            "round": round_num,
            "beta": beta,
            "proposals": proposals_summary,
            "revisions": revisions_summary,
            "metrics": {
                "js_divergence": js_divergence,
                "evidence_overlap": evidence_overlap,
            },
            "crit": crit_data or {},
            "pid": pid_data or {},
        })

        path = run_dir / "rounds" / f"round_{round_num:03d}" / "round_state.json"
        _write_json(path, data)
    except Exception:
        logger.debug("telemetry: failed to write round state", exc_info=True)


def write_crit_results(
    run_dir: Path,
    round_num: int,
    role: str,
    pillar_scores: dict,
    rho: float,
    reasoning: dict,
) -> None:
    """Write per-agent CRIT pillar scores and reasoning."""
    try:
        data = _round_floats({
            "round": round_num,
            "phase": "critique",
            "agent": role,
            "pillars": pillar_scores,
            "rho": rho,
            "reasoning": reasoning,
        })

        path = run_dir / "rounds" / f"round_{round_num:03d}" / f"crit_{role}.json"
        _write_json(path, data)
    except Exception:
        logger.debug("telemetry: failed to write CRIT results", exc_info=True)
