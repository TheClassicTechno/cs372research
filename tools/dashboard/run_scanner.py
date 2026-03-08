"""Filesystem scanner for debate run artifacts.

Reads the ``logging/runs/`` directory tree and exposes structured data
for the dashboard API.  Pure filesystem module — no FastAPI dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BASE = Path("logging/runs")
DAILY_PRICES_BASE = Path("data-pipeline/daily_prices/data")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _safe_json(path: Path) -> dict | list | None:
    """Read and parse a JSON file, returning None on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _get_run_status(run_dir: Path) -> str:
    """Determine run health from available files.

    Returns one of ``"complete"``, ``"partial"``, or ``"incomplete"``.
    """
    if not (run_dir / "manifest.json").exists():
        return "incomplete"
    if not (run_dir / "final").is_dir():
        return "incomplete"
    if not (run_dir / "final" / "final_portfolio.json").exists():
        return "partial"
    return "complete"


def _load_trajectories(run_dir: Path) -> list[dict]:
    """Load round metrics with fallback for old/partial runs.

    Prefers ``final/pid_crit_all_rounds.json`` (aggregated).  If missing,
    reconstructs from per-round ``metrics/`` files.
    """
    agg = run_dir / "final" / "pid_crit_all_rounds.json"
    data = _safe_json(agg)
    if isinstance(data, list) and data:
        return data

    # Fallback: reconstruct from per-round metric files
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.is_dir():
        return []

    rounds: list[dict] = []
    for rd in sorted(rounds_dir.glob("round_*")):
        try:
            round_num = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        entry: dict[str, Any] = {"round": round_num}
        metrics_dir = rd / "metrics"
        if not metrics_dir.is_dir():
            rounds.append(entry)
            continue

        for fname, key in [
            ("crit_scores.json", "crit"),
            ("pid_state.json", "pid"),
            ("js_divergence.json", "divergence"),
            ("evidence_overlap.json", "evidence"),
            ("js_divergence_propose.json", "divergence_propose"),
            ("evidence_overlap_propose.json", "evidence_propose"),
        ]:
            fdata = _safe_json(metrics_dir / fname)
            if fdata is not None:
                entry[key] = fdata
        rounds.append(entry)
    return rounds


def _extract_quality_metrics(run_dir: Path) -> dict:
    """Extract summary quality metrics from trajectories.

    Returns dict with ``final_rho_bar``, ``final_beta``, ``js_drop``,
    and ``reasoning_collapse`` flag.
    """
    traj = _load_trajectories(run_dir)
    if not traj:
        return {}

    rho_bars = []
    betas = []
    js_values = []

    for entry in traj:
        # Extract rho_bar
        crit = entry.get("crit")
        if isinstance(crit, dict):
            rb = crit.get("rho_bar")
            if rb is not None:
                rho_bars.append(rb)

        # Extract beta
        pid = entry.get("pid")
        if isinstance(pid, dict):
            b = pid.get("beta_new")
            if b is not None:
                betas.append(b)

        # Extract JS divergence
        div = entry.get("divergence")
        if isinstance(div, dict):
            js = div.get("js_divergence")
            if js is not None:
                js_values.append(js)

    result: dict[str, Any] = {}
    if rho_bars:
        result["final_rho_bar"] = rho_bars[-1]
        result["reasoning_collapse"] = (rho_bars[-1] - rho_bars[0]) < -0.05
    if betas:
        result["final_beta"] = betas[-1]
    if len(js_values) >= 2:
        result["js_drop"] = round(js_values[0] - js_values[-1], 4)

    return result


def _quarter_from_date(date_str: str) -> tuple[int, str] | None:
    """Parse an invest_quarter date like '2023-06-30' into (year, 'Q2').

    Also handles 'YYYYQN' format directly.
    """
    if not date_str:
        return None
    s = date_str.strip()
    # Handle "2023Q2" style
    if "Q" in s.upper():
        parts = s.upper().replace("Q", " Q").split()
        try:
            return int(parts[0]), parts[1]
        except (IndexError, ValueError):
            return None
    # Handle "2023-06-30" style
    try:
        month = int(s.split("-")[1])
    except (IndexError, ValueError):
        return None
    year = int(s.split("-")[0])
    q = {1: "Q1", 2: "Q1", 3: "Q1",
         4: "Q2", 5: "Q2", 6: "Q2",
         7: "Q3", 8: "Q3", 9: "Q3",
         10: "Q4", 11: "Q4", 12: "Q4"}.get(month)
    if q is None:
        return None
    return year, q


def compute_portfolio_performance(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    prices_base: Path = DAILY_PRICES_BASE,
) -> dict:
    """Compute portfolio return from final allocation and daily prices.

    Returns ``{initial_capital, final_value, profit, return_pct,
    allocations: {ticker: {weight, price_start, price_end, return_pct}},
    spy: {price_start, price_end, return_pct}}``.
    """
    run_dir = base_path / experiment / run_id
    manifest = _safe_json(run_dir / "manifest.json")
    portfolio = _safe_json(run_dir / "final" / "final_portfolio.json")

    if not manifest or not portfolio:
        return {"error": "Missing manifest or portfolio"}

    invest_q = manifest.get("invest_quarter")
    parsed = _quarter_from_date(invest_q) if invest_q else None
    if parsed is None:
        return {"error": f"Cannot parse invest_quarter: {invest_q}"}

    year, quarter = parsed
    initial_capital = 100_000.0

    # Load prices for each ticker
    allocations: dict[str, dict] = {}
    final_value = 0.0
    missing_tickers: list[str] = []

    for ticker, weight in portfolio.items():
        if weight == 0:
            continue
        # Map _CASH_ to no growth
        if ticker == "_CASH_":
            allocations[ticker] = {
                "weight": weight,
                "price_start": 1.0,
                "price_end": 1.0,
                "return_pct": 0.0,
            }
            final_value += weight * initial_capital
            continue

        price_file = prices_base / ticker / f"{year}_{quarter}.json"
        price_data = _safe_json(price_file)
        if not price_data or not price_data.get("daily_close"):
            missing_tickers.append(ticker)
            # Assume flat if no data
            allocations[ticker] = {
                "weight": weight,
                "price_start": None,
                "price_end": None,
                "return_pct": 0.0,
            }
            final_value += weight * initial_capital
            continue

        closes = price_data["daily_close"]
        p_start = closes[0]["close"]
        p_end = closes[-1]["close"]
        ticker_return = (p_end - p_start) / p_start
        final_value += weight * initial_capital * (1 + ticker_return)

        allocations[ticker] = {
            "weight": round(weight, 4),
            "price_start": round(p_start, 2),
            "price_end": round(p_end, 2),
            "return_pct": round(ticker_return * 100, 2),
        }

    profit = final_value - initial_capital
    return_pct = (profit / initial_capital) * 100

    result: dict[str, Any] = {
        "quarter": f"{year}_{quarter}",
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "return_pct": round(return_pct, 2),
        "allocations": allocations,
    }

    if missing_tickers:
        result["missing_tickers"] = missing_tickers

    # SPY benchmark
    spy_file = prices_base / "SPY" / f"{year}_{quarter}.json"
    spy_data = _safe_json(spy_file)
    if spy_data and spy_data.get("daily_close"):
        spy_closes = spy_data["daily_close"]
        spy_start = spy_closes[0]["close"]
        spy_end = spy_closes[-1]["close"]
        spy_ret = (spy_end - spy_start) / spy_start
        result["spy"] = {
            "price_start": round(spy_start, 2),
            "price_end": round(spy_end, 2),
            "return_pct": round(spy_ret * 100, 2),
        }

    return result


def _find_active_run(base_path: Path) -> tuple[str, str, Path] | None:
    """Find the most recently started run across all experiments.

    Returns ``(experiment, run_id, run_dir)`` or ``None``.
    """
    latest_mtime = 0.0
    best: tuple[str, str, Path] | None = None

    if not base_path.is_dir():
        return None

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                mtime = manifest_path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                best = (exp_dir.name, run_dir.name, run_dir)

    return best


def _read_text_safe(path: Path, max_chars: int = 8000) -> str | None:
    """Read a text file, truncating to *max_chars*."""
    try:
        text = path.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text
    except OSError:
        return None


def get_live_events(base_path: Path = DEFAULT_BASE) -> dict:
    """Scan the most recent run for debate events in write order.

    Returns ``{experiment, run_id, status, events: [...]}``.
    Each event has ``{id, round, phase, agent, content, mtime}``.
    """
    found = _find_active_run(base_path)
    if found is None:
        return {"experiment": None, "run_id": None, "status": "no_runs", "events": []}

    experiment, run_id, run_dir = found
    status = _get_run_status(run_dir)

    events: list[dict[str, Any]] = []

    rounds_dir = run_dir / "rounds"
    if not rounds_dir.is_dir():
        return {
            "experiment": experiment,
            "run_id": run_id,
            "status": status,
            "events": [],
        }

    for rd in sorted(rounds_dir.glob("round_*")):
        try:
            round_num = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Scan phases in the order they are written
        phase_specs: list[tuple[str, str, str, bool]] = []
        # (phase_label, subdir, filename, is_json)

        # Proposals
        proposals_dir = rd / "proposals"
        if proposals_dir.is_dir():
            for agent_dir in sorted(proposals_dir.iterdir()):
                if agent_dir.is_dir():
                    phase_specs.append(("PROPOSAL", "proposals", agent_dir.name, False))

        # Critiques
        critiques_dir = rd / "critiques"
        if critiques_dir.is_dir():
            for agent_dir in sorted(critiques_dir.iterdir()):
                if agent_dir.is_dir():
                    phase_specs.append(("CRITIQUE", "critiques", agent_dir.name, True))

        # Revisions
        revisions_dir = rd / "revisions"
        if revisions_dir.is_dir():
            for agent_dir in sorted(revisions_dir.iterdir()):
                if agent_dir.is_dir():
                    phase_specs.append(("REVISION", "revisions", agent_dir.name, False))

        # CRIT evaluations
        crit_dir = rd / "CRIT"
        if crit_dir.is_dir():
            for agent_dir in sorted(crit_dir.iterdir()):
                if agent_dir.is_dir():
                    phase_specs.append(("CRIT", "CRIT", agent_dir.name, False))

        # Metrics (single entries, not per-agent)
        metrics_dir = rd / "metrics"
        if metrics_dir.is_dir():
            for mfile in ["crit_scores.json", "pid_state.json"]:
                if (metrics_dir / mfile).exists():
                    phase_specs.append(("METRICS", "metrics", mfile, True))

        for phase_label, subdir, agent_or_file, is_json in phase_specs:
            if subdir == "metrics":
                fpath = rd / subdir / agent_or_file
                event_id = f"r{round_num}/{subdir}/{agent_or_file}"
                content_raw = _safe_json(fpath)
                if content_raw is None:
                    continue
                try:
                    mtime = fpath.stat().st_mtime
                except OSError:
                    mtime = 0.0
                events.append({
                    "id": event_id,
                    "round": round_num,
                    "phase": phase_label,
                    "agent": agent_or_file.replace(".json", ""),
                    "content": json.dumps(content_raw, indent=2),
                    "mtime": mtime,
                })
            else:
                agent_dir_path = rd / subdir / agent_or_file
                if is_json:
                    resp_file = agent_dir_path / "response.json"
                else:
                    resp_file = agent_dir_path / "response.txt"

                if not resp_file.exists():
                    continue

                event_id = f"r{round_num}/{subdir}/{agent_or_file}"

                try:
                    mtime = resp_file.stat().st_mtime
                except OSError:
                    mtime = 0.0

                if is_json:
                    content_data = _safe_json(resp_file)
                    content = json.dumps(content_data, indent=2) if content_data is not None else ""
                else:
                    content = _read_text_safe(resp_file) or ""

                entry: dict[str, Any] = {
                    "id": event_id,
                    "round": round_num,
                    "phase": phase_label,
                    "agent": agent_or_file,
                    "content": content,
                    "mtime": mtime,
                }

                # Include portfolio if available
                portfolio = _safe_json(agent_dir_path / "portfolio.json")
                if portfolio is not None:
                    entry["portfolio"] = portfolio

                events.append(entry)

    # Sort by mtime so events appear in the order they were written
    events.sort(key=lambda e: e["mtime"])

    return {
        "experiment": experiment,
        "run_id": run_id,
        "status": status,
        "events": events,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def list_experiments(base_path: Path = DEFAULT_BASE) -> list[str]:
    """Return sorted experiment directory names."""
    if not base_path.is_dir():
        return []
    return sorted(
        d.name for d in base_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _read_agent_profiles_from_config(run_dir: Path) -> dict[str, str] | None:
    """Read the agents mapping from a debate config YAML in final/.

    Returns {role: profile_name} or None if unavailable.
    """
    final_dir = run_dir / "final"
    if not final_dir.is_dir():
        return None
    for f in final_dir.iterdir():
        if f.suffix == ".yaml" and f.stem.startswith("debate"):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("agents"), dict):
                    return data["agents"]
            except Exception:
                continue
    return None


def list_runs(base_path: Path = DEFAULT_BASE, experiment: str = "default") -> list[dict]:
    """Return run summaries with status and quality metrics.

    If ``manifest.json`` is missing, returns a minimal entry with
    ``status: "incomplete"`` and no other metadata.
    """
    exp_dir = base_path / experiment
    if not exp_dir.is_dir():
        return []

    runs: list[dict] = []
    for run_dir in sorted(exp_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        status = _get_run_status(run_dir)
        entry: dict[str, Any] = {"run_id": run_dir.name, "status": status}

        manifest = _safe_json(run_dir / "manifest.json")
        if manifest is not None:
            for key in (
                "started_at", "completed_at", "model_name", "roles",
                "actual_rounds", "terminated_early", "termination_reason",
                "pid_enabled", "invest_quarter", "ticker_universe",
                "initial_beta", "final_beta", "config_paths", "agent_profiles",
            ):
                if key in manifest:
                    entry[key] = manifest[key]

            # Fallback: if manifest has no agent_profiles, read from config YAML
            if not entry.get("agent_profiles"):
                cfg_profiles = _read_agent_profiles_from_config(run_dir)
                if cfg_profiles:
                    entry["agent_profiles"] = cfg_profiles

        # Quality metrics
        if status != "incomplete":
            entry.update(_extract_quality_metrics(run_dir))

        runs.append(entry)

    return runs


def get_run_detail(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> dict | None:
    """Return full run detail: manifest + pid_config + final portfolio + round summaries.

    Returns ``None`` if the run directory does not exist.
    """
    run_dir = base_path / experiment / run_id
    if not run_dir.is_dir():
        return None

    detail: dict[str, Any] = {
        "run_id": run_id,
        "experiment": experiment,
        "status": _get_run_status(run_dir),
    }

    manifest = _safe_json(run_dir / "manifest.json")
    if manifest and not manifest.get("agent_profiles"):
        cfg_profiles = _read_agent_profiles_from_config(run_dir)
        if cfg_profiles:
            manifest["agent_profiles"] = cfg_profiles
    detail["manifest"] = manifest
    detail["pid_config"] = _safe_json(run_dir / "pid_config.json")
    detail["final_portfolio"] = _safe_json(run_dir / "final" / "final_portfolio.json")
    detail["quality"] = _extract_quality_metrics(run_dir)

    # Round summaries from round_state.json
    round_summaries = []
    rounds_dir = run_dir / "rounds"
    if rounds_dir.is_dir():
        for rd in sorted(rounds_dir.glob("round_*")):
            rs = _safe_json(rd / "round_state.json")
            if rs is not None:
                round_summaries.append(rs)
            else:
                try:
                    round_num = int(rd.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                round_summaries.append({"round": round_num, "incomplete": True})
    detail["round_summaries"] = round_summaries

    return detail


def get_pid_trajectory(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Extract PID fields per round from trajectories."""
    run_dir = base_path / experiment / run_id
    traj = _load_trajectories(run_dir)

    result = []
    for entry in traj:
        pid = entry.get("pid") or {}
        row: dict[str, Any] = {
            "round": entry.get("round"),
            "beta_in": entry.get("beta_in") or pid.get("beta_in"),
            "beta_new": pid.get("beta_new"),
            "tone_bucket": entry.get("tone_bucket") or pid.get("tone_bucket"),
            "e_t": pid.get("error", {}).get("e_t") if isinstance(pid.get("error"), dict) else pid.get("e_t"),
            "u_t": pid.get("u_t"),
            "quadrant": pid.get("quadrant"),
        }
        # Also include rho_bar for the chart
        crit = entry.get("crit") or {}
        row["rho_bar"] = crit.get("rho_bar")
        result.append(row)
    return result


def get_crit_trajectory(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Extract CRIT fields per round from trajectories."""
    run_dir = base_path / experiment / run_id
    traj = _load_trajectories(run_dir)

    result = []
    for entry in traj:
        crit = entry.get("crit") or {}
        row: dict[str, Any] = {
            "round": entry.get("round"),
            "rho_bar": crit.get("rho_bar"),
        }
        # Per-agent rho_i — try flat format first, then agent_scores format
        rho_i = crit.get("rho_i")
        if isinstance(rho_i, dict):
            row["rho_i"] = rho_i

        # Per-agent pillar scores — try agents.{role}.pillars first
        agents = crit.get("agents") or {}
        pillars: dict[str, dict] = {}
        for role, agent_data in agents.items():
            if isinstance(agent_data, dict) and "pillars" in agent_data:
                pillars[role] = agent_data["pillars"]

        # Fallback: agent_scores.{role}.rho_i / .pillar_scores format
        agent_scores = crit.get("agent_scores") or {}
        if isinstance(agent_scores, dict) and agent_scores:
            if not isinstance(rho_i, dict):
                row["rho_i"] = {
                    role: ad["rho_i"]
                    for role, ad in agent_scores.items()
                    if isinstance(ad, dict) and "rho_i" in ad
                }
            if not pillars:
                for role, ad in agent_scores.items():
                    if isinstance(ad, dict) and "pillar_scores" in ad:
                        pillars[role] = ad["pillar_scores"]

        if pillars:
            row["pillars"] = pillars

        result.append(row)
    return result


def get_divergence_trajectory(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Extract JS divergence + evidence overlap per phase per round.

    The aggregated ``pid_crit_all_rounds.json`` never contains propose-phase
    data, so we always read propose metrics directly from per-round files.
    """
    run_dir = base_path / experiment / run_id
    traj = _load_trajectories(run_dir)

    # Build a lookup of propose-phase metrics from per-round files,
    # since the aggregated file never includes them.
    propose_by_round: dict[int, dict] = {}
    rounds_dir = run_dir / "rounds"
    if rounds_dir.is_dir():
        for rd in sorted(rounds_dir.glob("round_*")):
            try:
                rn = int(rd.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            metrics_dir = rd / "metrics"
            if not metrics_dir.is_dir():
                continue
            div_p = _safe_json(metrics_dir / "js_divergence_propose.json")
            ev_p = _safe_json(metrics_dir / "evidence_overlap_propose.json")
            if div_p or ev_p:
                propose_by_round[rn] = {"div": div_p or {}, "ev": ev_p or {}}

    result = []
    for entry in traj:
        round_num = entry.get("round")

        # --- Propose phase (prefer trajectory entry, fall back to per-round files) ---
        div_p = entry.get("divergence_propose") or {}
        ev_p = entry.get("evidence_propose") or {}
        if not div_p and not ev_p and round_num in propose_by_round:
            div_p = propose_by_round[round_num]["div"]
            ev_p = propose_by_round[round_num]["ev"]
        if div_p or ev_p:
            result.append({
                "round": round_num,
                "phase": "propose",
                "js_divergence": div_p.get("js_divergence"),
                "mean_overlap": ev_p.get("mean_overlap"),
                "agent_confidences": div_p.get("agent_confidences"),
            })

        # --- Revise phase ---
        div_r = entry.get("divergence") or {}
        ev_r = entry.get("evidence") or {}
        js_r = div_r.get("js_divergence") or div_r.get("js")
        ov_r = ev_r.get("mean_overlap") or div_r.get("ov")
        if js_r is not None or ov_r is not None:
            result.append({
                "round": round_num,
                "phase": "revise",
                "js_divergence": js_r,
                "mean_overlap": ov_r,
                "agent_confidences": div_r.get("agent_confidences"),
            })
    return result


def get_portfolio_trajectory(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Read each round's revised portfolio per agent.

    Returns list of ``{round, allocations: {role: {ticker: weight}}, consensus: {ticker: weight}}``.
    """
    run_dir = base_path / experiment / run_id
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.is_dir():
        return []

    result = []
    for rd in sorted(rounds_dir.glob("round_*")):
        try:
            round_num = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        allocations: dict[str, dict] = {}
        rev_dir = rd / "revisions"
        if rev_dir.is_dir():
            for agent_dir in sorted(rev_dir.iterdir()):
                if not agent_dir.is_dir():
                    continue
                portfolio = _safe_json(agent_dir / "portfolio.json")
                if isinstance(portfolio, dict):
                    allocations[agent_dir.name] = portfolio

        # Compute consensus (mean across agents)
        consensus: dict[str, float] = {}
        if allocations:
            all_tickers: set[str] = set()
            for alloc in allocations.values():
                all_tickers.update(alloc.keys())
            n = len(allocations)
            for ticker in sorted(all_tickers):
                total = sum(alloc.get(ticker, 0.0) for alloc in allocations.values())
                consensus[ticker] = round(total / n, 4)

        result.append({
            "round": round_num,
            "allocations": allocations,
            "consensus": consensus,
        })
    return result


def get_round_detail(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    round_num: int = 1,
) -> dict | None:
    """Read round_state.json + agent text for proposals/critiques/revisions.

    Returns ``None`` if the round directory does not exist.
    """
    run_dir = base_path / experiment / run_id
    rd = run_dir / "rounds" / f"round_{round_num:03d}"
    if not rd.is_dir():
        return None

    detail: dict[str, Any] = {"round": round_num}
    detail["round_state"] = _safe_json(rd / "round_state.json")

    # Agent text
    agents: dict[str, dict] = {}
    for phase_name, subdir_name, is_json in [
        ("proposal", "proposals", False),
        ("critique", "critiques", True),
        ("revision", "revisions", False),
    ]:
        phase_dir = rd / subdir_name
        if not phase_dir.is_dir():
            continue
        for agent_dir in sorted(phase_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            role = agent_dir.name
            if role not in agents:
                agents[role] = {}

            if is_json:
                data = _safe_json(agent_dir / "response.json")
                agents[role][phase_name] = data
            else:
                resp_path = agent_dir / "response.txt"
                try:
                    agents[role][phase_name] = resp_path.read_text(encoding="utf-8")
                except OSError:
                    agents[role][phase_name] = None

            # Portfolio for proposals/revisions
            if not is_json:
                portfolio = _safe_json(agent_dir / "portfolio.json")
                if portfolio is not None:
                    agents[role][f"{phase_name}_portfolio"] = portfolio

    # CRIT request/response per agent
    crit_dir = rd / "CRIT"
    if crit_dir.is_dir():
        for agent_dir in sorted(crit_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            role = agent_dir.name
            if role not in agents:
                agents[role] = {}

            prompt_path = agent_dir / "prompt.txt"
            try:
                agents[role]["crit_request"] = prompt_path.read_text(encoding="utf-8")
            except OSError:
                agents[role]["crit_request"] = None

            response_path = agent_dir / "response.txt"
            try:
                agents[role]["crit_response"] = response_path.read_text(encoding="utf-8")
            except OSError:
                agents[role]["crit_response"] = None

    detail["agents"] = agents

    # Metrics
    metrics_dir = rd / "metrics"
    if metrics_dir.is_dir():
        detail["crit_scores"] = _safe_json(metrics_dir / "crit_scores.json")
        detail["pid_state"] = _safe_json(metrics_dir / "pid_state.json")
        detail["js_divergence"] = _safe_json(metrics_dir / "js_divergence.json")
        detail["evidence_overlap"] = _safe_json(metrics_dir / "evidence_overlap.json")

    return detail


def get_file_tree(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Recursive directory listing for the run directory.

    Returns ``[{name, path, type, size_bytes, children}]``.
    """
    run_dir = base_path / experiment / run_id
    if not run_dir.is_dir():
        return []

    def _walk(d: Path, prefix: str = "") -> list[dict]:
        entries = []
        for item in sorted(d.iterdir()):
            rel = f"{prefix}/{item.name}" if prefix else item.name
            if item.is_dir():
                entries.append({
                    "name": item.name,
                    "path": rel,
                    "type": "dir",
                    "children": _walk(item, rel),
                })
            else:
                try:
                    size = item.stat().st_size
                except OSError:
                    size = 0
                entries.append({
                    "name": item.name,
                    "path": rel,
                    "type": "file",
                    "size_bytes": size,
                })
        return entries

    return _walk(run_dir)


def read_run_file(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    relative_path: str = "",
) -> str:
    """Read any file within a run directory.

    Raises ``PermissionError`` if the resolved path escapes the run directory.
    Raises ``FileNotFoundError`` if the file does not exist.
    """
    run_root = (base_path / experiment / run_id).resolve()
    resolved = (run_root / relative_path).resolve()
    if not resolved.is_relative_to(run_root):
        raise PermissionError("Path traversal rejected")
    if not resolved.is_file():
        raise FileNotFoundError(f"Not found: {relative_path}")
    return resolved.read_text(encoding="utf-8")


def diff_run_configs(
    base_path: Path = DEFAULT_BASE,
    exp1: str = "",
    run1: str = "",
    exp2: str = "",
    run2: str = "",
) -> dict:
    """Compare two run manifests.

    Returns ``{only_left, only_right, different, shared}``.
    """
    m1 = _safe_json(base_path / exp1 / run1 / "manifest.json") or {}
    m2 = _safe_json(base_path / exp2 / run2 / "manifest.json") or {}

    keys1 = set(m1.keys())
    keys2 = set(m2.keys())

    only_left = {k: m1[k] for k in sorted(keys1 - keys2)}
    only_right = {k: m2[k] for k in sorted(keys2 - keys1)}
    shared = {}
    different = {}

    for k in sorted(keys1 & keys2):
        if m1[k] == m2[k]:
            shared[k] = m1[k]
        else:
            different[k] = {"left": m1[k], "right": m2[k]}

    return {
        "left": f"{exp1}/{run1}",
        "right": f"{exp2}/{run2}",
        "only_left": only_left,
        "only_right": only_right,
        "different": different,
        "shared": shared,
    }
