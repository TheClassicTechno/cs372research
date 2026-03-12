"""Filesystem scanner for debate run artifacts.

Reads the ``logging/runs/`` directory tree and exposes structured data
for the dashboard API.  Pure filesystem module — no FastAPI dependency.
"""

from __future__ import annotations

import json
import math
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


def _relative_run_dir(run_dir: Path) -> str:
    """Return run_dir as a path relative to the repo content root.

    Looks for ``logging/runs/`` in the resolved path and returns
    everything from that point.  Falls back to the full path if the
    marker is not found.
    """
    resolved = str(run_dir.resolve())
    marker = "logging/runs/"
    idx = resolved.find(marker)
    if idx != -1:
        return resolved[idx:]
    return resolved


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
            js = div.get("js_divergence") or div.get("js")
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


def _portfolio_value(
    portfolio: dict[str, float],
    prices_base: Path,
    year: str,
    quarter: str,
    initial_capital: float = 100_000.0,
) -> dict:
    """Compute performance metrics for a single portfolio allocation.

    Returns ``{initial_capital, final_value, profit, return_pct}``.
    Shared helper used by both consensus and per-agent performance.
    """
    final_value = 0.0
    for ticker, weight in portfolio.items():
        if weight == 0:
            continue
        if ticker == "_CASH_":
            final_value += weight * initial_capital
            continue
        price_file = prices_base / ticker / f"{year}_{quarter}.json"
        price_data = _safe_json(price_file)
        if not price_data or not price_data.get("daily_close"):
            final_value += weight * initial_capital
            continue
        closes = price_data["daily_close"]
        p_start = closes[0]["close"]
        p_end = closes[-1]["close"]
        ticker_return = (p_end - p_start) / p_start
        final_value += weight * initial_capital * (1 + ticker_return)

    profit = final_value - initial_capital
    return_pct = (profit / initial_capital) * 100
    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "return_pct": round(return_pct, 2),
    }


def compute_agent_performance(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    prices_base: Path = DAILY_PRICES_BASE,
) -> dict:
    """Compute per-agent portfolio performance from final-round allocations.

    Returns ``{agents: {role: {initial_capital, final_value, profit,
    return_pct}}}`` or ``{error: ...}`` on failure.
    """
    run_dir = base_path / experiment / run_id
    manifest = _safe_json(run_dir / "manifest.json")
    if not manifest:
        return {"error": "Missing manifest"}

    invest_q = manifest.get("invest_quarter")
    parsed = _quarter_from_date(invest_q) if invest_q else None
    if parsed is None:
        return {"error": f"Cannot parse invest_quarter: {invest_q}"}

    year, quarter = parsed

    trajectory = get_portfolio_trajectory(base_path, experiment, run_id)
    if not trajectory:
        return {"error": "No portfolio trajectory data"}

    last_round = trajectory[-1]
    allocations = last_round.get("allocations")
    if not allocations:
        return {"error": "No agent allocations in final round"}

    agents: dict[str, dict] = {}
    for role, portfolio in allocations.items():
        agents[role] = _portfolio_value(portfolio, prices_base, year, quarter)

    return {"agents": agents}


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
    """Read the agents mapping from a debate config YAML.

    Checks final/ first, then falls back to config_paths in the manifest.
    Returns {role: profile_name} or None if unavailable.
    """
    # Try final/ directory first
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        for f in final_dir.iterdir():
            if f.suffix == ".yaml" and f.stem.startswith("debate"):
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and isinstance(data.get("agents"), dict):
                        return data["agents"]
                except Exception:
                    continue

    # Fallback: read config_paths from manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for cp in manifest.get("config_paths", []):
                p = Path(cp)
                if p.is_file() and p.suffix == ".yaml":
                    data = yaml.safe_load(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and isinstance(data.get("agents"), dict):
                        return data["agents"]
        except Exception:
            pass

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

            # Compute elapsed seconds from timestamps
            if manifest.get("started_at") and manifest.get("completed_at"):
                try:
                    from datetime import datetime, timezone
                    started = datetime.fromisoformat(manifest["started_at"])
                    completed = datetime.fromisoformat(manifest["completed_at"])
                    entry["elapsed_s"] = round((completed - started).total_seconds(), 1)
                except (ValueError, TypeError):
                    pass

            # Ensure agent_profiles is a simple {role: profile_name} map.
            # The manifest may store full config objects; the config YAML
            # always has simple string profile names.
            ap = entry.get("agent_profiles")
            if not ap or not all(isinstance(v, str) for v in ap.values()):
                cfg_profiles = _read_agent_profiles_from_config(run_dir)
                if cfg_profiles:
                    entry["agent_profiles"] = cfg_profiles

        # Quality metrics
        if status != "incomplete":
            entry.update(_extract_quality_metrics(run_dir))

        # Portfolio performance (lightweight — reuses cached price files)
        if status != "incomplete" and manifest is not None:
            perf = compute_portfolio_performance(base_path, experiment, run_dir.name)
            if "error" not in perf:
                entry["portfolio_final_value"] = perf["final_value"]
                entry["portfolio_return_pct"] = perf["return_pct"]

        runs.append(entry)

    return runs


def _persist_metric(run_dir: Path, metric_name: str, data: Any) -> None:
    """Write a computed metric to the _dashboard artifact directory.

    Only writes if the content has changed, avoiding unnecessary disk writes.
    """
    dashboard_dir = run_dir / "_dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    path = dashboard_dir / f"{metric_name}.json"
    serialized = json.dumps(data, indent=2)
    if path.exists() and path.read_text() == serialized:
        return
    path.write_text(serialized)


def _compute_ticker_performance(
    tickers: list[str],
    prices_base: Path,
    year: str,
    quarter: str,
) -> list[dict]:
    """Compute per-ticker quarter performance (open, close, pct_change).

    Sorted by ticker. Excludes _CASH_. Returns [] if no price data.
    """
    result = []
    for ticker in sorted(tickers):
        if ticker == "_CASH_":
            continue
        price_file = prices_base / ticker / f"{year}_{quarter}.json"
        price_data = _safe_json(price_file)
        if not price_data or not price_data.get("daily_close"):
            continue
        closes = price_data["daily_close"]
        open_price = closes[0]["close"]
        close_price = closes[-1]["close"]
        pct_change = round((close_price - open_price) / open_price * 100, 2)
        result.append({
            "ticker": ticker,
            "open": round(open_price, 2),
            "close": round(close_price, 2),
            "pct_change": pct_change,
        })
    return result


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
        "run_dir": _relative_run_dir(run_dir),
    }

    manifest = _safe_json(run_dir / "manifest.json")
    if manifest:
        ap = manifest.get("agent_profiles")
        if not ap or not all(isinstance(v, str) for v in ap.values()):
            cfg_profiles = _read_agent_profiles_from_config(run_dir)
            if cfg_profiles:
                manifest["agent_profiles"] = cfg_profiles
    detail["manifest"] = manifest
    detail["pid_config"] = _safe_json(run_dir / "pid_config.json")
    detail["final_portfolio"] = _safe_json(run_dir / "final" / "final_portfolio.json")
    detail["quality"] = _extract_quality_metrics(run_dir)

    # Load debate + scenario config YAMLs from final/ directory.
    debate_config = None
    scenario_config = None
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        for f in sorted(final_dir.iterdir()):
            if f.suffix == ".yaml":
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        continue
                    if "debate_setup" in data:
                        debate_config = data
                    elif "invest_quarter" in data:
                        scenario_config = data
                except Exception:
                    continue
    detail["debate_config"] = debate_config
    detail["scenario_config"] = scenario_config

    # Ticker performance
    if manifest:
        invest_q = manifest.get("invest_quarter")
        parsed = _quarter_from_date(invest_q) if invest_q else None
        if parsed is not None:
            year, quarter = parsed
            tp = _compute_ticker_performance(
                manifest.get("ticker_universe", []),
                DAILY_PRICES_BASE, year, quarter,
            )
            detail["ticker_performance"] = tp
            _persist_metric(run_dir, "ticker_performance", tp)

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

    Reads exclusively from per-round metric files. Does NOT use the
    aggregated ``pid_crit_all_rounds.json`` which has duplicate entries
    per round and unreliable divergence values.

    Data sources (in priority order):
      1. New telemetry: ``rounds/round_NNN/metrics_propose.json`` etc.
      2. Legacy metrics: ``rounds/round_NNN/metrics/js_divergence.json`` etc.
    """
    run_dir = base_path / experiment / run_id
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.is_dir():
        return []

    # Collect per-phase data from per-round files.
    # Each dict maps round_num → {js, ov, confidences}.
    propose_data: dict[int, dict] = {}
    revision_data: dict[int, dict] = {}
    retry_data: dict[int, list[dict]] = {}

    for rd in sorted(rounds_dir.glob("round_*")):
        try:
            rn = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        metrics_dir = rd / "metrics"

        # --- Propose ---
        src = _safe_json(rd / "metrics_propose.json")
        if src:
            propose_data[rn] = {
                "js": src.get("js_divergence"),
                "ov": src.get("evidence_overlap"),
                "confidences": None,
            }
        elif metrics_dir.is_dir():
            div_p = _safe_json(metrics_dir / "js_divergence_propose.json")
            ev_p = _safe_json(metrics_dir / "evidence_overlap_propose.json")
            if div_p or ev_p:
                propose_data[rn] = {
                    "js": (div_p or {}).get("js_divergence"),
                    "ov": (ev_p or {}).get("mean_overlap"),
                    "confidences": (div_p or {}).get("agent_confidences"),
                }

        # --- Revision ---
        src = _safe_json(rd / "metrics_revision.json")
        if src:
            revision_data[rn] = {
                "js": src.get("js_divergence"),
                "ov": src.get("evidence_overlap"),
                "confidences": None,
            }
        elif metrics_dir.is_dir():
            div_r = _safe_json(metrics_dir / "js_divergence.json")
            ev_r = _safe_json(metrics_dir / "evidence_overlap.json")
            if div_r or ev_r:
                revision_data[rn] = {
                    "js": (div_r or {}).get("js_divergence"),
                    "ov": (ev_r or {}).get("mean_overlap"),
                    "confidences": (div_r or {}).get("agent_confidences"),
                }

        # --- Retries ---
        retries: list[dict] = []
        for tf in sorted(rd.glob("metrics_retry_*.json")):
            tr = _safe_json(tf)
            if tr:
                retries.append({
                    "js": tr.get("js_divergence"),
                    "ov": tr.get("evidence_overlap"),
                    "confidences": None,
                })
        if not retries and metrics_dir.is_dir():
            for rf in sorted(metrics_dir.glob("js_divergence_retry_*.json")):
                div_rt = _safe_json(rf)
                if div_rt is None:
                    continue
                suffix = rf.stem.replace("js_divergence_", "")
                ev_rt = _safe_json(metrics_dir / f"evidence_overlap_{suffix}.json")
                retries.append({
                    "js": div_rt.get("js_divergence"),
                    "ov": (ev_rt or {}).get("mean_overlap"),
                    "confidences": div_rt.get("agent_confidences"),
                })
        if retries:
            retry_data[rn] = retries

    # Build result rows in round order.
    all_rounds = sorted(set(propose_data) | set(revision_data) | set(retry_data))
    result: list[dict] = []
    prev_final: dict | None = None

    for rn in all_rounds:
        # Propose: use file data if available, else carry forward previous
        # round's last phase (round 2+ reuses prior revision as propose).
        if rn in propose_data:
            row = _div_row(rn, "propose", propose_data[rn])
            result.append(row)
        elif prev_final is not None:
            result.append({
                "round": rn,
                "phase": "propose",
                "js_divergence": prev_final["js_divergence"],
                "mean_overlap": prev_final["mean_overlap"],
                "agent_confidences": prev_final.get("agent_confidences"),
            })

        # Revision
        if rn in revision_data:
            row = _div_row(rn, "revise", revision_data[rn])
            result.append(row)
            prev_final = row

        # Retries
        if rn in retry_data:
            for idx, rt in enumerate(retry_data[rn], start=1):
                row = _div_row(rn, f"retry_{idx:03d}", rt)
                result.append(row)
                prev_final = row

    return result


def _div_row(round_num: int, phase: str, data: dict) -> dict:
    """Build a single divergence result row."""
    return {
        "round": round_num,
        "phase": phase,
        "js_divergence": data.get("js"),
        "mean_overlap": data.get("ov"),
        "agent_confidences": data.get("confidences"),
    }


def _read_phase_portfolios(phase_dir: Path) -> dict[str, dict]:
    """Read per-agent portfolio.json files from a phase directory.

    Returns ``{role: {ticker: weight}}``.
    """
    result: dict[str, dict] = {}
    if not phase_dir.is_dir():
        return result
    for agent_dir in sorted(phase_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        portfolio = _safe_json(agent_dir / "portfolio.json")
        if isinstance(portfolio, dict):
            result[agent_dir.name] = portfolio
    return result


def get_portfolio_trajectory(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
) -> list[dict]:
    """Read each round's proposed and revised portfolio per agent.

    Returns list of ``{round, proposals: {role: alloc}, revisions: {role: alloc},
    allocations: {role: alloc}, consensus: {ticker: weight}}``.
    ``allocations`` is an alias for ``revisions`` for backward compatibility.
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

        proposals = _read_phase_portfolios(rd / "proposals")
        revisions = _read_phase_portfolios(rd / "revisions")

        # Round 2+: propose phase is skipped so the proposals dir is empty.
        # The effective input was the prior round's revisions.
        if not proposals and result:
            proposals = result[-1].get("revisions", {})

        # Intervention retry phases (revisions_retry_001, revisions_retry_002, …)
        retries: list[dict[str, dict]] = []
        for retry_dir in sorted(rd.glob("revisions_retry_*")):
            retry_allocs = _read_phase_portfolios(retry_dir)
            if retry_allocs:
                retries.append(retry_allocs)

        # Compute consensus (mean across final revised agents)
        final_revisions = retries[-1] if retries else revisions
        consensus: dict[str, float] = {}
        if final_revisions:
            all_tickers: set[str] = set()
            for alloc in final_revisions.values():
                all_tickers.update(alloc.keys())
            n = len(final_revisions)
            for ticker in sorted(all_tickers):
                total = sum(alloc.get(ticker, 0.0) for alloc in final_revisions.values())
                consensus[ticker] = round(total / n, 4)

        entry: dict[str, Any] = {
            "round": round_num,
            "proposals": proposals,
            "revisions": revisions,
            "allocations": final_revisions,
            "consensus": consensus,
        }
        if retries:
            entry["retries"] = retries
        result.append(entry)
    return result


def compute_round_performance(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    prices_base: Path = DAILY_PRICES_BASE,
) -> list[dict]:
    """Compute per-round, per-phase, per-agent portfolio performance.

    Returns ``[{round, proposals: {role: perf}, revisions: {role: perf}}]``
    where each perf dict has ``{initial_capital, final_value, profit, return_pct}``.
    """
    run_dir = base_path / experiment / run_id
    manifest = _safe_json(run_dir / "manifest.json")
    if not manifest:
        return []

    invest_q = manifest.get("invest_quarter")
    parsed = _quarter_from_date(invest_q) if invest_q else None
    if parsed is None:
        return []

    year, quarter = parsed
    trajectory = get_portfolio_trajectory(base_path, experiment, run_id)

    result = []
    for entry in trajectory:
        round_entry: dict[str, Any] = {"round": entry["round"]}
        for phase in ("proposals", "revisions"):
            agents = entry.get(phase) or {}
            phase_perf: dict[str, dict] = {}
            for role, portfolio in agents.items():
                phase_perf[role] = _portfolio_value(
                    portfolio, prices_base, year, quarter,
                )
            round_entry[phase] = phase_perf
        # Intervention retry phases
        retries = entry.get("retries") or []
        if retries:
            retry_perfs = []
            for retry_allocs in retries:
                retry_perf: dict[str, dict] = {}
                for role, portfolio in retry_allocs.items():
                    retry_perf[role] = _portfolio_value(
                        portfolio, prices_base, year, quarter,
                    )
                retry_perfs.append(retry_perf)
            round_entry["retries"] = retry_perfs
        result.append(round_entry)
    return result


def _equal_weight_mean(portfolios: list[dict[str, float]]) -> dict[str, float]:
    """Compute the equal-weight mean portfolio across agents.

    Parameters: list of ``{ticker: weight}`` dicts.
    Returns averaged ``{ticker: mean_weight}``.
    """
    if not portfolios:
        return {}
    n = len(portfolios)
    agg: dict[str, float] = {}
    for portfolio in portfolios:
        for ticker, weight in portfolio.items():
            agg[ticker] = agg.get(ticker, 0.0) + weight
    return {ticker: round(total / n, 4) for ticker, total in agg.items()}


def _pv_with_delta(
    portfolio: dict[str, float] | None,
    prices_base: Path,
    year: str,
    quarter: str,
    initial_capital: float = 100_000.0,
) -> dict | None:
    """Compute portfolio value and delta vs initial capital.

    Returns ``{pv, delta_pct}`` or ``None`` if portfolio is None/empty.
    """
    if not portfolio:
        return None
    perf = _portfolio_value(portfolio, prices_base, year, quarter, initial_capital)
    pv = perf["final_value"]
    delta_pct = round((pv - initial_capital) / initial_capital * 100, 2)
    return {"pv": pv, "delta_pct": delta_pct}


def _l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """L1 distance between two portfolio weight vectors."""
    all_keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in all_keys)


def _sharpe_from_weights(
    portfolio: dict[str, float],
    prices_base: Path,
    year: str,
    quarter: str,
) -> float | None:
    """Compute annualized Sharpe ratio from daily portfolio returns.

    Uses up to 60 trading days in the investment quarter.
    Sharpe = (mean_daily_return / std_daily_return) * sqrt(252).
    Uses sample variance (N-1 denominator).
    Returns None if fewer than 2 daily returns or std < 1e-10.
    """
    ticker_closes_by_date: dict[str, dict[str, float]] = {}
    for ticker, weight in portfolio.items():
        if abs(weight) < 1e-8 or ticker == "_CASH_":
            continue
        price_file = prices_base / ticker / f"{year}_{quarter}.json"
        price_data = _safe_json(price_file)
        if price_data and price_data.get("daily_close"):
            ticker_closes_by_date[ticker] = {
                d["date"]: float(d["close"])
                for d in price_data["daily_close"]
            }
    if not ticker_closes_by_date:
        return None

    date_sets = [set(v.keys()) for v in ticker_closes_by_date.values()]
    common_dates = sorted(set.intersection(*date_sets))
    common_dates = common_dates[:61]
    if len(common_dates) < 2:
        return None

    ticker_returns_by_date: dict[str, dict[str, float]] = {}
    for ticker, closes_map in ticker_closes_by_date.items():
        returns = {}
        for i in range(1, len(common_dates)):
            prev_date = common_dates[i - 1]
            curr_date = common_dates[i]
            returns[curr_date] = (closes_map[curr_date] - closes_map[prev_date]) / closes_map[prev_date]
        ticker_returns_by_date[ticker] = returns

    return_dates = common_dates[1:]
    daily_returns: list[float] = []
    for date in return_dates:
        day_return = 0.0
        for ticker, weight in portfolio.items():
            if ticker in ticker_returns_by_date:
                day_return += float(weight) * ticker_returns_by_date[ticker][date]
        daily_returns.append(day_return)

    if len(daily_returns) < 2:
        return None

    mean_r = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(variance)
    if std < 1e-10:
        return None
    return round((mean_r / std) * math.sqrt(252), 4)


def compute_collapse_diagnostics(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    persist: bool = True,
) -> list[dict]:
    """Per-round agent collapse diagnostics.

    For each round, computes per-agent movement, toward_consensus,
    collapse_share, and dissent. Identifies collapse leader and index.
    """
    run_dir = base_path / experiment / run_id
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.is_dir():
        return []

    result = []
    for round_dir in sorted(rounds_dir.iterdir()):
        if not round_dir.name.startswith("round_"):
            continue
        try:
            round_num = int(round_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        proposals = _read_phase_portfolios(round_dir / "proposals")
        revisions = _read_phase_portfolios(round_dir / "revisions")
        if not proposals or not revisions:
            continue

        consensus = _equal_weight_mean(list(proposals.values()))
        if not consensus:
            continue

        agents: dict[str, dict] = {}
        roles = sorted(set(proposals) & set(revisions))
        for role in roles:
            prop = proposals[role]
            rev = revisions[role]
            movement = round(_l1_distance(prop, rev), 4)
            dist_prop_to_consensus = _l1_distance(prop, consensus)
            dist_rev_to_consensus = _l1_distance(rev, consensus)
            toward = round(dist_prop_to_consensus - dist_rev_to_consensus, 4)
            dissent = round(_l1_distance(rev, consensus), 4)
            agents[role] = {
                "movement": movement,
                "toward_consensus": toward,
                "dissent": dissent,
            }

        positive_toward = sum(
            max(agents[r]["toward_consensus"], 0) for r in roles
        )
        for role in roles:
            if positive_toward < 1e-10:
                agents[role]["collapse_share"] = None
            else:
                agents[role]["collapse_share"] = round(
                    max(agents[role]["toward_consensus"], 0) / positive_toward, 4
                )

        collapse_leader = None
        collapse_index = None
        shares = [
            (r, agents[r]["collapse_share"])
            for r in roles
            if agents[r]["collapse_share"] is not None
        ]
        if shares:
            shares.sort(key=lambda x: x[1], reverse=True)
            collapse_leader = shares[0][0]
            collapse_index = shares[0][1]

        result.append({
            "round": round_num,
            "agents": agents,
            "collapse_leader": collapse_leader,
            "collapse_index": collapse_index,
        })

    if persist:
        _persist_metric(run_dir, "collapse_diagnostics", result)
    return result


def compute_debate_impact(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "default",
    run_id: str = "",
    prices_base: Path = DAILY_PRICES_BASE,
) -> dict:
    """Compute debate impact: per-agent deltas, mean portfolios, Sharpe, summary, collapse.

    Returns ``{agent_deltas, mean_portfolios, sharpe, summary, collapse}``.
    """
    run_dir = base_path / experiment / run_id
    manifest = _safe_json(run_dir / "manifest.json")
    if not manifest:
        return {"error": "Missing manifest"}

    invest_q = manifest.get("invest_quarter")
    parsed = _quarter_from_date(invest_q) if invest_q else None
    if parsed is None:
        return {"error": f"Cannot parse invest_quarter: {invest_q}"}

    year, quarter = parsed
    trajectory = get_portfolio_trajectory(base_path, experiment, run_id)
    if not trajectory:
        return {"error": "No portfolio trajectory data"}

    first_round = trajectory[0]
    last_round = trajectory[-1]
    r1_proposals = first_round.get("proposals") or {}
    r1_revisions = first_round.get("revisions") or {}
    final_revisions = last_round.get("revisions") or {}

    # Read JS intervention portfolios per round
    rounds_dir = run_dir / "rounds"
    r1_js: dict[str, dict] = {}
    r2_js: dict[str, dict] = {}
    r2_revisions: dict[str, dict] = {}
    if rounds_dir.is_dir():
        r1_dir = rounds_dir / "round_001" / "revisions_retry_001"
        r1_js = _read_phase_portfolios(r1_dir)
        if len(trajectory) >= 2:
            r2_dir = rounds_dir / "round_002" / "revisions_retry_001"
            r2_js = _read_phase_portfolios(r2_dir)
            r2_revisions = trajectory[1].get("revisions") or {}

    # Read and validate judge portfolio
    final_portfolio = _safe_json(run_dir / "final" / "final_portfolio.json")
    judge_delta = None
    if isinstance(final_portfolio, dict):
        valid_weights = all(isinstance(v, (int, float)) for v in final_portfolio.values())
        if valid_weights and final_portfolio:
            judge_delta = _pv_with_delta(final_portfolio, prices_base, year, quarter)

    # Per-agent deltas with PV for every phase
    agent_deltas: dict[str, dict] = {}
    for role in sorted(set(r1_proposals) | set(final_revisions)):
        entry: dict[str, Any] = {}

        # Legacy fields (backward compat)
        if role in r1_proposals:
            entry["initial"] = _portfolio_value(
                r1_proposals[role], prices_base, year, quarter,
            )
        if role in final_revisions:
            entry["final"] = _portfolio_value(
                final_revisions[role], prices_base, year, quarter,
            )
        if "initial" in entry and "final" in entry:
            entry["delta_dollars"] = round(
                entry["final"]["final_value"] - entry["initial"]["final_value"], 2,
            )
            entry["delta_pct"] = round(
                entry["final"]["return_pct"] - entry["initial"]["return_pct"], 2,
            )

        # Extended PV fields for each phase
        entry["r1_proposal"] = _pv_with_delta(r1_proposals.get(role), prices_base, year, quarter)
        entry["r1_revision"] = _pv_with_delta(r1_revisions.get(role), prices_base, year, quarter)
        entry["r1_js"] = _pv_with_delta(r1_js.get(role), prices_base, year, quarter)
        entry["r2_revision"] = _pv_with_delta(r2_revisions.get(role), prices_base, year, quarter)
        entry["r2_js"] = _pv_with_delta(r2_js.get(role), prices_base, year, quarter)

        # Judge comparison
        entry["judge"] = None
        final_agent_pv = entry.get("r2_revision") or entry.get("r1_revision")
        if final_agent_pv and judge_delta:
            entry["judge"] = {
                "pv": judge_delta["pv"],
                "delta_pct": judge_delta["delta_pct"],
                "vs_agent_delta_pct": round(
                    final_agent_pv["delta_pct"] - judge_delta["delta_pct"], 2,
                ),
            }

        agent_deltas[role] = entry

    # Equal-weight mean portfolios
    mean_portfolios = _compute_round_means(
        r1_proposals, r1_revisions, "r1", prices_base, year, quarter,
    )

    # R1 JS intervention mean
    if r1_js:
        mean_r1_js = _equal_weight_mean(list(r1_js.values()))
        if mean_r1_js:
            mean_portfolios["r1_js"] = _portfolio_value(mean_r1_js, prices_base, year, quarter)

    # R2 means
    if len(trajectory) >= 2:
        r2_means = _compute_round_means(
            r1_revisions, r2_revisions, "r2", prices_base, year, quarter,
        )
        mean_portfolios.update(r2_means)

    # R2 JS intervention mean
    if r2_js:
        mean_r2_js = _equal_weight_mean(list(r2_js.values()))
        if mean_r2_js:
            mean_portfolios["r2_js"] = _portfolio_value(mean_r2_js, prices_base, year, quarter)

    # Sharpe ratios for equal-weight mean portfolio of each phase
    sharpe: dict[str, float | None] = {}
    for phase_key, phase_portfolios in [
        ("r1_proposal", r1_proposals),
        ("r1_revision", r1_revisions),
        ("r1_js", r1_js),
        ("r2_revision", r2_revisions),
        ("r2_js", r2_js),
    ]:
        if phase_portfolios:
            mean_p = _equal_weight_mean(list(phase_portfolios.values()))
            if mean_p:
                sharpe[phase_key] = _sharpe_from_weights(mean_p, prices_base, year, quarter)

    # --- Debate Summary (all server-side) ---
    mean_prop_return = None
    if "r1_proposals" in mean_portfolios:
        mean_prop_return = mean_portfolios["r1_proposals"]["return_pct"]

    final_rev_perf = (
        mean_portfolios.get("r2_revisions")
        or mean_portfolios.get("r1_revisions")
        or mean_portfolios.get("r1_proposals")
    )
    final_debate_return = final_rev_perf["return_pct"] if final_rev_perf else None

    debate_alpha = None
    if mean_prop_return is not None and final_debate_return is not None:
        debate_alpha = round(final_debate_return - mean_prop_return, 2)

    judge_return = None
    if judge_delta:
        judge_return = judge_delta["delta_pct"]

    agent_vs_judge = None
    if final_debate_return is not None and judge_return is not None:
        agent_vs_judge = round(final_debate_return - judge_return, 2)

    final_sharpe = sharpe.get("r2_revision") or sharpe.get("r1_revision")

    summary = {
        "mean_proposal_return": mean_prop_return,
        "final_debate_return": final_debate_return,
        "debate_alpha": debate_alpha,
        "judge_return": judge_return,
        "agent_vs_judge": agent_vs_judge,
        "final_sharpe": final_sharpe,
    }

    # Collapse diagnostics (persist=False to avoid double writes)
    collapse = compute_collapse_diagnostics(base_path, experiment, run_id, persist=False)

    result = {
        "agent_deltas": agent_deltas,
        "mean_portfolios": mean_portfolios,
        "sharpe": sharpe,
        "summary": summary,
        "collapse": collapse,
    }

    _persist_metric(run_dir, "debate_impact", result)
    return result


def _compute_round_means(
    proposals: dict,
    revisions: dict,
    prefix: str,
    prices_base: Path,
    year: int,
    quarter: int,
) -> dict[str, dict]:
    """Compute equal-weight mean portfolio perf for a round's proposals and revisions."""
    result: dict[str, dict] = {}
    mean_p = _equal_weight_mean(list(proposals.values()))
    mean_r = _equal_weight_mean(list(revisions.values()))
    if mean_p:
        result[f"{prefix}_proposals"] = _portfolio_value(
            mean_p, prices_base, year, quarter,
        )
    if mean_r:
        result[f"{prefix}_revisions"] = _portfolio_value(
            mean_r, prices_base, year, quarter,
        )
    return result


def _avg(values: list[float]) -> float:
    """Compute the mean of a list of floats, rounded to 2 decimal places."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _get_run_roles(base_path: Path, experiment: str, run_id: str) -> str:
    """Read sorted roles from a run's manifest and return as config key.

    Returns a comma-separated string like ``"risk, technical, value"``,
    or ``"unknown"`` if roles cannot be read.
    """
    manifest = _safe_json(base_path / experiment / run_id / "manifest.json")
    if manifest is None:
        return "unknown"
    roles = manifest.get("roles")
    if not roles or not isinstance(roles, list):
        return "unknown"
    return ", ".join(sorted(roles))


def compute_ablation_debate_impact(
    base_path: Path = DEFAULT_BASE,
    prices_base: Path = DAILY_PRICES_BASE,
) -> dict[str, dict]:
    """Aggregate debate impact across all runs per experiment, grouped by agent config.

    Returns ``{experiment: {configs: {config_key: {run_count, agent_deltas,
    mean_portfolios}}}}``.  Each config key is a sorted comma-separated role
    string (e.g. ``"risk, technical, value"``).
    """
    if not base_path.is_dir():
        return {}

    result: dict[str, dict] = {}
    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        experiment = exp_dir.name
        runs = list_runs(base_path, experiment)
        if not runs:
            continue

        # Collect (config_key, impact) pairs
        config_impacts: dict[str, list[dict]] = {}
        for run in runs:
            if run.get("status") != "complete":
                continue
            impact = compute_debate_impact(
                base_path, experiment, run["run_id"], prices_base,
            )
            if "error" not in impact:
                config_key = _get_run_roles(
                    base_path, experiment, run["run_id"],
                )
                config_impacts.setdefault(config_key, []).append(impact)

        if not config_impacts:
            continue

        configs: dict[str, dict] = {}
        for key in sorted(config_impacts):
            configs[key] = _aggregate_impacts(config_impacts[key])
        result[experiment] = {"configs": configs}

    return result


def _aggregate_impacts(impacts: list[dict]) -> dict:
    """Aggregate per-run debate impact dicts into experiment-level averages.

    Returns ``{run_count, agent_deltas, mean_portfolios}``.
    """
    n = len(impacts)

    # Collect per-agent deltas across runs
    all_roles: set[str] = set()
    for imp in impacts:
        all_roles.update(imp.get("agent_deltas", {}).keys())

    agent_agg: dict[str, dict] = {}
    for role in sorted(all_roles):
        init_returns: list[float] = []
        final_returns: list[float] = []
        deltas: list[float] = []
        for imp in impacts:
            ad = imp.get("agent_deltas", {}).get(role, {})
            if "initial" in ad and "final" in ad:
                init_returns.append(ad["initial"]["return_pct"])
                final_returns.append(ad["final"]["return_pct"])
            if "delta_pct" in ad:
                deltas.append(ad["delta_pct"])
        agent_agg[role] = {
            "mean_initial_return": _avg(init_returns),
            "mean_final_return": _avg(final_returns),
            "mean_delta_pct": _avg(deltas),
        }

    # Collect mean portfolio returns across runs, per round
    mean_portfolios: dict[str, dict] = {}
    for rnd in ("r1", "r2"):
        rnd_means = _aggregate_round_means(impacts, rnd)
        if rnd_means:
            mean_portfolios[rnd] = rnd_means

    # Aggregate Sharpe ratios across runs
    sharpe_agg = _aggregate_sharpe(impacts)

    return {
        "run_count": n,
        "agent_deltas": agent_agg,
        "mean_portfolios": mean_portfolios,
        "sharpe": sharpe_agg,
    }


def _aggregate_round_means(
    impacts: list[dict], prefix: str,
) -> dict:
    """Aggregate mean portfolio returns for a round prefix (r1/r2) across runs.

    Returns ``{proposals_return, revisions_return, critique_impact}``
    or empty dict if no data.
    """
    p_key = f"{prefix}_proposals"
    r_key = f"{prefix}_revisions"
    p_returns: list[float] = []
    r_returns: list[float] = []
    for imp in impacts:
        mp = imp.get("mean_portfolios", {})
        if p_key in mp:
            p_returns.append(mp[p_key]["return_pct"])
        if r_key in mp:
            r_returns.append(mp[r_key]["return_pct"])
    if not p_returns and not r_returns:
        return {}
    p_avg = _avg(p_returns)
    r_avg = _avg(r_returns)
    return {
        "proposals_return": p_avg,
        "revisions_return": r_avg,
        "critique_impact": round(r_avg - p_avg, 2),
    }


def _aggregate_sharpe(impacts: list[dict]) -> dict[str, float | None]:
    """Aggregate Sharpe ratios across runs by phase.

    Returns ``{r1_proposal: mean, r1_revision: mean, ...}`` with None
    for phases that have no data.
    """
    phases = ("r1_proposal", "r1_revision", "r1_js", "r2_revision", "r2_js")
    result: dict[str, float | None] = {}
    for phase in phases:
        values = []
        for imp in impacts:
            s = imp.get("sharpe", {})
            if s and s.get(phase) is not None:
                values.append(s[phase])
        result[phase] = round(sum(values) / len(values), 4) if values else None
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


# ------------------------------------------------------------------
# Paired statistical tests (paired t-test)
# ------------------------------------------------------------------
# Reproduces the statistical approach from scripts/compare_per_scenario.py:
#   scipy.stats.ttest_rel  — paired two-tailed t-test
#   scipy.stats.sem        — standard error of the mean
#   scipy.stats.t.interval — 95% confidence interval on the difference
#   numpy mean / std(ddof=1)

def _extract_collapse_ratio(run_dir: Path) -> float | None:
    """Compute collapse ratio for a single run.

    Baseline (no retry): JS_revision / JS_propose
    Intervention (retry exists): JS_retry / JS_propose

    Returns None if required metrics are missing.
    """
    r1 = run_dir / "rounds" / "round_001"
    propose = _safe_json(r1 / "metrics_propose.json")
    if not isinstance(propose, dict):
        return None
    js_propose = propose.get("js_divergence")
    if js_propose is None or js_propose == 0:
        return None

    # Prefer retry metrics if they exist (intervention condition)
    retry = _safe_json(r1 / "metrics_retry_001.json")
    if isinstance(retry, dict) and retry.get("js_divergence") is not None:
        return retry["js_divergence"] / js_propose

    revision = _safe_json(r1 / "metrics_revision.json")
    if not isinstance(revision, dict):
        return None
    js_revision = revision.get("js_divergence")
    if js_revision is None:
        return None
    return js_revision / js_propose


def compute_paired_tests(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "",
) -> dict:
    """Paired t-test comparing collapse ratios across two debate configs.

    Reproduces the statistical methodology from
    ``scripts/compare_per_scenario.py``: paired ``ttest_rel``, SEM,
    and 95 % CI on the mean difference.

    Groups runs by (debate_config_stem, scenario_stem), pairs by
    scenario across the two debate configs.

    Returns dict with config names, paired data, test statistics,
    and summary measures.
    """
    import numpy as np
    from scipy import stats

    exp_dir = base_path / experiment
    if not exp_dir.is_dir():
        return {"error": f"Experiment directory not found: {experiment}"}

    # Scan all runs and group by (debate_config, scenario)
    groups: dict[tuple[str, str], float] = {}
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        manifest = _safe_json(run_dir / "manifest.json")
        if not isinstance(manifest, dict):
            continue
        config_paths = manifest.get("config_paths")
        if not isinstance(config_paths, list) or len(config_paths) < 2:
            continue

        debate_stem = Path(config_paths[0]).stem
        scenario_stem = Path(config_paths[1]).stem

        ratio = _extract_collapse_ratio(run_dir)
        if ratio is None:
            continue
        groups[(debate_stem, scenario_stem)] = ratio

    # Identify the two debate configs
    debate_configs = sorted({k[0] for k in groups})
    if len(debate_configs) < 2:
        return {
            "pending": True,
            "message": f"Waiting for data — found {len(debate_configs)} of 2 debate configs",
            "configs_found": debate_configs,
        }
    if len(debate_configs) != 2:
        return {
            "error": f"Expected 2 debate configs, found {len(debate_configs)}",
            "configs_found": debate_configs,
        }

    config_a, config_b = debate_configs  # alphabetical

    # Build paired arrays matched by scenario
    scenarios_a = {k[1]: v for k, v in groups.items() if k[0] == config_a}
    scenarios_b = {k[1]: v for k, v in groups.items() if k[0] == config_b}
    common = sorted(set(scenarios_a) & set(scenarios_b))

    if len(common) < 2:
        return {
            "error": f"Too few paired scenarios ({len(common)}), need >= 2",
            "config_a": config_a,
            "config_b": config_b,
        }

    pairs = []
    a_ratios = []
    b_ratios = []
    for s in common:
        a_val = round(scenarios_a[s], 6)
        b_val = round(scenarios_b[s], 6)
        pairs.append({"scenario": s, "a": a_val, "b": b_val})
        a_ratios.append(a_val)
        b_ratios.append(b_val)

    # --- Statistical tests (mirrors compare_per_scenario.py) ----------
    v_a = np.array(a_ratios)
    v_b = np.array(b_ratios)
    diffs = v_b - v_a

    t_stat, p_val = stats.ttest_rel(v_b, v_a)
    mean_diff = float(np.mean(diffs))
    se_diff = float(stats.sem(diffs))

    mean_a = float(np.mean(v_a))
    sem_a = float(stats.sem(v_a))
    mean_b = float(np.mean(v_b))
    sem_b = float(stats.sem(v_b))

    ci_low, ci_high = stats.t.interval(
        0.95, len(diffs) - 1, loc=mean_diff, scale=se_diff,
    )

    n = len(common)
    n_b_greater = int(np.sum(v_b > v_a))
    n_a_greater = int(np.sum(v_a > v_b))

    return {
        "config_a": config_a,
        "config_b": config_b,
        "n_paired": n,
        "pairs": pairs,
        "ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "mean_diff": round(mean_diff, 6),
            "se_diff": round(se_diff, 6),
            "ci_95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        },
        "summary": {
            "a_mean": round(mean_a, 4),
            "a_sem": round(sem_a, 4),
            "a_std": round(float(np.std(v_a, ddof=1)), 4),
            "b_mean": round(mean_b, 4),
            "b_sem": round(sem_b, 4),
            "b_std": round(float(np.std(v_b, ddof=1)), 4),
            "n_b_greater": n_b_greater,
            "n_a_greater": n_a_greater,
        },
    }


# ------------------------------------------------------------------
# Financial metrics paired tests
# ------------------------------------------------------------------
# Reproduces the statistical approach and metric selection from
# scripts/compare_multiple_configs.py: paired ttest_rel per metric,
# mean ± SEM, 95% CI on mean difference.  Reads summary.json from
# simulation results directories, matched via invest_quarter.

# Metrics in display priority order (from compare_multiple_configs.py)
_FINANCIAL_METRIC_PRIORITIES = [
    "daily_metrics_excess_return_pct",
    "daily_metrics_annualized_sharpe",
    "daily_metrics_annualized_volatility",
    "daily_metrics_max_drawdown_pct",
    "daily_metrics_total_return_pct",
    "daily_metrics_annualized_sortino",
    "daily_metrics_calmar_ratio",
    "total_trades",
    "final_cash",
]

# Fields excluded when flattening episode_summaries
# (matches compare_per_scenario.py)
_FLATTEN_EXCLUDED = {
    "episode_id", "initial_cash", "final_positions",
    "final_prices", "position_values", "mean_revisions_metrics",
}

# Metrics excluded from the comparison table
# (matches compare_multiple_configs.py)
_METRIC_EXCLUSIONS = {
    "book_value", "calmar_ratio", "duration_seconds", "max_drawdown",
    "return_pct", "return_pct_with_cash_interest",
    "daily_metrics_trading_days", "daily_metrics_spy_return_pct",
    "max_drawdown_pct",
}


def _flatten_dict(
    d: dict,
    prefix: str = "",
    excluded_fields: set | None = None,
) -> dict[str, float]:
    """Recursively flatten a nested dict with underscore-joined keys.

    Reproduces ``flatten_dict`` from ``scripts/compare_per_scenario.py``.
    Only numeric values (int / float / bool) are included.
    """
    if excluded_fields is None:
        excluded_fields = set()
    flat: dict[str, float] = {}
    if not isinstance(d, dict):
        return flat
    for k, v in d.items():
        if k in excluded_fields:
            continue
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, prefix=f"{key}_", excluded_fields=excluded_fields))
        elif isinstance(v, (int, float, bool)) and v is not None:
            flat[key] = float(v)
    return flat


def _find_results_by_quarter(
    output_dir_str: str,
    debate_stem: str,
    invest_quarter: str,
) -> Path | None:
    """Locate a results directory matching (output_dir, debate_stem, invest_quarter).

    Scans ``{output_dir}/{debate_stem}[_NNN]/`` directories in reverse
    order (highest suffix first) and returns the first whose
    ``simulation_log.json`` records a matching ``invest_quarter``.
    """
    base = Path(output_dir_str)
    if not base.is_dir():
        return None

    candidates: list[Path] = []
    exact = base / debate_stem
    if exact.is_dir():
        candidates.append(exact)
    for i in range(1, 50):
        suffixed = base / f"{debate_stem}_{i:03d}"
        if suffixed.is_dir():
            candidates.append(suffixed)
        else:
            break

    # Most-recently-created (highest suffix) first
    for cand in reversed(candidates):
        sim_log = _safe_json(cand / "simulation_log.json")
        if not isinstance(sim_log, dict):
            continue
        cfg = sim_log.get("config")
        if isinstance(cfg, dict) and cfg.get("invest_quarter") == invest_quarter:
            return cand

    return None


def compute_financial_paired_tests(
    base_path: Path = DEFAULT_BASE,
    experiment: str = "",
    use_mean_revisions: bool = False,
) -> dict:
    """Paired t-tests on financial metrics across two debate configs.

    For each completed run in *experiment*, maps to the simulation
    results directory via ``invest_quarter`` matching, reads
    ``summary.json``, and flattens ``episode_summaries[0]`` using
    the same logic as ``scripts/compare_per_scenario.py``.

    When *use_mean_revisions* is True, reads the pre-computed
    ``mean_revisions_metrics`` (mean of agent revision portfolios)
    instead of the judge's portfolio metrics.

    Pairs metrics by scenario across the two debate configs and runs
    ``scipy.stats.ttest_rel`` per metric (matching
    ``scripts/compare_multiple_configs.py``).

    Returns dict with config names, per-metric statistics, and
    per-scenario raw values.
    """
    import numpy as np
    from scipy import stats

    exp_dir = base_path / experiment
    if not exp_dir.is_dir():
        return {"error": f"Experiment directory not found: {experiment}"}

    # config_stem → {scenario_stem → flat_metrics}
    config_metrics: dict[str, dict[str, dict[str, float]]] = {}
    # Cache scenario YAML reads (same scenario used by both configs)
    scenario_cache: dict[str, tuple[str, str]] = {}  # path → (iq, output_dir)

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        manifest = _safe_json(run_dir / "manifest.json")
        if not isinstance(manifest, dict):
            continue
        config_paths = manifest.get("config_paths")
        if not isinstance(config_paths, list) or len(config_paths) < 2:
            continue

        debate_stem = Path(config_paths[0]).stem
        scenario_path_str = config_paths[1]
        scenario_stem = Path(scenario_path_str).stem

        # Read scenario YAML (cached)
        if scenario_path_str not in scenario_cache:
            scenario_path = Path(scenario_path_str)
            try:
                sdata = yaml.safe_load(
                    scenario_path.read_text(encoding="utf-8"),
                )
            except Exception:
                continue
            if not isinstance(sdata, dict):
                continue
            iq = sdata.get("invest_quarter")
            od = sdata.get("output_dir", "results")
            if not iq:
                continue
            scenario_cache[scenario_path_str] = (iq, od)

        if scenario_path_str not in scenario_cache:
            continue
        invest_quarter, output_dir_str = scenario_cache[scenario_path_str]

        # Find matching results directory
        results_dir = _find_results_by_quarter(
            output_dir_str, debate_stem, invest_quarter,
        )
        if results_dir is None:
            continue

        # Read and flatten summary.json
        summary = _safe_json(results_dir / "summary.json")
        if not isinstance(summary, dict):
            continue
        ep_list = summary.get("episode_summaries")
        if not isinstance(ep_list, list) or not ep_list:
            continue

        if use_mean_revisions:
            rev = ep_list[0].get("mean_revisions_metrics")
            if not isinstance(rev, dict) or not rev:
                continue
            metrics = _flatten_dict(
                {"daily_metrics": rev}, excluded_fields=_FLATTEN_EXCLUDED,
            )
        else:
            metrics = _flatten_dict(ep_list[0], excluded_fields=_FLATTEN_EXCLUDED)
        if not metrics:
            continue

        config_metrics.setdefault(debate_stem, {})[scenario_stem] = metrics

    # Identify the two debate configs
    configs = sorted(config_metrics.keys())
    if len(configs) < 2:
        return {
            "pending": True,
            "message": f"Waiting for data — found {len(configs)} of 2 debate configs",
            "configs_found": configs,
        }
    if len(configs) != 2:
        return {
            "error": f"Expected 2 debate configs, found {len(configs)}",
            "configs_found": configs,
        }

    config_a, config_b = configs
    a_data = config_metrics[config_a]
    b_data = config_metrics[config_b]
    common = sorted(set(a_data) & set(b_data))

    if len(common) < 2:
        return {
            "error": f"Too few paired scenarios ({len(common)}), need >= 2",
            "config_a": config_a,
            "config_b": config_b,
        }

    # Determine metric list: prioritised metrics first, then any remaining
    all_keys: set[str] = set()
    for sc in common:
        all_keys.update(a_data[sc].keys())
        all_keys.update(b_data[sc].keys())
    all_keys -= _METRIC_EXCLUSIONS

    ordered_metrics = [m for m in _FINANCIAL_METRIC_PRIORITIES if m in all_keys]
    extras = sorted(all_keys - set(ordered_metrics))
    ordered_metrics.extend(extras)

    # Compute per-metric paired t-tests
    results_list: list[dict] = []
    for metric in ordered_metrics:
        vals_a: list[float] = []
        vals_b: list[float] = []
        per_scenario: list[dict] = []

        for sc in common:
            va = a_data[sc].get(metric)
            vb = b_data[sc].get(metric)
            if va is not None and vb is not None:
                vals_a.append(va)
                vals_b.append(vb)
                per_scenario.append({"scenario": sc, "a": va, "b": vb})

        if len(vals_a) < 2:
            continue

        v_a = np.array(vals_a)
        v_b = np.array(vals_b)
        diffs = v_b - v_a

        t_stat, p_val = stats.ttest_rel(v_b, v_a)
        mean_diff = float(np.mean(diffs))
        se_diff = float(stats.sem(diffs))
        ci_low, ci_high = stats.t.interval(
            0.95, len(diffs) - 1, loc=mean_diff, scale=se_diff,
        )

        # Handle NaN from ttest_rel (e.g. identical arrays)
        if math.isnan(p_val):
            p_val = 1.0
        if math.isnan(t_stat):
            t_stat = 0.0

        results_list.append({
            "metric": metric,
            "n": len(vals_a),
            "a_mean": round(float(np.mean(v_a)), 4),
            "a_sem": round(float(stats.sem(v_a)), 4),
            "b_mean": round(float(np.mean(v_b)), 4),
            "b_sem": round(float(stats.sem(v_b)), 4),
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "ci_95": [
                round(float(ci_low), 4),
                round(float(ci_high), 4),
            ],
            "per_scenario": per_scenario,
        })

    return {
        "config_a": config_a,
        "config_b": config_b,
        "n_paired": len(common),
        "source": "mean_revisions" if use_mean_revisions else "judge",
        "metrics": results_list,
    }
