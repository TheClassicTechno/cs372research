"""Rich-based terminal display for debate rounds.

Pure presentation layer — consumes existing data structures and renders them
as structured, colored console output. No logic changes.

Soft dependency on Rich: if not installed, all render functions gracefully
fall back to plain-text output via print().
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Soft Rich import — fall back to plain text if not available
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

_console: Any = Console(highlight=False) if RICH_AVAILABLE else None

# Exported API (helps prevent future refactors from accidentally breaking runner imports)
__all__ = [
    "render_round_header",
    "render_phase_label",
    "render_portfolio_table",
    "render_crit_summary",
    "render_crit_pillars",
    "render_crit_diagnostics",
    "render_divergence_metrics",
    "render_pid_update",
    "render_health_bar",
    "render_judge_result",
    "render_debate_end",
    "render_phase_metrics",
    "_reset_llm_tracker",
    "_llm_call_start",
    "_llm_call_end",
]

# ---------------------------------------------------------------------------
# LLM call tracker — thread-safe progress for parallel LLM calls
# ---------------------------------------------------------------------------
_llm_lock = threading.Lock()
_llm_calls: dict[str, dict] = {}  # call_id -> {role, phase, start_time, end_time}
_llm_total: int = 0


def _reset_llm_tracker(total: int = 0) -> None:
    global _llm_calls, _llm_total
    with _llm_lock:
        _llm_calls = {}
        _llm_total = total


def _llm_call_start(call_id: str, role: str, phase: str) -> None:
    with _llm_lock:
        _llm_calls[call_id] = {
            "role": role,
            "phase": phase,
            "start_time": time.monotonic(),
            "end_time": None,
        }


def _llm_call_end(call_id: str, response: str = "") -> None:
    with _llm_lock:
        if call_id in _llm_calls:
            _llm_calls[call_id]["end_time"] = time.monotonic()
    _render_llm_completion(call_id, response)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _score_color(val: float, green: float = 0.85, yellow: float = 0.70) -> str:
    if val >= green:
        return "green"
    if val >= yellow:
        return "yellow"
    return "red"


def _pillar_color(val: float) -> str:
    if val >= 0.9:
        return "green"
    if val >= 0.4:
        return "yellow"
    return "red"


_AGENT_COLORS = {
    "macro": "cyan",
    "value": "green",
    "risk": "red",
    "technical": "magenta",
    "sentiment": "blue",
    "devils_advocate": "bright_red",
    "judge": "yellow",
}


def _agent_color(role: str) -> str:
    return _AGENT_COLORS.get(role.lower(), "white")


# ---------------------------------------------------------------------------
# Response preview extraction — avoid JSON spam
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or unclosed ```)."""
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\s*```(?:json)?\s*\n([\s\S]*)", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_preview(response: str, max_chars: int = 400) -> str:
    """Return a clean, human-readable snippet from a model response.

    - Strips markdown code fences before parsing.
    - If response is JSON, prefer: justification, revision_notes,
      or first critique objection targeting another agent.
    - Otherwise treat as plain text.
    - Single-line, trimmed, truncated.
    """
    if not response:
        return ""

    text = response
    try:
        parsed = json.loads(_strip_code_fences(response))
        if isinstance(parsed, dict):
            # For critique responses, prefer showing an objection of another agent
            text = (
                parsed.get("justification", "")
                or parsed.get("revision_notes", "")
                or ""
            )
            if not text and "critiques" in parsed and isinstance(parsed["critiques"], list):
                for c in parsed["critiques"]:
                    if isinstance(c, dict):
                        obj = c.get("objection", "")
                        if obj:
                            target = c.get("target_role", "").upper()
                            text = f"→ {target}: {obj}" if target else obj
                            break
            if not text:
                # fall back but still avoid printing raw json if possible
                text = response
    except Exception:
        pass

    clean = str(text).replace("\n", " ").strip()
    if len(clean) > max_chars:
        clean = clean[: max_chars - 3] + "..."
    return clean


def _render_llm_completion(call_id: str, response: str = "") -> None:
    """Print a single completion line for one agent with optional response preview."""
    with _llm_lock:
        info = _llm_calls.get(call_id)
        if not info or info["end_time"] is None:
            return
        done = sum(1 for c in _llm_calls.values() if c["end_time"] is not None)
        total = max(_llm_total, len(_llm_calls))
        role_raw = info["role"]
        role = role_raw.upper()
        elapsed = info["end_time"] - info["start_time"]

    preview = _extract_preview(response, max_chars=400)

    # Spacing: completion line + optional indented snippet line for readability
    if not RICH_AVAILABLE or _console is None:
        print(f"  [{done}/{total}] {role:<12} {elapsed:5.1f}s", flush=True)
        if preview:
            print(f"      {preview}", flush=True)
        return

    line = Text()
    line.append(f"  [{done}/{total}] ", style="white")
    line.append(f"{role:<12}", style=_agent_color(role_raw))
    line.append(f" {elapsed:5.1f}s", style="bright_white")
    _console.print(line)

    if preview:
        snippet = Text("      ")
        snippet.append(preview, style="bright_green")
        _console.print(snippet)
        _console.print()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_round_header(
    round_num: int,
    max_rounds: int,
    beta: float,
    tone_bucket: str,
    universe: list[str],
    invest_quarter: str | None = None,
) -> None:
    """Clean status header for debate rounds (better spacing + β color)."""
    bar = "─" * 72
    quarter_str = f"   {invest_quarter}" if invest_quarter else ""

    if not RICH_AVAILABLE or _console is None:
        print("\n" + bar)
        print(f" Round {round_num} / {max_rounds}   β={beta:.3f}   tone={tone_bucket}{quarter_str}")
        print(bar)
        if universe:
            print(f" Universe: {' '.join(universe)}")
        return

    _console.print()
    _console.print(f"[white]{bar}[/white]")
    header = Text()
    header.append(f" Round {round_num} / {max_rounds}", style="bright_white")
    header.append("   β=", style="white")
    header.append(f"{beta:.3f}", style="bright_yellow")
    header.append("   tone=", style="white")
    header.append(str(tone_bucket), style="bright_magenta")
    if invest_quarter:
        header.append("   ", style="white")
        header.append(invest_quarter, style="bright_white")
    _console.print(header)
    _console.print(f"[white]{bar}[/white]")
    if universe:
        _console.print(Text(" Universe: ", style="white") + Text(" ".join(universe), style="green"))


def render_phase_label(phase: str) -> None:
    """Print a phase label (Propose / Critique / Revise). Adds breathing room."""
    if not RICH_AVAILABLE or _console is None:
        print(f"\n\n  ── {phase} ──", flush=True)
        return
    _console.print("\n")
    _console.print(f"  [bright_white]── {phase} ──[/bright_white]")
    _console.print()


def render_portfolio_table(agents: list[dict], phase: str = "Proposals") -> None:
    """Render an aligned table of allocations across agents (decimals only, no percent symbols)."""
    if not agents:
        return

    # Deduplicate by role (keep last)
    seen: dict[str, dict] = {}
    for a in agents:
        seen[a.get("role", "?")] = a
    agents = list(seen.values())

    roles = [a.get("role", "?") for a in agents]
    allocations = [a.get("action_dict", {}).get("allocation", {}) for a in agents]
    confidences = [a.get("action_dict", {}).get("confidence", 0.5) for a in agents]

    all_tickers: set[str] = set()
    for alloc in allocations:
        all_tickers.update(alloc.keys())

    def _avg(t: str) -> float:
        return sum(al.get(t, 0.0) for al in allocations) / max(len(allocations), 1)

    # Sort tickers by average weight, show only meaningful ones
    ticker_order = sorted(all_tickers, key=_avg, reverse=True)
    ticker_order = [t for t in ticker_order if any(al.get(t, 0.0) > 0.005 for al in allocations)]
    if not ticker_order:
        return

    if not RICH_AVAILABLE or _console is None:
        print()
        header = "Ticker  " + "  ".join(f"{r.upper():>10}" for r in roles)
        print(header)
        print("-" * len(header))
        for t in ticker_order:
            row = [t.ljust(6)]
            for alloc in allocations:
                w = alloc.get(t, 0.0)
                row.append(f"{w:>10.2f}" if w > 0.005 else f"{'':>10}")
            print("  ".join(row))
        print()
        return

    _console.print()
    table = Table(
        title=phase,
        box=box.SIMPLE_HEAVY,
        title_style="bright_green",
        header_style="bright_green",
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Ticker", style="white", no_wrap=True)
    for role in roles:
        table.add_column(role.upper(), justify="right", style=_agent_color(role), no_wrap=True)

    for t in ticker_order:
        row = [t]
        for alloc in allocations:
            w = alloc.get(t, 0.0)
            row.append(f"{w:.2f}" if w > 0.005 else "")
        table.add_row(*row)

    _console.print(table)
    _console.print()  # spacing after table


def render_crit_summary(rho_bar: float, agent_scores: dict[str, Any]) -> None:
    """Display CRIT aggregate + per-agent ρ scores (spaced, readable)."""
    if not RICH_AVAILABLE or _console is None:
        print("\n  CRIT Reasoning Quality")
        color_label = "GREEN" if rho_bar >= 0.85 else "YELLOW" if rho_bar >= 0.70 else "RED"
        print(f"    ρ̄ = {rho_bar:.3f} [{color_label}]")
        for role, cr in agent_scores.items():
            rho_i = cr.get("rho_i", 0.0) if isinstance(cr, dict) else getattr(cr, "rho_bar", cr)
            print(f"    {role:<12} {rho_i:.3f}")
        return

    lines = Text()
    c = _score_color(rho_bar)
    lines.append("  ρ̄ = ", style="white")
    lines.append(f"{rho_bar:.3f}\n", style=c)

    for role, cr in agent_scores.items():
        rho_i = cr.get("rho_i", 0.0) if isinstance(cr, dict) else (cr.rho_bar if hasattr(cr, "rho_bar") else cr)
        rc = _score_color(float(rho_i))
        lines.append(f"  {role:<12}", style=_agent_color(role))
        lines.append(f" {float(rho_i):.3f}\n", style=rc)

    panel = Panel(
        lines,
        title="[bright_white]CRIT Reasoning Quality[/bright_white]",
        border_style="blue",
        expand=False,
    )
    _console.print()
    _console.print(panel)
    _console.print()


def render_crit_pillars(agent_scores: dict[str, Any]) -> None:
    """Render CRIT pillar breakdown table with color coding."""
    if not agent_scores:
        return

    pillar_keys = ["LV", "ES", "AC", "CA"]
    pillar_full = {"LV": "Validity", "ES": "Evidence", "AC": "Alternatives", "CA": "Causal"}

    if not RICH_AVAILABLE or _console is None:
        print("\n  CRIT Pillars")
        header = "Agent".ljust(12) + "  " + "  ".join(f"{k:>6}" for k in pillar_keys)
        print("   " + header)
        for role, cr in agent_scores.items():
            pillars = cr.get("pillars", {}) if isinstance(cr, dict) else getattr(getattr(cr, "pillar_scores", {}), "__dict__", {})
            vals = [pillars.get(k, 0.0) for k in pillar_keys]
            print(f"   {role:<12}  " + "  ".join(f"{v:>6.2f}" for v in vals))
        return

    table = Table(
        title="CRIT Pillars",
        box=box.SIMPLE_HEAVY,
        title_style="bright_green",
        header_style="bright_green",
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Agent", style="white", no_wrap=True, min_width=12)
    for k in pillar_keys:
        table.add_column(f"{k}\n{pillar_full[k]}", justify="center", no_wrap=True)

    # rows
    for role, cr in agent_scores.items():
        data = cr if isinstance(cr, dict) else {}
        pillars = data.get("pillars", {})
        row = [f"[{_agent_color(role)}]{role}[/{_agent_color(role)}]"]
        for k in pillar_keys:
            v = float(pillars.get(k, 0.0))
            row.append(f"[{_pillar_color(v)}]{v:.2f}[/{_pillar_color(v)}]")
        table.add_row(*row)

    # averages
    table.add_section()
    n = max(len(agent_scores), 1)
    avg_row = ["[white]avg[/white]"]
    weakest_key = None
    weakest_val = 2.0
    for k in pillar_keys:
        total = 0.0
        for cr in agent_scores.values():
            data = cr if isinstance(cr, dict) else {}
            total += float(data.get("pillars", {}).get(k, 0.0))
        avg = total / n
        if avg < weakest_val:
            weakest_val = avg
            weakest_key = k
        avg_row.append(f"[{_pillar_color(avg)}]{avg:.2f}[/{_pillar_color(avg)}]")
    table.add_row(*avg_row)

    _console.print(table)
    _console.print()
    if weakest_key:
        _console.print(f"  [red]Weakest pillar → {weakest_key} ({pillar_full[weakest_key]})[/red]")
        _console.print()


def render_crit_diagnostics(agent_scores: dict[str, Any]) -> None:
    """Show detected CRIT diagnostic flags (only categories that fired)."""
    diag_keys = ["contradictions", "unsupported_claims", "ignored_critiques", "premature_certainty", "causal_overreach", "conclusion_drift"]

    triggered: dict[str, list[str]] = {}
    for role, cr in agent_scores.items():
        if isinstance(cr, dict):
            diag = cr.get("diagnostics", {})
            for k in diag_keys:
                if diag.get(k, False):
                    triggered.setdefault(k, []).append(role)

    if not triggered:
        return

    if not RICH_AVAILABLE or _console is None:
        print("\n  CRIT Diagnostics")
        for label, roles in triggered.items():
            print(f"    {label}: {', '.join(roles)}")
        return

    lines = Text()
    for label, roles in triggered.items():
        lines.append(f"  {label}: ", style="red")
        lines.append(", ".join(roles) + "\n", style="red")

    panel = Panel(
        lines,
        title="[red]CRIT Diagnostics[/red]",
        border_style="red",
        expand=False,
    )
    _console.print(panel)
    _console.print()


def render_divergence_metrics(js: float, ov: float) -> None:
    """Display JS divergence and portfolio overlap."""
    if not RICH_AVAILABLE or _console is None:
        print("\n  Divergence Metrics")
        print(f"    JS divergence     {js:.4f}")
        print(f"    Portfolio overlap {ov:.3f}")
        return

    lines = Text()
    lines.append("  JS divergence     ", style="bright_white")
    lines.append(f"{js:.4f}\n", style="bright_cyan")
    lines.append("  Portfolio overlap ", style="bright_white")
    lines.append(f"{ov:.3f}", style="bright_cyan")

    panel = Panel(
        lines,
        title="[bright_white]Divergence Metrics[/bright_white]",
        border_style="cyan",
        expand=False,
    )
    _console.print(panel)
    _console.print()


def render_pid_update(
    beta_old: float,
    beta_new: float,
    quadrant: str,
    rho_star: float,
    rho_bar: float,
    error: float,
) -> None:
    """Display PID controller state update (β highlighted)."""
    if not RICH_AVAILABLE or _console is None:
        print("\n  PID Controller")
        print(f"    β: {beta_old:.3f} → {beta_new:.3f}")
        print(f"    quadrant: {quadrant}")
        print(f"    target ρ={rho_star:.2f}  actual ρ={rho_bar:.3f}  error={error:+.3f}")
        return

    quad_colors = {
        "converged": "green",
        "exploring": "yellow",
        "quality_gap": "red",
        "stuck": "bright_red",
    }
    qc = quad_colors.get(quadrant, "white")

    lines = Text()
    lines.append("  β: ", style="white")
    lines.append(f"{beta_old:.3f}", style="bright_yellow")
    lines.append(" → ", style="white")
    lines.append(f"{beta_new:.3f}\n", style="bright_yellow")

    lines.append("  quadrant: ", style="white")
    lines.append(f"{quadrant}\n", style=qc)

    lines.append("\n  target ρ = ", style="white")
    lines.append(f"{rho_star:.2f}", style="bright_white")
    lines.append("   actual ρ = ", style="white")
    lines.append(f"{rho_bar:.3f}", style=_score_color(rho_bar))
    lines.append("   error = ", style="white")
    lines.append(f"{error:+.3f}", style=("red" if error > 0 else "bright_green"))

    panel = Panel(
        lines,
        title="[bright_white]PID Controller[/bright_white]",
        border_style="magenta",
        expand=False,
    )
    _console.print(panel)
    _console.print()


def render_health_bar(
    round_num: int,
    rho_bar: float,
    js: float,
    beta_old: float,
    beta_new: float,
    weakest_pillar: str | None = None,
    stable_rounds: int = 0,
    convergence_window: int = 2,
) -> None:
    """Compact one-line round summary for scanning long debates."""
    if not RICH_AVAILABLE or _console is None:
        wp = f" | Weak: {weakest_pillar}" if weakest_pillar else ""
        conv = f" | stable={stable_rounds}/{convergence_window}" if stable_rounds > 0 else ""
        print(f"\n  ▸ Round {round_num} | ρ̄:{rho_bar:.3f} | JS:{js:.4f} | β:{beta_old:.3f}→{beta_new:.3f}{wp}{conv}")
        return

    line = Text("  ▸ ", style="white")
    line.append(f"Round {round_num}", style="bright_white")
    line.append(" │ ρ̄:", style="white")
    line.append(f"{rho_bar:.3f}", style=_score_color(rho_bar))
    line.append(" │ JS:", style="white")
    line.append(f"{js:.4f}", style="cyan")
    line.append(" │ β:", style="white")
    line.append(f"{beta_old:.3f}", style="bright_yellow")
    line.append("→", style="white")
    line.append(f"{beta_new:.3f}", style="bright_yellow")

    if weakest_pillar:
        line.append(" │ Weak: ", style="white")
        line.append(weakest_pillar, style="red")
    if stable_rounds > 0:
        line.append(f" │ stable={stable_rounds}/{convergence_window}", style="green")

    _console.print(line)
    _console.print()


def render_judge_result(final_action: dict) -> None:
    """Display the judge's final portfolio decision (clean, no JSON spam)."""
    alloc = final_action.get("allocation", {}) or {}
    conf = float(final_action.get("confidence", 0.5))
    justification = str(final_action.get("justification", "") or "")

    sorted_alloc = sorted(alloc.items(), key=lambda x: -x[1])
    active = [(t, w) for t, w in sorted_alloc if w > 0.005]

    if not RICH_AVAILABLE or _console is None:
        print("\n" + "═" * 60)
        print(f"  JUDGE FINAL DECISION  conf={conf:.2f}")
        print("─" * 60)
        for t, w in active:
            print(f"  {t:<6} {w:.2f}")
        if justification:
            print("\n  " + justification)
        print("═" * 60)
        return

    lines = Text()
    lines.append("Confidence: ", style="white")
    lines.append(f"{conf:.2f}\n\n", style=_score_color(conf, green=0.80, yellow=0.65))

    for t, w in active:
        lines.append(f"{t:<6}", style="white")
        lines.append(f" {float(w):.2f}\n", style="bright_white")

    if justification:
        lines.append("\n")
        lines.append(justification, style="green")

    panel = Panel(
        lines,
        title="[yellow]Judge Final Decision[/yellow]",
        border_style="yellow",
        expand=False,
    )
    _console.print()
    _console.print(panel)
    _console.print()


def render_debate_end(
    terminated_early: bool,
    reason: str,
    total_rounds: int,
    logged_dir: str | None = None,
) -> None:
    """Display end-of-debate summary."""
    if not RICH_AVAILABLE or _console is None:
        status = f"Converged ({reason})" if terminated_early else f"Completed {total_rounds} rounds"
        print(f"\n  {'━' * 70}")
        print(f"  DEBATE {status}")
        if logged_dir:
            print(f"  Logged → {logged_dir}")
        print(f"  {'━' * 70}")
        return

    bar = "━" * 70
    status = f"Converged ({reason})" if terminated_early else f"Completed {total_rounds} rounds"
    style = "green" if terminated_early else "white"
    _console.print(f"\n[bright_white]{bar}[/bright_white]")
    _console.print(f"  [{style}]DEBATE {status}[/{style}]")
    if logged_dir:
        _console.print(f"  [white]Logged → {logged_dir}[/white]")
    _console.print(f"[bright_white]{bar}[/bright_white]")
    _console.print()


# ---------------------------------------------------------------------------
# Phase metrics wrapper (compatibility with runner)
# ---------------------------------------------------------------------------

def render_phase_metrics(phase_data: dict, rho_star: float = 0.80) -> None:
    """Render all metrics for a round from the phase_data dict.

    Compatibility wrapper used by the runner. Pure presentation.
    """
    crit = phase_data.get("crit", {}) or {}
    pid = phase_data.get("pid", {}) or {}
    div = phase_data.get("divergence", {}) or {}
    conv = phase_data.get("convergence", {}) or {}

    rho_bar = float(crit.get("rho_bar", 0.0))
    agents = crit.get("agents", {}) or {}

    # CRIT
    render_crit_summary(rho_bar, agents)
    render_crit_pillars(agents)
    render_crit_diagnostics(agents)

    # divergence
    js = float(div.get("js", 0.0))
    ov = float(div.get("ov", 0.0))
    render_divergence_metrics(js, ov)

    # PID
    beta_old = float(pid.get("beta_old", phase_data.get("beta_in", 0.5)))
    beta_new = float(pid.get("beta_new", beta_old))
    quadrant = str(pid.get("quadrant", "unknown"))
    error = float(pid.get("e_t", 0.0))

    render_pid_update(
        beta_old=beta_old,
        beta_new=beta_new,
        quadrant=quadrant,
        rho_star=float(rho_star),
        rho_bar=rho_bar,
        error=error,
    )

    # health bar
    weakest = _find_weakest_pillar(agents)
    render_health_bar(
        round_num=int(phase_data.get("round", 0)),
        rho_bar=rho_bar,
        js=js,
        beta_old=beta_old,
        beta_new=beta_new,
        weakest_pillar=weakest,
        stable_rounds=int(conv.get("stable_rounds", 0)),
        convergence_window=int(conv.get("convergence_window", 2)),
    )


def _find_weakest_pillar(agents: dict) -> str | None:
    """Find the weakest average pillar across agents (phase_data dict format)."""
    pillar_keys = ["LV", "ES", "AC", "CA"]
    pillar_full = {"LV": "Validity", "ES": "Evidence", "AC": "Alternatives", "CA": "Causal"}
    if not agents:
        return None

    n = max(len(agents), 1)
    weakest_key = None
    weakest_val = 2.0
    for k in pillar_keys:
        total = 0.0
        for d in agents.values():
            if isinstance(d, dict):
                total += float(d.get("pillars", {}).get(k, 0.0))
        avg = total / n
        if avg < weakest_val:
            weakest_val = avg
            weakest_key = k

    if weakest_key:
        return f"{weakest_key} ({pillar_full[weakest_key]})"
    return None