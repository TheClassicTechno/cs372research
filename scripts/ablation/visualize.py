"""Auto-generate visualization plots from ablation results.

Produces:
  - Heatmap: steady_state_error over (Kp, Ki)
  - Heatmap: oscillation_rate over (Kp, Ki)
  - Contour: convergence_rate
  - Scatter: beta_overshoot vs settling_round
  - Stability region plot (stable vs unstable gains)
  - ρ contraction rate heatmap

Gracefully skips if matplotlib is not available or data is insufficient.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_plots(results: list[dict], output_dir: Path) -> list[str]:
    """Generate all visualization plots. Returns list of generated file paths.

    Skips gracefully if matplotlib is unavailable or data is insufficient.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping visualization")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    completed = [r for r in results if r.get("status") == "completed"]
    if len(completed) < 3:
        logger.warning("Too few completed runs (%d) for plots", len(completed))
        return []

    generated: list[str] = []

    # Extract gain sweep data for heatmaps
    gain_runs = _extract_gain_grid(completed)

    if gain_runs:
        for name, func in [
            ("steady_state_error_heatmap", _plot_sse_heatmap),
            ("oscillation_heatmap", _plot_oscillation_heatmap),
            ("convergence_contour", _plot_convergence_contour),
            ("stability_region", _plot_stability_region),
            ("kappa_heatmap", _plot_kappa_heatmap),
        ]:
            try:
                path = plots_dir / f"{name}.png"
                func(gain_runs, path)
                generated.append(str(path))
                logger.info("Generated plot: %s", path)
            except Exception as exc:
                logger.warning("Failed to generate %s: %s", name, exc)

    # Scatter: beta_overshoot vs settling_round (uses all completed runs)
    try:
        path = plots_dir / "overshoot_vs_settling.png"
        _plot_overshoot_scatter(completed, path)
        generated.append(str(path))
        logger.info("Generated plot: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate overshoot scatter: %s", exc)

    # Paranoia vs realignment scatter
    try:
        path = plots_dir / "paranoia_vs_realignment.png"
        _plot_paranoia_scatter(completed, path)
        generated.append(str(path))
        logger.info("Generated plot: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate paranoia scatter: %s", exc)

    return generated


def _extract_gain_grid(results: list[dict]) -> list[dict]:
    """Extract runs from the gains group that vary Kp or Ki."""
    return [r for r in results if r.get("group") in ("gains", "random_gain_samples",
                                                       "high_gain_stress", "interactions")]


def _plot_sse_heatmap(runs: list[dict], path: Path) -> None:
    """Heatmap of steady_state_error over (Kp, Ki)."""
    kp_vals, ki_vals, z_vals = [], [], []
    for r in runs:
        kp = r.get("Kp")
        ki = r.get("Ki")
        sse = r.get("steady_state_error")
        if kp is not None and ki is not None and sse is not None:
            kp_vals.append(kp)
            ki_vals.append(ki)
            z_vals.append(sse)

    if len(z_vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(kp_vals, ki_vals, c=z_vals, cmap="RdYlGn_r",
                         s=100, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="Steady State Error (ρ* - ρ_final)")
    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Ki (Integral Gain)")
    ax.set_title("Steady State Error — PID Gain Space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_oscillation_heatmap(runs: list[dict], path: Path) -> None:
    """Heatmap of oscillation flags over (Kp, Ki)."""
    kp_vals, ki_vals, z_vals = [], [], []
    for r in runs:
        kp = r.get("Kp")
        ki = r.get("Ki")
        osc = r.get("rho_oscillation_flag")
        if kp is not None and ki is not None and osc is not None:
            kp_vals.append(kp)
            ki_vals.append(ki)
            z_vals.append(1.0 if osc else 0.0)

    if len(z_vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mcolors.ListedColormap(["#2ecc71", "#e74c3c"])
    scatter = ax.scatter(kp_vals, ki_vals, c=z_vals, cmap=cmap,
                         s=100, edgecolors="black", linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label="ρ Oscillation (0=stable, 1=oscillatory)",
                 ticks=[0, 1])
    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Ki (Integral Gain)")
    ax.set_title("ρ Oscillation Detection — PID Gain Space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_convergence_contour(runs: list[dict], path: Path) -> None:
    """Scatter plot of convergence window met over (Kp, Ki)."""
    kp_vals, ki_vals, z_vals = [], [], []
    for r in runs:
        kp = r.get("Kp")
        ki = r.get("Ki")
        conv = r.get("convergence_window_met")
        if kp is not None and ki is not None and conv is not None:
            kp_vals.append(kp)
            ki_vals.append(ki)
            z_vals.append(1.0 if conv else 0.0)

    if len(z_vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])
    scatter = ax.scatter(kp_vals, ki_vals, c=z_vals, cmap=cmap,
                         s=100, edgecolors="black", linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label="Converged (0=no, 1=yes)", ticks=[0, 1])
    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Ki (Integral Gain)")
    ax.set_title("Convergence (Window) — PID Gain Space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stability_region(runs: list[dict], path: Path) -> None:
    """Stability region: control_stable and behavioral_stable over (Kp, Ki)."""
    kp_vals, ki_vals, labels = [], [], []
    for r in runs:
        kp = r.get("Kp")
        ki = r.get("Ki")
        cs = r.get("control_stable")
        bs = r.get("behavioral_stable")
        if kp is not None and ki is not None:
            kp_vals.append(kp)
            ki_vals.append(ki)
            if cs and bs:
                labels.append(2)  # Both stable
            elif cs:
                labels.append(1)  # Control only
            else:
                labels.append(0)  # Unstable

    if len(labels) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mcolors.ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])
    scatter = ax.scatter(kp_vals, ki_vals, c=labels, cmap=cmap,
                         s=100, edgecolors="black", linewidths=0.5, vmin=0, vmax=2)
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Unstable", "Control Only", "Both Stable"])
    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Ki (Integral Gain)")
    ax.set_title("Stability Regions — PID Gain Space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_kappa_heatmap(runs: list[dict], path: Path) -> None:
    """Heatmap of empirical contraction rate κ over (Kp, Ki)."""
    kp_vals, ki_vals, z_vals = [], [], []
    for r in runs:
        kp = r.get("Kp")
        ki = r.get("Ki")
        kappa = r.get("empirical_kappa")
        if kp is not None and ki is not None and kappa is not None:
            kp_vals.append(kp)
            ki_vals.append(ki)
            z_vals.append(kappa)

    if len(z_vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(kp_vals, ki_vals, c=z_vals, cmap="RdYlGn_r",
                         s=100, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="Empirical κ (< 1 = contraction)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Ki (Integral Gain)")
    ax.set_title("Empirical Contraction Rate κ — PID Gain Space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_overshoot_scatter(results: list[dict], path: Path) -> None:
    """Scatter: beta_overshoot vs settling_round, colored by control_stable."""
    x, y, c = [], [], []
    for r in results:
        overshoot = r.get("beta_overshoot")
        settling = r.get("settling_round")
        stable = r.get("behavioral_stable")
        if overshoot is not None and settling is not None:
            x.append(overshoot)
            y.append(settling)
            c.append(1.0 if stable else 0.0)

    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])
    scatter = ax.scatter(x, y, c=c, cmap=cmap, s=60, edgecolors="black",
                         linewidths=0.5, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label="Behavioral Stable", ticks=[0, 1])
    ax.set_xlabel("Beta Overshoot (max(β) - initial_β)")
    ax.set_ylabel("Settling Round")
    ax.set_title("Overshoot vs Settling Time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_paranoia_scatter(results: list[dict], path: Path) -> None:
    """Scatter: paranoia_rate vs realignment_rate."""
    x, y, labels = [], [], []
    for r in results:
        paranoia = r.get("paranoia_rate")
        realign = r.get("realignment_rate")
        if paranoia is not None and realign is not None:
            x.append(paranoia)
            y.append(realign)
            labels.append(r.get("group", ""))

    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=60, edgecolors="black", linewidths=0.5, alpha=0.7)

    # Diagonal: net_effect = 0 line
    lim = max(max(x + [0.01]), max(y + [0.01])) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="net_effect = 0")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Paranoia Rate P(T→F | correct)")
    ax.set_ylabel("Realignment Rate P(F→T | incorrect)")
    ax.set_title("Paranoia Tax vs Realignment Benefit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
