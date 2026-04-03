"""
6 comparison plots: Empirical (amber) vs. Simulated (blue).
All plots use a consistent dark theme matching the dashboard palette.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stylized_facts import StylizedFacts

# ── Style ─────────────────────────────────────────────────────────────────────

_STYLE: dict = {
    "figure.facecolor":  "#0a0a0a",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#333333",
    "text.color":        "#cccccc",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#222222",
    "grid.alpha":        0.5,
    "axes.titlecolor":   "#dddddd",
    "legend.facecolor":  "#1a1a1a",
    "legend.edgecolor":  "#333333",
}

COLOR_EMP   = "#f59e0b"   # amber — empirical
COLOR_SIM   = "#3b82f6"   # blue  — simulated
COLOR_GAUSS = "#666666"   # grey  — Gaussian reference


# ── Public API ────────────────────────────────────────────────────────────────


def plot_all(
    emp: StylizedFacts,
    sim: StylizedFacts,
    output_dir: str,
) -> list[str]:
    """
    Generate all 6 comparison plots and save to output_dir.
    Returns list of paths actually saved (skips plots with missing data).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    with plt.rc_context(_STYLE):
        for fn in (
            _plot_qq,
            _plot_return_distribution,
            _plot_acf,
            _plot_spread,
            _plot_intraday,
            _plot_signature,
        ):
            path = fn(emp, sim, output_dir)
            if path:
                saved.append(path)
    return saved


# ── Internal plot helpers ─────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _plot_qq(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 1: Q-Q — scaled IET vs. Exp(1/mean) quantiles."""
    if not emp.qq and not sim.qq:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Q-Q plot — inter-event times scaled by mean rate vs. Exp(1)",
        fontsize=13, y=1.01,
    )

    for ax, sf, color in zip(axes, [emp, sim], [COLOR_EMP, COLOR_SIM]):
        if not sf.qq:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#555")
            continue
        th = np.asarray(sf.qq["theoretical"])
        em = np.asarray(sf.qq["empirical"])
        lim = float(np.nanpercentile(np.concatenate([th, em]), 99))
        ax.scatter(th, em, s=1.5, alpha=0.4, color=color, rasterized=True)
        ax.plot([0, lim], [0, lim], "--", color="#ffffff", alpha=0.2, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(sf.label or ("Empirical" if color == COLOR_EMP else "Simulated"),
                     fontsize=11)
        ax.set_xlabel("Exp(1) theoretical quantiles")
        ax.set_ylabel("Scaled inter-event times")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return _save(fig, f"{out}/qq_plot.png")


def _plot_return_distribution(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 2: Log₁₀-density of normalised log-returns (fat tails)."""
    if not emp.returns and not sim.returns:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))

    if emp.returns:
        ax.plot(
            emp.returns["bin_centers"], emp.returns["log_density"],
            color=COLOR_EMP, linewidth=1.6,
            label=f"{emp.label or 'Empirical'}  κ={emp.returns['kurtosis']:.1f}",
        )
        ax.plot(
            emp.returns["bin_centers"], emp.returns["log_gaussian"],
            color=COLOR_GAUSS, linewidth=1, linestyle="--", label="Gaussian",
        )
    if sim.returns:
        ax.plot(
            sim.returns["bin_centers"], sim.returns["log_density"],
            color=COLOR_SIM, linewidth=1.6,
            label=f"{sim.label or 'Simulated'}  κ={sim.returns['kurtosis']:.1f}",
        )

    ax.set_xlabel("Normalised log-return  (z-score)")
    ax.set_ylabel("log₁₀(density)")
    ax.set_title("Return distribution — fat tails", fontsize=13)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, f"{out}/return_distribution.png")


def _plot_acf(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 3: ACF of |returns| — volatility clustering."""
    if not emp.acf and not sim.acf:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))

    if emp.acf:
        ax.plot(emp.acf["lags"], emp.acf["acf"],
                color=COLOR_EMP, linewidth=1.5, alpha=0.85,
                label=emp.label or "Empirical")
    if sim.acf:
        ax.plot(sim.acf["lags"], sim.acf["acf"],
                color=COLOR_SIM, linewidth=1.5, alpha=0.85,
                label=sim.label or "Simulated")

    ax.axhline(0, color="#ffffff", alpha=0.12, linewidth=0.5)
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("ACF")
    ax.set_title("ACF of |returns| — volatility clustering", fontsize=13)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, f"{out}/acf_absolute_returns.png")


def _plot_spread(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 4: Bid-ask spread distribution."""
    if not emp.spread and not sim.spread:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))

    if emp.spread:
        ax.plot(
            emp.spread["bin_centers"], emp.spread["density"],
            color=COLOR_EMP, linewidth=1.6,
            label=f"{emp.label or 'Empirical'}  med={emp.spread['median']:.2f}",
        )
    if sim.spread:
        ax.plot(
            sim.spread["bin_centers"], sim.spread["density"],
            color=COLOR_SIM, linewidth=1.6,
            label=f"{sim.label or 'Simulated'}  med={sim.spread['median']:.2f}",
        )

    ax.set_xlabel("Bid-ask spread (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Spread distribution", fontsize=13)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, f"{out}/spread_distribution.png")


def _plot_intraday(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 5: Intraday event-rate profile."""
    if not emp.intraday and not sim.intraday:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))

    if emp.intraday:
        centers = np.asarray(emp.intraday["bin_centers_min"])
        width   = (centers[1] - centers[0]) * 0.4 if len(centers) > 1 else 1.0
        ax.bar(centers - width / 2, emp.intraday["rates"],
               width=width, alpha=0.75, color=COLOR_EMP,
               label=emp.label or "Empirical")
    if sim.intraday:
        centers = np.asarray(sim.intraday["bin_centers_min"])
        width   = (centers[1] - centers[0]) * 0.4 if len(centers) > 1 else 1.0
        ax.bar(centers + width / 2, sim.intraday["rates"],
               width=width, alpha=0.75, color=COLOR_SIM,
               label=sim.label or "Simulated")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Events per second")
    ax.set_title("Intraday activity profile", fontsize=13)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, f"{out}/intraday_volume.png")


def _plot_signature(emp: StylizedFacts, sim: StylizedFacts, out: str) -> str | None:
    """Plot 6: Signature plot — realised vol vs. Δt."""
    if not emp.signature and not sim.signature:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))

    if emp.signature:
        rv = np.where(np.isfinite(emp.signature["realized_vol"]),
                      emp.signature["realized_vol"], np.nan)
        ax.plot(emp.signature["dt"], rv,
                "o-", color=COLOR_EMP, linewidth=1.6, markersize=5,
                label=emp.label or "Empirical")
    if sim.signature:
        rv = np.where(np.isfinite(sim.signature["realized_vol"]),
                      sim.signature["realized_vol"], np.nan)
        ax.plot(sim.signature["dt"], rv,
                "s-", color=COLOR_SIM, linewidth=1.6, markersize=5,
                label=sim.label or "Simulated")

    ax.set_xscale("log")
    ax.set_xlabel("Sampling interval Δt (seconds)")
    ax.set_ylabel("Realised volatility (per second)")
    ax.set_title("Signature plot — microstructure noise at high frequency", fontsize=13)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, f"{out}/signature_plot.png")
