"""
Generate six matplotlib plots comparing all four market-making strategies.

Plots produced
--------------
1. cumulative_pnl.png      — median + 25/75 percentile band per strategy
2. pnl_distribution.png    — box plot of episode-end PnL
3. inventory_trajectories.png — inventory paths for one episode (seed_start)
4. sharpe_comparison.png   — bar chart with ±1 std error bars
5. fillrate_vs_risk.png    — scatter: fill rate vs inventory std
6. pnl_decomposition.png   — stacked bar: spread PnL + inventory PnL

Usage
-----
    uv run python -m quantflow.evaluation.plots \\
        --input results/evaluation/results.parquet \\
        --output-dir results/evaluation/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from quantflow.evaluation.compare import AGENT_ORDER, AGENT_COLORS


plt.style.use("seaborn-v0_8-whitegrid")

_FIG   = (10, 6)
_DPI   = 150
_ALPHA = 0.22   # band fill transparency


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color(agent: str) -> str:
    return AGENT_COLORS.get(agent, "#555555")


def _legend_patches(agents: list[str]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=_color(a), label=a)
        for a in agents
    ]


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dest = path / name
    fig.savefig(dest, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {dest}")


# ── Plot 1: Cumulative PnL Curves ─────────────────────────────────────────────

def plot_cumulative_pnl(
    traj_df:   pd.DataFrame,
    agents:    list[str],
    output:    Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    for agent in agents:
        color = _color(agent)
        sub   = traj_df[traj_df["agent"] == agent]

        # Pivot: rows=step, cols=episode_id
        pivot = sub.pivot_table(
            index="step", columns="episode_id", values="cumulative_pnl"
        )
        pivot = pivot.ffill()   # forward-fill truncated episodes

        steps  = pivot.index.to_numpy()
        median = pivot.median(axis=1).to_numpy()
        p25    = pivot.quantile(0.25, axis=1).to_numpy()
        p75    = pivot.quantile(0.75, axis=1).to_numpy()

        ax.plot(steps, median, color=color, linewidth=1.8, label=agent)
        ax.fill_between(steps, p25, p75, color=color, alpha=_ALPHA)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Cumulative PnL", fontsize=11)
    ax.set_title("Cumulative PnL — Median ± IQR  (50 episodes)", fontsize=13)
    ax.legend(handles=_legend_patches(agents), fontsize=10)
    _save(fig, output, "cumulative_pnl.png")


# ── Plot 2: PnL Distribution ──────────────────────────────────────────────────

def plot_pnl_distribution(
    summary_df: pd.DataFrame,
    agents:     list[str],
    output:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    data   = [summary_df[summary_df["agent"] == a]["final_pnl"].to_numpy() for a in agents]
    colors = [_color(a) for a in agents]

    bp = ax.boxplot(
        data,
        patch_artist = True,
        notch        = False,
        widths       = 0.5,
        medianprops  = {"color": "white", "linewidth": 2.0},
        flierprops   = {"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]:
        whisker.set(color="#555555", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="#555555", linewidth=1.2)
    for flier, color in zip(bp["fliers"], colors):
        flier.set(markerfacecolor=color, markeredgecolor="none")

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, fontsize=10)
    ax.set_ylabel("Episode PnL", fontsize=11)
    ax.set_title("PnL Distribution per Strategy  (50 episodes)", fontsize=13)
    _save(fig, output, "pnl_distribution.png")


# ── Plot 3: Inventory Trajectories ────────────────────────────────────────────

def plot_inventory_trajectories(
    traj_df:    pd.DataFrame,
    agents:     list[str],
    episode_id: int,
    output:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    for agent in agents:
        sub = traj_df[
            (traj_df["agent"]      == agent) &
            (traj_df["episode_id"] == episode_id)
        ]
        if sub.empty:
            continue
        ax.plot(
            sub["step"].to_numpy(),
            sub["inventory"].to_numpy(),
            color     = _color(agent),
            linewidth = 1.6,
            label     = agent,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Inventory (lots)", fontsize=11)
    ax.set_title(f"Inventory Trajectories  (episode seed {episode_id})", fontsize=13)
    ax.legend(handles=_legend_patches(agents), fontsize=10)
    _save(fig, output, "inventory_trajectories.png")


# ── Plot 4: Sharpe Comparison ─────────────────────────────────────────────────

def plot_sharpe_comparison(
    summary_df: pd.DataFrame,
    agents:     list[str],
    output:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    means  = [summary_df[summary_df["agent"] == a]["sharpe"].mean() for a in agents]
    stds   = [summary_df[summary_df["agent"] == a]["sharpe"].std()  for a in agents]
    colors = [_color(a) for a in agents]
    x      = np.arange(len(agents))

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, width=0.5,
                  capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ypos = bar.get_height() + (stds[bars.index(bar)] if mean >= 0 else -stds[bars.index(bar)])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.02 * (1 if mean >= 0 else -1),
            f"{mean:+.3f}",
            ha="center", va="bottom" if mean >= 0 else "top",
            fontsize=9, color="#222222",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe Ratio Comparison  (mean ± 1 std)", fontsize=13)
    _save(fig, output, "sharpe_comparison.png")


# ── Plot 5: Fill Rate vs Inventory Risk ───────────────────────────────────────

def plot_fillrate_vs_risk(
    summary_df: pd.DataFrame,
    agents:     list[str],
    output:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    for agent in agents:
        sub    = summary_df[summary_df["agent"] == agent]
        x_mean = sub["fill_rate"].mean()
        y_mean = sub["inventory_std"].mean()
        color  = _color(agent)

        ax.scatter(x_mean, y_mean, color=color, s=120, zorder=5)
        ax.annotate(
            agent,
            xy       = (x_mean, y_mean),
            xytext   = (6, 4),
            textcoords="offset points",
            fontsize = 9,
            color    = color,
        )

    ax.set_xlabel("Mean Fill Rate  (fills / step)", fontsize=11)
    ax.set_ylabel("Mean Inventory Std  (lots)", fontsize=11)
    ax.set_title("Fill Rate vs Inventory Risk Tradeoff", fontsize=13)
    _save(fig, output, "fillrate_vs_risk.png")


# ── Plot 6: PnL Decomposition ─────────────────────────────────────────────────

def plot_pnl_decomposition(
    summary_df: pd.DataFrame,
    agents:     list[str],
    output:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG)

    x             = np.arange(len(agents))
    spread_means  = [summary_df[summary_df["agent"] == a]["spread_pnl"].mean()    for a in agents]
    inv_means     = [summary_df[summary_df["agent"] == a]["inventory_pnl"].mean() for a in agents]

    # Stacked: spread_pnl as base, inventory_pnl stacked on top
    ax.bar(x, spread_means, width=0.5, label="Spread PnL",    color="#4CAF50", alpha=0.85)
    ax.bar(x, inv_means,    width=0.5, label="Inventory PnL", color="#F44336", alpha=0.75,
           bottom=spread_means)

    # Total marker
    totals = [s + i for s, i in zip(spread_means, inv_means)]
    ax.scatter(x, totals, color="black", s=60, zorder=5, label="Total PnL")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=10)
    ax.set_ylabel("PnL (mean across episodes)", fontsize=11)
    ax.set_title("PnL Decomposition: Spread Revenue vs Inventory P&L", fontsize=13)
    ax.legend(fontsize=10)
    _save(fig, output, "pnl_decomposition.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all(
    results_path:     str | Path,
    output_dir:       str | Path = "results/evaluation/plots",
    trajectories_path: str | Path | None = None,
    episode_id:       int        = 0,
) -> None:
    """
    Generate all six plots from previously saved Parquet files.

    Parameters
    ----------
    results_path : path-like
        Path to ``results.parquet`` (episode summaries).
    output_dir : path-like
        Directory to write PNG files into.
    trajectories_path : path-like, optional
        Path to ``trajectories.parquet``.  Inferred from ``results_path``'s
        parent directory if not provided.
    episode_id : int
        Episode index used for the inventory trajectory plot.
    """
    results_path  = Path(results_path)
    output_dir    = Path(output_dir)

    if trajectories_path is None:
        trajectories_path = results_path.parent / "trajectories.parquet"

    summary_df = pd.read_parquet(results_path)
    agents     = [a for a in AGENT_ORDER if a in summary_df["agent"].unique()]

    print(f"Generating plots for: {agents}")
    print(f"Output → {output_dir}\n")

    # Trajectory data (needed for plots 1 & 3)
    traj_df: pd.DataFrame | None = None
    if Path(trajectories_path).exists():
        traj_df = pd.read_parquet(trajectories_path)
    else:
        print(f"  [warn] trajectories.parquet not found at {trajectories_path}")
        print(f"  [warn] plots 1 and 3 will be skipped\n")

    if traj_df is not None:
        print("Plot 1 — Cumulative PnL Curves")
        plot_cumulative_pnl(traj_df, agents, output_dir)

    print("Plot 2 — PnL Distribution")
    plot_pnl_distribution(summary_df, agents, output_dir)

    if traj_df is not None:
        print("Plot 3 — Inventory Trajectories")
        plot_inventory_trajectories(traj_df, agents, episode_id, output_dir)

    print("Plot 4 — Sharpe Ratio Comparison")
    plot_sharpe_comparison(summary_df, agents, output_dir)

    print("Plot 5 — Fill Rate vs Inventory Risk")
    plot_fillrate_vs_risk(summary_df, agents, output_dir)

    print("Plot 6 — PnL Decomposition")
    plot_pnl_decomposition(summary_df, agents, output_dir)

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate evaluation plots")
    p.add_argument("--input",       required=True,
                   help="Path to results.parquet")
    p.add_argument("--trajectories", default=None,
                   help="Path to trajectories.parquet (auto-detected if omitted)")
    p.add_argument("--output-dir",  default="results/evaluation/plots")
    p.add_argument("--episode-id",  type=int, default=0,
                   help="Episode index for inventory trajectory plot (default 0)")
    args = p.parse_args()

    generate_all(
        results_path      = args.input,
        output_dir        = args.output_dir,
        trajectories_path = args.trajectories,
        episode_id        = args.episode_id,
    )


if __name__ == "__main__":
    main()
