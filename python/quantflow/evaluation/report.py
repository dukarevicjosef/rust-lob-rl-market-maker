"""
Load ``results.parquet`` and print a formatted evaluation report.

Sections
--------
1. Performance table — PnL, Sharpe, MaxDD, FillRate, InvStd, Q2T
2. PnL decomposition — Spread PnL vs Inventory PnL per strategy

Usage
-----
    uv run python -m quantflow.evaluation.report \\
        --input results/evaluation/results.parquet \\
        --output-dir results/evaluation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantflow.evaluation.compare import AGENT_ORDER


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame, agent: str) -> dict:
    g = df[df["agent"] == agent]
    return {
        "n":           len(g),
        "pnl_mean":    g["final_pnl"].mean(),
        "pnl_std":     g["final_pnl"].std(),
        "sharpe":      g["sharpe"].mean(),
        "sharpe_std":  g["sharpe"].std(),
        "max_dd":      g["max_drawdown"].mean(),
        "fill_rate":   g["fill_rate"].mean(),
        "inv_std":     g["inventory_std"].mean(),
        "q2t":         g["quote_to_trade"].mean(),
        "spread_pnl":  g["spread_pnl"].mean(),
        "inv_pnl":     g["inventory_pnl"].mean(),
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _rule(n: int = 100) -> str:
    return "─" * n


def _section(title: str) -> str:
    return f"\n{title}\n{_rule(len(title))}"


# ── Report ────────────────────────────────────────────────────────────────────

def report(
    input_path:  str | Path,
    output_dir:  str | Path | None = None,
) -> pd.DataFrame:
    """
    Print and optionally save the evaluation report.

    Returns a DataFrame with one row per agent and aggregated metrics.
    """
    df     = pd.read_parquet(input_path)
    agents = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    stats  = {a: _agg(df, a) for a in agents}

    lines: list[str] = []

    # ── Performance table ──────────────────────────────────────────────────────

    w = 100
    header = (
        f"{'Agent':<20} "
        f"{'PnL (mean±std)':>18} "
        f"{'Sharpe':>14} "
        f"{'MaxDD':>9} "
        f"{'FillRate':>10} "
        f"{'InvStd':>8} "
        f"{'Q2T':>7}"
    )
    lines += [_section("Performance Summary"), header, _rule(w)]

    for agent in agents:
        m = stats[agent]
        lines.append(
            f"{agent:<20} "
            f"{m['pnl_mean']:>+9.2f}±{m['pnl_std']:<7.2f} "
            f"{m['sharpe']:>+10.3f}±{m['sharpe_std']:<3.3f} "
            f"{m['max_dd']:>9.3f} "
            f"{m['fill_rate']:>10.4f} "
            f"{m['inv_std']:>8.2f} "
            f"{m['q2t']:>7.1f}"
        )
    lines.append(_rule(w))

    # ── PnL decomposition ──────────────────────────────────────────────────────

    dw = 58
    dhead = f"{'Agent':<20} {'Spread PnL':>12} {'Inventory PnL':>14} {'Total PnL':>10}"
    lines += [_section("PnL Decomposition  (mean across episodes)"), dhead, _rule(dw)]

    for agent in agents:
        m    = stats[agent]
        tot  = m["spread_pnl"] + m["inv_pnl"]
        lines.append(
            f"{agent:<20} "
            f"{m['spread_pnl']:>+12.2f} "
            f"{m['inv_pnl']:>+14.2f} "
            f"{tot:>+10.2f}"
        )
    lines.append(_rule(dw))

    # ── SAC vs best baseline delta ─────────────────────────────────────────────

    if "SAC" in stats and "Optimized AS" in stats:
        sac = stats["SAC"]
        opt = stats["Optimized AS"]
        lines += [
            _section("SAC vs Optimized AS"),
            f"  ΔPnL    = {sac['pnl_mean'] - opt['pnl_mean']:+.2f}",
            f"  ΔSharpe = {sac['sharpe']   - opt['sharpe']:+.3f}",
            f"  ΔMaxDD  = {sac['max_dd']   - opt['max_dd']:+.3f}",
        ]

    # ── Metadata ───────────────────────────────────────────────────────────────

    n_ep = stats[agents[0]]["n"] if agents else 0
    lines += ["", f"Episodes per agent: {n_ep}   |   Input: {input_path}"]

    report_str = "\n".join(lines)
    print(report_str)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text(report_str + "\n")
        print(f"\nReport saved → {out / 'report.txt'}")

    return pd.DataFrame(stats).T


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate evaluation report from results.parquet")
    p.add_argument("--input",      required=True, help="Path to results.parquet")
    p.add_argument("--output-dir", default=None,  help="Directory to save report.txt")
    args = p.parse_args()
    report(args.input, args.output_dir)


if __name__ == "__main__":
    main()
