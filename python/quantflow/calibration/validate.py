"""
Stylized Facts Validation CLI.

Usage:
  uv run python -m quantflow.calibration.validate \\
    --params  data/btcusdt/calibration/hawkes_params.json \\
    --events  data/btcusdt/processed/2026-04-02_events.parquet \\
    --output  results/calibration/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .simulate_calibrated import simulate_from_calibration
from .stylized_facts import StylizedFacts
from .plot_stylized_facts import plot_all


# ── Session aggregation ───────────────────────────────────────────────────────


def _aggregate_sessions(sessions: list[dict]) -> dict:
    """
    Concatenate N sessions into a single aggregate session for stats.

    Prices are chained continuously: each session's price series is shifted so
    that its starting price equals the ending price of the previous session.
    This prevents artificial return spikes at session boundaries.
    """
    all_events:  list[tuple[float, int]] = []
    all_mid:     list[np.ndarray]        = []
    all_spreads: list[np.ndarray]        = []
    t_offset    = 0.0
    last_price  = None   # ending price of the previous session

    for s in sessions:
        t_max_s = s["t_max"]
        for t, d in s["events"]:
            all_events.append((t + t_offset, d))

        if len(s["mid_prices"]) > 0:
            mp = s["mid_prices"].copy()
            if last_price is not None:
                # Shift all prices so this session starts at last_price
                price_shift = last_price - mp[0, 1]
                mp[:, 1] += price_shift
            mp[:, 0] += t_offset
            all_mid.append(mp)
            last_price = float(mp[-1, 1])

        if len(s["spreads"]) > 0:
            sp = s["spreads"].copy()
            sp[:, 0] += t_offset
            all_spreads.append(sp)

        t_offset += t_max_s + 1.0

    return {
        "events":     all_events,
        "mid_prices": np.vstack(all_mid)    if all_mid    else np.empty((0, 2)),
        "spreads":    np.vstack(all_spreads) if all_spreads else np.empty((0, 2)),
        "t_max":      t_offset,
    }


# ── Report ────────────────────────────────────────────────────────────────────


def _match(emp_v: float, sim_v: float, rtol: float = 0.30) -> str:
    if not (np.isfinite(emp_v) and np.isfinite(sim_v)):
        return "—"
    err = abs(emp_v - sim_v) / max(abs(emp_v), 1e-12)
    if err < rtol * 0.5:
        return "✓"
    if err < rtol:
        return "≈"
    return "✗"


def _get(d: dict, *keys, default: float = float("nan")) -> float:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return float(d) if np.isfinite(float(d)) else default


def _print_report(
    emp:       StylizedFacts,
    sim:       StylizedFacts,
    saved:     list[str],
    t_max:     float,
    n_sessions: int,
) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  STYLIZED FACTS VALIDATION — BTCUSDT                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Empirical : {t_max:.0f}s of BTCUSDT (2026-04-02)")
    print(f"  Simulated : {n_sessions} sessions × {t_max:.0f}s, calibrated Hawkes")
    print()

    hdr  = f"  {'Stylized Fact':<26} │ {'Empirical':>11} │ {'Simulated':>11} │ {'Match':>5}"
    sep  = "  " + "─" * 26 + "─┼─" + "─" * 11 + "─┼─" + "─" * 11 + "─┼─" + "─" * 5
    print(hdr)
    print(sep)

    def row(label: str, ev: float, sv: float, fmt: str = ".4f") -> None:
        es = f"{ev:{fmt}}" if np.isfinite(ev) else "—"
        ss = f"{sv:{fmt}}" if np.isfinite(sv) else "—"
        print(f"  {label:<26} │ {es:>11} │ {ss:>11} │ {_match(ev, sv):>5}")

    # Return distribution
    ek = _get(emp.returns, "kurtosis")
    sk = _get(sim.returns, "kurtosis")
    ew = _get(emp.returns, "skewness")
    sw = _get(sim.returns, "skewness")
    row("Return kurtosis",  ek, sk, ".2f")
    row("Return skewness",  ew, sw, ".3f")

    # ACF
    ea1  = _get(emp.acf, "acf") if not emp.acf else (emp.acf.get("acf") or [float("nan")])[0]
    sa1  = _get(sim.acf, "acf") if not sim.acf else (sim.acf.get("acf") or [float("nan")])[0]
    ea1  = float(ea1) if isinstance(ea1, (int, float)) else float("nan")
    sa1  = float(sa1) if isinstance(sa1, (int, float)) else float("nan")
    ea100_list = emp.acf.get("acf", []) if emp.acf else []
    sa100_list = sim.acf.get("acf", []) if sim.acf else []
    ea100 = float(ea100_list[99]) if len(ea100_list) > 99 else float("nan")
    sa100 = float(sa100_list[99]) if len(sa100_list) > 99 else float("nan")
    row("ACF |r| lag=1s",   ea1,   sa1,   ".4f")
    row("ACF |r| lag=100s", ea100, sa100, ".4f")

    # Spread
    em_med = _get(emp.spread, "median")
    sm_med = _get(sim.spread, "median")
    row("Spread median (USD)", em_med, sm_med, ".4f")

    # Event rate
    e_rates = emp.intraday.get("rates", []) if emp.intraday else []
    s_rates = sim.intraday.get("rates", []) if sim.intraday else []
    e_rate  = float(np.mean(e_rates)) if e_rates else float("nan")
    s_rate  = float(np.mean(s_rates)) if s_rates else float("nan")
    row("Events per second",   e_rate, s_rate, ".2f")

    # Signature vol
    def _rv_at(sf: StylizedFacts, target_dt: float) -> float:
        if not sf.signature:
            return float("nan")
        dts = sf.signature.get("dt", [])
        rvs = sf.signature.get("realized_vol", [])
        for dt_, rv in zip(dts, rvs):
            if abs(dt_ - target_dt) < target_dt * 0.1:
                return float(rv)
        return float("nan")

    row("Signature vol (Δt=1s)",  _rv_at(emp, 1.0),  _rv_at(sim, 1.0),  ".6f")
    row("Signature vol (Δt=60s)", _rv_at(emp, 60.0), _rv_at(sim, 60.0), ".6f")

    print()
    print("  ✓ = good match (<15%)   ≈ = approximate (<30%)   ✗ = poor match")
    print()

    if saved:
        print("  Plots saved:")
        for p in saved:
            print(f"    {p}")
        print()

    print("  Known limitations:")
    print("    - Intraday profile is flat (stationary Hawkes, no U-shape)")
    print("    - Spread model is a simplified mean-reverting noise process")
    print("    - Long-memory (Hurst exponent) may underfit with exponential kernels")
    print()


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_validation(
    params_path:  str | Path,
    events_path:  str | Path,
    output_dir:   str | Path,
    n_sessions:   int   = 10,
    t_max:        float = 600.0,
    verbose:      bool  = True,
) -> tuple[StylizedFacts, StylizedFacts]:
    """
    Full validation pipeline: empirical facts → simulation → 6 plots → report.

    Returns
    -------
    (emp, sim) : pair of StylizedFacts objects
    """
    params_path = Path(params_path)
    events_path = Path(events_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  STYLIZED FACTS — COMPUTING                                         ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")

    # 1 — Empirical
    if verbose:
        print(f"\n  [1/3] Empirical facts from {events_path.name} …")
    t0 = time.perf_counter()
    if events_path.suffix == ".npz":
        emp = StylizedFacts.from_npz(str(events_path), t_max=t_max)
    else:
        emp = StylizedFacts.from_parquet(str(events_path), t_max=t_max)
    if verbose:
        print(f"        {time.perf_counter()-t0:.1f}s")

    # 2 — Simulation
    if verbose:
        print(f"\n  [2/3] Simulating {n_sessions} × {t_max:.0f}s sessions …")
    t0 = time.perf_counter()
    sessions = simulate_from_calibration(
        params_path=params_path,
        n_sessions=n_sessions,
        t_max=t_max,
    )
    agg = _aggregate_sessions(sessions)
    sim = StylizedFacts.from_session(agg)
    if verbose:
        n_ev = len(agg["events"])
        print(f"        {time.perf_counter()-t0:.1f}s — {n_ev:,} simulated events")

    # 3 — Plots
    if verbose:
        print("\n  [3/3] Generating 6 comparison plots …")
    saved = plot_all(emp, sim, str(output_dir))

    # 4 — Report
    if verbose:
        _print_report(emp, sim, saved, t_max, n_sessions)

    return emp, sim


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantflow.calibration.validate",
        description="Stylized facts validation: empirical BTCUSDT vs. calibrated Hawkes.",
    )
    p.add_argument("--params",     required=True,
                   help="hawkes_params.json from calibrate CLI")
    p.add_argument("--events",     required=True,
                   help="Events file: .parquet (with prices) or .npz (IET + intraday only)")
    p.add_argument("--output",     required=True,
                   help="Directory for saved plot PNGs")
    p.add_argument("--n-sessions", type=int,   default=10,
                   help="Number of sessions to simulate (default 10)")
    p.add_argument("--t-max",      type=float, default=600.0,
                   help="Session length in seconds (default 600)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run_validation(
            params_path=args.params,
            events_path=args.events,
            output_dir=args.output,
            n_sessions=args.n_sessions,
            t_max=args.t_max,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
