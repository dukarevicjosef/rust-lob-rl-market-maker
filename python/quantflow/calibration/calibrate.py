"""
CLI: Hawkes MLE calibration with GoF diagnostics and cross-excitation report.

Usage
-----
    python -m quantflow.calibration.calibrate \\
        --events data/btcusdt/processed/2026-01-15_events.parquet \\
        --output results/hawkes_2026-01-15.json \\
        --max-iter 1000 \\
        --dims 0 1 2 3

Or as a library:

    from quantflow.calibration.calibrate import run_calibration
    result = run_calibration(events_path="...", output_path="...")
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .event_classifier import EventClassifier
from .hawkes_mle import HawkesMLE, CalibrationResult
from .goodness_of_fit import HawkesGoodnessOfFit


# ── formatting helpers ────────────────────────────────────────────────────────


def _fmt_float(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_bool(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _bar(v: float, max_v: float, width: int = 12) -> str:
    """ASCII bar proportional to v/max_v."""
    if max_v <= 0:
        return " " * width
    filled = int(round(min(v / max_v, 1.0) * width))
    return "█" * filled + "░" * (width - filled)


# ── main calibration function ─────────────────────────────────────────────────


def run_calibration(
    events_path:  str | Path,
    output_path:  str | Path | None = None,
    max_iter:     int = 500,
    n_restarts:   int = 3,
    skip_gof:     bool = False,
    dims:         list[int] | None = None,
    min_events:   int = 100,
    verbose:      bool = True,
) -> CalibrationResult:
    """
    Load events, run MLE calibration, optionally compute GoF, print report.

    Returns the CalibrationResult (also saved to output_path if provided).
    """
    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # ── load and classify ─────────────────────────────────────────────────────
    if verbose:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  HAWKES MLE CALIBRATION                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print(f"  Source:   {events_path}")
        print()

    clf  = EventClassifier(min_events_per_dim=min_events)
    data = clf.load_and_classify(events_path)

    if verbose:
        data.print_summary()
        print()

    if data.total_events == 0:
        raise ValueError("No events found in the input file.")

    # ── calibrate ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    mle = HawkesMLE(
        n_dims=data.n_dims,
        dim_names=EventClassifier.DIM_NAMES,
        max_iter=max_iter,
        n_restarts=n_restarts,
        min_events=min_events,
    )

    result = mle.calibrate(data, dims=dims)

    elapsed = time.perf_counter() - t0

    # ── GoF ───────────────────────────────────────────────────────────────────
    all_times = [data.times_for_dim(d) for d in range(data.n_dims)]
    gof_results: dict[int, dict] = {}

    if not skip_gof:
        for p in result.dim_params:
            gof = HawkesGoodnessOfFit(p, all_times, data.t_span)
            gof_results[p.dim] = gof.summary()

    # ── report ────────────────────────────────────────────────────────────────
    if verbose:
        _print_calibration_report(result, gof_results, elapsed)

    # ── save ──────────────────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        if verbose:
            print(f"\n  Saved → {output_path}")

    return result


# ── report printer ────────────────────────────────────────────────────────────


def _print_calibration_report(
    result:      CalibrationResult,
    gof_results: dict[int, dict],
    elapsed_s:   float,
) -> None:
    D    = result.n_dims
    has_gof = len(gof_results) > 0

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  CALIBRATION RESULTS                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Observation window : {result.t_span:.1f}s  ({result.t_span/3600:.2f}h)")
    print(f"  Total events       : {result.total_events:,}")
    print(f"  Calibrated dims    : {result.calibrated_dims}")
    print(f"  Skipped dims       : {result.skipped_dims}")
    print(f"  Wall-clock         : {elapsed_s:.1f}s")
    print()

    # ── per-dimension table ───────────────────────────────────────────────────
    hdr_gof = "  KS Stat   KS p    GoF  " if has_gof else ""
    print(
        f"  {'Dim':>3} │ {'Name':<20} │ {'μ':>8} │ {'Σα/β':>6} │ "
        f"{'LL':>12} │ {'N':>7} │{hdr_gof}"
    )
    sep = "─" * (3 + 3 + 20 + 3 + 8 + 3 + 6 + 3 + 12 + 3 + 7 + (25 if has_gof else 0))
    print(f"  {sep}")

    for p in sorted(result.dim_params, key=lambda x: x.dim):
        gof_str = ""
        if has_gof and p.dim in gof_results:
            g = gof_results[p.dim]
            gof_str = (
                f"  {g['ks_statistic']:>6.4f}  {g['ks_p_value']:>6.4f}  "
                f"{_fmt_bool(g['ks_passed']):<4}"
            )
        print(
            f"  {p.dim:>3} │ {p.dim_name:<20} │ {p.mu:>8.4f} │ "
            f"{p.branching_ratio:>6.4f} │ {p.log_likelihood:>12.2f} │ "
            f"{p.n_events:>7,} │{gof_str}"
        )

    print()

    # ── excitation matrix heatmap (ASCII) ────────────────────────────────────
    if len(result.dim_params) > 0:
        mat   = result.excitation_matrix()
        max_v = np.max(mat) if np.max(mat) > 0 else 1.0

        print("  EXCITATION MATRIX  α_ij / β_ij  (row=target, col=source)")
        print("  " + " " * 24 + "  ".join(f"{j:>4}" for j in range(D)))
        print("  " + "─" * (24 + 6 * D))

        for i in result.calibrated_dims:
            name_short = result.dim_names[i][:20]
            row_vals   = "  ".join(
                f"{mat[i,j]:>4.2f}" if j in result.calibrated_dims else "    "
                for j in range(D)
            )
            print(f"  {i:>2} {name_short:<20}  {row_vals}")

        print()

        # ── top cross-excitations ─────────────────────────────────────────────
        top = result.top_cross_excitations(n=8)
        if top:
            print("  TOP CROSS-EXCITATIONS")
            print(f"  {'Source':<22} → {'Target':<22}  Strength  Bar")
            print("  " + "─" * 70)
            max_strength = top[0][2] if top else 1.0
            for tgt, src, strength in top:
                bar = _bar(strength, max_strength, width=14)
                sname = result.dim_names[src][:20]
                tname = result.dim_names[tgt][:20]
                print(f"  {sname:<22} → {tname:<22}  {strength:>6.4f}    {bar}")
            print()

    # ── GoF summary ──────────────────────────────────────────────────────────
    if has_gof:
        n_pass = sum(1 for g in gof_results.values() if g["ks_passed"])
        n_tot  = len(gof_results)
        print(f"  GOODNESS OF FIT  (Time-Rescaling + KS test, α=0.05)")
        print(f"  Passed: {n_pass}/{n_tot} dimensions")
        for dim, g in sorted(gof_results.items()):
            status = "✓" if g["ks_passed"] else "✗"
            print(
                f"    {status} Dim {dim:>2} {g['dim_name']:<22} "
                f"τ̄={g['tau_mean']:.3f}  KS={g['ks_statistic']:.4f}  p={g['ks_p_value']:.4f}"
            )
        print()


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantflow.calibration.calibrate",
        description="Hawkes MLE calibration from classified market events.",
    )
    p.add_argument(
        "--events", required=True,
        help="Path to *_events.parquet produced by the event classifier.",
    )
    p.add_argument(
        "--output", default=None,
        help="Optional JSON output path for CalibrationResult.",
    )
    p.add_argument(
        "--max-iter", type=int, default=500,
        help="L-BFGS-B maximum iterations per dimension (default 500).",
    )
    p.add_argument(
        "--n-restarts", type=int, default=3,
        help="Number of random restarts (default 3).",
    )
    p.add_argument(
        "--skip-gof", action="store_true",
        help="Skip Time-Rescaling GoF computation (faster).",
    )
    p.add_argument(
        "--dims", nargs="+", type=int, default=None,
        help="Specific dimension indices to calibrate (default: all active).",
    )
    p.add_argument(
        "--min-events", type=int, default=100,
        help="Minimum events per dimension to attempt calibration (default 100).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        run_calibration(
            events_path=args.events,
            output_path=args.output,
            max_iter=args.max_iter,
            n_restarts=args.n_restarts,
            skip_gof=args.skip_gof,
            dims=args.dims,
            min_events=args.min_events,
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
