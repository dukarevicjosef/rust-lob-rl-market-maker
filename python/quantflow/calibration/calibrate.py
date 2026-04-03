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

import numpy as np

from .event_classifier import EventClassifier, HawkesEventData
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


# ── npz loader ───────────────────────────────────────────────────────────────


def _load_npz(path: Path, min_events: int) -> HawkesEventData:
    """
    Load a classified_events.npz produced by the event classifier pipeline.

    Expected keys: times_dim0 … times_dim{N-1}, all_times.
    Falls back to event_summary.json in the same directory for metadata.
    """
    import json as _json

    d      = np.load(path, allow_pickle=True)
    n_dims = sum(1 for k in d.keys() if k.startswith("times_dim"))

    all_times_list: list[np.ndarray] = [
        d[f"times_dim{i}"].astype(np.float64) for i in range(n_dims)
    ]

    # t_span from the merged stream
    all_t  = d["all_times"].astype(np.float64) if "all_times" in d else np.concatenate(all_times_list)
    t_start = float(all_t[0])  if len(all_t) > 0 else 0.0
    t_end   = float(all_t[-1]) if len(all_t) > 0 else 0.0
    t_span  = t_end - t_start

    # Optionally enrich from sidecar summary
    summary_path = path.parent / "event_summary.json"
    dim_stats_raw: list[dict] = []
    if summary_path.exists():
        try:
            s = _json.loads(summary_path.read_text())
            dim_stats_raw = s.get("dim_stats", [])
        except Exception:
            pass

    dim_events: list[dict] = []
    dim_stats:  list[dict] = []
    total = 0

    for i in range(n_dims):
        times = all_times_list[i]
        inter = np.diff(times) if len(times) > 1 else np.array([], dtype=np.float64)
        total += len(times)

        if dim_stats_raw and i < len(dim_stats_raw):
            base = dict(dim_stats_raw[i])
        else:
            base = {
                "dim": i,
                "name": EventClassifier.DIM_NAMES[i] if i < len(EventClassifier.DIM_NAMES) else f"dim_{i}",
            }

        base.update({
            "dim":              i,
            "name":             base.get("name", f"dim_{i}"),
            "count":            len(times),
            "rate_per_sec":     len(times) / t_span if t_span > 0 else 0.0,
            "pct_of_total":     0.0,
            "mean_inter_time":  float(np.mean(inter))   if len(inter) > 0 else float("inf"),
            "median_inter_time":float(np.median(inter)) if len(inter) > 0 else float("inf"),
            "std_inter_time":   float(np.std(inter))    if len(inter) > 0 else 0.0,
            "mean_quantity":    0.0,
            "active":           len(times) >= min_events,
        })

        dim_events.append({
            "times":       times,
            "prices":      np.zeros(len(times), dtype=np.float64),
            "quantities":  np.zeros(len(times), dtype=np.float64),
            "inter_times": inter,
        })
        dim_stats.append(base)

    for s in dim_stats:
        s["pct_of_total"] = s["count"] / total * 100 if total > 0 else 0.0

    return HawkesEventData(
        dim_events=dim_events,
        dim_stats=dim_stats,
        t_start=t_start,
        t_end=t_end,
        t_span=t_span,
        total_events=total,
        source_path=str(path),
    )


# ── data truncation helper ────────────────────────────────────────────────────


def _truncate_data(data: HawkesEventData, max_seconds: float) -> HawkesEventData:
    """
    Return a copy of HawkesEventData restricted to events in [t_start, t_start + max_seconds].
    Useful to reduce calibration time on large full-day datasets.
    """
    t_cutoff = data.t_start + max_seconds
    new_dim_events: list[dict] = []
    new_dim_stats:  list[dict] = []
    total = 0

    for i in range(data.n_dims):
        src     = data.dim_events[i]
        times   = src["times"]
        prices  = src["prices"]
        qtys    = src["quantities"]

        mask  = times <= t_cutoff
        times = times[mask]
        inter = np.diff(times) if len(times) > 1 else np.array([], dtype=np.float64)
        total += len(times)

        old = data.dim_stats[i]
        t_span = max_seconds if max_seconds > 0 else 1.0
        new_dim_events.append({
            "times":       times,
            "prices":      prices[mask],
            "quantities":  qtys[mask],
            "inter_times": inter,
        })
        new_dim_stats.append({
            **old,
            "count":            len(times),
            "rate_per_sec":     len(times) / t_span,
            "pct_of_total":     0.0,
            "mean_inter_time":  float(np.mean(inter))   if len(inter) > 0 else float("inf"),
            "median_inter_time":float(np.median(inter)) if len(inter) > 0 else float("inf"),
            "std_inter_time":   float(np.std(inter))    if len(inter) > 0 else 0.0,
            "active":           len(times) >= 50,
        })

    for s in new_dim_stats:
        s["pct_of_total"] = s["count"] / total * 100 if total > 0 else 0.0

    return HawkesEventData(
        dim_events=new_dim_events,
        dim_stats=new_dim_stats,
        t_start=data.t_start,
        t_end=min(data.t_end, t_cutoff),
        t_span=min(data.t_span, max_seconds),
        total_events=total,
        source_path=data.source_path,
    )


# ── main calibration function ─────────────────────────────────────────────────


def run_calibration(
    events_path:  str | Path,
    output_path:  str | Path | None = None,
    max_iter:     int = 500,
    n_restarts:   int = 3,
    skip_gof:     bool = False,
    dims:         list[int] | None = None,
    min_events:   int = 100,
    max_seconds:  float | None = None,
    verbose:      bool = True,
) -> CalibrationResult:
    """
    Load events, run MLE calibration, optionally compute GoF, print report.

    Parameters
    ----------
    max_seconds : float | None
        Truncate event data to the first max_seconds of the observation window.
        Useful for large datasets (e.g., use 600.0 for 10 minutes of BTC data).
        Parameter estimates from a well-behaved 10-minute window are typically
        as accurate as those from a full day.
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

    if events_path.suffix == ".npz":
        data = _load_npz(events_path, min_events)
    else:
        clf  = EventClassifier(min_events_per_dim=min_events)
        data = clf.load_and_classify(events_path)

    # Truncate to first max_seconds of data for faster calibration on large datasets
    if max_seconds is not None and data.t_span > max_seconds:
        data = _truncate_data(data, max_seconds)
        if verbose:
            print(f"  [Truncated to first {max_seconds:.0f}s — {data.total_events:,} events]")
            print()

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
    p.add_argument(
        "--max-seconds", type=float, default=None,
        help=(
            "Truncate input to the first N seconds of the observation window. "
            "Useful for large datasets (e.g., 600 for 10 minutes of BTC tick data). "
            "Parameter estimates converge well beyond a few thousand events per dim."
        ),
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
            max_seconds=args.max_seconds,
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
