"""
Multi-session Hawkes calibration.

Supports two strategies:
  A) Concatenate all sessions → single calibration (default)
  B) Calibrate per session → median parameters across sessions

Usage
-----
    # Option A: concatenated
    uv run python -m quantflow.calibration.calibrate_multi \
        --sessions data/btcusdt/sessions/ \
        --output data/btcusdt/calibration/hawkes_params_v2.json

    # Option B: per-session median
    uv run python -m quantflow.calibration.calibrate_multi \
        --sessions data/btcusdt/sessions/ \
        --strategy median \
        --output data/btcusdt/calibration/hawkes_params_v2.json

    # Also accepts processed event files directly
    uv run python -m quantflow.calibration.calibrate_multi \
        --events data/btcusdt/processed/2026-04-02_events.parquet \
                 data/btcusdt/processed/2026-04-10_events.parquet \
        --output data/btcusdt/calibration/hawkes_params_v2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .calibrate import run_calibration, _truncate_data
from .event_classifier import EventClassifier, HawkesEventData
from .hawkes_mle import CalibrationResult, HawkesParams


def _find_event_files(sessions_dir: Path) -> list[Path]:
    """
    Find event parquet files in a sessions directory.

    Searches for both processed *_events.parquet files and raw session
    files that can be processed.
    """
    files = sorted(sessions_dir.glob("*_events.parquet"))
    if not files:
        files = sorted(sessions_dir.rglob("*_events.parquet"))
    return files


def _load_session_data(
    path: Path,
    max_seconds: float | None = None,
    min_events: int = 100,
) -> HawkesEventData:
    """Load a single session's event data."""
    clf = EventClassifier(min_events_per_dim=min_events)
    data = clf.load_and_classify(path)
    if max_seconds is not None and data.t_span > max_seconds:
        data = _truncate_data(data, max_seconds)
    return data


def _concat_sessions(sessions: list[HawkesEventData]) -> HawkesEventData:
    """
    Concatenate multiple sessions with 1-second gaps between them.
    Shifts timestamps so sessions are contiguous.
    """
    if len(sessions) == 1:
        return sessions[0]

    import polars as pl

    combined_dim_events: list[dict] = []
    combined_dim_stats: list[dict] = []
    n_dims = sessions[0].n_dims
    total = 0

    for dim in range(n_dims):
        all_times: list[np.ndarray] = []
        all_prices: list[np.ndarray] = []
        all_qtys: list[np.ndarray] = []
        offset = 0.0

        for sess in sessions:
            times = sess.dim_events[dim]["times"].copy()
            if len(times) == 0:
                continue
            times = times - times[0] + offset if len(times) > 0 else times
            all_times.append(times)
            all_prices.append(sess.dim_events[dim]["prices"])
            all_qtys.append(sess.dim_events[dim]["quantities"])
            offset = float(times[-1]) + 1.0 if len(times) > 0 else offset + 1.0

        if all_times:
            merged_t = np.concatenate(all_times)
            merged_p = np.concatenate(all_prices)
            merged_q = np.concatenate(all_qtys)
        else:
            merged_t = np.array([], dtype=np.float64)
            merged_p = np.array([], dtype=np.float64)
            merged_q = np.array([], dtype=np.float64)

        inter = np.diff(merged_t) if len(merged_t) > 1 else np.array([], dtype=np.float64)
        total += len(merged_t)

        combined_dim_events.append({
            "times": merged_t,
            "prices": merged_p,
            "quantities": merged_q,
            "inter_times": inter,
        })

    t_span = sum(s.t_span for s in sessions) + len(sessions) - 1
    for dim in range(n_dims):
        t = combined_dim_events[dim]["times"]
        inter = combined_dim_events[dim]["inter_times"]
        combined_dim_stats.append({
            "dim": dim,
            "name": EventClassifier.DIM_NAMES[dim] if dim < len(EventClassifier.DIM_NAMES) else f"dim_{dim}",
            "count": len(t),
            "rate_per_sec": len(t) / t_span if t_span > 0 else 0.0,
            "pct_of_total": len(t) / total * 100 if total > 0 else 0.0,
            "mean_inter_time": float(np.mean(inter)) if len(inter) > 0 else float("inf"),
            "median_inter_time": float(np.median(inter)) if len(inter) > 0 else float("inf"),
            "std_inter_time": float(np.std(inter)) if len(inter) > 0 else 0.0,
            "mean_quantity": 0.0,
            "active": len(t) >= 100,
        })

    sources = ", ".join(s.source_path for s in sessions)
    return HawkesEventData(
        dim_events=combined_dim_events,
        dim_stats=combined_dim_stats,
        t_start=0.0,
        t_end=t_span,
        t_span=t_span,
        total_events=total,
        source_path=sources,
    )


def _median_params(results: list[CalibrationResult]) -> CalibrationResult:
    """
    Compute median parameters across multiple per-session calibrations.
    For each dimension, take the element-wise median of mu, alpha, beta.
    """
    n_dims = results[0].n_dims
    dim_names = results[0].dim_names

    # Collect all calibrated dims across sessions
    all_dims = set()
    for r in results:
        all_dims.update(r.calibrated_dims)

    merged_params: list[HawkesParams] = []
    calibrated: list[int] = []
    skipped: list[int] = []

    for dim in range(n_dims):
        params_for_dim = [r.params_for(dim) for r in results if r.params_for(dim) is not None]
        if len(params_for_dim) < 2:
            if params_for_dim:
                merged_params.append(params_for_dim[0])
                calibrated.append(dim)
            else:
                skipped.append(dim)
            continue

        mu_arr = np.array([p.mu for p in params_for_dim])
        alpha_stack = np.stack([p.alpha for p in params_for_dim])
        beta_stack = np.stack([p.beta for p in params_for_dim])
        n_events_arr = np.array([p.n_events for p in params_for_dim])

        med_mu = float(np.median(mu_arr))
        med_alpha = np.median(alpha_stack, axis=0)
        med_beta = np.median(beta_stack, axis=0)
        br = float(np.sum(med_alpha / np.maximum(med_beta, 1e-9)))

        merged_params.append(HawkesParams(
            dim=dim,
            dim_name=dim_names[dim] if dim < len(dim_names) else f"dim_{dim}",
            n_dims=n_dims,
            mu=med_mu,
            alpha=med_alpha,
            beta=med_beta,
            log_likelihood=0.0,
            branching_ratio=min(br, 0.95),
            n_events=int(np.median(n_events_arr)),
        ))
        calibrated.append(dim)

    total_events = sum(r.total_events for r in results)
    total_tspan = sum(r.t_span for r in results)

    return CalibrationResult(
        dim_params=merged_params,
        dim_names=dim_names,
        t_span=total_tspan,
        total_events=total_events,
        source_path=f"median of {len(results)} sessions",
        calibrated_dims=calibrated,
        skipped_dims=skipped,
        meta={"strategy": "median", "n_sessions": len(results)},
    )


def calibrate_multi(
    event_files: list[Path],
    output_path: Path | None = None,
    per_session_dir: Path | None = None,
    strategy: str = "concat",
    max_iter: int = 500,
    n_restarts: int = 3,
    max_seconds: float | None = None,
    min_events: int = 100,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Calibrate Hawkes parameters from multiple event files.

    Parameters
    ----------
    strategy : str
        "concat" — concatenate all sessions, calibrate once (Option A)
        "median" — calibrate per session, take median params (Option B)
    per_session_dir : Path, optional
        If set, save individual session calibrations here.
    """
    if verbose:
        print()
        print("=" * 70)
        print("  MULTI-SESSION HAWKES CALIBRATION")
        print("=" * 70)
        print(f"  Strategy:  {strategy}")
        print(f"  Sessions:  {len(event_files)}")
        for f in event_files:
            print(f"    - {f.name}")
        print()

    if strategy == "concat":
        sessions = []
        for f in event_files:
            if verbose:
                print(f"  Loading {f.name}...")
            data = _load_session_data(f, max_seconds=max_seconds, min_events=min_events)
            sessions.append(data)
            if verbose:
                print(f"    {data.total_events:,} events, {data.t_span:.0f}s")

        combined = _concat_sessions(sessions)
        if verbose:
            print(f"\n  Combined: {combined.total_events:,} events, {combined.t_span:.0f}s")
            print()

        from .hawkes_mle import HawkesMLE
        from .goodness_of_fit import HawkesGoodnessOfFit

        mle = HawkesMLE(
            n_dims=combined.n_dims,
            dim_names=EventClassifier.DIM_NAMES,
            max_iter=max_iter,
            n_restarts=n_restarts,
            min_events=min_events,
        )

        t0 = time.perf_counter()
        result = mle.calibrate(combined)
        elapsed = time.perf_counter() - t0

        result.meta["strategy"] = "concat"
        result.meta["n_sessions"] = len(event_files)
        result.meta["session_files"] = [str(f) for f in event_files]
        result.meta["calibration_time_s"] = round(elapsed, 1)

        if verbose:
            print(f"\n  Calibration completed in {elapsed:.1f}s")
            for p in result.dim_params:
                print(f"    dim {p.dim:2d} {p.dim_name:20s}  "
                      f"mu={p.mu:.4f}  br={p.branching_ratio:.3f}  N={p.n_events:,}")

    elif strategy == "median":
        per_session_results: list[CalibrationResult] = []

        for i, f in enumerate(event_files):
            if verbose:
                print(f"  [{i+1}/{len(event_files)}] Calibrating {f.name}...")

            sess_result = run_calibration(
                events_path=f,
                output_path=None,
                max_iter=max_iter,
                n_restarts=n_restarts,
                min_events=min_events,
                max_seconds=max_seconds,
                verbose=False,
            )
            per_session_results.append(sess_result)

            if per_session_dir is not None:
                per_session_dir.mkdir(parents=True, exist_ok=True)
                label = f.stem.replace("_events", "")
                sess_path = per_session_dir / f"{label}.json"
                sess_result.save(sess_path)
                if verbose:
                    print(f"    → {sess_path}")

        result = _median_params(per_session_results)

        if verbose:
            print(f"\n  Median parameters across {len(per_session_results)} sessions:")
            for p in result.dim_params:
                print(f"    dim {p.dim:2d} {p.dim_name:20s}  "
                      f"mu={p.mu:.4f}  br={p.branching_ratio:.3f}")

    else:
        raise ValueError(f"Unknown strategy: {strategy!r} (expected 'concat' or 'median')")

    # Check branching ratios
    if verbose:
        print("\n  Branching ratio check:")
        for p in result.dim_params:
            status = "OK" if p.branching_ratio < 1.0 else "WARN"
            print(f"    dim {p.dim:2d}: {p.branching_ratio:.3f}  [{status}]")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        if verbose:
            print(f"\n  Saved → {output_path}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-session Hawkes MLE calibration",
    )
    p.add_argument("--sessions", type=str, default=None,
                   help="Directory containing *_events.parquet files")
    p.add_argument("--events", type=str, nargs="+", default=None,
                   help="Explicit list of event parquet files")
    p.add_argument("--output", type=str,
                   default="data/btcusdt/calibration/hawkes_params_v2.json",
                   help="Output path for calibration result")
    p.add_argument("--per-session-dir", type=str, default=None,
                   help="Directory to save per-session calibrations (median strategy)")
    p.add_argument("--strategy", type=str, default="concat",
                   choices=["concat", "median"],
                   help="Calibration strategy: concat or median")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Truncate each session to this many seconds")
    p.add_argument("--min-events", type=int, default=100)
    args = p.parse_args()

    if args.events:
        files = [Path(f) for f in args.events]
    elif args.sessions:
        files = _find_event_files(Path(args.sessions))
    else:
        p.error("Provide --sessions or --events")

    if not files:
        p.error("No event files found")

    per_session_dir = Path(args.per_session_dir) if args.per_session_dir else None

    calibrate_multi(
        event_files=files,
        output_path=Path(args.output),
        per_session_dir=per_session_dir,
        strategy=args.strategy,
        max_iter=args.max_iter,
        n_restarts=args.n_restarts,
        max_seconds=args.max_seconds,
        min_events=args.min_events,
    )


if __name__ == "__main__":
    main()
