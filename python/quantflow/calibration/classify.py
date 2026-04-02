"""
Event classifier CLI — prepares normalised MarketEvents for Hawkes MLE calibration.

Usage
-----
Single day:
  uv run python -m quantflow.calibration.classify \\
    --input data/btcusdt/processed/2026-04-02_events.parquet \\
    --output data/btcusdt/calibration/

Multiple days (auto-discover):
  uv run python -m quantflow.calibration.classify \\
    --input-dir data/btcusdt/processed/ \\
    --output data/btcusdt/calibration/

Multiple days (explicit):
  uv run python -m quantflow.calibration.classify \\
    --input-dir data/btcusdt/processed/ \\
    --dates 2026-04-01 2026-04-02 \\
    --output data/btcusdt/calibration/

Output files
------------
  event_summary.json          — per-dimension statistics
  intraday_profile.json       — event counts/rates per 30-min bin
  classified_events.npz       — numpy arrays: times_dim0 … times_dim11,
                                all_times, all_dims
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .event_classifier import EventClassifier


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify MarketEvents for Hawkes calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",     type=str, help="Single *_events.parquet file")
    source.add_argument("--input-dir", type=str, help="Directory with *_events.parquet files")

    p.add_argument("--dates",          nargs="+", default=None,
                   help="Date strings (YYYY-MM-DD) to load from --input-dir")
    p.add_argument("--output",         type=str, required=True,
                   help="Output directory for calibration artefacts")
    p.add_argument("--min-events",     type=int, default=100,
                   help="Min events per dim to be considered active (default: 100)")
    p.add_argument("--bin-minutes",    type=int, default=30,
                   help="Bin size in minutes for intraday profile (default: 30)")
    p.add_argument("--max-inter-time", type=float, default=60.0,
                   help="Max allowed inter-event gap in seconds (default: 60.0)")
    p.add_argument("--min-price",      type=float, default=0.01,
                   help="Minimum valid price (default: 0.01)")
    p.add_argument("--no-filter",      action="store_true",
                   help="Skip outlier filtering step")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf = EventClassifier(min_events_per_dim=args.min_events)

    # ── Step 1: load ──────────────────────────────────────────────────────────
    print("Loading events…")
    if args.input:
        data = clf.load_and_classify(args.input)
    else:
        data = clf.load_multiple_days(args.input_dir, dates=args.dates)

    # ── Step 2: summary ───────────────────────────────────────────────────────
    data.print_summary()

    # ── Step 3: intraday profile ──────────────────────────────────────────────
    print(f"\nComputing intraday profile ({args.bin_minutes}-minute bins)…")
    profile = clf.compute_intraday_profile(data, bin_minutes=args.bin_minutes)

    # ── Step 4: outlier filtering ─────────────────────────────────────────────
    if not args.no_filter:
        before = data.total_events
        data = clf.filter_outliers(
            data,
            max_inter_time=args.max_inter_time,
            min_price=args.min_price,
        )
        dropped = before - data.total_events
        pct = dropped / before * 100 if before > 0 else 0.0
        print(f"\nOutlier filtering: removed {dropped:,} events ({pct:.2f}%)")
        print(f"Remaining: {data.total_events:,}")
    else:
        print("\nOutlier filtering skipped (--no-filter).")

    # ── Step 5: save artefacts ────────────────────────────────────────────────
    # event_summary.json
    summary_path = output_dir / "event_summary.json"
    summary = {
        "source_path":   data.source_path,
        "t_span":        data.t_span,
        "total_events":  data.total_events,
        "active_dims":   data.active_dims,
        "dim_stats":     data.dim_stats,
        "filter": {
            "applied":        not args.no_filter,
            "max_inter_time": args.max_inter_time,
            "min_price":      args.min_price,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")

    # intraday_profile.json
    profile_path = output_dir / "intraday_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"Wrote {profile_path}")

    # classified_events.npz
    npz_data: dict[str, np.ndarray] = {}
    for dim in range(data.n_dims):
        npz_data[f"times_dim{dim}"] = data.times_for_dim(dim)

    all_times, all_dims = data.all_times_flat()
    npz_data["all_times"] = all_times
    npz_data["all_dims"]  = all_dims

    npz_path = output_dir / "classified_events.npz"
    np.savez_compressed(npz_path, **npz_data)
    size_mb = npz_path.stat().st_size / 1_000_000
    print(f"Wrote {npz_path}  ({size_mb:.2f} MB)")

    # ── Step 6: final report ──────────────────────────────────────────────────
    active = data.active_dims
    print(f"\n{'─'*60}")
    print(f"  Active dimensions:  {len(active)}/{data.n_dims}")
    print(f"  Output directory:   {output_dir.resolve()}")
    print(f"  Ready for Hawkes MLE calibration.")
    print(f"{'─'*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
