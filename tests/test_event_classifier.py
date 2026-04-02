from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantflow.calibration import EventClassifier, HawkesEventData


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_parquet(df: pl.DataFrame, directory: Path, name: str = "events.parquet") -> Path:
    path = directory / name
    df.write_parquet(path)
    return path


def _make_df(
    timestamps: list[float],
    event_types: list[int],
    prices: list[float] | None = None,
    quantities: list[float] | None = None,
) -> pl.DataFrame:
    n = len(timestamps)
    return pl.DataFrame({
        "timestamp":  timestamps,
        "event_type": event_types,
        "price":      prices if prices is not None else [50_000.0] * n,
        "quantity":   quantities if quantities is not None else [0.01] * n,
    })


# ── load_and_classify ─────────────────────────────────────────────────────────

def test_empty_parquet_returns_zero_counts():
    clf = EventClassifier()
    df = _make_df([], [])
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    assert data.total_events == 0
    assert data.active_dims == []
    for dim in range(12):
        assert len(data.times_for_dim(dim)) == 0


def test_missing_column_raises():
    clf = EventClassifier()
    with tempfile.TemporaryDirectory() as tmp:
        bad_df = pl.DataFrame({"timestamp": [1.0], "event_type": [0]})
        path = _write_parquet(bad_df, Path(tmp))
        with pytest.raises(ValueError, match="Missing required column"):
            clf.load_and_classify(path)


def test_only_market_buy_events():
    clf = EventClassifier(min_events_per_dim=1)
    timestamps = [float(i) for i in range(200)]
    event_types = [0] * 200   # all MarketBuy
    df = _make_df(timestamps, event_types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    assert data.total_events == 200
    assert len(data.times_for_dim(0)) == 200
    for dim in range(1, 12):
        assert len(data.times_for_dim(dim)) == 0
    assert data.active_dims == [0]


def test_mixed_events_correct_distribution():
    clf = EventClassifier(min_events_per_dim=1)
    # 60 MarketBuy (dim 0), 40 MarketSell (dim 1), 10 LimitBuyBest (dim 2)
    ts = sorted([float(i) for i in range(110)])
    types = [0] * 60 + [1] * 40 + [2] * 10
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    assert data.total_events == 110
    assert len(data.times_for_dim(0)) == 60
    assert len(data.times_for_dim(1)) == 40
    assert len(data.times_for_dim(2)) == 10
    for dim in range(3, 12):
        assert len(data.times_for_dim(dim)) == 0


def test_active_dims_threshold():
    clf = EventClassifier(min_events_per_dim=50)
    ts = [float(i) for i in range(160)]
    # dim 0: 100 events, dim 1: 60 events, dim 2: 0 events
    types = [0] * 100 + [1] * 60
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    assert 0 in data.active_dims
    assert 1 in data.active_dims
    for dim in range(2, 12):
        assert dim not in data.active_dims


# ── all_times_flat ────────────────────────────────────────────────────────────

def test_all_times_flat_sorted():
    clf = EventClassifier(min_events_per_dim=1)
    # Interleaved events: dim 0 at odd seconds, dim 1 at even seconds
    ts = [float(i) for i in range(20)]
    types = [i % 2 for i in range(20)]
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    times, dims = data.all_times_flat()
    assert len(times) == 20
    assert np.all(np.diff(times) >= 0), "all_times_flat must be chronologically sorted"
    assert set(dims.tolist()) == {0, 1}


# ── compute_intraday_profile ──────────────────────────────────────────────────

def test_intraday_profile_bin_sums_equal_total():
    clf = EventClassifier(min_events_per_dim=1)
    # 3600 events over 3600 seconds → 1 event/sec
    ts = [float(i) for i in range(3600)]
    types = [i % 12 for i in range(3600)]
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    profile = clf.compute_intraday_profile(data, bin_minutes=30)

    assert sum(profile["counts"]) == data.total_events
    # per_dim counts also sum to total
    per_dim_total = sum(
        sum(profile["per_dim"][dim]) for dim in range(12)
    )
    assert per_dim_total == data.total_events


# ── load_multiple_days ────────────────────────────────────────────────────────

def test_load_multiple_days_no_timestamp_overlap():
    clf = EventClassifier(min_events_per_dim=1)
    day1 = _make_df([0.0, 1.0, 2.0], [0, 1, 0])
    day2 = _make_df([0.0, 1.0, 2.0], [1, 0, 1])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_parquet(day1, tmp_path, "2026-04-01_events.parquet")
        _write_parquet(day2, tmp_path, "2026-04-02_events.parquet")

        data = clf.load_multiple_days(tmp_path, dates=["2026-04-01", "2026-04-02"])

    assert data.total_events == 6

    times, _ = data.all_times_flat()
    # No overlap: all diffs must be >= 0
    assert np.all(np.diff(times) >= 0)
    # Day 2 timestamps must be strictly greater than day 1's last timestamp
    # (due to the 1-second gap)
    t0 = data.times_for_dim(0)
    t1 = data.times_for_dim(1)
    all_t = np.sort(np.concatenate([t0, t1]))
    assert all_t[3] > all_t[2]   # first day-2 event comes after last day-1 event


def test_load_multiple_days_missing_file_raises():
    clf = EventClassifier()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            clf.load_multiple_days(Path(tmp), dates=["9999-01-01"])


# ── filter_outliers ───────────────────────────────────────────────────────────

def test_filter_outliers_removes_long_gaps():
    clf = EventClassifier(min_events_per_dim=1)
    # dim 0: normal events at t=0,1,2; then a 120s gap; then t=122,123
    ts = [0.0, 1.0, 2.0, 122.0, 123.0]
    types = [0] * 5
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    filtered = clf.filter_outliers(data, max_inter_time=60.0)

    # t=122 should be dropped (preceded by 120s gap > 60s); t=123 has gap 1s → kept
    remaining = len(filtered.times_for_dim(0))
    assert remaining == 4   # 0,1,2 kept; 122 dropped; 123 kept


def test_filter_outliers_removes_zero_price():
    clf = EventClassifier(min_events_per_dim=1)
    ts = [float(i) for i in range(5)]
    types = [0] * 5
    prices = [50_000.0, 0.0, 50_000.0, 50_000.0, 50_000.0]
    df = _make_df(ts, types, prices=prices)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    filtered = clf.filter_outliers(data, min_price=0.01)
    assert len(filtered.times_for_dim(0)) == 4


def test_filter_outliers_removes_zero_quantity():
    clf = EventClassifier(min_events_per_dim=1)
    ts = [float(i) for i in range(4)]
    types = [0] * 4
    quantities = [0.01, 0.0, 0.01, 0.01]
    df = _make_df(ts, types, quantities=quantities)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(df, Path(tmp))
        data = clf.load_and_classify(path)

    filtered = clf.filter_outliers(data)
    assert len(filtered.times_for_dim(0)) == 3


# ── CLI smoke test ────────────────────────────────────────────────────────────

def test_classify_cli_creates_output_files():
    from quantflow.calibration.classify import main

    clf_check = EventClassifier(min_events_per_dim=1)
    ts = [float(i) for i in range(500)]
    types = [i % 12 for i in range(500)]
    df = _make_df(ts, types)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = _write_parquet(df, tmp_path, "test_events.parquet")
        output_dir = tmp_path / "calibration"

        rc = main([
            "--input", str(input_path),
            "--output", str(output_dir),
            "--min-events", "1",
        ])
        assert rc == 0

        assert (output_dir / "event_summary.json").exists()
        assert (output_dir / "intraday_profile.json").exists()
        assert (output_dir / "classified_events.npz").exists()

        # Verify npz content
        npz = np.load(output_dir / "classified_events.npz")
        assert "all_times" in npz
        assert "all_dims" in npz
        assert len(npz["all_times"]) == 500

        # Verify JSON is valid
        summary = json.loads((output_dir / "event_summary.json").read_text())
        assert summary["total_events"] == 500
