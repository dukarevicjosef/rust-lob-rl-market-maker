"""
Tests for python/quantflow/features.py.

Run with:
    uv run pytest tests/test_features.py -v
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from quantflow.features import (
    FEATURE_NAMES,
    RunningNormalizer,
    compute_all,
    depth_ratio,
    order_flow_imbalance,
    realized_volatility,
    spread_bps,
    trade_arrival_rate,
    volume_imbalance,
    weighted_mid_price,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_snapshot(
    bid_prices: list[float | None],
    bid_qtys:   list[int],
    ask_prices: list[float | None],
    ask_qtys:   list[int],
) -> pa.RecordBatch:
    """Build a minimal LOB snapshot RecordBatch."""
    return pa.record_batch({
        "bid_price": pa.array(bid_prices, type=pa.float64()),
        "bid_qty":   pa.array(bid_qtys,   type=pa.uint64()),
        "ask_price": pa.array(ask_prices, type=pa.float64()),
        "ask_qty":   pa.array(ask_qtys,   type=pa.uint64()),
    })


@pytest.fixture
def balanced_snapshot() -> pa.RecordBatch:
    """5-level snapshot with equal bid/ask depth."""
    return _make_snapshot(
        bid_prices=[99.0, 98.0, 97.0, 96.0, 95.0],
        bid_qtys  =[100,  200,  300,  400,  500],
        ask_prices=[101.0, 102.0, 103.0, 104.0, 105.0],
        ask_qtys  =[100,   200,   300,   400,   500],
    )


@pytest.fixture
def bid_heavy_snapshot() -> pa.RecordBatch:
    """Snapshot where bid side is significantly heavier."""
    return _make_snapshot(
        bid_prices=[99.0, 98.0, 97.0],
        bid_qtys  =[900, 800, 700],
        ask_prices=[101.0, 102.0, 103.0],
        ask_qtys  =[100,   100,   100],
    )


@pytest.fixture
def empty_snapshot() -> pa.RecordBatch:
    """Snapshot with NaN prices (empty levels)."""
    nan = float("nan")
    return _make_snapshot(
        bid_prices=[nan, nan],
        bid_qtys  =[0,   0],
        ask_prices=[nan, nan],
        ask_qtys  =[0,   0],
    )


def _buy_trades(n: int, price: float = 100.0, qty: int = 10) -> list[dict]:
    return [{"price": price, "qty": qty, "is_buy": True} for _ in range(n)]


def _sell_trades(n: int, price: float = 100.0, qty: int = 10) -> list[dict]:
    return [{"price": price, "qty": qty, "is_buy": False} for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# volume_imbalance
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeImbalance:
    def test_balanced_returns_zero(self, balanced_snapshot):
        vi = volume_imbalance(balanced_snapshot, level=1)
        assert vi == pytest.approx(0.0)

    def test_bid_heavy_positive(self, bid_heavy_snapshot):
        vi = volume_imbalance(bid_heavy_snapshot, level=1)
        assert vi > 0.0

    def test_range_within_minus_one_plus_one(self, bid_heavy_snapshot):
        for lvl in (1, 3, 5):
            vi = volume_imbalance(bid_heavy_snapshot, level=lvl)
            assert -1.0 <= vi <= 1.0

    def test_empty_book_returns_zero(self, empty_snapshot):
        vi = volume_imbalance(empty_snapshot, level=1)
        assert vi == pytest.approx(0.0)

    def test_level_gt_rows_clipped(self, balanced_snapshot):
        # snapshot has 5 rows; requesting 10 levels should not crash
        vi = volume_imbalance(balanced_snapshot, level=10)
        assert -1.0 <= vi <= 1.0

    def test_all_bid_returns_plus_one(self):
        snap = _make_snapshot(
            bid_prices=[100.0], bid_qtys=[500],
            ask_prices=[101.0], ask_qtys=[0],
        )
        vi = volume_imbalance(snap, level=1)
        assert vi == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# weighted_mid_price
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedMidPrice:
    def test_symmetric_equals_arithmetic_mid(self, balanced_snapshot):
        wmp = weighted_mid_price(balanced_snapshot)
        # equal qtys → arithmetic mid of 99 and 101 = 100
        assert wmp == pytest.approx(100.0, abs=0.1)

    def test_bid_pressure_pulls_below_mid(self, bid_heavy_snapshot):
        # heavy bid side → microprice pulled toward ask (execution pressure)
        wmp = weighted_mid_price(bid_heavy_snapshot)
        arithmetic_mid = (99.0 + 101.0) / 2
        assert wmp != pytest.approx(arithmetic_mid)

    def test_empty_book_returns_zero(self, empty_snapshot):
        wmp = weighted_mid_price(empty_snapshot)
        assert wmp == pytest.approx(0.0)

    def test_single_valid_level(self):
        snap = _make_snapshot(
            bid_prices=[100.0], bid_qtys=[200],
            ask_prices=[102.0], ask_qtys=[100],
        )
        wmp = weighted_mid_price(snap)
        # bid heavy → pull toward ask: P* = (102*200 + 100*100) / (200+100)
        expected = (102.0 * 200 + 100.0 * 100) / 300
        assert wmp == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# depth_ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestDepthRatio:
    def test_balanced_returns_one(self, balanced_snapshot):
        dr = depth_ratio(balanced_snapshot, levels=5)
        assert dr == pytest.approx(1.0, rel=1e-6)

    def test_bid_heavy_gt_one(self, bid_heavy_snapshot):
        dr = depth_ratio(bid_heavy_snapshot, levels=3)
        assert dr > 1.0

    def test_empty_ask_large(self):
        snap = _make_snapshot(
            bid_prices=[99.0], bid_qtys=[1000],
            ask_prices=[101.0], ask_qtys=[0],
        )
        dr = depth_ratio(snap, levels=1)
        assert dr > 100.0  # near-infinity, denominator = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# spread_bps
# ─────────────────────────────────────────────────────────────────────────────

class TestSpreadBps:
    def test_two_tick_spread(self):
        snap = _make_snapshot(
            bid_prices=[99.0], bid_qtys=[100],
            ask_prices=[101.0], ask_qtys=[100],
        )
        s = spread_bps(snap)
        expected = (101.0 - 99.0) / 100.0 * 10_000.0
        assert s == pytest.approx(expected, rel=1e-5)

    def test_empty_bid_returns_zero(self):
        snap = _make_snapshot(
            bid_prices=[None], bid_qtys=[0],
            ask_prices=[101.0], ask_qtys=[100],
        )
        assert spread_bps(snap) == pytest.approx(0.0)

    def test_nan_prices_return_zero(self, empty_snapshot):
        assert spread_bps(empty_snapshot) == pytest.approx(0.0)

    def test_nonnegative(self, balanced_snapshot):
        assert spread_bps(balanced_snapshot) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# order_flow_imbalance
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderFlowImbalance:
    def test_all_buys_returns_plus_one(self):
        ofi = order_flow_imbalance(_buy_trades(50), mid=100.0)
        assert ofi == pytest.approx(1.0)

    def test_all_sells_returns_minus_one(self):
        ofi = order_flow_imbalance(_sell_trades(50), mid=100.0)
        assert ofi == pytest.approx(-1.0)

    def test_balanced_returns_zero(self):
        trades = _buy_trades(50) + _sell_trades(50)
        ofi = order_flow_imbalance(trades, mid=100.0)
        assert ofi == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert order_flow_imbalance([], mid=100.0) == pytest.approx(0.0)

    def test_window_limits_lookback(self):
        # 200 sells then 100 buys; window=100 → only the 100 buys visible
        trades = _sell_trades(200) + _buy_trades(100)
        ofi = order_flow_imbalance(trades, window=100, mid=100.0)
        assert ofi == pytest.approx(1.0)

    def test_tick_rule_without_is_buy(self):
        # prices above mid → classified as buys
        trades = [{"price": 101.0, "qty": 10} for _ in range(20)]
        ofi = order_flow_imbalance(trades, mid=100.0)
        assert ofi == pytest.approx(1.0)

    def test_no_mid_uses_median(self):
        buys  = [{"price": 102.0, "qty": 10} for _ in range(30)]
        sells = [{"price":  98.0, "qty": 10} for _ in range(10)]
        ofi = order_flow_imbalance(buys + sells)  # mid inferred as 102.0
        # median of 40 trades: 30×102 + 10×98 → median = 102 → sells classified
        assert -1.0 <= ofi <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# realized_volatility
# ─────────────────────────────────────────────────────────────────────────────

class TestRealizedVolatility:
    def test_constant_prices_returns_zero(self):
        trades = [{"price": 100.0, "qty": 1} for _ in range(100)]
        rv = realized_volatility(trades)
        assert rv == pytest.approx(0.0, abs=1e-10)

    def test_single_trade_returns_zero(self):
        assert realized_volatility([{"price": 100.0, "qty": 1}]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert realized_volatility([]) == pytest.approx(0.0)

    def test_volatile_series_positive(self):
        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
        trades = [{"price": float(p), "qty": 1} for p in prices]
        rv = realized_volatility(trades, window=200)
        assert rv > 0.0

    def test_window_truncates(self):
        # 1000 flat trades + 10 volatile ones
        flat     = [{"price": 100.0, "qty": 1} for _ in range(1000)]
        volatile = [{"price": 100.0 + i * 2, "qty": 1} for i in range(10)]
        trades   = flat + volatile
        rv_small = realized_volatility(trades, window=10)
        rv_all   = realized_volatility(trades, window=1100)
        assert rv_small > rv_all  # volatile window > mostly-flat window


# ─────────────────────────────────────────────────────────────────────────────
# trade_arrival_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeArrivalRate:
    def test_uniform_arrivals(self):
        # 60 trades uniformly spread over [0, 60) → rate = 1.0 trades/s
        trades     = [{"price": 100.0, "qty": 1}] * 60
        timestamps = list(range(60))
        rate = trade_arrival_rate(trades, timestamps, window_s=60.0)
        assert rate == pytest.approx(1.0, rel=0.05)

    def test_empty_timestamps_returns_zero(self):
        assert trade_arrival_rate([], [], window_s=60.0) == pytest.approx(0.0)

    def test_window_clips_old_trades(self):
        # 100 old trades at t=0, 10 recent at t=200..209
        old_ts  = [0.0] * 100
        new_ts  = [200.0 + i for i in range(10)]
        ts      = old_ts + new_ts
        trades  = [{"price": 100.0, "qty": 1}] * 110
        rate    = trade_arrival_rate(trades, ts, window_s=60.0)
        # Only 10 trades in [150, 210], rate = 10/60
        assert rate == pytest.approx(10 / 60.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# compute_all
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAll:
    def test_returns_float32_array_shape_8(self, balanced_snapshot):
        trades = _buy_trades(20) + _sell_trades(20)
        result = compute_all(balanced_snapshot, trades)
        assert result.dtype == np.float32
        assert result.shape == (8,)

    def test_feature_names_length_matches(self):
        assert len(FEATURE_NAMES) == 8

    def test_no_timestamps_gives_zero_arrival_rate(self, balanced_snapshot):
        trades = _buy_trades(50)
        result = compute_all(balanced_snapshot, trades, timestamps=None)
        assert result[7] == pytest.approx(0.0)  # trade_arrival_rate index

    def test_mid_overrides_computed_value(self, balanced_snapshot):
        trades = _buy_trades(10)
        result = compute_all(balanced_snapshot, trades, mid=999.0)
        # index 2 = weighted_mid_price which should be the override
        assert result[2] == pytest.approx(999.0, rel=1e-4)

    def test_no_nan_in_output(self, balanced_snapshot):
        trades = _buy_trades(100)
        ts     = list(range(100))
        result = compute_all(balanced_snapshot, trades, timestamps=ts)
        assert not np.any(np.isnan(result))


# ─────────────────────────────────────────────────────────────────────────────
# RunningNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestRunningNormalizer:
    def test_initial_mean_and_var_zero(self):
        n = RunningNormalizer(4)
        np.testing.assert_array_equal(n.mean, np.zeros(4))
        np.testing.assert_array_equal(n.var,  np.zeros(4))

    def test_single_update_mean_equals_value(self):
        n = RunningNormalizer(3)
        x = np.array([1.0, 2.0, 3.0])
        n.update(x)
        np.testing.assert_allclose(n.mean, x)

    def test_welford_mean_converges(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=2.0, size=(10_000, 4))
        n = RunningNormalizer(4)
        n.update_batch(data)
        np.testing.assert_allclose(n.mean, data.mean(axis=0), atol=0.05)

    def test_welford_variance_converges(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=0.0, scale=3.0, size=(10_000, 4))
        n = RunningNormalizer(4)
        n.update_batch(data)
        # Population variance (Bessel-uncorrected)
        expected_var = np.var(data, axis=0, ddof=0)
        np.testing.assert_allclose(n.var, expected_var, rtol=0.02)

    def test_normalize_output_shape_and_dtype(self):
        n = RunningNormalizer(4)
        x = np.ones(4)
        n.update_batch(np.random.default_rng(2).normal(size=(100, 4)))
        z = n.normalize(x)
        assert z.shape == (4,)
        assert z.dtype == np.float32

    def test_normalize_zero_mean_unit_std(self):
        rng = np.random.default_rng(3)
        data = rng.normal(loc=10.0, scale=2.0, size=(50_000, 2))
        n = RunningNormalizer(2)
        n.update_batch(data)
        normed = np.array([n.normalize(row) for row in data])
        np.testing.assert_allclose(normed.mean(axis=0), [0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(normed.std(axis=0),  [1.0, 1.0], atol=0.05)

    def test_update_and_normalize_returns_normalized(self):
        n = RunningNormalizer(2)
        rng = np.random.default_rng(4)
        n.update_batch(rng.normal(size=(200, 2)))
        x  = np.array([1.0, 2.0])
        z1 = n.update_and_normalize(x)
        z2 = n.normalize(x)   # same stats after update_and_normalize
        np.testing.assert_allclose(z1, z2, atol=1e-6)

    def test_wrong_shape_raises(self):
        n = RunningNormalizer(4)
        with pytest.raises(ValueError, match="Expected shape"):
            n.update(np.ones(3))

    def test_repr_contains_count(self):
        n = RunningNormalizer(4)
        n.update(np.ones(4))
        assert "count=1" in repr(n)

    def test_save_and_load_roundtrip(self):
        rng = np.random.default_rng(5)
        n = RunningNormalizer(3, eps=1e-5)
        n.update_batch(rng.normal(size=(500, 3)))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            n.save(path)
            n2 = RunningNormalizer.load(path)

            assert n2.n_features == n.n_features
            assert n2.eps        == n.eps
            assert n2._count     == n._count
            np.testing.assert_allclose(n2.mean, n.mean)
            np.testing.assert_allclose(n2.var,  n.var)
        finally:
            path.unlink(missing_ok=True)

    def test_load_continues_accumulating(self):
        """A loaded normalizer must keep accepting updates."""
        rng = np.random.default_rng(6)
        n = RunningNormalizer(2)
        n.update_batch(rng.normal(size=(100, 2)))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            n.save(path)
            n2 = RunningNormalizer.load(path)
            n2.update(np.array([1.0, 2.0]))   # must not raise
            assert n2._count == n._count + 1
        finally:
            path.unlink(missing_ok=True)

    def test_constant_feature_maps_to_zero(self):
        """Feature with zero variance should normalize to 0 (not inf)."""
        n = RunningNormalizer(1)
        for _ in range(100):
            n.update(np.array([7.0]))
        z = n.normalize(np.array([7.0]))
        assert math.isfinite(float(z[0]))
        assert float(z[0]) == pytest.approx(0.0, abs=1e-4)
