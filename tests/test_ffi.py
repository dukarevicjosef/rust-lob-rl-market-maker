"""
Tests for the quantflow PyO3 extension module.

Run with:
    uv run maturin develop --release
    uv run pytest tests/test_ffi.py -v
"""
from __future__ import annotations

import pyarrow as pa
import pytest

import quantflow


# ── OrderBook ─────────────────────────────────────────────────────────────────

class TestOrderBook:
    def test_empty_book_has_no_best(self):
        book = quantflow.OrderBook()
        assert book.best_bid() is None
        assert book.best_ask() is None
        assert book.mid_price() is None
        assert book.spread() is None

    def test_add_resting_bid(self):
        book = quantflow.OrderBook()
        report = book.add_limit_order("bid", 99.5, 100)
        assert report["order_id"] == 1
        assert report["filled_qty"] == 0
        assert report["remaining_qty"] == 100
        assert report["is_fully_filled"] is False
        assert report["trades"] == []

    def test_best_bid_after_insert(self):
        book = quantflow.OrderBook()
        book.add_limit_order("bid", 99.5, 100)
        bid = book.best_bid()
        assert bid is not None
        price, qty = bid
        assert abs(price - 99.5) < 1e-6
        assert qty == 100

    def test_best_ask_after_insert(self):
        book = quantflow.OrderBook()
        book.add_limit_order("ask", 100.5, 50)
        ask = book.best_ask()
        assert ask is not None
        price, qty = ask
        assert abs(price - 100.5) < 1e-6
        assert qty == 50

    def test_mid_price_and_spread(self):
        book = quantflow.OrderBook()
        book.add_limit_order("bid", 99.0, 100)
        book.add_limit_order("ask", 101.0, 100)
        assert abs(book.mid_price() - 100.0) < 1e-6
        assert abs(book.spread() - 2.0) < 1e-6

    def test_crossing_order_generates_trade(self):
        book = quantflow.OrderBook()
        book.add_limit_order("ask", 100.0, 100)
        report = book.add_limit_order("bid", 101.0, 50)
        assert len(report["trades"]) == 1
        trade = report["trades"][0]
        assert abs(trade["price"] - 100.0) < 1e-6
        assert trade["qty"] == 50

    def test_cancel_removes_order(self):
        book = quantflow.OrderBook()
        report = book.add_limit_order("bid", 99.0, 100)
        oid = report["order_id"]
        cancelled = book.cancel_order(oid)
        assert cancelled is not None
        assert cancelled["order_id"] == oid
        assert book.best_bid() is None

    def test_cancel_unknown_order_returns_none(self):
        book = quantflow.OrderBook()
        assert book.cancel_order(999_999) is None

    def test_snapshot_returns_record_batch(self):
        book = quantflow.OrderBook()
        book.add_limit_order("bid", 99.0, 200)
        book.add_limit_order("bid", 98.0, 150)
        book.add_limit_order("ask", 101.0, 100)
        book.add_limit_order("ask", 102.0, 80)

        batch = book.snapshot(5)  # returns pa.RecordBatch directly
        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 5
        assert "bid_price" in batch.schema.names
        assert "bid_qty"   in batch.schema.names
        assert "ask_price" in batch.schema.names
        assert "ask_qty"   in batch.schema.names

    def test_snapshot_best_prices_correct(self):
        book = quantflow.OrderBook()
        book.add_limit_order("bid", 99.0, 200)
        book.add_limit_order("ask", 101.0, 100)

        batch = book.snapshot(3)
        assert abs(batch["bid_price"][0].as_py() - 99.0) < 1e-6
        assert abs(batch["ask_price"][0].as_py() - 101.0) < 1e-6

    def test_side_aliases_accepted(self):
        book = quantflow.OrderBook()
        book.add_limit_order("buy",  99.0, 10)
        book.add_limit_order("sell", 101.0, 10)
        assert book.best_bid() is not None
        assert book.best_ask() is not None

    def test_unknown_side_raises(self):
        book = quantflow.OrderBook()
        with pytest.raises(Exception):
            book.add_limit_order("long", 99.0, 10)


# ── HawkesSimulator ───────────────────────────────────────────────────────────

class TestHawkesSimulator:
    def test_create_default(self):
        sim = quantflow.HawkesSimulator.new()
        assert sim is not None

    def test_create_with_config(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 300.0, "snapshot_interval": 10})
        assert sim is not None

    def test_step_returns_none_before_reset(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 10.0})
        # Not yet reset — pending is empty → step returns None immediately
        result = sim.step()
        assert result is None

    def test_reset_and_step(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 60.0, "snapshot_interval": 1000})
        sim.reset(42)
        event = sim.step()
        assert event is not None
        assert "sim_time" in event
        assert "event_type" in event
        assert "trades" in event
        assert "has_snapshot" in event
        assert 0 <= event["event_type"] <= 11
        assert event["sim_time"] >= 0.0

    def test_trades_produce_dicts_with_correct_keys(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 300.0, "snapshot_interval": 10000})
        sim.reset(42)
        all_trades = []
        while (e := sim.step()) is not None:
            all_trades.extend(e["trades"])
        assert len(all_trades) > 0, "Expected at least one trade in 300 s simulation"
        t = all_trades[0]
        assert "price" in t
        assert "qty" in t
        assert "maker_id" in t
        assert "taker_id" in t

    def test_simulate_day_returns_arrow_batch(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 60.0})
        result = sim.simulate_day(42)
        assert result["n_events"] > 0
        assert result["sim_time"] > 0.0

        batch = result["trades"]
        assert isinstance(batch, pa.RecordBatch)
        assert batch.schema.field("price").type == pa.float64()
        assert batch.schema.field("qty").type == pa.uint64()
        assert batch.num_rows == result["n_trades"]

    def test_simulate_day_different_seeds_differ(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 60.0})
        r1 = sim.simulate_day(42)
        r2 = sim.simulate_day(99)
        # Different seeds should produce different event counts with very high probability
        assert r1["n_events"] != r2["n_events"] or r1["n_trades"] != r2["n_trades"]

    def test_get_book_returns_order_book(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 60.0, "snapshot_interval": 1000})
        sim.reset(42)
        # Advance a few events so the book has content
        for _ in range(20):
            if sim.step() is None:
                break
        book = sim.get_book()
        assert isinstance(book, quantflow.OrderBook)

    def test_mid_price_after_reset(self):
        sim = quantflow.HawkesSimulator.new({"t_max": 60.0})
        sim.reset(42)
        mid = sim.mid_price()
        assert mid is not None
        assert mid > 0.0


# ── AvellanedaStoikov ─────────────────────────────────────────────────────────

class TestAvellanedaStoikov:
    def test_create_with_fixed_sigma(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.1, kappa=1.5, t_end=3600.0, sigma=0.02)
        assert abs(strat.gamma - 0.1) < 1e-9
        assert abs(strat.kappa - 1.5) < 1e-9
        assert abs(strat.sigma - 0.02) < 1e-9
        assert strat.sigma_auto is False

    def test_create_with_auto_sigma(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.05, kappa=0.5, t_end=3600.0)
        assert strat.sigma_auto is True

    def test_compute_quotes_symmetric_at_zero_inventory(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.1, kappa=1.5, t_end=3600.0, sigma=0.01)
        bid, ask = strat.compute_quotes(mid=100.0, inventory=0, t=0.0)
        assert ask > bid
        # Symmetric around mid at zero inventory
        assert abs((bid + ask) / 2 - 100.0) < 1e-6

    def test_long_inventory_shifts_quotes_down(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.1, kappa=1.5, t_end=3600.0, sigma=0.01)
        bid0, ask0 = strat.compute_quotes(100.0, 0, 0.0)
        bid_long, ask_long = strat.compute_quotes(100.0, 20, 0.0)
        assert bid_long < bid0
        assert ask_long < ask0

    def test_spread_narrows_toward_end_of_day(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.1, kappa=1.5, t_end=3600.0, sigma=0.01)
        bid_early, ask_early = strat.compute_quotes(100.0, 0, 0.0)
        bid_late, ask_late = strat.compute_quotes(100.0, 0, 3500.0)
        spread_early = ask_early - bid_early
        spread_late  = ask_late  - bid_late
        assert spread_late < spread_early

    def test_repr_contains_params(self):
        strat = quantflow.AvellanedaStoikov(gamma=0.3, kappa=2.0, t_end=1800.0)
        r = repr(strat)
        assert "0.3" in r
        assert "2.0" in r
        assert "1800" in r
