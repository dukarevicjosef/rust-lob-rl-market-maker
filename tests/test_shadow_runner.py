"""Unit tests for the shadow mode fill simulation logic."""
from __future__ import annotations

import pytest

from quantflow.paper_trading.shadow_runner import ShadowConfig, ShadowQuote


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(
    bid_price: float | None = None,
    bid_qty: float = 0.001,
    ask_price: float | None = None,
    ask_qty: float = 0.001,
    position: float = 0.0,
    cash: float = 0.0,
    maker_fee_bps: float = 0.0,
    max_position: float = 0.01,
) -> dict:
    return {
        "active_bid": ShadowQuote(bid_price, bid_qty) if bid_price else None,
        "active_ask": ShadowQuote(ask_price, ask_qty) if ask_price else None,
        "position": position,
        "cash": cash,
        "fees": 0.0,
        "fills": 0,
        "fee_rate": maker_fee_bps / 10_000.0,
        "max_position": max_position,
    }


def _check_fill(state: dict, trade_price: float, trade_qty: float = 0.001) -> dict:
    """
    Replicate ShadowRunner._check_fill logic for unit testing.
    Returns updated state.
    """
    fee_rate = state["fee_rate"]

    # Bid fill
    bid = state["active_bid"]
    if bid is not None and trade_price <= bid.price:
        fill_qty = min(trade_qty, bid.qty)
        fill_price = bid.price
        fee = fill_price * fill_qty * fee_rate
        if abs(state["position"] + fill_qty) <= state["max_position"]:
            state["position"] += fill_qty
            state["cash"] -= fill_price * fill_qty + fee
            state["fees"] += fee
            state["fills"] += 1
            state["active_bid"] = None

    # Ask fill
    ask = state["active_ask"]
    if ask is not None and trade_price >= ask.price:
        fill_qty = min(trade_qty, ask.qty)
        fill_price = ask.price
        fee = fill_price * fill_qty * fee_rate
        if abs(state["position"] - fill_qty) <= state["max_position"]:
            state["position"] -= fill_qty
            state["cash"] += fill_price * fill_qty - fee
            state["fees"] += fee
            state["fills"] += 1
            state["active_ask"] = None

    return state


# ── Fill simulation tests ────────────────────────────────────────────────────

class TestShadowFills:

    def test_trade_at_bid_triggers_buy(self):
        """Trade at bid price -> shadow BUY fill."""
        s = _make_state(bid_price=75500.0)
        s = _check_fill(s, trade_price=75500.0)
        assert s["fills"] == 1
        assert s["position"] == 0.001
        assert s["active_bid"] is None

    def test_trade_below_bid_triggers_buy(self):
        """Trade below bid price -> shadow BUY fill."""
        s = _make_state(bid_price=75500.0)
        s = _check_fill(s, trade_price=75490.0)
        assert s["fills"] == 1
        assert s["position"] == 0.001

    def test_trade_at_ask_triggers_sell(self):
        """Trade at ask price -> shadow SELL fill."""
        s = _make_state(ask_price=75550.0)
        s = _check_fill(s, trade_price=75550.0)
        assert s["fills"] == 1
        assert s["position"] == -0.001
        assert s["active_ask"] is None

    def test_trade_above_ask_triggers_sell(self):
        """Trade above ask price -> shadow SELL fill."""
        s = _make_state(ask_price=75550.0)
        s = _check_fill(s, trade_price=75560.0)
        assert s["fills"] == 1
        assert s["position"] == -0.001

    def test_trade_between_bid_ask_no_fill(self):
        """Trade between bid and ask -> no fill."""
        s = _make_state(bid_price=75500.0, ask_price=75550.0)
        s = _check_fill(s, trade_price=75525.0)
        assert s["fills"] == 0
        assert s["position"] == 0.0
        assert s["active_bid"] is not None
        assert s["active_ask"] is not None


class TestShadowPnL:

    def test_round_trip_pnl(self):
        """Buy at bid, sell at ask -> positive PnL."""
        s = _make_state(bid_price=75500.0, ask_price=75550.0)
        # Buy fill
        s = _check_fill(s, trade_price=75500.0)
        assert s["position"] == pytest.approx(0.001)
        # New ask quote
        s["active_ask"] = ShadowQuote(75550.0, 0.001)
        # Sell fill
        s = _check_fill(s, trade_price=75550.0)
        assert s["position"] == pytest.approx(0.0)
        # PnL = sell - buy = 75.55 - 75.50 = 0.05
        assert s["cash"] == pytest.approx(0.05)

    def test_fee_with_zero_bps(self):
        """0 bps fee -> fees stay at 0."""
        s = _make_state(bid_price=75500.0, maker_fee_bps=0.0)
        s = _check_fill(s, trade_price=75500.0)
        assert s["fees"] == 0.0

    def test_fee_with_2_bps(self):
        """2 bps fee -> correct fee calculation."""
        s = _make_state(bid_price=75500.0, maker_fee_bps=2.0)
        s = _check_fill(s, trade_price=75500.0)
        # fee = 75500 * 0.001 * 0.0002 = 0.01510
        expected_fee = 75500.0 * 0.001 * (2.0 / 10_000.0)
        assert s["fees"] == pytest.approx(expected_fee)
        assert s["fills"] == 1


class TestShadowLimits:

    def test_position_limit_prevents_fill(self):
        """Fill that would exceed max_position is rejected."""
        s = _make_state(
            bid_price=75500.0, bid_qty=0.01,
            position=0.005, max_position=0.01,
        )
        # Filling 0.01 would bring position to 0.015 > 0.01
        s = _check_fill(s, trade_price=75500.0, trade_qty=0.01)
        assert s["fills"] == 0
        assert s["position"] == 0.005

    def test_no_quote_no_fill(self):
        """No active quote -> no fill regardless of trade price."""
        s = _make_state()  # no bid or ask
        s = _check_fill(s, trade_price=75500.0)
        assert s["fills"] == 0
