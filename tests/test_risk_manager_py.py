"""Python-layer tests for the RiskManager PyO3 binding."""
from __future__ import annotations

import pytest

quantflow = pytest.importorskip("quantflow")
RiskManager = quantflow.RiskManager

_EXPECTED_STATUS_KEYS = {
    "is_killed",
    "position",
    "daily_pnl",
    "peak_pnl",
    "drawdown",
    "open_orders",
    "violations_today",
    "position_utilization",
    "loss_utilization",
}


# ── 1. check_order raises ValueError on violation ─────────────────────────────

def test_check_order_raises_on_violation():
    rm = RiskManager({"max_order_size": 0.05})
    # 0.01 is fine
    assert rm.check_order("buy", 0.01, 50_000.0) is True
    # 0.10 > 0.05 → should raise
    with pytest.raises(ValueError, match="Order size"):
        rm.check_order("buy", 0.10, 50_000.0)


# ── 2. status() returns dict with all required keys ───────────────────────────

def test_status_returns_complete_dict():
    rm = RiskManager()
    st = rm.status()
    assert isinstance(st, dict), "status() must return a dict"
    missing = _EXPECTED_STATUS_KEYS - set(st.keys())
    assert not missing, f"status() dict missing keys: {missing}"
    # Spot-check types
    assert isinstance(st["is_killed"], bool)
    assert isinstance(st["position"], float)
    assert isinstance(st["open_orders"], int)


# ── 3. Round-trip: place → fill → cancel → check_order ───────────────────────

def test_round_trip_place_fill_cancel():
    rm = RiskManager({"max_position": 0.1, "max_open_orders": 2})

    # 1. Place two orders — increments open_orders
    rm.on_order_placed()
    rm.on_order_placed()
    assert rm.status()["open_orders"] == 2

    # 2. Fill the first bid — position increases, open_orders decrements
    rm.on_fill("buy", 0.03, 67_000.0)
    assert abs(rm.position - 0.03) < 1e-9
    assert rm.status()["open_orders"] == 1

    # 3. Cancel the second order — open_orders back to 0
    rm.on_order_cancelled()
    assert rm.status()["open_orders"] == 0

    # 4. Now a new buy should pass (position 0.03, adding 0.05 = 0.08 < 0.1)
    rm.on_order_placed()   # simulate placing before check (order accepted)
    assert rm.check_order("buy", 0.05, 67_000.0) is True

    # 5. PnL update: loss of 60 → below max_daily_loss=100, should NOT kill
    rm.update_pnl(-60.0, 0.0)
    assert not rm.is_killed

    # 6. Loss of 110 → kill switch fires
    rm.update_pnl(-110.0, 0.0)
    assert rm.is_killed

    with pytest.raises(ValueError, match="KILL SWITCH"):
        rm.check_order("sell", 0.03, 67_000.0)

    # 7. Reset and verify orders are accepted again
    rm.reset_kill_switch()
    rm.reset_daily()
    assert rm.check_order("sell", 0.03, 67_000.0) is True
