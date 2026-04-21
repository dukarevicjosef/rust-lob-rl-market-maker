"""Tests for minimum spread floor in paper trading."""
from __future__ import annotations


def _compute_spread_floor(
    mid: float,
    bid: float,
    ask: float,
    maker_fee_bps: float = 2.0,
    min_spread_multiplier: float = 1.5,
) -> tuple[float, float, bool]:
    """
    Replicate the spread floor logic from PaperTradingRunner._refresh_quotes.
    Returns (final_bid, final_ask, floor_triggered).
    """
    rt_fee_rate = 2.0 * maker_fee_bps / 10_000.0
    min_spread = mid * rt_fee_rate * min_spread_multiplier
    agent_spread = ask - bid

    if agent_spread < min_spread:
        half_min = min_spread / 2.0
        return mid - half_min, mid + half_min, True
    return bid, ask, False


class TestSpreadFloor:

    def test_narrow_spread_widened(self):
        """mid=75000, spread=20 < min_spread=45 -> widened."""
        mid = 75_000.0
        bid, ask, triggered = _compute_spread_floor(mid, 74_990.0, 75_010.0)
        # min_spread = 75000 * 0.0004 * 1.5 = 45
        assert triggered is True
        assert ask - bid == 45.0
        assert bid == mid - 22.5
        assert ask == mid + 22.5

    def test_wide_spread_unchanged(self):
        """mid=75000, spread=100 > min_spread=45 -> no change."""
        mid = 75_000.0
        bid, ask, triggered = _compute_spread_floor(mid, 74_950.0, 75_050.0)
        assert triggered is False
        assert bid == 74_950.0
        assert ask == 75_050.0

    def test_multiplier_1x_breakeven(self):
        """min_spread_multiplier=1.0 -> min_spread=30 (exact RT fee cost)."""
        mid = 75_000.0
        # spread=20 < 30
        bid, ask, triggered = _compute_spread_floor(
            mid, 74_990.0, 75_010.0, min_spread_multiplier=1.0,
        )
        # min_spread = 75000 * 0.0004 * 1.0 = 30
        assert triggered is True
        assert ask - bid == 30.0

    def test_multiplier_2x_conservative(self):
        """min_spread_multiplier=2.0 -> min_spread=60."""
        mid = 75_000.0
        # spread=50 < 60
        bid, ask, triggered = _compute_spread_floor(
            mid, 74_975.0, 75_025.0, min_spread_multiplier=2.0,
        )
        assert triggered is True
        assert ask - bid == 60.0

    def test_exact_min_spread_no_trigger(self):
        """Spread exactly at min_spread -> no widening."""
        mid = 75_000.0
        # min_spread = 45, set spread = 45
        bid, ask, triggered = _compute_spread_floor(mid, mid - 22.5, mid + 22.5)
        assert triggered is False
        assert bid == mid - 22.5
        assert ask == mid + 22.5
