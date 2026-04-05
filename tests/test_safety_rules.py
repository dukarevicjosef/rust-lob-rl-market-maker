"""Unit tests for the safety-rule layer (quantflow/envs/safety_rules.py)."""
from __future__ import annotations

import pytest
from quantflow.envs.safety_rules import apply_safety_rules, RulesTriggered

# ── helpers ────────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    inventory_soft_limit  = 30,
    inventory_hard_limit  = 40,
    tick_size             = 0.01,
    vol_spread_threshold  = 2.0,
    vol_spread_multiplier = 2.0,
)

MID       = 100.0
TICK      = _DEFAULTS["tick_size"]
BID_QUOTE = MID - 0.05    # normal agent quote: 5 ticks below mid
ASK_QUOTE = MID + 0.05    # normal agent quote: 5 ticks above mid


def _call(inventory: int, **overrides):
    kwargs = {
        "bid_p":           BID_QUOTE,
        "ask_p":           ASK_QUOTE,
        "mid":             MID,
        "inventory":       inventory,
        "current_vol":     0.001,
        "vol_ema":         0.0,      # disabled unless overridden
        "last_quote_mid":  None,
    }
    kwargs.update(overrides)
    return apply_safety_rules(**kwargs, **_DEFAULTS)


# ── Inventory soft limit ───────────────────────────────────────────────────────

def test_inventory_soft_long_suppresses_bid():
    """inv=+35 (≥ soft_limit=30, < hard_limit=40): only ask placed."""
    bid, ask, rules = _call(inventory=35)
    assert bid is None
    assert ask == pytest.approx(ASK_QUOTE)
    assert rules.inventory_soft is True
    assert rules.inventory_hard is False


def test_inventory_soft_short_suppresses_ask():
    """inv=-35: only bid placed."""
    bid, ask, rules = _call(inventory=-35)
    assert bid == pytest.approx(BID_QUOTE)
    assert ask is None
    assert rules.inventory_soft is True
    assert rules.inventory_hard is False


# ── Inventory hard limit ───────────────────────────────────────────────────────

def test_inventory_hard_long_forces_aggressive_ask():
    """inv=+45 (≥ hard_limit=40): aggressive ask 1 tick below mid."""
    bid, ask, rules = _call(inventory=45)
    assert bid is None
    assert ask == pytest.approx(MID - TICK)
    assert rules.inventory_hard is True


def test_inventory_hard_short_forces_aggressive_bid():
    """inv=-45: aggressive bid 1 tick above mid."""
    bid, ask, rules = _call(inventory=-45)
    assert bid == pytest.approx(MID + TICK)
    assert ask is None
    assert rules.inventory_hard is True


# ── Quote pull ────────────────────────────────────────────────────────────────

def test_quote_pull_fires_on_large_mid_move():
    """Mid moved 3 ticks since last quote: both sides pulled."""
    last_mid = MID - 3 * TICK          # 3 ticks away from current MID
    bid, ask, rules = _call(inventory=0, last_quote_mid=last_mid)
    assert bid is None
    assert ask is None
    assert rules.quote_pull is True


def test_quote_pull_does_not_fire_on_small_mid_move():
    """Mid moved only 0.5 ticks: quotes pass through unchanged."""
    last_mid = MID - 0.5 * TICK
    bid, ask, rules = _call(inventory=0, last_quote_mid=last_mid)
    assert bid == pytest.approx(BID_QUOTE)
    assert ask == pytest.approx(ASK_QUOTE)
    assert rules.quote_pull is False


# ── Vol regime ────────────────────────────────────────────────────────────────

def test_vol_regime_widens_spread():
    """current_vol = 3× EMA: half-spread should double."""
    ema = 0.001
    vol = 3.0 * ema          # well above threshold of 2×
    bid, ask, rules = _call(inventory=0, current_vol=vol, vol_ema=ema)
    assert rules.vol_regime is True
    # Original half-spread = 0.05; multiplier=2 → new half-spread = 0.10
    assert ask - MID == pytest.approx(0.10, abs=1e-9)
    assert MID - bid == pytest.approx(0.10, abs=1e-9)


def test_vol_regime_does_not_fire_at_normal_vol():
    """current_vol = 1× EMA (below 2× threshold): quotes unchanged."""
    ema = 0.001
    vol = ema                # same as EMA — no regime
    bid, ask, rules = _call(inventory=0, current_vol=vol, vol_ema=ema)
    assert rules.vol_regime is False
    assert bid == pytest.approx(BID_QUOTE)
    assert ask == pytest.approx(ASK_QUOTE)


def test_vol_regime_disabled_when_ema_zero():
    """EMA=0.0 (first steps): vol-regime rule must not fire regardless of current_vol."""
    bid, ask, rules = _call(inventory=0, current_vol=1.0, vol_ema=0.0)
    assert rules.vol_regime is False


# ── Combined: vol regime + inventory soft ─────────────────────────────────────

def test_combined_high_vol_and_inventory_soft():
    """High vol widens spread AND inventory at soft limit suppresses bid side."""
    ema = 0.001
    vol = 3.0 * ema
    bid, ask, rules = _call(inventory=35, current_vol=vol, vol_ema=ema)
    # vol regime fires first → widens spread
    assert rules.vol_regime is True
    # then inventory soft fires → bid suppressed
    assert rules.inventory_soft is True
    assert bid is None
    # ask should be widened
    assert ask is not None
    assert ask > MID
    assert ask - MID == pytest.approx(0.10, abs=1e-9)
