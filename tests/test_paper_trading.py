"""
Unit tests for the paper trading stack.

Tests run without a live exchange connection and without the compiled
quantflow extension (they are skipped gracefully if unavailable).
"""
from __future__ import annotations

import os
import math

import pytest
import numpy as np

from quantflow.paper_trading.config import PaperTradingConfig
from quantflow.paper_trading.lob_tracker import LOBTracker
from quantflow.paper_trading.obs_builder import ObservationBuilder


# ── 1. PaperTradingConfig.from_env() ─────────────────────────────────────────

def test_config_from_env_defaults(monkeypatch):
    """from_env() uses testnet=True when BINANCE_TESTNET is unset."""
    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET",
                "BINANCE_SYMBOL", "MODEL_PATH"):
        monkeypatch.delenv(var, raising=False)

    cfg = PaperTradingConfig.from_env()
    assert cfg.testnet is True
    assert cfg.symbol == "BTCUSDT"
    assert cfg.api_key == ""

def test_config_from_env_env_override(monkeypatch):
    """from_env() reads BINANCE_TESTNET=false."""
    monkeypatch.setenv("BINANCE_API_KEY",    "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET",    "false")
    monkeypatch.setenv("BINANCE_SYMBOL",     "ETHUSDT")

    cfg = PaperTradingConfig.from_env()
    assert cfg.testnet is False
    assert cfg.symbol == "ETHUSDT"
    assert cfg.api_key == "key"


# ── 2. ObservationBuilder key structure ───────────────────────────────────────

def test_obs_keys():
    """build_obs() returns all 14 v2 observation keys."""
    cfg = PaperTradingConfig()
    bld = ObservationBuilder(cfg)
    lob = LOBTracker()

    # Seed the LOB with a minimal book
    lob.apply_depth_snapshot({
        "b": [["66000.0", "0.5"], ["65990.0", "1.0"]],
        "a": [["66010.0", "0.5"], ["66020.0", "1.0"]],
    })

    obs = bld.build_obs(lob=lob, position_btc=0.0, cash=0.0, session_time=0.0)

    expected_keys = {
        "lob_state", "volume_imbalance", "spread", "mid_price_return",
        "volatility", "inventory", "pnl", "time_remaining",
        "ofi_short", "ofi_long", "trade_arrival_rate", "vol_short",
        "spread_percentile", "agent_fill_imbalance", "sigma_regime",
    }
    assert set(obs.keys()) == expected_keys

    # sigma_regime is fixed at 0.5 in live mode
    assert abs(float(obs["sigma_regime"][0]) - 0.5) < 1e-6


# ── 3. compute_quotes normalisation at mid=66000 ──────────────────────────────

def test_compute_quotes_at_mid():
    """
    AS quote prices should bracket mid symmetrically for a flat inventory
    and produce a spread > tick_size.
    """
    quantflow = pytest.importorskip("quantflow")

    cfg  = PaperTradingConfig()
    mid  = 66_000.0
    strat = quantflow.AvellanedaStoikov(
        gamma=0.1, kappa=cfg.base_kappa, t_end=cfg.t_max_session, sigma=cfg.sigma_fixed
    )
    bid_p, ask_p = strat.compute_quotes(mid=mid, inventory=0, t=0.0)

    assert bid_p < mid < ask_p, "Quotes must bracket mid at zero inventory"
    spread = ask_p - bid_p
    assert spread > cfg.tick_size, f"Spread {spread:.4f} should exceed tick_size {cfg.tick_size}"
    # Sanity: spread shouldn't be more than 1 % of mid for zero inventory
    assert spread < 0.01 * mid


# ── 4. Integration: LOBTracker + ObservationBuilder round-trip ────────────────

def test_obs_builder_round_trip():
    """
    After several market updates and a synthetic fill, obs shapes and
    types must match the expected observation space bounds.
    """
    cfg = PaperTradingConfig()
    bld = ObservationBuilder(cfg)
    lob = LOBTracker()

    prices = [65_990.0, 66_000.0, 66_010.0, 65_995.0, 66_005.0]
    for p in prices:
        half_spread = 5.0
        lob.apply_depth_snapshot({
            "b": [[str(p - half_spread), "0.3"]],
            "a": [[str(p + half_spread), "0.3"]],
        })
        bld.update_market(lob, wall_time=float(prices.index(p)))
        bld.record_trade(float(prices.index(p)), side_sign=1, qty_btc=0.01)

    bld.record_fill(wall_time=5.0, side="BUY", price=65_995.0)

    obs = bld.build_obs(
        lob          = lob,
        position_btc = 0.001,
        cash         = -65.0,
        session_time = 60.0,
    )

    for key, arr in obs.items():
        assert isinstance(arr, np.ndarray), f"{key} should be ndarray"
        assert arr.dtype == np.float32,     f"{key} dtype should be float32"
        assert np.all(np.isfinite(arr)),    f"{key} contains non-finite values"

    assert obs["lob_state"].shape == (20,)
    assert obs["inventory"].shape  == (1,)
    assert float(obs["time_remaining"][0]) <= 1.0
    assert float(obs["time_remaining"][0]) >= 0.0
    # inventory should be > 0 for long position
    assert float(obs["inventory"][0]) > 0.0
