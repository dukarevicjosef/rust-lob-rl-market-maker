"""
Observation builder for live paper trading.

Mirrors ``MarketMakingEnv._build_obs()`` exactly so that the SAC agent
receives observations in the same normalised space it was trained on.

Key mapping differences between simulation and live:
- Quantities: sim uses integer lots; live uses BTC. Normalise with qty_scale_btc.
- Inventory: map position_btc → fractional lots via training_inventory_limit.
- PnL: divide by pnl_scale (≈ mid × max_position_btc).
- Time: wall-clock seconds since session start / t_max_session.
- sigma_regime: fixed 0.5 (live regime is unknown; use midpoint of training range).
"""
from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np

from quantflow.obs_features import (
    compute_agent_fill_imbalance,
    compute_order_flow_imbalance,
    compute_realized_vol,
    compute_spread_percentile,
    compute_trade_arrival_rate,
)
from quantflow.paper_trading.config import PaperTradingConfig
from quantflow.paper_trading.lob_tracker import LOBTracker


class ObservationBuilder:
    """
    Stateful rolling-buffer manager that produces agent observations.

    Call ``update_market()`` every time a depth or trade event arrives,
    ``record_fill()`` on every agent order fill, then ``build_obs()`` when
    the quoting loop needs a fresh observation.
    """

    def __init__(self, cfg: PaperTradingConfig) -> None:
        self._cfg     = cfg
        self._levels  = 5
        self._vol_window = 30

        # Mid-return history for realized vol (matches _realized_vol)
        self._mid_returns: collections.deque = collections.deque(maxlen=self._vol_window)
        self._prev_mid:    float | None = None

        # v2 rolling buffers
        self._trade_times:   collections.deque = collections.deque(maxlen=1000)
        self._trade_sides:   collections.deque = collections.deque(maxlen=1000)
        self._trade_volumes: collections.deque = collections.deque(maxlen=1000)

        self._mid_prices_short: collections.deque = collections.deque(maxlen=100)
        self._mid_prices_long:  collections.deque = collections.deque(maxlen=500)
        self._spread_history:   collections.deque = collections.deque(maxlen=200)

        self._agent_fill_history: collections.deque = collections.deque(maxlen=20)

    # ── ingestion ─────────────────────────────────────────────────────────────

    def update_market(
        self,
        lob: LOBTracker,
        wall_time: float,
    ) -> None:
        """
        Called after each depth update.  Updates mid-return buffer.
        """
        mid = lob.mid()
        if mid is None:
            return
        if self._prev_mid is not None and self._prev_mid > 0.0:
            r = math.log(mid / self._prev_mid)
            self._mid_returns.append(r)
        self._mid_prices_short.append(mid)
        self._mid_prices_long.append(mid)
        self._prev_mid = mid
        spread = lob.spread()
        if spread is not None:
            self._spread_history.append(spread)

    def record_trade(
        self,
        wall_time: float,
        side_sign: int,   # +1 = buy aggressor, -1 = sell aggressor
        qty_btc:   float,
    ) -> None:
        """Called on each aggTrade message to populate OFI buffers."""
        self._trade_times.append(wall_time)
        self._trade_sides.append(side_sign)
        # Normalise qty to comparable scale for OFI computation.
        # We store fractional lots so the OFI formula stays scale-invariant.
        self._trade_volumes.append(qty_btc / self._cfg.qty_scale_btc)

    def record_fill(
        self,
        wall_time: float,
        side:      str,   # "BUY" or "SELL"
        price:     float,
    ) -> None:
        """Called when one of the agent's resting orders is filled."""
        side_sign = 1 if side.upper() == "BUY" else -1
        self._agent_fill_history.append((wall_time, side_sign, price))

    def reset(self) -> None:
        self._mid_returns.clear()
        self._prev_mid = None
        self._trade_times.clear()
        self._trade_sides.clear()
        self._trade_volumes.clear()
        self._mid_prices_short.clear()
        self._mid_prices_long.clear()
        self._spread_history.clear()
        self._agent_fill_history.clear()

    # ── observation construction ───────────────────────────────────────────────

    def build_obs(
        self,
        lob:          LOBTracker,
        position_btc: float,
        cash:         float,
        session_time: float,   # seconds since session start
    ) -> dict[str, np.ndarray]:
        """
        Build the agent observation dict.

        Parameters
        ----------
        lob
            Current LOB snapshot.
        position_btc
            Signed net position in BTC (positive = long).
        cash
            Cumulative realised cash PnL in USDT.
        session_time
            Elapsed session time in seconds.

        Returns
        -------
        Dict matching MarketMakingEnv.observation_space (v2 14 keys).
        """
        cfg = self._cfg
        mid = lob.mid()
        if mid is None:
            mid = self._prev_mid or 0.0

        snap   = lob.snapshot(self._levels)
        bp_lst = snap["bid_price"]
        bq_lst = snap["bid_qty"]
        ap_lst = snap["ask_price"]
        aq_lst = snap["ask_qty"]

        lob_state = np.zeros(self._levels * 4, dtype=np.float32)
        v_bid = v_ask = 0.0

        for i in range(self._levels):
            # bid level
            bp = bp_lst[i] if bp_lst[i] is not None else mid
            bq = bq_lst[i] if bq_lst[i] is not None else 0.0
            lob_state[i * 2]     = float(np.clip((bp - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[i * 2 + 1] = float(np.clip(bq / cfg.qty_scale_btc, 0.0, 1.0))
            v_bid += bq

            # ask level
            offset = self._levels * 2
            ap = ap_lst[i] if ap_lst[i] is not None else mid
            aq = aq_lst[i] if aq_lst[i] is not None else 0.0
            lob_state[offset + i * 2]     = float(np.clip((ap - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[offset + i * 2 + 1] = float(np.clip(aq / cfg.qty_scale_btc, 0.0, 1.0))
            v_ask += aq

        vol_imbalance = (v_bid - v_ask) / (v_bid + v_ask + 1e-9)

        spread_raw  = lob.spread() or 0.0
        spread_norm = float(np.clip(spread_raw / (mid + 1e-9), 0.0, 1.0))

        r_last = self._mid_returns[-1] if self._mid_returns else 0.0
        r_clip = float(np.clip(r_last, -0.1, 0.1) / 0.1)

        vol_rv = self._realized_vol()
        vol    = float(np.clip(vol_rv / 0.05, 0.0, 1.0))

        # Inventory: map position_btc to the same normalised range as training.
        # Training used integer lots in [-inventory_limit, +inventory_limit].
        # Live: position_btc / max_position_btc gives a fraction in [-1, 1],
        # which equals lots / training_inventory_limit directly.
        inv_norm = float(np.clip(
            position_btc / (cfg.max_position_btc + 1e-9),
            -1.0, 1.0,
        ))

        # PnL: mark-to-market value / pnl_scale
        mtm_pnl  = cash + position_btc * mid
        pnl_norm = float(np.clip(mtm_pnl / (cfg.pnl_scale + 1e-9), -1.0, 1.0))

        t_rem = float(np.clip(
            1.0 - session_time / (cfg.t_max_session + 1e-9),
            0.0, 1.0,
        ))

        obs: dict[str, Any] = {
            "lob_state":        lob_state,
            "volume_imbalance": np.array([vol_imbalance], dtype=np.float32),
            "spread":           np.array([spread_norm],   dtype=np.float32),
            "mid_price_return": np.array([r_clip],        dtype=np.float32),
            "volatility":       np.array([vol],           dtype=np.float32),
            "inventory":        np.array([inv_norm],      dtype=np.float32),
            "pnl":              np.array([pnl_norm],      dtype=np.float32),
            "time_remaining":   np.array([t_rem],         dtype=np.float32),
        }

        # v2 regime features
        obs["ofi_short"] = np.array([
            compute_order_flow_imbalance(self._trade_sides, self._trade_volumes, window=50)
        ], dtype=np.float32)
        obs["ofi_long"] = np.array([
            compute_order_flow_imbalance(self._trade_sides, self._trade_volumes, window=300)
        ], dtype=np.float32)
        obs["trade_arrival_rate"] = np.array([
            compute_trade_arrival_rate(self._trade_times, session_time, window_sec=10.0)
        ], dtype=np.float32)
        obs["vol_short"] = np.array([
            compute_realized_vol(self._mid_prices_short, window=50)
        ], dtype=np.float32)
        obs["spread_percentile"] = np.array([
            compute_spread_percentile(spread_raw, self._spread_history)
        ], dtype=np.float32)
        obs["agent_fill_imbalance"] = np.array([
            compute_agent_fill_imbalance(self._agent_fill_history)
        ], dtype=np.float32)
        # sigma_regime: fixed midpoint of training range [0.7, 1.5] → 0.5
        obs["sigma_regime"] = np.array([0.5], dtype=np.float32)

        return obs

    # ── internal helpers ───────────────────────────────────────────────────────

    def _realized_vol(self) -> float:
        """Quadratic variation of log mid-returns — matches MarketMakingEnv._realized_vol."""
        if len(self._mid_returns) < 2:
            return 0.0
        arr = np.array(self._mid_returns, dtype=np.float64)
        return float(np.sqrt(np.sum(arr ** 2)))

    # ── AS inventory scaling ───────────────────────────────────────────────────

    def inventory_for_as(self, position_btc: float) -> int:
        """
        Convert live BTC position to training-space lot inventory for the
        Avellaneda-Stoikov formula.

        Training used inventory_limit=50 lots; live max_position=max_position_btc.
        The linear mapping preserves the agent's risk-aversion semantics.
        """
        cfg = self._cfg
        frac = position_btc / (cfg.max_position_btc + 1e-9)
        return int(round(frac * cfg.training_inventory_limit))
