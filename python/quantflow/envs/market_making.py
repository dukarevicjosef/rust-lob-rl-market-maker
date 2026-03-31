"""
Gymnasium market-making environment backed by the Rust LOB engine.

The agent controls risk-aversion (γ) and spread adjustment (κ-offset) each
step; the Rust AS formula translates those into bid/ask quotes which are
placed on the live Hawkes-driven LOB.

Observation space (Dict)
------------------------
lob_state           Box(20,)  — 5 bid levels + 5 ask levels, each (Δprice, qty)
                               Δprice = (level_price − mid) / mid  (signed)
                               qty    = qty / max_qty_scale         (clipped [0,1])
volume_imbalance    Box(1,)   — (V_bid − V_ask) / (V_bid + V_ask)
spread              Box(1,)   — bid-ask spread / mid
mid_price_return    Box(1,)   — log(mid_t / mid_{t-1}), clipped
volatility          Box(1,)   — rolling realized vol (quadratic variation)
inventory           Box(1,)   — signed inventory / inventory_limit  → [−1, 1]
pnl                 Box(1,)   — PnL / (initial_mid × inventory_limit)
time_remaining      Box(1,)   — 1 − t / T                           → [0, 1]

Action space  Box(2,)
-----------------
action[0]   γ  ∈ [0.01, 1.0]   risk-aversion parameter passed to AS
action[1]   κ-offset ∈ [−0.5, 0.5]  multiplier: κ = κ_base × (1 + offset)

Reward
------
R = Δpnl  −  φ·|q|  −  ψ·q²  −  λ·max(0, |q|−K)

where q = inventory, K = inventory_limit.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import quantflow


# ── Default config ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    # Simulator
    "t_max":              3600.0,   # seconds per episode
    "snapshot_interval":  200,      # events between book snapshots (for speed)
    # Episode structure
    "episode_length":     10_000,   # max env steps per episode
    "events_per_step":    50,       # Hawkes events to advance per step()
    "warm_up_events":     1_000,    # events before first agent action
    # AS base parameters
    "base_kappa":         1.5,
    "quote_qty":          10,       # lots placed per side
    # Inventory
    "inventory_limit":    50,
    # Observation normalisation
    "lob_levels":         5,        # levels per side in observation
    "max_qty_scale":      500.0,    # qty normalisation denominator
    "vol_window":         30,       # mid-price returns kept for vol estimate
    "pnl_scale":          None,     # set to initial_mid × inventory_limit
    # Reward weights
    "phi":                0.005,    # linear inventory penalty
    "psi":                0.0001,   # quadratic inventory penalty
    "lambda_hard":        0.1,      # hard breach penalty
    "initial_mid":        100.0,    # used for pnl_scale and normalisation
    # Seed
    "seed":               42,
}


# ── Environment ────────────────────────────────────────────────────────────────

class MarketMakingEnv(gym.Env):
    """
    Gymnasium wrapper around the Hawkes-driven LOB simulator.

    The agent adjusts AS parameters every ``events_per_step`` market events.
    Quotes are placed/refreshed after each action.  Fill detection happens
    by matching ``maker_id`` in the trades returned by the simulator's step.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # Episode parameters
        self.episode_length:  int   = int(cfg["episode_length"])
        self.events_per_step: int   = int(cfg["events_per_step"])
        self.warm_up_events:  int   = int(cfg["warm_up_events"])
        self.inventory_limit: int   = int(cfg["inventory_limit"])
        self.quote_qty:       int   = int(cfg["quote_qty"])
        self.lob_levels:      int   = int(cfg["lob_levels"])
        self.base_kappa:      float = float(cfg["base_kappa"])
        self.max_qty_scale:   float = float(cfg["max_qty_scale"])
        self.vol_window:      int   = int(cfg["vol_window"])
        self.initial_mid:     float = float(cfg["initial_mid"])
        self.pnl_scale:       float = float(cfg.get("pnl_scale") or
                                            self.initial_mid * self.inventory_limit)
        self.t_max:           float = float(cfg["t_max"])
        # Reward weights
        self.phi         = float(cfg["phi"])
        self.psi         = float(cfg["psi"])
        self.lambda_hard = float(cfg["lambda_hard"])

        # Spaces
        lob_dim = self.lob_levels * 4   # 5 bid (Δp,q) + 5 ask (Δp,q)
        self.observation_space = spaces.Dict({
            "lob_state":         spaces.Box(-1.0, 1.0, shape=(lob_dim,),  dtype=np.float32),
            "volume_imbalance":  spaces.Box(-1.0, 1.0, shape=(1,),        dtype=np.float32),
            "spread":            spaces.Box( 0.0, 1.0, shape=(1,),        dtype=np.float32),
            "mid_price_return":  spaces.Box(-1.0, 1.0, shape=(1,),        dtype=np.float32),
            "volatility":        spaces.Box( 0.0, 1.0, shape=(1,),        dtype=np.float32),
            "inventory":         spaces.Box(-1.0, 1.0, shape=(1,),        dtype=np.float32),
            "pnl":               spaces.Box(-1.0, 1.0, shape=(1,),        dtype=np.float32),
            "time_remaining":    spaces.Box( 0.0, 1.0, shape=(1,),        dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low  = np.array([0.01, -0.5], dtype=np.float32),
            high = np.array([1.00,  0.5], dtype=np.float32),
        )

        # Build simulator (no reset yet — gymnasium contract requires reset() first)
        self._sim: quantflow.HawkesSimulator = quantflow.HawkesSimulator.new({
            "t_max":             self.t_max,
            "snapshot_interval": int(cfg["snapshot_interval"]),
        })
        self._strat: quantflow.AvellanedaStoikov | None = None

        # State (initialised in reset)
        self._inventory:  int   = 0
        self._cash:       float = 0.0
        self._step_count: int   = 0
        self._sim_time:   float = 0.0
        self._prev_mid:   float = self.initial_mid
        self._prev_pnl:   float = 0.0
        self._bid_id:     int | None = None
        self._ask_id:     int | None = None
        self._mid_returns: deque[float] = deque(maxlen=self.vol_window)
        self._exhausted:  bool = False

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else int(self.cfg["seed"])

        self._sim.reset(rng_seed)

        self._inventory  = 0
        self._cash       = 0.0
        self._step_count = 0
        self._sim_time   = 0.0
        self._prev_pnl   = 0.0
        self._bid_id     = None
        self._ask_id     = None
        self._mid_returns.clear()
        self._exhausted  = False

        # Initial mid for normalisation
        mid = self._sim.mid_price()
        self._prev_mid = mid if mid is not None else self.initial_mid

        # Warm-up: run events without placing quotes so the book settles
        for _ in range(self.warm_up_events):
            event = self._sim.step()
            if event is None:
                break
            self._sim_time = event["sim_time"]
            self._update_mid_returns(event["sim_time"])

        # Build strategy (sigma auto from warm-up data — use base kappa)
        self._strat = quantflow.AvellanedaStoikov(
            gamma=0.1, kappa=self.base_kappa, t_end=self.t_max,
        )

        obs = self._build_obs()
        return obs, {}

    # ── step ───────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        gamma       = float(np.clip(action[0], 0.01, 1.0))
        kappa_off   = float(np.clip(action[1], -0.5, 0.5))
        kappa       = self.base_kappa * (1.0 + kappa_off)

        # Cancel stale quotes
        self._cancel_quotes()

        # Compute and place new quotes
        mid = self._sim.mid_price()
        if mid is None:
            mid = self._prev_mid

        bid_p, ask_p = quantflow.AvellanedaStoikov(
            gamma=gamma, kappa=kappa, t_end=self.t_max, sigma=0.02
        ).compute_quotes(mid=mid, inventory=self._inventory, t=self._sim_time)

        self._place_quotes(bid_p, ask_p)

        # Advance N events
        fill_pnl = 0.0
        for _ in range(self.events_per_step):
            event = self._sim.step()
            if event is None:
                self._exhausted = True
                break
            self._sim_time = event["sim_time"]
            self._update_mid_returns(event["sim_time"])
            fill_pnl += self._process_fills(event["trades"])

        # Mark-to-market PnL
        current_mid = self._sim.mid_price()
        if current_mid is None:
            current_mid = self._prev_mid
        pnl = self._cash + self._inventory * current_mid

        # Reward
        delta_pnl = pnl - self._prev_pnl
        q = self._inventory
        reward = (
            delta_pnl
            - self.phi * abs(q)
            - self.psi * q ** 2
            - self.lambda_hard * max(0.0, abs(q) - self.inventory_limit)
        )

        self._prev_pnl  = pnl
        self._prev_mid  = current_mid
        self._step_count += 1

        terminated = (self._step_count >= self.episode_length) or self._exhausted
        truncated  = False

        obs  = self._build_obs()
        info = {
            "pnl":        pnl,
            "inventory":  self._inventory,
            "sim_time":   self._sim_time,
            "fill_pnl":   fill_pnl,
            "mid":        current_mid,
        }
        return obs, float(reward), terminated, truncated, info

    # ── internals ──────────────────────────────────────────────────────────────

    def _cancel_quotes(self) -> None:
        if self._bid_id is not None:
            self._sim.cancel_agent_order(self._bid_id)
            self._bid_id = None
        if self._ask_id is not None:
            self._sim.cancel_agent_order(self._ask_id)
            self._ask_id = None

    def _place_quotes(self, bid_p: float, ask_p: float) -> None:
        if bid_p <= 0.0 or ask_p <= bid_p:
            return
        inv = self._inventory
        qty = self.quote_qty
        if inv + qty <= self.inventory_limit:
            try:
                self._bid_id = self._sim.place_limit_order("bid", bid_p, qty)
            except Exception:
                self._bid_id = None
        if inv - qty >= -self.inventory_limit:
            try:
                self._ask_id = self._sim.place_limit_order("ask", ask_p, qty)
            except Exception:
                self._ask_id = None

    def _process_fills(self, trades: list[dict]) -> float:
        """Update cash/inventory for any fills against agent's orders."""
        pnl_delta = 0.0
        for t in trades:
            maker_id = t["maker_id"]
            price    = t["price"]
            qty      = t["qty"]
            if self._bid_id is not None and maker_id == self._bid_id:
                self._inventory += qty
                self._cash      -= price * qty
                self._bid_id     = None
                pnl_delta       += qty * price  # gross proceeds tracked in cash
            elif self._ask_id is not None and maker_id == self._ask_id:
                self._inventory -= qty
                self._cash      += price * qty
                self._ask_id     = None
        return pnl_delta

    def _update_mid_returns(self, _sim_time: float) -> None:
        mid = self._sim.mid_price()
        if mid is not None and self._prev_mid > 0.0:
            r = math.log(mid / self._prev_mid)
            self._mid_returns.append(r)

    def _realized_vol(self) -> float:
        if len(self._mid_returns) < 2:
            return 0.0
        arr = np.array(self._mid_returns, dtype=np.float64)
        return float(np.sqrt(np.sum(arr ** 2)))

    def _build_obs(self) -> dict[str, np.ndarray]:
        mid = self._sim.mid_price()
        if mid is None:
            mid = self._prev_mid

        book  = self._sim.get_book()
        rb    = book.snapshot(self.lob_levels)

        bid_prices = rb["bid_price"].to_pylist()
        bid_qtys   = rb["bid_qty"].to_pylist()
        ask_prices = rb["ask_price"].to_pylist()
        ask_qtys   = rb["ask_qty"].to_pylist()

        # LOB state: interleaved (Δprice, norm_qty) per level, bids then asks
        lob_state = np.zeros(self.lob_levels * 4, dtype=np.float32)
        v_bid = v_ask = 0.0
        for i in range(self.lob_levels):
            # Bids
            bp = bid_prices[i] if bid_prices[i] == bid_prices[i] else mid  # NaN → mid
            bq = bid_qtys[i] if bid_qtys[i] else 0
            lob_state[i * 2]     = float(np.clip((bp - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[i * 2 + 1] = float(np.clip(bq / self.max_qty_scale, 0.0, 1.0))
            v_bid += bq
            # Asks
            offset = self.lob_levels * 2
            ap = ask_prices[i] if ask_prices[i] == ask_prices[i] else mid
            aq = ask_qtys[i] if ask_qtys[i] else 0
            lob_state[offset + i * 2]     = float(np.clip((ap - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[offset + i * 2 + 1] = float(np.clip(aq / self.max_qty_scale, 0.0, 1.0))
            v_ask += aq

        vol_imbalance = (v_bid - v_ask) / (v_bid + v_ask + 1e-9)
        spread_raw    = book.spread()
        spread_norm   = float(np.clip((spread_raw or 0.0) / (mid + 1e-9), 0.0, 1.0))

        # Mid-price return (last observation)
        r_last = self._mid_returns[-1] if self._mid_returns else 0.0
        r_clip = float(np.clip(r_last, -0.1, 0.1) / 0.1)  # → [−1, 1]

        vol  = float(np.clip(self._realized_vol() / 0.05, 0.0, 1.0))
        inv  = float(np.clip(self._inventory / (self.inventory_limit + 1e-9), -1.0, 1.0))

        current_pnl = self._cash + self._inventory * mid
        pnl_norm    = float(np.clip(current_pnl / (self.pnl_scale + 1e-9), -1.0, 1.0))
        t_rem       = float(np.clip(1.0 - self._sim_time / (self.t_max + 1e-9), 0.0, 1.0))

        return {
            "lob_state":        lob_state,
            "volume_imbalance": np.array([vol_imbalance], dtype=np.float32),
            "spread":           np.array([spread_norm],   dtype=np.float32),
            "mid_price_return": np.array([r_clip],        dtype=np.float32),
            "volatility":       np.array([vol],           dtype=np.float32),
            "inventory":        np.array([inv],           dtype=np.float32),
            "pnl":              np.array([pnl_norm],      dtype=np.float32),
            "time_remaining":   np.array([t_rem],         dtype=np.float32),
        }


# ── Registration ───────────────────────────────────────────────────────────────

gym.register(
    id="quantflow/MarketMaking-v0",
    entry_point="quantflow.envs.market_making:MarketMakingEnv",
    max_episode_steps=10_000,
)
