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

Reward v1 (default)
------
R = Δpnl  −  φ·|q|  −  ψ·q²  −  λ·max(0, |q|−K)

Reward v2
---------
R = Δpnl + rt_bonus − φ_eff·|q| − ψ·q² − λ·max(0,|q|−K) − terminal_penalty

where φ_eff is modulated by trend alignment (asymmetric inventory penalty).
"""
from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import quantflow


# ── Default config ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    # Simulator
    "t_max":              3600.0,
    "snapshot_interval":  200,
    # Episode structure
    "episode_length":     10_000,
    "events_per_step":    50,
    "warm_up_events":     1_000,
    # AS base parameters
    "base_kappa":         1.5,
    "quote_qty":          10,
    # Inventory
    "inventory_limit":    50,
    # Observation normalisation
    "lob_levels":         5,
    "max_qty_scale":      500.0,
    "vol_window":         30,
    "pnl_scale":          None,
    # Reward weights (v1 and shared)
    "phi":                0.005,    # linear inventory penalty
    "psi":                0.0001,   # quadratic inventory penalty
    "lambda_hard":        0.1,      # hard breach penalty (alias: lambda_breach)
    "initial_mid":        100.0,
    # Reward v2 extensions
    "reward_config": {
        "phi":                  0.01,
        "psi":                  0.001,
        "lambda_breach":        1.0,
        "rt_weight":            0.5,
        "asymmetric_strength":  0.3,
        "terminal_weight":      2.0,
        "reward_version":       "v1",
    },
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

    Set ``reward_config["reward_version"] = "v2"`` to enable the extended
    reward with round-trip bonus, asymmetric inventory penalty, and terminal
    position penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        # Merge nested reward_config
        rc = {**_DEFAULT_CONFIG["reward_config"], **cfg.get("reward_config", {})}
        cfg["reward_config"] = rc
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

        # v1 reward weights (top-level config, backward-compatible)
        self.phi         = float(cfg["phi"])
        self.psi         = float(cfg["psi"])
        self.lambda_hard = float(cfg["lambda_hard"])

        # v2 reward config
        self.reward_version:      str   = rc["reward_version"]
        self._rc_phi:             float = float(rc["phi"])
        self._rc_psi:             float = float(rc["psi"])
        self._rc_lambda:          float = float(rc["lambda_breach"])
        self.rt_weight:           float = float(rc["rt_weight"])
        self.asymmetric_strength: float = float(rc["asymmetric_strength"])
        self.terminal_weight:     float = float(rc["terminal_weight"])

        # Spaces
        lob_dim = self.lob_levels * 4
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

        # Simulator
        self._sim: quantflow.HawkesSimulator = quantflow.HawkesSimulator.new({
            "t_max":             self.t_max,
            "snapshot_interval": int(cfg["snapshot_interval"]),
        })
        self._strat: quantflow.AvellanedaStoikov | None = None

        # Episode state (initialised in reset)
        self._inventory:  int   = 0
        self._cash:       float = 0.0
        self._step_count: int   = 0
        self._sim_time:   float = 0.0
        self._prev_mid:   float = self.initial_mid
        self._prev_pnl:   float = 0.0
        self._bid_id:     int | None = None
        self._ask_id:     int | None = None
        self._mid_returns: collections.deque = collections.deque(maxlen=self.vol_window)
        self._exhausted:  bool = False

        # v2 per-step fill tracking (reset each step)
        self._step_buys:      int   = 0
        self._step_sells:     int   = 0
        self._step_buy_value: float = 0.0
        self._step_sell_value: float = 0.0

        # v2 mid-price history for trend estimation
        self._mid_price_history: collections.deque = collections.deque(maxlen=300)

        # v2 state shared between step() and _compute_reward_v2()
        self._current_pnl:    float = 0.0
        self._current_spread: float = 0.0
        self._terminated:     bool  = False

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
        self._mid_price_history.clear()
        self._exhausted  = False
        self._current_pnl    = 0.0
        self._current_spread = 0.0
        self._terminated     = False

        mid = self._sim.mid_price()
        self._prev_mid = mid if mid is not None else self.initial_mid

        for _ in range(self.warm_up_events):
            event = self._sim.step()
            if event is None:
                break
            self._sim_time = event["sim_time"]
            self._update_mid_returns()

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

        # Reset per-step fill accumulators
        self._step_buys       = 0
        self._step_sells      = 0
        self._step_buy_value  = 0.0
        self._step_sell_value = 0.0

        self._cancel_quotes()

        mid = self._sim.mid_price()
        if mid is None:
            mid = self._prev_mid

        bid_p, ask_p = quantflow.AvellanedaStoikov(
            gamma=gamma, kappa=kappa, t_end=self.t_max, sigma=0.02
        ).compute_quotes(mid=mid, inventory=self._inventory, t=self._sim_time)

        self._place_quotes(bid_p, ask_p)

        fill_pnl = 0.0
        for _ in range(self.events_per_step):
            event = self._sim.step()
            if event is None:
                self._exhausted = True
                break
            self._sim_time = event["sim_time"]
            self._update_mid_returns()
            fill_pnl += self._process_fills(event["trades"])

        current_mid = self._sim.mid_price()
        if current_mid is None:
            current_mid = self._prev_mid

        # Update spread for terminal penalty
        book = self._sim.get_book()
        self._current_spread = book.spread() or 0.01

        # Mid-price history for trend (v2)
        self._mid_price_history.append(current_mid)

        pnl = self._cash + self._inventory * current_mid
        self._current_pnl = pnl

        self._step_count += 1
        terminated = (self._step_count >= self.episode_length) or self._exhausted
        self._terminated = terminated

        # Dispatch reward
        if self.reward_version == "v2":
            reward, components = self._compute_reward_v2()
        else:
            reward, components = self._compute_reward_v1(pnl)

        self._prev_pnl  = pnl
        self._prev_mid  = current_mid

        obs  = self._build_obs()
        info: dict[str, Any] = {
            "pnl":        pnl,
            "inventory":  self._inventory,
            "sim_time":   self._sim_time,
            "fill_pnl":   fill_pnl,
            "mid":        current_mid,
            "reward_components": components,
        }
        return obs, float(reward), terminated, False, info

    # ── reward functions ────────────────────────────────────────────────────────

    def _compute_reward_v1(self, pnl: float) -> tuple[float, dict]:
        """
        Original reward: PnL change minus linear and quadratic inventory
        penalties and a hard-breach term.
        """
        delta_pnl     = pnl - self._prev_pnl
        q             = self._inventory
        inv_penalty   = self.phi * abs(q)
        risk_penalty  = self.psi * q ** 2
        breach_penalty = self.lambda_hard * max(0.0, abs(q) - self.inventory_limit)

        reward = delta_pnl - inv_penalty - risk_penalty - breach_penalty
        components = {
            "delta_pnl":       delta_pnl,
            "rt_bonus":        0.0,
            "inv_penalty":     inv_penalty,
            "risk_penalty":    risk_penalty,
            "breach_penalty":  breach_penalty,
            "terminal_penalty": 0.0,
            "trend_alignment": 0.0,
            "round_trips":     0,
        }
        return reward, components

    def _compute_reward_v2(self) -> tuple[float, dict]:
        """
        Extended reward with:
        - Round-trip spread-capture bonus
        - Asymmetric inventory penalty modulated by short-term price trend
        - Terminal open-position penalty
        """
        delta_pnl = self._current_pnl - self._prev_pnl

        # ── Round-trip bonus ────────────────────────────────────────────────
        round_trips = min(self._step_buys, self._step_sells)
        rt_bonus = 0.0
        if round_trips > 0 and self._step_buys > 0 and self._step_sells > 0:
            avg_buy  = self._step_buy_value  / self._step_buys
            avg_sell = self._step_sell_value / self._step_sells
            spread_captured = avg_sell - avg_buy
            rt_bonus = round_trips * max(0.0, spread_captured) * self.rt_weight

        # ── Trend-aligned inventory penalty ────────────────────────────────
        # Short-term trend from last 50 mid-price observations.
        trend = 0.0
        if len(self._mid_price_history) >= 50:
            recent = self._mid_price_history[-1]
            past   = self._mid_price_history[-50]
            if past > 0.0:
                trend = (recent - past) / past

        inv = self._inventory
        if inv != 0 and abs(trend) > 1e-6:
            # +1: inventory and trend in same direction (riding momentum — less risky)
            # −1: inventory and trend opposing (adverse selection — more risky)
            aligned = (inv > 0 and trend > 0) or (inv < 0 and trend < 0)
            trend_alignment = 1.0 if aligned else -1.0
        else:
            trend_alignment = 0.0

        # trend aligned → factor < 1 (smaller penalty); opposing → factor > 1
        asymmetric_factor = 1.0 - self.asymmetric_strength * trend_alignment

        inv_penalty    = self._rc_phi * abs(inv) * asymmetric_factor
        risk_penalty   = self._rc_psi * inv ** 2
        breach_penalty = self._rc_lambda * max(0.0, abs(inv) - self.inventory_limit)

        # ── Terminal open-position penalty ──────────────────────────────────
        terminal_penalty = 0.0
        if self._terminated:
            spread = self._current_spread if self._current_spread > 0.0 else 0.01
            terminal_penalty = self.terminal_weight * abs(inv) * spread

        reward = (
            delta_pnl
            + rt_bonus
            - inv_penalty
            - risk_penalty
            - breach_penalty
            - terminal_penalty
        )
        components = {
            "delta_pnl":        delta_pnl,
            "rt_bonus":         rt_bonus,
            "inv_penalty":      inv_penalty,
            "risk_penalty":     risk_penalty,
            "breach_penalty":   breach_penalty,
            "terminal_penalty": terminal_penalty,
            "trend_alignment":  trend_alignment,
            "round_trips":      round_trips,
        }
        return reward, components

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
        pnl_delta = 0.0
        for t in trades:
            maker_id = t["maker_id"]
            price    = t["price"]
            qty      = t["qty"]
            if self._bid_id is not None and maker_id == self._bid_id:
                self._inventory      += qty
                self._cash           -= price * qty
                self._bid_id          = None
                pnl_delta            += qty * price
                # v2 tracking
                self._step_buys      += qty
                self._step_buy_value += price * qty
            elif self._ask_id is not None and maker_id == self._ask_id:
                self._inventory       -= qty
                self._cash            += price * qty
                self._ask_id           = None
                # v2 tracking
                self._step_sells      += qty
                self._step_sell_value += price * qty
        return pnl_delta

    def _update_mid_returns(self) -> None:
        mid = self._sim.mid_price()
        if mid is not None and mid > 0.0 and self._prev_mid > 0.0:
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

        lob_state = np.zeros(self.lob_levels * 4, dtype=np.float32)
        v_bid = v_ask = 0.0
        for i in range(self.lob_levels):
            bp = bid_prices[i] if bid_prices[i] == bid_prices[i] else mid
            bq = bid_qtys[i] if bid_qtys[i] else 0
            lob_state[i * 2]     = float(np.clip((bp - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[i * 2 + 1] = float(np.clip(bq / self.max_qty_scale, 0.0, 1.0))
            v_bid += bq
            offset = self.lob_levels * 2
            ap = ask_prices[i] if ask_prices[i] == ask_prices[i] else mid
            aq = ask_qtys[i] if ask_qtys[i] else 0
            lob_state[offset + i * 2]     = float(np.clip((ap - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[offset + i * 2 + 1] = float(np.clip(aq / self.max_qty_scale, 0.0, 1.0))
            v_ask += aq

        vol_imbalance = (v_bid - v_ask) / (v_bid + v_ask + 1e-9)
        spread_raw    = book.spread()
        spread_norm   = float(np.clip((spread_raw or 0.0) / (mid + 1e-9), 0.0, 1.0))

        r_last = self._mid_returns[-1] if self._mid_returns else 0.0
        r_clip = float(np.clip(r_last, -0.1, 0.1) / 0.1)

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
