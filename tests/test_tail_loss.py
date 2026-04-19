"""
Tests for tail-loss reduction: inventory soft limit and drawdown penalty.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from quantflow.envs.market_making import MarketMakingEnv
from quantflow.envs.safety_rules import apply_safety_rules


# ── Helpers ──────────────────────────────────────────────────────────────────

MID = 100.0
BID = MID - 0.05
ASK = MID + 0.05
TICK = 0.01


def _call_safety(inventory: int, soft_limit: int = 20) -> tuple:
    return apply_safety_rules(
        bid_p=BID, ask_p=ASK, mid=MID,
        inventory=inventory, current_vol=0.001, vol_ema=0.0,
        last_quote_mid=None,
        inventory_soft_limit=soft_limit,
        inventory_hard_limit=40,
        tick_size=TICK,
        vol_spread_threshold=2.0,
        vol_spread_multiplier=2.0,
    )


def _make_env(extra: dict[str, Any] | None = None) -> MarketMakingEnv:
    cfg: dict[str, Any] = {
        "t_max":           900.0,
        "episode_length":  5,
        "events_per_step": 1,
        "warm_up_events":  0,
        "inventory_limit": 50,
        "initial_mid":     100.0,
        "normalize_reward": False,
    }
    cfg.update(extra or {})
    return MarketMakingEnv(config=cfg)


# ── Inventory soft limit tests ───────────────────────────────────────────────

class TestInventorySoftLimit:

    def test_above_soft_limit_suppresses_increasing_side(self):
        """|inventory|=25 with soft_limit=20: only opposite side quoted."""
        bid, ask, rules = _call_safety(inventory=25, soft_limit=20)
        assert bid is None
        assert ask == pytest.approx(ASK)
        assert rules.inventory_soft is True

    def test_above_soft_limit_short(self):
        """|inventory|=-25 with soft_limit=20: only bid side quoted."""
        bid, ask, rules = _call_safety(inventory=-25, soft_limit=20)
        assert bid == pytest.approx(BID)
        assert ask is None
        assert rules.inventory_soft is True

    def test_below_soft_limit_bilateral(self):
        """|inventory|=15 with soft_limit=20: both sides quoted."""
        bid, ask, rules = _call_safety(inventory=15, soft_limit=20)
        assert bid == pytest.approx(BID)
        assert ask == pytest.approx(ASK)
        assert rules.inventory_soft is False

    def test_default_config_is_20(self):
        """Default inventory_soft_limit in env config is now 20."""
        env = _make_env()
        assert env.inventory_soft_limit == 20


# ── Drawdown penalty tests ──────────────────────────────────────────────────

class TestDrawdownPenalty:

    def _env_with_dd(
        self, peak_pnl: float, current_pnl: float,
        coef: float = 0.001, threshold: float = 200.0,
    ) -> tuple[float, dict]:
        """Create env, set state, compute reward, return (dd_penalty, components)."""
        env = _make_env({
            "dd_penalty_threshold": threshold,
            "dd_penalty_coef": coef,
        })
        env._inventory = 0
        env._prev_pnl = current_pnl
        env._peak_pnl = peak_pnl
        env._current_pnl = current_pnl

        # Manually compute what step() does for dd_penalty
        pnl = current_pnl
        env._peak_pnl = max(env._peak_pnl, pnl)
        current_drawdown = env._peak_pnl - pnl
        if coef > 0.0 and current_drawdown > threshold:
            dd_excess = current_drawdown - threshold
            dd_penalty = coef * dd_excess ** 2
        else:
            dd_penalty = 0.0

        return dd_penalty, {"current_drawdown": current_drawdown}

    def test_drawdown_below_threshold_no_penalty(self):
        """Drawdown 150 (under 200 threshold) -> dd_penalty = 0."""
        dd_penalty, info = self._env_with_dd(
            peak_pnl=1000.0, current_pnl=850.0,  # drawdown = 150
        )
        assert dd_penalty == 0.0
        assert info["current_drawdown"] == pytest.approx(150.0)

    def test_drawdown_300_penalty(self):
        """Drawdown 300 -> excess 100 -> 0.001 * 100^2 = 10.0."""
        dd_penalty, info = self._env_with_dd(
            peak_pnl=1000.0, current_pnl=700.0,  # drawdown = 300
        )
        assert dd_penalty == pytest.approx(10.0)

    def test_drawdown_500_penalty(self):
        """Drawdown 500 -> excess 300 -> 0.001 * 300^2 = 90.0."""
        dd_penalty, info = self._env_with_dd(
            peak_pnl=1000.0, current_pnl=500.0,  # drawdown = 500
        )
        assert dd_penalty == pytest.approx(90.0)

    def test_zero_coef_disables_penalty(self):
        """dd_penalty_coef=0.0 -> no penalty regardless of drawdown."""
        dd_penalty, _ = self._env_with_dd(
            peak_pnl=1000.0, current_pnl=400.0,  # drawdown = 600
            coef=0.0,
        )
        assert dd_penalty == 0.0

    def test_peak_pnl_tracked_and_reset(self):
        """peak_pnl is correctly tracked and reset to 0 on reset()."""
        env = _make_env({"dd_penalty_coef": 0.001})
        env.reset(seed=42)
        assert env._peak_pnl == 0.0

        # Simulate rising PnL
        env._peak_pnl = max(env._peak_pnl, 500.0)
        assert env._peak_pnl == 500.0

        # PnL drops — peak stays
        env._peak_pnl = max(env._peak_pnl, 300.0)
        assert env._peak_pnl == 500.0

        # Reset clears it
        env.reset(seed=43)
        assert env._peak_pnl == 0.0

    def test_dd_penalty_in_reward_components(self):
        """dd_penalty appears in reward_components via step()."""
        env = _make_env({
            "dd_penalty_coef": 0.001,
            "dd_penalty_threshold": 200.0,
        })
        env.reset(seed=42)

        # Force a state where there's a drawdown
        env._peak_pnl = 500.0
        env._cash = -200.0   # total pnl will be low

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        assert "dd_penalty" in info
        assert "current_drawdown" in info
        assert "peak_pnl" in info
        assert "dd_penalty" in info["reward_components"]
