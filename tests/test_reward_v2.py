"""
Unit and regression tests for the MarketMakingEnv reward functions.

Test inventory
--------------
1. Reward v1 regression — v1 produces identical numbers to the pre-v2 formula.
2. Round-trip bonus positive — buy and sell in the same step → rt_bonus > 0.
3. Round-trip bonus zero — only buys this step → rt_bonus == 0.
4. Asymmetric penalty (aligned) — long inventory + rising price → smaller penalty.
5. Asymmetric penalty (opposing) — long inventory + falling price → larger penalty.
6. Terminal penalty applied — env terminates → terminal_penalty > 0 when inv != 0.
7. Terminal penalty zero — env terminates with flat inventory → terminal_penalty == 0.
8. asymmetric_strength=0 symmetry — aligned vs opposing penalties are equal.
9. Gymnasium env_checker — environment passes gym's built-in checks.
"""
from __future__ import annotations

import math
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantflow.envs.market_making import MarketMakingEnv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_env(extra: dict[str, Any] | None = None) -> MarketMakingEnv:
    cfg: dict[str, Any] = {
        "t_max":           900.0,
        "episode_length":  5,
        "events_per_step": 1,
        "warm_up_events":  0,
        "inventory_limit": 50,
        "initial_mid":     100.0,
    }
    cfg.update(extra or {})
    return MarketMakingEnv(config=cfg)


def _prime_env(env: MarketMakingEnv) -> None:
    """Run one no-op step so _prev_pnl / _prev_mid are initialised."""
    env._prev_pnl = 0.0
    env._prev_mid = 100.0
    env._sim_time  = 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRewardV1Regression:
    """v1 reward must match the hand-computed formula exactly."""

    def test_v1_formula(self) -> None:
        env = _make_env()
        env.phi         = 0.005
        env.psi         = 0.0001
        env.lambda_hard = 0.1
        env._inventory  = 5
        env._prev_pnl   = 10.0

        pnl  = 15.0   # delta = +5
        reward, comps = env._compute_reward_v1(pnl)

        inv = 5
        expected_delta    = 5.0
        expected_inv_pen  = 0.005 * inv
        expected_risk_pen = 0.0001 * inv ** 2
        expected_breach   = 0.0   # |5| < 50
        expected_reward   = (expected_delta - expected_inv_pen
                             - expected_risk_pen - expected_breach)

        assert math.isclose(reward, expected_reward, rel_tol=1e-9)
        assert math.isclose(comps["delta_pnl"],     expected_delta,    rel_tol=1e-9)
        assert math.isclose(comps["inv_penalty"],   expected_inv_pen,  rel_tol=1e-9)
        assert math.isclose(comps["risk_penalty"],  expected_risk_pen, rel_tol=1e-9)
        assert comps["rt_bonus"]         == 0.0
        assert comps["terminal_penalty"] == 0.0
        assert comps["round_trips"]      == 0

    def test_v1_breach_penalty(self) -> None:
        env = _make_env()
        env.lambda_hard = 0.1
        env._inventory  = 60   # exceeds limit=50
        env._prev_pnl   = 0.0
        pnl = 0.0
        reward, comps = env._compute_reward_v1(pnl)
        expected_breach = 0.1 * (60 - 50)
        assert math.isclose(comps["breach_penalty"], expected_breach, rel_tol=1e-9)
        assert reward < 0.0


class TestRoundTripBonus:

    def _env_with_fills(
        self,
        buys: int, buy_value: float,
        sells: int, sell_value: float,
    ) -> MarketMakingEnv:
        env = _make_env({
            "reward_config": {
                "reward_version": "v2",
                "rt_weight": 0.5,
                "phi": 0.0,
                "psi": 0.0,
                "lambda_breach": 0.0,
                "asymmetric_strength": 0.0,
                "terminal_weight": 0.0,
            },
        })
        env._step_buys       = buys
        env._step_sells      = sells
        env._step_buy_value  = buy_value
        env._step_sell_value = sell_value
        env._inventory       = 0
        env._current_pnl     = 0.0
        env._prev_pnl        = 0.0
        env._terminated      = False
        env._current_spread  = 0.01
        return env

    def test_rt_bonus_positive(self) -> None:
        """Buy at 99.90, sell at 100.10 → capture 0.20 spread per round-trip."""
        env = self._env_with_fills(
            buys=10, buy_value=999.0,
            sells=10, sell_value=1001.0,
        )
        _, comps = env._compute_reward_v2()
        assert comps["rt_bonus"] > 0.0
        assert comps["round_trips"] == 10

    def test_rt_bonus_zero_no_sells(self) -> None:
        """Only buys this step — no completed round-trips."""
        env = self._env_with_fills(
            buys=5, buy_value=500.0,
            sells=0, sell_value=0.0,
        )
        _, comps = env._compute_reward_v2()
        assert comps["rt_bonus"] == 0.0
        assert comps["round_trips"] == 0

    def test_rt_bonus_zero_inverted_spread(self) -> None:
        """avg_sell < avg_buy → negative spread captured → bonus clamped to 0."""
        env = self._env_with_fills(
            buys=5, buy_value=505.0,   # avg_buy = 101
            sells=5, sell_value=495.0, # avg_sell = 99
        )
        _, comps = env._compute_reward_v2()
        assert comps["rt_bonus"] == 0.0


class TestAsymmetricInventoryPenalty:

    def _env_with_inv_and_trend(
        self,
        inventory: int,
        trend_direction: str,  # "up", "down", "flat"
        strength: float = 0.3,
    ) -> MarketMakingEnv:
        env = _make_env({
            "reward_config": {
                "reward_version": "v2",
                "phi": 0.01,
                "psi": 0.0,
                "lambda_breach": 0.0,
                "rt_weight": 0.0,
                "asymmetric_strength": strength,
                "terminal_weight": 0.0,
            },
        })
        env._inventory      = inventory
        env._current_pnl    = 0.0
        env._prev_pnl       = 0.0
        env._terminated     = False
        env._current_spread = 0.01
        env._step_buys      = 0
        env._step_sells     = 0
        env._step_buy_value  = 0.0
        env._step_sell_value = 0.0

        # Populate mid-price history with a clear trend.
        base = 100.0
        history: list[float] = []
        if trend_direction == "up":
            history = [base + i * 0.02 for i in range(60)]
        elif trend_direction == "down":
            history = [base - i * 0.02 for i in range(60)]
        else:
            history = [base] * 60
        for p in history:
            env._mid_price_history.append(p)
        return env

    def test_aligned_penalty_smaller(self) -> None:
        """Long inventory + rising prices → aligned → penalty < unmodulated baseline."""
        env_aligned = self._env_with_inv_and_trend(10, "up",   strength=0.3)
        env_base    = self._env_with_inv_and_trend(10, "flat", strength=0.0)
        _, c_aligned = env_aligned._compute_reward_v2()
        _, c_base    = env_base._compute_reward_v2()
        assert c_aligned["inv_penalty"] < c_base["inv_penalty"]
        assert c_aligned["trend_alignment"] == 1.0

    def test_opposing_penalty_larger(self) -> None:
        """Long inventory + falling prices → adverse selection → penalty > baseline."""
        env_opposing = self._env_with_inv_and_trend(10, "down", strength=0.3)
        env_base     = self._env_with_inv_and_trend(10, "flat", strength=0.0)
        _, c_opposing = env_opposing._compute_reward_v2()
        _, c_base     = env_base._compute_reward_v2()
        assert c_opposing["inv_penalty"] > c_base["inv_penalty"]
        assert c_opposing["trend_alignment"] == -1.0


class TestTerminalPenalty:

    def _base_env(self, **rc_overrides: Any) -> MarketMakingEnv:
        rc = {
            "reward_version":       "v2",
            "phi":                  0.0,
            "psi":                  0.0,
            "lambda_breach":        0.0,
            "rt_weight":            0.0,
            "asymmetric_strength":  0.0,
            "terminal_weight":      2.0,
        }
        rc.update(rc_overrides)
        env = _make_env({"reward_config": rc})
        env._step_buys      = 0
        env._step_sells     = 0
        env._step_buy_value  = 0.0
        env._step_sell_value = 0.0
        env._current_pnl    = 0.0
        env._prev_pnl       = 0.0
        return env

    def test_terminal_penalty_applied(self) -> None:
        """Non-zero inventory at episode end incurs terminal_penalty > 0."""
        env = self._base_env()
        env._inventory       = 5
        env._terminated      = True
        env._current_spread  = 0.02

        _, comps = env._compute_reward_v2()
        expected = 2.0 * 5 * 0.02
        assert math.isclose(comps["terminal_penalty"], expected, rel_tol=1e-9)
        assert comps["terminal_penalty"] > 0.0

    def test_terminal_penalty_zero_flat(self) -> None:
        """Zero inventory at episode end → no terminal penalty."""
        env = self._base_env()
        env._inventory      = 0
        env._terminated     = True
        env._current_spread = 0.02

        _, comps = env._compute_reward_v2()
        assert comps["terminal_penalty"] == 0.0

    def test_no_terminal_penalty_mid_episode(self) -> None:
        """Non-zero inventory but not terminated → no terminal penalty."""
        env = self._base_env()
        env._inventory      = 5
        env._terminated     = False
        env._current_spread = 0.02

        _, comps = env._compute_reward_v2()
        assert comps["terminal_penalty"] == 0.0


class TestAsymmetricStrengthZero:

    def test_symmetry_with_strength_zero(self) -> None:
        """With asymmetric_strength=0, aligned and opposing penalties must be equal."""
        def _make(trend: str) -> MarketMakingEnv:
            env = _make_env({
                "reward_config": {
                    "reward_version":      "v2",
                    "phi":                 0.01,
                    "psi":                 0.0,
                    "lambda_breach":       0.0,
                    "rt_weight":           0.0,
                    "asymmetric_strength": 0.0,
                    "terminal_weight":     0.0,
                },
            })
            env._inventory      = 10
            env._current_pnl    = 0.0
            env._prev_pnl       = 0.0
            env._terminated     = False
            env._current_spread = 0.01
            env._step_buys      = 0
            env._step_sells     = 0
            env._step_buy_value  = 0.0
            env._step_sell_value = 0.0
            base = 100.0
            if trend == "up":
                history = [base + i * 0.02 for i in range(60)]
            else:
                history = [base - i * 0.02 for i in range(60)]
            for p in history:
                env._mid_price_history.append(p)
            return env

        env_up, env_down = _make("up"), _make("down")
        _, c_up   = env_up._compute_reward_v2()
        _, c_down = env_down._compute_reward_v2()
        assert math.isclose(
            c_up["inv_penalty"], c_down["inv_penalty"], rel_tol=1e-9
        ), (
            f"Expected equal penalties with strength=0, got "
            f"{c_up['inv_penalty']} vs {c_down['inv_penalty']}"
        )


class TestGymEnvChecker:

    def test_env_checker(self) -> None:
        """Gymnasium's env_checker must raise no errors or warnings."""
        from gymnasium.utils.env_checker import check_env

        env = _make_env({"episode_length": 10, "warm_up_events": 0})
        # check_env raises AssertionError or warnings on spec violations.
        check_env(env, warn=True, skip_render_check=True)
