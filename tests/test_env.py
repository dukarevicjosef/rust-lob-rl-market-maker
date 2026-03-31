"""
Tests for MarketMakingEnv.

Run with:
    uv run pytest tests/test_env.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from quantflow.envs.market_making import MarketMakingEnv

# ── Shared fixture ─────────────────────────────────────────────────────────────

FAST_CFG = {
    "t_max":             120.0,
    "episode_length":    30,
    "warm_up_events":    100,
    "events_per_step":   20,
    "snapshot_interval": 5000,
    "lob_levels":        5,
    "seed":              42,
}


@pytest.fixture
def env() -> MarketMakingEnv:
    e = MarketMakingEnv(FAST_CFG)
    e.reset(seed=42)
    return e


# ── Gymnasium env_checker ──────────────────────────────────────────────────────

def test_env_checker():
    """Gymnasium's official compliance checker must pass."""
    from gymnasium.utils.env_checker import check_env
    e = MarketMakingEnv(FAST_CFG)
    check_env(e, skip_render_check=True)


# ── Spaces ────────────────────────────────────────────────────────────────────

def test_observation_space_is_dict():
    e = MarketMakingEnv(FAST_CFG)
    assert isinstance(e.observation_space, gym.spaces.Dict)


def test_observation_space_keys():
    e = MarketMakingEnv(FAST_CFG)
    expected = {
        "lob_state", "volume_imbalance", "spread", "mid_price_return",
        "volatility", "inventory", "pnl", "time_remaining",
    }
    assert set(e.observation_space.spaces.keys()) == expected


def test_action_space_shape():
    e = MarketMakingEnv(FAST_CFG)
    assert e.action_space.shape == (2,)
    np.testing.assert_allclose(e.action_space.low,  [0.01, -0.5])
    np.testing.assert_allclose(e.action_space.high, [1.00,  0.5])


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_obs_and_info():
    e = MarketMakingEnv(FAST_CFG)
    obs, info = e.reset(seed=1)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_reset_obs_in_space():
    e = MarketMakingEnv(FAST_CFG)
    obs, _ = e.reset(seed=1)
    assert e.observation_space.contains(obs), f"obs not in space: {obs}"


def test_reset_clears_inventory_and_pnl():
    e = MarketMakingEnv(FAST_CFG)
    # Run some steps to dirty state
    e.reset(seed=1)
    for _ in range(5):
        e.step(e.action_space.sample())
    # Then reset
    obs, _ = e.reset(seed=2)
    assert e._inventory == 0
    assert e._cash == 0.0
    assert e._step_count == 0


def test_reset_different_seeds_differ():
    e = MarketMakingEnv(FAST_CFG)
    obs1, _ = e.reset(seed=10)
    obs2, _ = e.reset(seed=99)
    # mid_price_return comes from different Hawkes realisations → should differ
    # (not guaranteed on the very first obs but extremely likely)
    differ = any(
        not np.allclose(obs1[k], obs2[k])
        for k in obs1
    )
    assert differ, "Two different seeds produced identical observations"


# ── step ──────────────────────────────────────────────────────────────────────

def test_step_returns_five_tuple(env: MarketMakingEnv):
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_obs_in_space(env: MarketMakingEnv):
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs), f"step obs not in space:\n{obs}"


def test_step_action_space_sampled(env: MarketMakingEnv):
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)


# ── Observation ranges ────────────────────────────────────────────────────────

def test_lob_state_range(env: MarketMakingEnv):
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
        lob = obs["lob_state"]
        assert lob.shape == (FAST_CFG["lob_levels"] * 4,)
        assert np.all(lob >= -1.0 - 1e-5), f"lob_state below -1: {lob.min()}"
        assert np.all(lob <=  1.0 + 1e-5), f"lob_state above +1: {lob.max()}"


def test_scalar_obs_in_unit_range(env: MarketMakingEnv):
    scalar_keys = {
        "volume_imbalance": (-1.0, 1.0),
        "spread":           ( 0.0, 1.0),
        "mid_price_return": (-1.0, 1.0),
        "volatility":       ( 0.0, 1.0),
        "inventory":        (-1.0, 1.0),
        "pnl":              (-1.0, 1.0),
        "time_remaining":   ( 0.0, 1.0),
    }
    for _ in range(15):
        obs, *_ = env.step(env.action_space.sample())
        for key, (lo, hi) in scalar_keys.items():
            val = float(obs[key].item())
            assert lo - 1e-5 <= val <= hi + 1e-5, (
                f"{key}={val:.4f} outside [{lo}, {hi}]"
            )


# ── PnL and inventory logic ───────────────────────────────────────────────────

def test_inventory_changes_on_fill():
    """Run many steps; with enough fills inventory should deviate from 0."""
    cfg = {**FAST_CFG, "episode_length": 50, "events_per_step": 50}
    e = MarketMakingEnv(cfg)
    e.reset(seed=42)
    max_inv = 0
    for _ in range(50):
        _, _, term, _, info = e.step(np.array([0.1, 0.0], dtype=np.float32))
        max_inv = max(max_inv, abs(info["inventory"]))
        if term:
            break
    assert max_inv > 0, "Inventory never changed — fills may be broken"


def test_pnl_changes_over_episode():
    cfg = {**FAST_CFG, "episode_length": 50, "events_per_step": 50}
    e = MarketMakingEnv(cfg)
    e.reset(seed=42)
    pnls = []
    for _ in range(20):
        _, _, term, _, info = e.step(np.array([0.1, 0.0], dtype=np.float32))
        pnls.append(info["pnl"])
        if term:
            break
    # PnL must vary (inventory × mid changes even without fills)
    assert len(set(f"{p:.6f}" for p in pnls)) > 1, "PnL never changed"


def test_inventory_bounded_by_limit():
    cfg = {**FAST_CFG, "episode_length": 100, "events_per_step": 100,
           "inventory_limit": 20}
    e = MarketMakingEnv(cfg)
    e.reset(seed=42)
    for _ in range(100):
        _, _, term, _, info = e.step(np.array([0.05, 0.0], dtype=np.float32))
        assert abs(info["inventory"]) <= cfg["inventory_limit"] + 10 + 1, (
            f"|inventory| = {abs(info['inventory'])} exceeds limit"
        )
        if term:
            break


# ── Termination ───────────────────────────────────────────────────────────────

def test_terminates_after_episode_length():
    cfg = {**FAST_CFG, "episode_length": 10}
    e = MarketMakingEnv(cfg)
    e.reset(seed=42)
    terminated = False
    for i in range(15):
        _, _, terminated, _, _ = e.step(e.action_space.sample())
        if terminated:
            assert i == 9, f"Terminated at step {i}, expected step 9"
            break
    assert terminated, "Episode never terminated"


# ── Gymnasium registration ─────────────────────────────────────────────────────

def test_gym_make():
    """Environment can be constructed through gymnasium.make."""
    e = gym.make("quantflow/MarketMaking-v0")
    obs, _ = e.reset()
    assert isinstance(obs, dict)
    e.close()
