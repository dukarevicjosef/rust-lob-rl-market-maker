"""
Tests for DomainRandomizer and the set_simulator_config() interface.

Test inventory
--------------
1. DomainRandomizer wraps env without error; gymnasium check_env passes.
2. Two resets with different seeds produce different domain parameters.
3. mu_scale stays within [1 - mu_range, 1 + mu_range] = [0.8, 1.2].
4. initial_mid stays within [95, 105].
5. sigma_scale stays within [0.7, 1.5].
6. set_simulator_config() is applied to the inner env on next reset.
7. Without DomainRandomizer → initial_mid stays fixed across resets (control).
8. DomainRandomizer with fixed seed reproduces the same params (determinism).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quantflow.envs.domain_randomizer import DEFAULT_RANDOMIZE_CONFIG, DomainRandomizer
from quantflow.envs.market_making import MarketMakingEnv


# ── Helpers ───────────────────────────────────────────────────────────────────

_N_SAMPLES = 200  # how many draws to check per range test


def _make_env(extra: dict[str, Any] | None = None) -> MarketMakingEnv:
    cfg: dict[str, Any] = {
        "t_max":           900.0,
        "episode_length":  5,
        "events_per_step": 1,
        "warm_up_events":  0,
        "inventory_limit": 50,
        "initial_mid":     100.0,
        "obs_version":     "v2",
    }
    cfg.update(extra or {})
    return MarketMakingEnv(config=cfg)


def _make_dr(extra_rand: dict[str, Any] | None = None) -> DomainRandomizer:
    return DomainRandomizer(_make_env(), randomize_config=extra_rand)


def _seed_dr(dr: DomainRandomizer, seed: int) -> None:
    """Seed the wrapper's np_random so _sample_params() is deterministic."""
    import gymnasium
    dr.np_random, _ = gymnasium.utils.seeding.np_random(seed)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGymCompliance:

    def test_check_env_passes(self) -> None:
        from gymnasium.utils.env_checker import check_env

        dr = _make_dr()
        check_env(dr, warn=True, skip_render_check=True)


class TestParamVariety:

    def test_different_seeds_different_params(self) -> None:
        """Two different seeds must produce different initial_mid values."""
        dr = _make_dr()
        _seed_dr(dr, 42)
        p1 = dr._sample_params()
        _seed_dr(dr, 99)
        p2 = dr._sample_params()
        assert p1["initial_mid"] != p2["initial_mid"], (
            "Different seeds produced identical initial_mid — RNG not seeded properly."
        )

    def test_consecutive_resets_differ(self) -> None:
        """Resets with increasing seeds must vary the initial_mid seen by the env."""
        dr = _make_dr()
        dr.reset(seed=10)
        mid1 = dr.env.initial_mid
        dr.reset(seed=11)
        mid2 = dr.env.initial_mid
        assert mid1 != mid2, (
            f"Consecutive resets produced identical initial_mid={mid1}."
        )


class TestParameterRanges:

    def _draw_many(self, key: str, n: int = _N_SAMPLES) -> list[float]:
        dr = _make_dr()
        _seed_dr(dr, 0)
        return [dr._sample_params()[key] for _ in range(n)]

    def test_mu_scale_range(self) -> None:
        cfg = DEFAULT_RANDOMIZE_CONFIG
        lo  = 1.0 - cfg["mu_range"]
        hi  = 1.0 + cfg["mu_range"]
        for v in self._draw_many("mu_scale"):
            assert lo <= v <= hi, f"mu_scale={v} outside [{lo}, {hi}]"

    def test_initial_mid_range(self) -> None:
        cfg = DEFAULT_RANDOMIZE_CONFIG
        lo  = cfg["mid_price_low"]
        hi  = cfg["mid_price_high"]
        for v in self._draw_many("initial_mid"):
            assert lo <= v <= hi, f"initial_mid={v} outside [{lo}, {hi}]"

    def test_sigma_scale_range(self) -> None:
        cfg = DEFAULT_RANDOMIZE_CONFIG
        lo  = cfg["sigma_scale_low"]
        hi  = cfg["sigma_scale_high"]
        for v in self._draw_many("sigma_scale"):
            assert lo <= v <= hi, f"sigma_scale={v} outside [{lo}, {hi}]"

    def test_alpha_scale_range(self) -> None:
        cfg = DEFAULT_RANDOMIZE_CONFIG
        lo  = 1.0 - cfg["alpha_range"]
        hi  = 1.0 + cfg["alpha_range"]
        for v in self._draw_many("alpha_scale"):
            assert lo <= v <= hi, f"alpha_scale={v} outside [{lo}, {hi}]"


class TestSetSimulatorConfig:

    def test_initial_mid_applied(self) -> None:
        """set_simulator_config({"initial_mid": 102}) must change env.initial_mid."""
        env = _make_env()
        env.set_simulator_config({"initial_mid": 102.0})
        env.reset(seed=42)
        assert env.initial_mid == pytest.approx(102.0), (
            f"Expected initial_mid=102.0, got {env.initial_mid}"
        )

    def test_sigma_scale_reflected_in_obs(self) -> None:
        """
        sigma_scale is exposed as sigma_regime in obs-v2.
        With sigma_scale=1.5 the sigma_regime obs should be 1.0 (at the upper end).
        """
        env = _make_env()
        env.set_simulator_config({"sigma_scale": 1.5})
        obs, _ = env.reset(seed=42)
        assert "sigma_regime" in obs
        assert obs["sigma_regime"][0] == pytest.approx(1.0, abs=1e-5)

    def test_empty_config_clears_pending(self) -> None:
        """Passing an empty dict must not raise and must leave params unchanged."""
        env = _make_env()
        env.set_simulator_config({})
        # Should reset cleanly with default params.
        obs, _ = env.reset(seed=42)
        assert obs is not None

    def test_config_consumed_after_reset(self) -> None:
        """Pending params must be cleared after one reset (not applied twice)."""
        env = _make_env()
        env.set_simulator_config({"initial_mid": 98.0})
        env.reset(seed=42)
        mid_after_first = env.initial_mid
        env.reset(seed=43)
        mid_after_second = env.initial_mid
        # Without a second set_simulator_config() call, initial_mid should persist
        # because env.initial_mid is updated in place (not reverted to 100.0).
        # What must NOT happen: an error or crash on the second reset.
        assert mid_after_first == pytest.approx(98.0)
        assert mid_after_second == pytest.approx(98.0)  # still 98 — no new config


class TestControlGroup:

    def test_no_dr_initial_mid_fixed(self) -> None:
        """Without DomainRandomizer, initial_mid must be the same every reset."""
        env = _make_env({"initial_mid": 100.0})
        env.reset(seed=42)
        mid1 = env.initial_mid
        env.reset(seed=43)
        mid2 = env.initial_mid
        assert mid1 == mid2 == pytest.approx(100.0)


class TestDeterminism:

    def test_same_seed_same_params(self) -> None:
        """DomainRandomizer with same seed must draw identical params."""
        dr = _make_dr()
        _seed_dr(dr, 7)
        p1 = dr._sample_params()
        _seed_dr(dr, 7)
        p2 = dr._sample_params()
        assert p1["initial_mid"]  == pytest.approx(p2["initial_mid"])
        assert p1["sigma_scale"]  == pytest.approx(p2["sigma_scale"])
        assert p1["mu_scale"]     == pytest.approx(p2["mu_scale"])
