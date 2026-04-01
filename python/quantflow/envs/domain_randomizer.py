"""
DomainRandomizer: Gymnasium Wrapper that perturbs Hawkes simulator parameters
before every episode reset so the agent never trains on an identical market.

Each reset samples a new parameter vector from configurable uniform ranges.
The perturbations act on:

  initial_mid   — Starting price level (Python controls this directly).
  sigma_scale   — Multiplier on lognormal order-size σ; higher → larger
                  individual trades → more price impact → higher realised vol.
  mu_scale      — Multiplier on lognormal order-size μ; higher → bigger
                  average order size → more market activity.
  alpha_scale   — Stored for observability; Hawkes excitation reconstruction
                  is not supported at this layer without a full re-init.
  beta_scale    — Stored for observability; same caveat as alpha_scale.

Usage
-----
    from quantflow.envs.market_making import MarketMakingEnv
    from quantflow.envs.domain_randomizer import DomainRandomizer

    base_env = MarketMakingEnv(config={"obs_version": "v2", ...})
    env      = DomainRandomizer(base_env)
    obs, _   = env.reset(seed=42)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.utils.seeding import np_random as gym_np_random

from quantflow.envs.market_making import MarketMakingEnv


DEFAULT_RANDOMIZE_CONFIG: dict[str, float] = {
    "mu_range":         0.20,    # ±20% on lognormal order-size μ
    "alpha_range":      0.15,    # ±15% (stored; not applied to Hawkes process)
    "beta_range":       0.10,    # ±10% (stored; not applied to Hawkes process)
    "mid_price_low":    95.0,
    "mid_price_high":  105.0,
    "sigma_scale_low":   0.7,
    "sigma_scale_high":  1.5,
}


class DomainRandomizer(gym.Wrapper):
    """
    Perturbs MarketMakingEnv simulator parameters before each episode.

    Parameters
    ----------
    env : MarketMakingEnv
        The environment to wrap.
    randomize_config : dict, optional
        Override any key from ``DEFAULT_RANDOMIZE_CONFIG``.
    """

    def __init__(
        self,
        env: MarketMakingEnv,
        randomize_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.randomize_config: dict[str, Any] = {
            **DEFAULT_RANDOMIZE_CONFIG,
            **(randomize_config or {}),
        }

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None, **kwargs: Any):
        # Re-seed the wrapper's own np_random so that _sample_params() is
        # deterministic: the same `seed` always yields the same domain params.
        if seed is not None:
            self.np_random, self._np_random_seed = gym_np_random(seed)
        params = self._sample_params()
        # Push params into the inner env before it resets the simulator.
        self.env.set_simulator_config(params)
        return self.env.reset(seed=seed, options=options, **kwargs)

    # ── Parameter sampling ────────────────────────────────────────────────────

    def _sample_params(self) -> dict[str, float]:
        """
        Draw one parameter vector uniformly within the configured ranges.

        Uses ``self.np_random`` — Gymnasium's seeded RNG — so parameter
        sequences are fully reproducible given a fixed seed.
        """
        rng = self.np_random
        cfg = self.randomize_config
        return {
            "mu_scale": float(rng.uniform(
                1.0 - cfg["mu_range"],
                1.0 + cfg["mu_range"],
            )),
            "alpha_scale": float(rng.uniform(
                1.0 - cfg["alpha_range"],
                1.0 + cfg["alpha_range"],
            )),
            "beta_scale": float(rng.uniform(
                1.0 - cfg["beta_range"],
                1.0 + cfg["beta_range"],
            )),
            "initial_mid": float(rng.uniform(
                cfg["mid_price_low"],
                cfg["mid_price_high"],
            )),
            "sigma_scale": float(rng.uniform(
                cfg["sigma_scale_low"],
                cfg["sigma_scale_high"],
            )),
        }
