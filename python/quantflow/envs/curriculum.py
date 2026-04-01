"""
CurriculumWrapper: 3-stage difficulty progression for the MarketMakingEnv.

Stage progression
-----------------
Easy   — low volatility, minimal Hawkes clustering, wide spreads.
         Agent learns the basics: inventory management, spread capture.
Medium — default parameters.
         Agent learns: Hawkes clustering, adverse selection.
Hard   — elevated volatility, strong clustering, tighter spreads.
         Agent learns: stress regimes, burst activity.

The wrapper injects stage-appropriate simulator parameters via
``set_simulator_config()`` before each episode reset.  Stage advancement
is automatic (based on step count) or manual via ``set_stage()``.

Convenience
-----------
    from quantflow.envs.curriculum import CurriculumWrapper, compute_stage_steps

    steps = compute_stage_steps(200_000)  # (60000, 100000, 40000)
    env   = CurriculumWrapper(base_env, stage_steps=steps)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from quantflow.envs.market_making import MarketMakingEnv


# ── Stage definitions ─────────────────────────────────────────────────────────

STAGES: dict[str, dict[str, Any]] = {
    "easy": {
        "sigma_scale":   0.3,
        "mu_scale":      0.7,
        "alpha_scale":   0.3,
        "initial_mid":   100.0,
        "description":   "Low vol, minimal clustering, wide spreads",
    },
    "medium": {
        "sigma_scale":   1.0,
        "mu_scale":      1.0,
        "alpha_scale":   1.0,
        "initial_mid":   100.0,
        "description":   "Normal parameters",
    },
    "hard": {
        "sigma_scale":   1.5,
        "mu_scale":      1.3,
        "alpha_scale":   1.5,
        "initial_mid":   100.0,
        "description":   "High vol, strong clustering, tight spreads",
    },
}

_STAGE_NAMES: list[str] = ["easy", "medium", "hard"]


def compute_stage_steps(total_steps: int) -> tuple[int, int, int]:
    """
    Compute per-stage step budgets from a total training budget.

    Split: 30 % Easy → 50 % Medium → 20 % Hard.
    The remainder after integer division goes to Hard so that
    ``easy + medium + hard == total_steps`` exactly.

    Parameters
    ----------
    total_steps : int
        Total number of environment steps planned for training.

    Returns
    -------
    tuple[int, int, int]
        ``(easy_steps, medium_steps, hard_steps)``

    Examples
    --------
    >>> compute_stage_steps(50_000)
    (15000, 25000, 10000)
    >>> compute_stage_steps(1_000_000)
    (300000, 500000, 200000)
    """
    easy   = int(total_steps * 0.30)
    medium = int(total_steps * 0.50)
    hard   = total_steps - easy - medium
    return (easy, medium, hard)


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CurriculumWrapper(gym.Wrapper):
    """
    Automatically increases market difficulty across three training stages.

    Parameters
    ----------
    env : MarketMakingEnv
        The environment to wrap.
    stage_steps : tuple[int, int, int]
        Training steps per stage before auto-advancing.
        Use ``compute_stage_steps(total)`` for proportional splits.
    auto_advance : bool
        When True (default), advances to the next stage automatically
        once ``stage_steps[current]`` steps have elapsed.
    """

    STAGES = STAGES

    def __init__(
        self,
        env: MarketMakingEnv,
        stage_steps: tuple[int, int, int] = (15_000, 25_000, 10_000),
        auto_advance: bool = True,
    ) -> None:
        super().__init__(env)
        self.stage_steps  = stage_steps
        self.auto_advance = auto_advance

        self._current_stage:    int = 0
        self._steps_in_stage:   int = 0
        self._total_steps:      int = 0

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_stage(self) -> int:
        return self._current_stage

    @property
    def current_stage_name(self) -> str:
        return _STAGE_NAMES[self._current_stage]

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, **kwargs: Any):
        stage_cfg = STAGES[self.current_stage_name]
        self.env.set_simulator_config(stage_cfg)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._steps_in_stage += 1
        self._total_steps    += 1

        info["curriculum_stage"]          = self.current_stage_name
        info["curriculum_stage_idx"]      = self._current_stage
        info["curriculum_steps_in_stage"] = self._steps_in_stage

        if self.auto_advance and self._current_stage < 2:
            if self._steps_in_stage >= self.stage_steps[self._current_stage]:
                self._advance_stage()

        return obs, reward, terminated, truncated, info

    # ── Stage control ─────────────────────────────────────────────────────────

    def _advance_stage(self) -> None:
        self._current_stage   = min(2, self._current_stage + 1)
        self._steps_in_stage  = 0

    def set_stage(self, stage: int) -> None:
        """
        Manually jump to a specific stage (0 = easy, 1 = medium, 2 = hard).

        Does not reset step counters.
        """
        self._current_stage  = max(0, min(2, stage))
        self._steps_in_stage = 0
