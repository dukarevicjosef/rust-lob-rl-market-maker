"""
Tests for CurriculumWrapper and compute_stage_steps.
"""
from __future__ import annotations

import pytest

from quantflow.envs.curriculum import (
    CurriculumWrapper,
    STAGES,
    _STAGE_NAMES,
    compute_stage_steps,
)
from quantflow.envs.market_making import MarketMakingEnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_env(seed: int = 42) -> MarketMakingEnv:
    return MarketMakingEnv(config={
        "episode_length":  100,
        "events_per_step": 10,
        "warm_up_events":  20,
        "seed":            seed,
    })


def _make_wrapper(
    stage_steps: tuple[int, int, int] = (5, 8, 3),
    auto_advance: bool = True,
    seed: int = 42,
) -> CurriculumWrapper:
    env = _make_env(seed)
    return CurriculumWrapper(env, stage_steps=stage_steps, auto_advance=auto_advance)


# ── compute_stage_steps ───────────────────────────────────────────────────────

def test_stage_steps_sum_equals_total():
    total = 200_000
    easy, medium, hard = compute_stage_steps(total)
    assert easy + medium + hard == total


def test_stage_steps_proportions():
    easy, medium, hard = compute_stage_steps(1_000_000)
    assert easy   == 300_000
    assert medium == 500_000
    assert hard   == 200_000


def test_stage_steps_small_total():
    easy, medium, hard = compute_stage_steps(10)
    assert easy + medium + hard == 10
    assert easy >= 0 and medium >= 0 and hard >= 0


# ── Initial state ─────────────────────────────────────────────────────────────

def test_wrapper_starts_in_easy_stage():
    wrapper = _make_wrapper()
    assert wrapper.current_stage      == 0
    assert wrapper.current_stage_name == "easy"


def test_reset_applies_easy_config(monkeypatch):
    """reset() must call set_simulator_config with the easy stage dict."""
    applied: list[dict] = []
    wrapper = _make_wrapper()
    monkeypatch.setattr(wrapper.env, "set_simulator_config", lambda cfg: applied.append(cfg))
    wrapper.reset()
    assert len(applied) == 1
    assert applied[0]["sigma_scale"] == STAGES["easy"]["sigma_scale"]


# ── Auto-advance ──────────────────────────────────────────────────────────────

def test_auto_advance_easy_to_medium():
    """After stage_steps[0] steps, stage advances to medium automatically."""
    stage_steps = (5, 8, 3)
    wrapper = _make_wrapper(stage_steps=stage_steps)
    wrapper.reset()

    for _ in range(stage_steps[0]):
        obs, reward, terminated, truncated, info = wrapper.step(wrapper.action_space.sample())
        if terminated or truncated:
            wrapper.reset()

    assert wrapper.current_stage      == 1
    assert wrapper.current_stage_name == "medium"


def test_auto_advance_medium_to_hard():
    """After stage_steps[0] + stage_steps[1] total steps, stage is hard."""
    stage_steps = (5, 8, 3)
    wrapper = _make_wrapper(stage_steps=stage_steps)
    wrapper.reset()

    for _ in range(stage_steps[0] + stage_steps[1]):
        obs, reward, terminated, truncated, info = wrapper.step(wrapper.action_space.sample())
        if terminated or truncated:
            wrapper.reset()

    assert wrapper.current_stage      == 2
    assert wrapper.current_stage_name == "hard"


def test_no_advance_beyond_hard():
    """Stage never exceeds 2 (hard) regardless of step count."""
    stage_steps = (2, 3, 2)
    wrapper = _make_wrapper(stage_steps=stage_steps)
    wrapper.reset()

    for _ in range(50):
        obs, reward, terminated, truncated, info = wrapper.step(wrapper.action_space.sample())
        if terminated or truncated:
            wrapper.reset()

    assert wrapper.current_stage == 2


# ── Info dict ─────────────────────────────────────────────────────────────────

def test_info_contains_curriculum_keys():
    """step() must inject curriculum metadata into the info dict."""
    wrapper = _make_wrapper()
    wrapper.reset()
    _, _, _, _, info = wrapper.step(wrapper.action_space.sample())

    assert "curriculum_stage"          in info
    assert "curriculum_stage_idx"      in info
    assert "curriculum_steps_in_stage" in info


def test_info_stage_idx_matches_current_stage():
    wrapper = _make_wrapper()
    wrapper.reset()
    _, _, _, _, info = wrapper.step(wrapper.action_space.sample())
    assert info["curriculum_stage_idx"] == wrapper.current_stage
    assert info["curriculum_stage"]     == wrapper.current_stage_name


# ── Manual override ───────────────────────────────────────────────────────────

def test_set_stage_manual():
    """set_stage() jumps to the requested stage without touching step counters."""
    wrapper = _make_wrapper()
    wrapper.reset()
    wrapper.set_stage(2)
    assert wrapper.current_stage      == 2
    assert wrapper.current_stage_name == "hard"


def test_set_stage_clamps_to_valid_range():
    wrapper = _make_wrapper()
    wrapper.set_stage(-5)
    assert wrapper.current_stage == 0
    wrapper.set_stage(99)
    assert wrapper.current_stage == 2


# ── Disabled auto-advance ─────────────────────────────────────────────────────

def test_no_auto_advance_when_disabled():
    stage_steps = (3, 5, 2)
    wrapper = _make_wrapper(stage_steps=stage_steps, auto_advance=False)
    wrapper.reset()

    for _ in range(20):
        obs, reward, terminated, truncated, info = wrapper.step(wrapper.action_space.sample())
        if terminated or truncated:
            wrapper.reset()

    # Stage must remain at 0 since auto_advance=False
    assert wrapper.current_stage == 0
