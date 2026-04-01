"""
A/B comparison: training without vs. with CurriculumWrapper.

Variant A (no-curriculum): 50K steps on fixed medium parameters.
Variant B (curriculum):     50K steps split 30/50/20 across easy→medium→hard.

Additionally tracks eval_reward every 5K steps during training (convergence
analysis) and evaluates both trained models across 3 market regimes.

Usage
-----
    python -m quantflow.training.ab_curriculum

Optional env vars
-----------------
    WANDB_PROJECT     — W&B project name (set to "" to disable)
    AB_STEPS          — total training steps per variant (default 50_000)
    AB_EVAL_EPISODES  — evaluation episodes per regime (default 10)
    AB_EVAL_INTERVAL  — convergence eval every N steps (default 5_000)
"""
from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from quantflow.envs.curriculum import CurriculumWrapper, compute_stage_steps
from quantflow.envs.market_making import MarketMakingEnv

import gymnasium as gym


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_STEPS:    int = int(os.environ.get("AB_STEPS",          50_000))
EVAL_EPISODES:  int = int(os.environ.get("AB_EVAL_EPISODES",  10))
EVAL_INTERVAL:  int = int(os.environ.get("AB_EVAL_INTERVAL",  5_000))
FIRST_EVAL_SEED: int = 4_000
TRAIN_SEED:      int = 42

WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "quantflow-ab")

_BASE_CFG: dict[str, Any] = {
    "t_max":           3_600.0,
    "episode_length":  2_000,
    "events_per_step": 50,
    "warm_up_events":  500,
    "inventory_limit": 50,
    "obs_version":     "v2",
    "reward_config": {
        "phi":                  0.01,
        "psi":                  0.001,
        "lambda_breach":        1.0,
        "rt_weight":            0.5,
        "asymmetric_strength":  0.3,
        "terminal_weight":      2.0,
        "reward_version":       "v2",
    },
}

REGIMES: dict[str, dict[str, float]] = {
    "STANDARD": {},
    "HIGH-VOL": {"sigma_scale": 2.0, "mu_scale": 1.3},
    "LOW-LIQ":  {"mu_scale": 0.5},
}

REGIME_LABELS: dict[str, str] = {
    "STANDARD": "Standard",
    "HIGH-VOL": "High-Vol  (σ×2, μ×1.3)",
    "LOW-LIQ":  "Low-Liq   (μ×0.5)",
}


@dataclass
class VariantSpec:
    name:            str
    use_curriculum:  bool


VARIANTS: list[VariantSpec] = [
    VariantSpec(name="no-curriculum", use_curriculum=False),
    VariantSpec(name="curriculum",    use_curriculum=True),
]


# ── Environment factories ─────────────────────────────────────────────────────

def _make_base_env(seed: int) -> MarketMakingEnv:
    return MarketMakingEnv(config={**_BASE_CFG, "seed": seed})


def _make_train_env(variant: VariantSpec, seed: int) -> gym.Env:
    base = _make_base_env(seed)
    if variant.use_curriculum:
        stage_steps = compute_stage_steps(TRAIN_STEPS)
        return CurriculumWrapper(base, stage_steps=stage_steps)
    return base


# ── Convergence tracking ──────────────────────────────────────────────────────

@dataclass
class ConvergencePoint:
    step:         int
    mean_reward:  float


class _ConvergenceCallback(BaseCallback):
    """
    Evaluates mean episode reward every ``eval_interval`` steps.

    Evaluation runs ``n_eval_episodes`` deterministic episodes on a
    fresh ``MarketMakingEnv`` (standard parameters, no curriculum wrapper).
    """

    def __init__(
        self,
        eval_interval:   int,
        n_eval_episodes: int,
        eval_seed:       int,
        verbose:         int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_interval   = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed       = eval_seed
        self.history:        list[ConvergencePoint] = []
        self._next_eval:     int = eval_interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_interval

        rewards: list[float] = []
        for i in range(self.n_eval_episodes):
            env    = _make_base_env(self.eval_seed + i)
            obs, _ = env.reset(seed=self.eval_seed + i)
            done   = False
            ep_rew = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                done    = terminated or truncated
                ep_rew += float(reward)
            rewards.append(ep_rew)

        mean_rew = statistics.mean(rewards)
        self.history.append(ConvergencePoint(self.num_timesteps, mean_rew))

        if self.verbose >= 1:
            print(
                f"  [convergence] step={self.num_timesteps:,}  "
                f"mean_reward={mean_rew:+.4f}"
            )
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def _train(variant: VariantSpec) -> tuple[PPO, list[ConvergencePoint]]:
    conv_cb = _ConvergenceCallback(
        eval_interval   = EVAL_INTERVAL,
        n_eval_episodes = max(3, EVAL_EPISODES // 3),
        eval_seed       = FIRST_EVAL_SEED + 500,
    )
    vec_env = make_vec_env(
        lambda: _make_train_env(variant, TRAIN_SEED),
        n_envs=1,
        seed=TRAIN_SEED,
    )
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=0,
        seed=TRAIN_SEED,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
    )
    model.learn(total_timesteps=TRAIN_STEPS, callback=conv_cb)
    return model, conv_cb.history


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    pnl:    float
    sharpe: float


def _evaluate_episode(
    model:         PPO,
    seed:          int,
    regime_params: dict[str, float],
) -> EpisodeResult:
    env = _make_base_env(seed)
    if regime_params:
        env.set_simulator_config(regime_params)
    obs, _ = env.reset(seed=seed)
    done   = False

    pnl_history: list[float] = [0.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        pnl_history.append(info["pnl"])

    rets = [b - a for a, b in zip(pnl_history[:-1], pnl_history[1:])]
    if len(rets) > 1:
        m = statistics.mean(rets)
        v = statistics.variance(rets)
        sharpe = (m / math.sqrt(v)) if v > 1e-10 else 0.0
    else:
        sharpe = 0.0

    return EpisodeResult(pnl=pnl_history[-1], sharpe=sharpe)


def _evaluate_regime(
    model:         PPO,
    regime_params: dict[str, float],
) -> list[EpisodeResult]:
    return [
        _evaluate_episode(model, FIRST_EVAL_SEED + i, regime_params)
        for i in range(EVAL_EPISODES)
    ]


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(m: float, s: float) -> str:
    return f"{m:+.4f} ± {s:.4f}"


def _print_convergence(
    variant_names:    list[str],
    conv_histories:   list[list[ConvergencePoint]],
) -> None:
    col_w  = 18
    name_w = 10

    all_steps = sorted({pt.step for hist in conv_histories for pt in hist})
    if not all_steps:
        return

    sep = "─" * (name_w + col_w * len(variant_names))
    print()
    print(sep)
    print(f"  Convergence: mean episode reward every {EVAL_INTERVAL:,} steps")
    print(sep)
    header = f"{'Step':<{name_w}}" + "".join(
        f"{n:>{col_w}}" for n in variant_names
    )
    print(header)
    print(sep)

    for step in all_steps:
        row = f"{step:<{name_w},}"
        for hist in conv_histories:
            pts = [pt for pt in hist if pt.step == step]
            if pts:
                row += f"{pts[0].mean_reward:>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    print(sep)
    print()


def _print_regime_report(
    variant_names:  list[str],
    regime_results: dict[str, list[list[EpisodeResult]]],
) -> list[list[float]]:
    col_w  = 30
    name_w = 40
    sep    = "─" * (name_w + col_w * len(variant_names))

    print()
    print(sep)
    print(f"  A/B Curriculum Learning  (reward=v2, obs=v2 for both)")
    print(sep)
    print(f"{'':>{name_w}}" + "".join(
        f"  {n + ' (mean±std)':<{col_w - 2}}" for n in variant_names
    ))
    print(sep)

    robustness: list[list[float]] = [[] for _ in variant_names]

    for regime_key, per_variant in regime_results.items():
        label = REGIME_LABELS[regime_key]
        print(f"\n{label.upper()} REGIME ({EVAL_EPISODES} episodes)")

        for metric_name, extractor in [
            ("PnL",             lambda r: r.pnl),
            ("Sharpe",          lambda r: r.sharpe),
            ("Pos. Sharpe (%)", lambda r: 100.0 * float(r.sharpe > 0.0)),
        ]:
            row = f"  {metric_name:<{name_w - 2}}"
            for vi, results in enumerate(per_variant):
                vals = [extractor(r) for r in results]
                m    = statistics.mean(vals)
                s    = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"  {_fmt(m, s):<{col_w - 2}}"
                if metric_name == "Sharpe":
                    robustness[vi].append(m)
            print(row)

    print()
    print(sep)
    print(f"\n  Robustness Score = mean(Sharpe across all {len(REGIMES)} regimes)")
    for name, scores in zip(variant_names, robustness):
        mean_score = statistics.mean(scores) if scores else 0.0
        print(f"  {name}: {mean_score:+.4f}")
    print()

    return robustness


# ── W&B ──────────────────────────────────────────────────────────────────────

def _try_wandb_log(
    variant_names:   list[str],
    regime_results:  dict[str, list[list[EpisodeResult]]],
    conv_histories:  list[list[ConvergencePoint]],
    train_times:     list[float],
    robustness:      list[list[float]],
) -> None:
    if not WANDB_PROJECT:
        return
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        print("[ab_curriculum] wandb not installed — skipping W&B logging")
        return

    wandb.init(project=WANDB_PROJECT, name="ab_curriculum", reinit=True)
    summary: dict[str, float] = {}

    for regime_key, per_variant in regime_results.items():
        for name, results in zip(variant_names, per_variant):
            pnls   = [r.pnl    for r in results]
            sharps = [r.sharpe for r in results]
            summary[f"{name}/{regime_key}/pnl_mean"]       = statistics.mean(pnls)
            summary[f"{name}/{regime_key}/sharpe_mean"]    = statistics.mean(sharps)
            summary[f"{name}/{regime_key}/pos_sharpe_pct"] = statistics.mean(
                [100.0 * float(r.sharpe > 0.0) for r in results]
            )

    for name, rob in zip(variant_names, robustness):
        summary[f"{name}/robustness_score"] = statistics.mean(rob) if rob else 0.0

    for name, t in zip(variant_names, train_times):
        summary[f"{name}/train_time_s"] = t

    # Convergence curves as step-keyed scalars
    for name, hist in zip(variant_names, conv_histories):
        for pt in hist:
            wandb.log({f"{name}/convergence/mean_reward": pt.mean_reward}, step=pt.step)

    wandb.log(summary)
    wandb.finish()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    stage_steps = compute_stage_steps(TRAIN_STEPS)
    print(f"Training {len(VARIANTS)} variants × {TRAIN_STEPS:,} steps …")
    print(f"  no-curriculum: fixed medium parameters throughout")
    print(
        f"  curriculum:    {stage_steps[0]:,} easy → "
        f"{stage_steps[1]:,} medium → "
        f"{stage_steps[2]:,} hard"
    )
    print(f"  convergence eval every {EVAL_INTERVAL:,} steps")
    print(f"Evaluating {EVAL_EPISODES} episodes × {len(REGIMES)} regimes\n")

    trained_models:  list[PPO]                      = []
    conv_histories:  list[list[ConvergencePoint]]   = []
    train_times:     list[float]                    = []

    for variant in VARIANTS:
        print(f"  Training {variant.name} …", end="", flush=True)
        t0                = time.perf_counter()
        model, conv_hist  = _train(variant)
        elapsed           = time.perf_counter() - t0
        train_times.append(elapsed)
        conv_histories.append(conv_hist)
        print(f" done ({elapsed:.1f}s)")
        trained_models.append(model)

    variant_names = [v.name for v in VARIANTS]
    _print_convergence(variant_names, conv_histories)

    print()
    regime_results: dict[str, list[list[EpisodeResult]]] = {}
    for regime_key, regime_params in REGIMES.items():
        per_variant: list[list[EpisodeResult]] = []
        for variant, model in zip(VARIANTS, trained_models):
            print(
                f"  Evaluating {variant.name} / {REGIME_LABELS[regime_key]} …",
                end="",
                flush=True,
            )
            results = _evaluate_regime(model, regime_params)
            per_variant.append(results)
            print(" done")
        regime_results[regime_key] = per_variant

    robustness = _print_regime_report(variant_names, regime_results)
    _try_wandb_log(variant_names, regime_results, conv_histories, train_times, robustness)


if __name__ == "__main__":
    main()
