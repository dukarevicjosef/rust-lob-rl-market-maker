"""
A/B comparison of reward v1 vs reward v2.

Trains two identical PPO agents for 50_000 steps each (seed=42), then
evaluates both on 20 held-out episodes (seeds 2000–2019) and prints a
formatted comparison table.

Usage
-----
    python -m quantflow.training.ab_reward

Optional env vars
-----------------
    WANDB_PROJECT   — W&B project name (set to "" to disable W&B logging)
    AB_STEPS        — training steps per variant (default 50_000)
    AB_EVAL_SEEDS   — number of evaluation seeds (default 20)
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from quantflow.envs.market_making import MarketMakingEnv


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_STEPS: int = int(os.environ.get("AB_STEPS", 50_000))
EVAL_SEEDS:  int = int(os.environ.get("AB_EVAL_SEEDS", 20))
FIRST_EVAL_SEED: int = 2_000
TRAIN_SEED:  int = 42

WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "quantflow-ab")

_BASE_CFG: dict[str, Any] = {
    "t_max":           3_600.0,
    "episode_length":  2_000,
    "events_per_step": 50,
    "warm_up_events":  500,
    "inventory_limit": 50,
}


@dataclass
class VariantSpec:
    name:          str
    reward_version: str
    extra_cfg:     dict[str, Any] = field(default_factory=dict)


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name           = "v1",
        reward_version = "v1",
        extra_cfg      = {},
    ),
    VariantSpec(
        name           = "v2",
        reward_version = "v2",
        extra_cfg      = {
            "reward_config": {
                "phi":                  0.01,
                "psi":                  0.001,
                "lambda_breach":        1.0,
                "rt_weight":            0.5,
                "asymmetric_strength":  0.3,
                "terminal_weight":      2.0,
                "reward_version":       "v2",
            },
        },
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_env(variant: VariantSpec, seed: int) -> MarketMakingEnv:
    cfg = {
        **_BASE_CFG,
        "seed": seed,
        "reward_config": {"reward_version": variant.reward_version},
        **variant.extra_cfg,
    }
    return MarketMakingEnv(config=cfg)


def _train(variant: VariantSpec) -> PPO:
    """Train a PPO agent for TRAIN_STEPS steps and return it."""
    vec_env = make_vec_env(
        lambda: _make_env(variant, TRAIN_SEED),
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
    model.learn(total_timesteps=TRAIN_STEPS)
    return model


@dataclass
class EpisodeResult:
    pnl:       float
    fills:     int
    max_inv:   int
    n_steps:   int


def _evaluate_episode(model: PPO, variant: VariantSpec, seed: int) -> EpisodeResult:
    env = _make_env(variant, seed)
    obs, _ = env.reset(seed=seed)
    done    = False
    fills   = 0
    max_inv = 0
    n_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done     = terminated or truncated
        max_inv  = max(max_inv, abs(info["inventory"]))
        n_steps += 1

    return EpisodeResult(
        pnl     = float(info["pnl"]),
        fills   = fills,
        max_inv = max_inv,
        n_steps = n_steps,
    )


def _evaluate(model: PPO, variant: VariantSpec) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for i in range(EVAL_SEEDS):
        seed = FIRST_EVAL_SEED + i
        results.append(_evaluate_episode(model, variant, seed))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, stdev)."""
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _print_table(
    variant_names: list[str],
    metrics: dict[str, list[list[float]]],
) -> None:
    """
    Print a formatted comparison table.

    metrics: {metric_name: [[values_v1], [values_v2], ...]}
    """
    col_w  = 24
    name_w = 20

    # Header
    header = f"{'Metric':<{name_w}}" + "".join(
        f"{'  ' + n + ' (mean±std)':<{col_w}}" for n in variant_names
    )
    sep = "─" * len(header)
    print()
    print(sep)
    print("  A/B Reward Comparison")
    print(sep)
    print(header)
    print(sep)

    for metric, per_variant in metrics.items():
        row = f"{metric:<{name_w}}"
        for values in per_variant:
            m, s = _stats(values)
            row += f"  {m:+.4f} ± {s:.4f}".ljust(col_w)
        print(row)

    print(sep)
    print()


# ── W&B integration ───────────────────────────────────────────────────────────

def _try_wandb_log(
    variant_names: list[str],
    metrics: dict[str, list[list[float]]],
    train_times: list[float],
) -> None:
    if not WANDB_PROJECT:
        return
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        print("[ab_reward] wandb not installed — skipping W&B logging")
        return

    wandb.init(project=WANDB_PROJECT, name="ab_reward", reinit=True)
    summary: dict[str, float] = {}
    for metric, per_variant in metrics.items():
        for name, values in zip(variant_names, per_variant):
            m, s = _stats(values)
            summary[f"{name}/{metric}/mean"] = m
            summary[f"{name}/{metric}/std"]  = s
    for name, t in zip(variant_names, train_times):
        summary[f"{name}/train_time_s"] = t
    wandb.log(summary)
    wandb.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Training {len(VARIANTS)} variants × {TRAIN_STEPS:,} steps each …")
    print(f"Evaluating on {EVAL_SEEDS} episodes (seeds {FIRST_EVAL_SEED}–"
          f"{FIRST_EVAL_SEED + EVAL_SEEDS - 1})\n")

    trained_models: list[PPO]              = []
    eval_results:   list[list[EpisodeResult]] = []
    train_times:    list[float]            = []

    for variant in VARIANTS:
        print(f"  Training reward={variant.reward_version} …", end="", flush=True)
        t0 = time.perf_counter()
        model = _train(variant)
        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)
        print(f" done ({elapsed:.1f}s)")
        trained_models.append(model)

    print()
    for variant, model in zip(VARIANTS, trained_models):
        print(f"  Evaluating {variant.name} …", end="", flush=True)
        results = _evaluate(model, variant)
        eval_results.append(results)
        print(f" done")

    # Collect metrics
    metrics: dict[str, list[list[float]]] = {
        "PnL":         [[r.pnl     for r in res] for res in eval_results],
        "Max |inv|":   [[float(r.max_inv) for r in res] for res in eval_results],
        "Steps/ep":    [[float(r.n_steps) for r in res] for res in eval_results],
        "Train time s":[[t]                              for t in train_times],
    }

    variant_names = [v.name for v in VARIANTS]
    _print_table(variant_names, metrics, )

    # Determine winner by mean PnL
    mean_pnls = [statistics.mean(m) for m in metrics["PnL"]]
    best_idx  = int(np.argmax(mean_pnls))
    print(f"  Winner by mean PnL: reward={VARIANTS[best_idx].name} "
          f"({mean_pnls[best_idx]:+.4f})\n")

    _try_wandb_log(variant_names, metrics, train_times)


if __name__ == "__main__":
    main()
