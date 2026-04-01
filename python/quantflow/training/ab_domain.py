"""
A/B comparison: training without vs. with DomainRandomizer.

Both variants use reward_version="v2" and obs_version="v2".
The only variable is whether training episodes are randomised over Hawkes
simulator parameters (market regime).

Evaluation runs across three fixed regimes so that regime-specific and
aggregate robustness can be compared side-by-side.

Usage
-----
    python -m quantflow.training.ab_domain

Optional env vars
-----------------
    WANDB_PROJECT     — W&B project name (set to "" to disable)
    AB_STEPS          — training steps per variant (default 50_000)
    AB_EVAL_EPISODES  — evaluation episodes per regime (default 10)
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
from stable_baselines3.common.env_util import make_vec_env

from quantflow.envs.domain_randomizer import DomainRandomizer
from quantflow.envs.market_making import MarketMakingEnv


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_STEPS:    int = int(os.environ.get("AB_STEPS", 50_000))
EVAL_EPISODES:  int = int(os.environ.get("AB_EVAL_EPISODES", 10))
FIRST_EVAL_SEED: int = 3_000
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

# Evaluation regimes: each specifies simulator params injected before reset.
REGIMES: dict[str, dict[str, float]] = {
    "STANDARD":  {},
    "HIGH-VOL":  {"sigma_scale": 2.0, "mu_scale": 1.3},
    "LOW-LIQ":   {"mu_scale":    0.5},   # alpha_scale=0.5 not applied (Hawkes only)
}

REGIME_LABELS: dict[str, str] = {
    "STANDARD": "Standard",
    "HIGH-VOL":  "High-Vol  (σ×2, μ×1.3)",
    "LOW-LIQ":   "Low-Liq   (μ×0.5)",
}


@dataclass
class VariantSpec:
    name:           str
    use_domain_rand: bool
    extra_cfg:      dict[str, Any] = field(default_factory=dict)


VARIANTS: list[VariantSpec] = [
    VariantSpec(name="no-DR",   use_domain_rand=False),
    VariantSpec(name="with-DR", use_domain_rand=True),
]


# ── Environment factories ─────────────────────────────────────────────────────

def _make_base_env(seed: int) -> MarketMakingEnv:
    return MarketMakingEnv(config={**_BASE_CFG, "seed": seed})


def _make_train_env(variant: VariantSpec, seed: int) -> gym.Env:
    base = _make_base_env(seed)
    if variant.use_domain_rand:
        return DomainRandomizer(base)
    return base


# gym import needed for type annotation
import gymnasium as gym  # noqa: E402


# ── Training ──────────────────────────────────────────────────────────────────

def _train(variant: VariantSpec) -> PPO:
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
    model.learn(total_timesteps=TRAIN_STEPS)
    return model


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
    env    = _make_base_env(seed)
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

    return EpisodeResult(
        pnl    = pnl_history[-1],
        sharpe = sharpe,
    )


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


def _print_report(
    variant_names: list[str],
    regime_results: dict[str, list[list[EpisodeResult]]],
) -> tuple[list[float], list[float]]:
    """Print the comparison table. Returns (mean_sharpes_no_dr, mean_sharpes_with_dr)."""
    col_w  = 30
    name_w = 40
    sep    = "─" * (name_w + col_w * len(variant_names))

    print()
    print(sep)
    print(f"  A/B Domain Randomization  (reward=v2, obs=v2 for both)")
    print(sep)
    print(f"{'':>{name_w}}" + "".join(
        f"  {n + ' (mean±std)':<{col_w - 2}}" for n in variant_names
    ))
    print(sep)

    robustness: list[list[float]] = [[] for _ in variant_names]

    for regime_key, per_variant in regime_results.items():
        label = REGIME_LABELS[regime_key]
        n     = EVAL_EPISODES
        print(f"\n{label.upper()} REGIME ({n} episodes)")

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

    # Robustness score
    print(f"\n  Robustness Score = mean(Sharpe across all {len(REGIMES)} regimes)")
    for name, scores in zip(variant_names, robustness):
        mean_score = statistics.mean(scores) if scores else 0.0
        print(f"  {name}: {mean_score:+.4f}")
    print()

    return robustness


# ── W&B ──────────────────────────────────────────────────────────────────────

def _try_wandb_log(
    variant_names:  list[str],
    regime_results: dict[str, list[list[EpisodeResult]]],
    train_times:    list[float],
    robustness:     list[list[float]],
) -> None:
    if not WANDB_PROJECT:
        return
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        print("[ab_domain] wandb not installed — skipping W&B logging")
        return

    wandb.init(project=WANDB_PROJECT, name="ab_domain", reinit=True)
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
    wandb.log(summary)
    wandb.finish()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Training {len(VARIANTS)} variants × {TRAIN_STEPS:,} steps …")
    print(f"  no-DR:   fixed simulator parameters")
    print(f"  with-DR: DomainRandomizer (μ±20%, σ∈[0.7,1.5], mid∈[95,105])")
    print(f"Evaluating {EVAL_EPISODES} episodes × {len(REGIMES)} regimes\n")

    trained_models: list[PPO] = []
    train_times:    list[float] = []

    for variant in VARIANTS:
        print(f"  Training {variant.name} …", end="", flush=True)
        t0      = time.perf_counter()
        model   = _train(variant)
        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)
        print(f" done ({elapsed:.1f}s)")
        trained_models.append(model)

    print()
    regime_results: dict[str, list[list[EpisodeResult]]] = {}
    for regime_key, regime_params in REGIMES.items():
        per_variant: list[list[EpisodeResult]] = []
        for variant, model in zip(VARIANTS, trained_models):
            print(f"  Evaluating {variant.name} / {REGIME_LABELS[regime_key]} …",
                  end="", flush=True)
            results = _evaluate_regime(model, regime_params)
            per_variant.append(results)
            print(" done")
        regime_results[regime_key] = per_variant

    variant_names = [v.name for v in VARIANTS]
    robustness    = _print_report(variant_names, regime_results)
    _try_wandb_log(variant_names, regime_results, train_times, robustness)


if __name__ == "__main__":
    main()
