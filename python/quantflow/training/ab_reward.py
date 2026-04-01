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

import math
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
    name:           str
    reward_version: str
    extra_cfg:      dict[str, Any] = field(default_factory=dict)


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
    pnl:              float
    max_inv:          int
    inv_std:          float
    n_steps:          int
    n_fills:          int          # steps where ≥1 agent fill occurred
    sharpe:           float
    max_drawdown:     float        # largest peak-to-trough drop in episode PnL
    # v2-only components (0.0 for v1)
    mean_rt_bonus:    float
    mean_term_pen:    float


def _evaluate_episode(model: PPO, variant: VariantSpec, seed: int) -> EpisodeResult:
    env   = _make_env(variant, seed)
    obs, _ = env.reset(seed=seed)
    done   = False

    inv_history:  list[int]   = []
    pnl_history:  list[float] = [0.0]
    n_fills       = 0          # steps where ≥1 agent fill occurred
    rt_bonuses:   list[float] = []
    term_pens:    list[float] = []
    prev_inv      = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        inv  = info["inventory"]
        pnl  = info["pnl"]
        inv_history.append(abs(inv))
        pnl_history.append(pnl)

        # Count fill EVENTS (not lots).
        # A fill occurred if inventory changed (buy or sell fill) or if
        # fill_pnl is non-zero (buy fill that was netted by a same-step sell).
        if inv != prev_inv or info.get("fill_pnl", 0.0) != 0.0:
            n_fills += 1
        prev_inv = inv

        comps = info.get("reward_components", {})
        rt_bonuses.append(comps.get("rt_bonus", 0.0))
        term_pens.append(comps.get("terminal_penalty", 0.0))

    # Sharpe: step-to-step PnL returns
    rets = [b - a for a, b in zip(pnl_history[:-1], pnl_history[1:])]
    if len(rets) > 1:
        m = statistics.mean(rets)
        v = statistics.variance(rets)
        sharpe = (m / math.sqrt(v)) if v > 1e-10 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown over episode PnL curve
    peak = pnl_history[0]
    max_dd = 0.0
    for p in pnl_history:
        if p > peak:
            peak = p
        dd = peak - p
        if dd > max_dd:
            max_dd = dd

    inv_std = statistics.stdev(inv_history) if len(inv_history) > 1 else 0.0

    return EpisodeResult(
        pnl           = pnl_history[-1],
        max_inv       = max(inv_history) if inv_history else 0,
        inv_std       = inv_std,
        n_steps       = len(rets),
        n_fills       = n_fills,
        sharpe        = sharpe,
        max_drawdown  = max_dd,
        mean_rt_bonus = statistics.mean(rt_bonuses) if rt_bonuses else 0.0,
        mean_term_pen = statistics.mean(term_pens)  if term_pens  else 0.0,
    )


def _evaluate(model: PPO, variant: VariantSpec) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for i in range(EVAL_SEEDS):
        results.append(_evaluate_episode(model, variant, FIRST_EVAL_SEED + i))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> tuple[float, float]:
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _print_table(
    variant_names: list[str],
    metrics: dict[str, list[list[float]]],
    v2_only_metrics: set[str],
) -> None:
    col_w  = 26
    name_w = 22

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
        tag = "  [v2]" if metric in v2_only_metrics else ""
        row = f"{metric + tag:<{name_w}}"
        for i, values in enumerate(per_variant):
            if metric in v2_only_metrics and variant_names[i] != "v2":
                row += f"  {'—':>{col_w - 2}}"
            else:
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
    print(f"Evaluating on {EVAL_SEEDS} episodes "
          f"(seeds {FIRST_EVAL_SEED}–{FIRST_EVAL_SEED + EVAL_SEEDS - 1})\n")

    trained_models: list[PPO]                 = []
    eval_results:   list[list[EpisodeResult]] = []
    train_times:    list[float]               = []

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

    # Fill rate: round_trips / n_steps (proxy; each round-trip = 1 buy + 1 sell fill)
    fill_rates = [
        [r.n_fills / max(r.n_steps, 1) * 100.0 for r in res]
        for res in eval_results
    ]
    win_rates = [
        [100.0 if r.pnl > 0.0 else 0.0 for r in res]
        for res in eval_results
    ]

    metrics: dict[str, list[list[float]]] = {
        "PnL":              [[r.pnl          for r in res] for res in eval_results],
        "Sharpe":           [[r.sharpe        for r in res] for res in eval_results],
        "Max Drawdown":     [[r.max_drawdown  for r in res] for res in eval_results],
        "Fill Rate (%)":    fill_rates,
        "Inventory Std":    [[r.inv_std       for r in res] for res in eval_results],
        "Max |inv|":        [[float(r.max_inv) for r in res] for res in eval_results],
        "Win Rate (%)":     win_rates,
        "Mean RT Bonus":    [[r.mean_rt_bonus for r in res] for res in eval_results],
        "Mean Term Pen":    [[r.mean_term_pen for r in res] for res in eval_results],
        "Train time (s)":   [[t]                            for t in train_times],
    }

    v2_only: set[str] = {"Mean RT Bonus", "Mean Term Pen"}
    variant_names = [v.name for v in VARIANTS]
    _print_table(variant_names, metrics, v2_only)

    mean_pnls = [statistics.mean(m) for m in metrics["PnL"]]
    best_idx  = int(np.argmax(mean_pnls))
    print(f"  Winner by mean PnL: reward={VARIANTS[best_idx].name} "
          f"({mean_pnls[best_idx]:+.4f})\n")

    _try_wandb_log(variant_names, metrics, train_times)


if __name__ == "__main__":
    main()
