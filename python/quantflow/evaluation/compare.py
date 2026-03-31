"""
Run all four strategies for N out-of-sample episodes and save results.

Output files
------------
``<output_dir>/results.parquet``      — per-episode summary metrics
``<output_dir>/trajectories.parquet`` — per-step (pnl, inventory, reward)

Usage
-----
    uv run python -m quantflow.evaluation.compare \\
        --model-path runs/sac_test/best_model.zip \\
        --n-episodes 50 --seed-start 1000 \\
        --output-dir results/evaluation
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from quantflow.envs.market_making import MarketMakingEnv


# ── Constants ─────────────────────────────────────────────────────────────────

AGENT_COLORS: dict[str, str] = {
    "SAC":             "#2196F3",
    "Optimized AS":    "#4CAF50",
    "Static AS":       "#FF9800",
    "Naive Symmetric": "#9E9E9E",
}

AGENT_ORDER: list[str] = [
    "Naive Symmetric",
    "Static AS",
    "Optimized AS",
    "SAC",
]


# ── Fixed policies ─────────────────────────────────────────────────────────────

def _naive_policy(obs: dict) -> np.ndarray:       # noqa: ARG001
    return np.array([0.5,  0.0], dtype=np.float32)

def _static_as_policy(obs: dict) -> np.ndarray:   # noqa: ARG001
    return np.array([0.1,  0.0], dtype=np.float32)

def _optim_as_policy(obs: dict) -> np.ndarray:    # noqa: ARG001
    return np.array([0.05, 0.0], dtype=np.float32)


# ── Rollout ───────────────────────────────────────────────────────────────────

def _rollout(
    env:       MarketMakingEnv,
    policy_fn: Callable[[dict], np.ndarray],
    seed:      int,
) -> tuple[dict, pd.DataFrame]:
    """
    Run one episode; return (summary_dict, trajectory_df).

    PnL decomposition
    -----------------
    total_pnl     = cash + inventory × final_mid
    inventory_pnl = inventory × (final_mid − initial_mid)   [price drift]
    spread_pnl    = total_pnl − inventory_pnl                [spread capture]
    """
    obs, _ = env.reset(seed=seed)
    initial_mid: float = env._sim.mid_price() or env.initial_mid

    step_rewards: list[float] = []
    pnl_history:  list[float] = []
    inv_history:  list[int]   = []
    sim_times:    list[float] = []
    fill_count = 0
    prev_inv   = 0
    done       = False
    info: dict = {}

    while not done:
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        step_rewards.append(float(reward))
        pnl_history.append(float(info["pnl"]))
        inv = int(info["inventory"])
        inv_history.append(inv)
        sim_times.append(float(info.get("sim_time", 0.0)))
        if abs(inv - prev_inv) > 0:
            fill_count += 1
        prev_inv = inv

    n     = len(step_rewards)
    arr_r = np.array(step_rewards, dtype=np.float64)
    arr_p = np.array(pnl_history,  dtype=np.float64)

    sharpe  = float(arr_r.mean() / (arr_r.std() + 1e-9) * np.sqrt(n))
    peak    = np.maximum.accumulate(arr_p)
    max_dd  = float(np.max(peak - arr_p)) if n > 0 else 0.0

    final_pnl     = float(arr_p[-1]) if n > 0 else 0.0
    final_mid     = float(info.get("mid",       initial_mid))
    final_inv     = int  (info.get("inventory", 0))
    inventory_pnl = final_inv * (final_mid - initial_mid)
    spread_pnl    = final_pnl - inventory_pnl

    summary = {
        "final_pnl":      final_pnl,
        "total_reward":   float(arr_r.sum()),
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "fill_rate":      fill_count / n if n > 0 else 0.0,
        "inventory_std":  float(np.std(inv_history)),
        "quote_to_trade": (2 * n) / (fill_count + 1e-9),
        "spread_pnl":     spread_pnl,
        "inventory_pnl":  inventory_pnl,
        "n_steps":        n,
    }

    traj_df = pd.DataFrame({
        "step":           np.arange(n),
        "sim_time":       sim_times,
        "cumulative_pnl": arr_p,
        "inventory":      inv_history,
        "reward":         step_rewards,
    })

    return summary, traj_df


# ── Public API ────────────────────────────────────────────────────────────────

def compare(
    model_path:  str | Path,
    n_episodes:  int                   = 50,
    seed_start:  int                   = 1000,
    env_config:  dict[str, Any] | None = None,
    output_dir:  str | Path            = "results/evaluation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all four strategies and persist Parquet files.

    Returns
    -------
    summary_df : DataFrame
        One row per (agent, episode).
    traj_df : DataFrame
        One row per (agent, episode, step).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SAC.load(model_path)

    def sac_policy(obs: dict) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    policies: dict[str, Callable] = {
        "Naive Symmetric": _naive_policy,
        "Static AS":       _static_as_policy,
        "Optimized AS":    _optim_as_policy,
        "SAC":             sac_policy,
    }

    summary_rows:  list[dict]          = []
    traj_frames:   list[pd.DataFrame]  = []

    for name in AGENT_ORDER:
        policy_fn = policies[name]
        print(f"  {name:<20} [{n_episodes} eps] ", end="", flush=True)
        ep_pnls: list[float] = []

        for i in range(n_episodes):
            env = MarketMakingEnv(env_config)
            summary, traj = _rollout(env, policy_fn, seed=seed_start + i)
            summary["agent"]      = name
            summary["episode_id"] = i
            summary_rows.append(summary)

            traj["agent"]      = name
            traj["episode_id"] = i
            traj_frames.append(traj)
            ep_pnls.append(summary["final_pnl"])

        mu, sigma = float(np.mean(ep_pnls)), float(np.std(ep_pnls))
        print(f"pnl={mu:+.2f}±{sigma:.2f}")

    summary_df = pd.DataFrame(summary_rows)
    traj_df    = pd.concat(traj_frames, ignore_index=True)

    summary_df.to_parquet(output_dir / "results.parquet",      index=False)
    traj_df.to_parquet   (output_dir / "trajectories.parquet", index=False)

    print(f"\nSaved → {output_dir / 'results.parquet'}")
    print(f"Saved → {output_dir / 'trajectories.parquet'}")

    return summary_df, traj_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compare four market-making strategies")
    p.add_argument("--model-path",  required=True)
    p.add_argument("--n-episodes",  type=int, default=50)
    p.add_argument("--seed-start",  type=int, default=1000)
    p.add_argument("--output-dir",  default="results/evaluation")
    args = p.parse_args()

    print(f"\nStrategy comparison — {args.n_episodes} episodes per agent\n")
    compare(
        model_path = args.model_path,
        n_episodes = args.n_episodes,
        seed_start = args.seed_start,
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
