"""
Out-of-sample evaluation and baseline comparison for trained SAC agents.

Four agents are compared across 100 episodes:
  1. Naive Symmetric   — fixed action (γ=0.5, κ-offset=0)
  2. Static AS         — fixed action (γ=0.1, κ-offset=0)   [conservative]
  3. Optimized AS      — fixed action (γ=0.05, κ-offset=0)  [best from grid search]
  4. SAC               — trained policy (deterministic)

Metrics per agent
-----------------
PnL mean±std · Sharpe · Max Drawdown · Fill Rate · Inventory Std · Quote-to-Trade

Usage
-----
    uv run python -m quantflow.training.evaluate runs/sac/best_model.zip
    uv run python -m quantflow.training.evaluate runs/sac/best_model.zip --episodes 200
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import SAC

from quantflow.envs.market_making import MarketMakingEnv


# ── Per-episode result container ───────────────────────────────────────────────

@dataclass
class EpisodeResult:
    total_reward:   float
    final_pnl:      float
    sharpe:         float
    max_drawdown:   float
    fill_rate:      float
    inventory_std:  float
    quote_to_trade: float
    n_steps:        int


# ── Single episode rollout ─────────────────────────────────────────────────────

def rollout_episode(
    env:       MarketMakingEnv,
    policy_fn: Callable[[dict], np.ndarray],
    seed:      int,
) -> EpisodeResult:
    """
    Run one complete episode and return structured metrics.

    Parameters
    ----------
    env : MarketMakingEnv
        A freshly constructed environment (not yet reset).
    policy_fn : callable
        Maps an observation dict to a numpy action array.
    seed : int
        Seed passed to env.reset().
    """
    obs, _ = env.reset(seed=seed)

    step_rewards: list[float] = []
    pnl_history:  list[float] = []
    inv_history:  list[int]   = []
    fill_count                = 0
    prev_inv                  = 0
    done                      = False
    info: dict                = {}

    while not done:
        action                      = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done                        = term or trunc

        step_rewards.append(float(reward))
        pnl_history.append(float(info["pnl"]))
        inv = int(info["inventory"])
        inv_history.append(inv)
        if abs(inv - prev_inv) > 0:
            fill_count += 1
        prev_inv = inv

    n     = len(step_rewards)
    arr_r = np.array(step_rewards)
    arr_p = np.array(pnl_history)

    # Sharpe: annualised to episode length
    sharpe = float(arr_r.mean() / (arr_r.std() + 1e-9) * np.sqrt(n))

    # Max drawdown on PnL curve
    peak     = np.maximum.accumulate(arr_p)
    max_dd   = float(np.max(peak - arr_p)) if n > 0 else 0.0

    # Quote-to-trade: 2 resting orders per step / fills
    q2t = (2 * n) / (fill_count + 1e-9)

    return EpisodeResult(
        total_reward   = float(arr_r.sum()),
        final_pnl      = float(arr_p[-1]) if n > 0 else 0.0,
        sharpe         = sharpe,
        max_drawdown   = max_dd,
        fill_rate      = fill_count / n if n > 0 else 0.0,
        inventory_std  = float(np.std(inv_history)),
        quote_to_trade = q2t,
        n_steps        = n,
    )


# ── Baseline policies ──────────────────────────────────────────────────────────

def _naive_policy(obs: dict) -> np.ndarray:          # noqa: ARG001
    """γ=0.5, κ-offset=0 — wide symmetric quotes, no inventory adaptation."""
    return np.array([0.5, 0.0], dtype=np.float32)


def _static_as_policy(obs: dict) -> np.ndarray:      # noqa: ARG001
    """γ=0.1, κ-offset=0 — conservative AS, low risk aversion."""
    return np.array([0.1, 0.0], dtype=np.float32)


def _optimized_as_policy(obs: dict) -> np.ndarray:   # noqa: ARG001
    """γ=0.05, κ-offset=0 — best γ from Rust grid search."""
    return np.array([0.05, 0.0], dtype=np.float32)


# ── Aggregate helper ───────────────────────────────────────────────────────────

def _aggregate(results: list[EpisodeResult]) -> dict[str, float]:
    return {
        "pnl_mean":    float(np.mean([r.final_pnl      for r in results])),
        "pnl_std":     float(np.std( [r.final_pnl      for r in results])),
        "sharpe_mean": float(np.mean([r.sharpe          for r in results])),
        "sharpe_std":  float(np.std( [r.sharpe          for r in results])),
        "max_dd":      float(np.mean([r.max_drawdown    for r in results])),
        "fill_rate":   float(np.mean([r.fill_rate       for r in results])),
        "inv_std":     float(np.mean([r.inventory_std   for r in results])),
        "q2t":         float(np.mean([r.quote_to_trade  for r in results])),
    }


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate(
    model_path:  str | Path,
    n_episodes:  int                   = 100,
    env_config:  dict[str, Any] | None = None,
    seed_offset: int                   = 10_000,
    use_wandb:   bool                  = False,
    wandb_project: str                 = "quantflow-mm",
    wandb_name:    str | None          = None,
) -> dict[str, dict[str, float]]:
    """
    Evaluate a trained SAC model against three baselines.

    Parameters
    ----------
    model_path : path-like
        Path to a ``.zip`` file produced by ``SAC.save()``.
    n_episodes : int
        Number of out-of-sample episodes per agent.
    env_config : dict, optional
        Environment config overrides (uses defaults if omitted).
    seed_offset : int
        Seeds used are ``[seed_offset, seed_offset + n_episodes)``.
        Choose a range disjoint from training seeds to avoid look-ahead bias.
    use_wandb : bool
        Log results to W&B as a summary table.
    wandb_project : str
        W&B project name.
    wandb_name : str, optional
        W&B run name for the evaluation run.

    Returns
    -------
    dict
        Agent name → aggregate metric dict.
    """
    model = SAC.load(model_path)

    def sac_policy(obs: dict) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    agents: dict[str, Callable[[dict], np.ndarray]] = {
        "Naive Symmetric": _naive_policy,
        "Static AS":       _static_as_policy,
        "Optimized AS":    _optimized_as_policy,
        "SAC":             sac_policy,
    }

    all_results: dict[str, dict[str, float]] = {}

    for name, policy_fn in agents.items():
        print(f"  {name:<20} [{n_episodes} episodes] …", end="", flush=True)
        episode_results = []
        for i in range(n_episodes):
            env = MarketMakingEnv(env_config)
            episode_results.append(
                rollout_episode(env, policy_fn, seed=seed_offset + i)
            )
        all_results[name] = _aggregate(episode_results)
        m = all_results[name]
        print(f" PnL={m['pnl_mean']:+.2f}±{m['pnl_std']:.2f}  Sharpe={m['sharpe_mean']:+.3f}")

    if use_wandb:
        _log_wandb(all_results, wandb_project, wandb_name)

    return all_results


def _log_wandb(
    results:  dict[str, dict[str, float]],
    project:  str,
    run_name: str | None,
) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping W&B logging")
        return

    wandb.init(project=project, name=run_name or "eval", job_type="evaluation")

    # One row per agent as a W&B Table
    cols    = ["agent", "pnl_mean", "pnl_std", "sharpe_mean", "sharpe_std",
               "max_dd", "fill_rate", "inv_std", "q2t"]
    table   = wandb.Table(columns=cols)

    for name, m in results.items():
        table.add_data(
            name,
            m["pnl_mean"], m["pnl_std"],
            m["sharpe_mean"], m["sharpe_std"],
            m["max_dd"], m["fill_rate"], m["inv_std"], m["q2t"],
        )

    # Flat summary scalars (SAC vs best baseline)
    sac = results.get("SAC", {})
    opt = results.get("Optimized AS", {})
    wandb.summary["sac_pnl_mean"]         = sac.get("pnl_mean", 0.0)
    wandb.summary["sac_sharpe_mean"]      = sac.get("sharpe_mean", 0.0)
    wandb.summary["delta_pnl_vs_opt_as"]  = sac.get("pnl_mean", 0.0) - opt.get("pnl_mean", 0.0)
    wandb.summary["delta_sharpe_vs_opt_as"] = (
        sac.get("sharpe_mean", 0.0) - opt.get("sharpe_mean", 0.0)
    )
    wandb.log({"eval_results": table})
    wandb.finish()


# ── Pretty-print table ─────────────────────────────────────────────────────────

def print_table(results: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'Agent':<20} "
        f"{'PnL (mean±std)':>18} "
        f"{'Sharpe':>12} "
        f"{'MaxDD':>9} "
        f"{'FillRate':>10} "
        f"{'InvStd':>8} "
        f"{'Q2T':>7}"
    )
    rule = "─" * len(header)
    print(rule)
    print(header)
    print(rule)
    for name, m in results.items():
        print(
            f"{name:<20} "
            f"{m['pnl_mean']:>+9.2f}±{m['pnl_std']:<7.2f} "
            f"{m['sharpe_mean']:>+10.3f}±{m['sharpe_std']:<1.3f} "
            f"{m['max_dd']:>9.3f} "
            f"{m['fill_rate']:>10.4f} "
            f"{m['inv_std']:>8.2f} "
            f"{m['q2t']:>7.1f}"
        )
    print(rule)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate trained SAC model against baselines"
    )
    p.add_argument("model_path",       type=str,
                   help="Path to saved SAC model (.zip)")
    p.add_argument("--episodes",       type=int,  default=100,
                   help="Number of out-of-sample episodes (default 100)")
    p.add_argument("--seed-offset",    type=int,  default=10_000,
                   help="Starting seed (default 10000; keep disjoint from training)")
    p.add_argument("--wandb",          action="store_true",
                   help="Log results to Weights & Biases")
    p.add_argument("--wandb-project",  type=str,  default="quantflow-mm")
    p.add_argument("--wandb-name",     type=str,  default=None)
    args = p.parse_args()

    print(f"\nOut-of-sample evaluation — {args.episodes} episodes\n")
    results = evaluate(
        model_path    = args.model_path,
        n_episodes    = args.episodes,
        seed_offset   = args.seed_offset,
        use_wandb     = args.wandb,
        wandb_project = args.wandb_project,
        wandb_name    = args.wandb_name,
    )
    print()
    print_table(results)


if __name__ == "__main__":
    main()
