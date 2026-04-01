"""
SAC training pipeline for the Hawkes-driven market-making environment.

Observation space is a gymnasium Dict — SB3 requires ``MultiInputPolicy``
(not ``MlpPolicy``) which routes each sub-space through a CombinedExtractor
before passing the concatenated vector to the MLP trunk.

TensorBoard is not required. Metrics are logged to stdout and optionally
to Weights & Biases via direct ``wandb.log()`` calls (no TB sync needed).

Usage
-----
    # Quick run (50k steps)
    uv run python -m quantflow.training.train --timesteps 50000

    # Full training with W&B
    uv run python -m quantflow.training.train --wandb --wandb-project quantflow-mm

    # Final 2M-step run
    uv run python -m quantflow.training.train --final --run-dir runs/sac_final
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from quantflow.envs.market_making import MarketMakingEnv

# W&B is optional — imported lazily
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Hyperparameter config ──────────────────────────────────────────────────────

@dataclass
class SACConfig:
    # Policy
    policy:        str         = "MultiInputPolicy"  # required for Dict obs
    # SAC hyperparameters
    learning_rate: float       = 3e-4
    buffer_size:   int         = 1_000_000
    batch_size:    int         = 256
    tau:           float       = 0.005
    gamma:         float       = 0.99
    ent_coef:      str | float = "auto"              # automatic entropy tuning
    net_arch:      list[int]   = field(default_factory=lambda: [256, 256])
    # Training duration
    total_timesteps: int = 1_000_000
    # Callback / evaluation
    eval_freq:          int   = 10_000
    n_eval_episodes:    int   = 5
    # AS baseline for delta logging
    as_baseline_gamma:        float = 0.1
    as_baseline_kappa_offset: float = 0.0


# Default environment config — reward v2 tuned parameters
_DEFAULT_ENV_CFG: dict[str, Any] = {
    "reward_config": {
        "phi":                 0.01,
        "psi":                 0.001,
        "lambda_breach":       1.0,
        "rt_weight":           0.5,
        "asymmetric_strength": 0.3,
        "terminal_weight":     2.0,
        "reward_version":      "v2",
    },
}

# Shorter episode config used inside callbacks to keep evaluation fast.
# normalize_reward=False so eval metrics reflect true economic values,
# not the normalized signal seen by the agent during training.
_EVAL_ENV_CFG: dict[str, Any] = {
    "episode_length":   500,
    "events_per_step":  50,
    "warm_up_events":   500,
    "normalize_reward": False,
}


# ── Callbacks ─────────────────────────────────────────────────────────────────

class QuantflowEvalCallback(BaseCallback):
    """
    Evaluate the current SAC policy and a static AS baseline every
    ``eval_freq`` environment steps.

    Logged keys
    -----------
    eval/mean_reward, eval/mean_pnl, eval/mean_sharpe, eval/mean_inventory_std
    baseline/mean_reward, baseline/mean_pnl, baseline/mean_sharpe
    delta/pnl, delta/sharpe
    """

    def __init__(
        self,
        eval_env_config: dict[str, Any],
        eval_freq:       int         = 10_000,
        n_eval_episodes: int         = 5,
        save_path:       Path | None = None,
        as_gamma:        float       = 0.1,
        as_kappa_offset: float       = 0.0,
        use_wandb:       bool        = False,
        verbose:         int         = 1,
    ) -> None:
        super().__init__(verbose)
        self._eval_env_cfg  = eval_env_config
        self._eval_freq     = eval_freq
        self._n_eval        = n_eval_episodes
        self._save_path     = Path(save_path) if save_path else None
        self._as_action     = np.array([as_gamma, as_kappa_offset], dtype=np.float32)
        self._use_wandb     = use_wandb and _WANDB_AVAILABLE
        self._best_reward   = -np.inf

    # ── BaseCallback interface ─────────────────────────────────────────────────

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        sac_m = self._run_episodes(
            lambda obs: self.model.predict(obs, deterministic=True)[0]
        )
        as_m = self._run_episodes(lambda _: self._as_action)

        metrics = {
            "eval/mean_reward":        sac_m["mean_reward"],
            "eval/mean_pnl":           sac_m["mean_pnl"],
            "eval/mean_sharpe":        sac_m["mean_sharpe"],
            "eval/mean_inventory_std": sac_m["mean_inv_std"],
            "baseline/mean_reward":    as_m["mean_reward"],
            "baseline/mean_pnl":       as_m["mean_pnl"],
            "baseline/mean_sharpe":    as_m["mean_sharpe"],
            "delta/pnl":               sac_m["mean_pnl"]    - as_m["mean_pnl"],
            "delta/sharpe":            sac_m["mean_sharpe"] - as_m["mean_sharpe"],
        }

        # SB3 stdout logger (no TensorBoard needed)
        for k, v in metrics.items():
            self.logger.record(k, v)
        self.logger.dump(step=self.num_timesteps)

        # Direct W&B logging (independent of TensorBoard)
        if self._use_wandb:
            _wandb.log(metrics, step=self.num_timesteps)

        if self.verbose >= 1:
            print(
                f"[eval @ {self.num_timesteps:,d}] "
                f"reward={sac_m['mean_reward']:+.3f}  "
                f"pnl={sac_m['mean_pnl']:+.2f}  "
                f"sharpe={sac_m['mean_sharpe']:+.3f} | "
                f"ΔPnL={sac_m['mean_pnl']-as_m['mean_pnl']:+.2f}  "
                f"ΔSharpe={sac_m['mean_sharpe']-as_m['mean_sharpe']:+.3f}"
            )

        if sac_m["mean_reward"] > self._best_reward and self._save_path:
            self._best_reward = sac_m["mean_reward"]
            self.model.save(self._save_path / "best_model")
            if self.verbose >= 1:
                print(f"  → new best ({self._best_reward:.3f}) — model saved")

        return True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _run_episodes(self, policy_fn) -> dict[str, float]:
        rewards, pnls, sharpes, inv_stds = [], [], [], []

        for seed in range(self._n_eval):
            env = MarketMakingEnv(self._eval_env_cfg)
            obs, _ = env.reset(seed=1000 + seed)
            raw_rewards, inv_hist = [], []
            done = False
            info: dict = {}

            while not done:
                obs, _rew, term, trunc, info = env.step(policy_fn(obs))
                # Use raw_reward for all eval metrics — normalize_reward=False
                # in eval env so raw_reward == rew, but explicit for safety.
                raw_rewards.append(float(info.get("raw_reward", _rew)))
                inv_hist.append(int(info["inventory"]))
                done = term or trunc

            arr = np.array(raw_rewards)
            n   = len(arr)
            rewards.append(float(arr.sum()))
            pnls.append(float(info["pnl"]))
            sharpes.append(float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(n)))
            inv_stds.append(float(np.std(inv_hist)))

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_pnl":    float(np.mean(pnls)),
            "mean_sharpe": float(np.mean(sharpes)),
            "mean_inv_std": float(np.mean(inv_stds)),
        }


# ── Training entry point ───────────────────────────────────────────────────────

def train(
    sac_cfg:    SACConfig,
    env_config: dict[str, Any] | None = None,
    run_dir:    Path | str | None     = None,
    use_wandb:  bool                  = False,
    wandb_project: str                = "quantflow-mm",
    wandb_name:    str | None         = None,
) -> SAC:
    """
    Train a SAC market-making agent.

    Parameters
    ----------
    sac_cfg : SACConfig
        All SAC hyperparameters.
    env_config : dict, optional
        Overrides for ``_DEFAULT_CONFIG`` in ``MarketMakingEnv``.
    run_dir : path-like, optional
        Root directory for TensorBoard logs and checkpoints.
        Defaults to ``runs/sac``.
    use_wandb : bool
        Enable Weights & Biases logging (requires ``wandb`` to be installed
        and ``wandb login`` to have been run).
    wandb_project : str
        W&B project name.
    wandb_name : str, optional
        W&B run name; auto-generated if omitted.

    Returns
    -------
    SAC
        The trained model (also saved to ``run_dir/final_model.zip``).
    """
    run_dir = Path(run_dir or "runs/sac")
    run_dir.mkdir(parents=True, exist_ok=True)

    merged_env_cfg = {**_DEFAULT_ENV_CFG, **(env_config or {})}
    env = MarketMakingEnv(merged_env_cfg)

    model = SAC(
        policy        = sac_cfg.policy,
        env           = env,
        learning_rate = sac_cfg.learning_rate,
        buffer_size   = sac_cfg.buffer_size,
        batch_size    = sac_cfg.batch_size,
        tau           = sac_cfg.tau,
        gamma         = sac_cfg.gamma,
        ent_coef      = sac_cfg.ent_coef,
        policy_kwargs = {"net_arch": sac_cfg.net_arch},
        verbose       = 1,
        # tensorboard_log omitted — not required; W&B logs directly
    )

    wandb_active = False
    if use_wandb:
        if _WANDB_AVAILABLE:
            _wandb.init(
                project          = wandb_project,
                name             = wandb_name,
                config           = {**sac_cfg.__dict__, **(env_config or {})},
                sync_tensorboard = False,   # no TensorBoard dependency
                save_code        = False,
            )
            wandb_active = True
            print(f"W&B run: {_wandb.run.url}")
        else:
            print("wandb not installed — skipping W&B logging")

    eval_cfg  = {**merged_env_cfg, **_EVAL_ENV_CFG}
    callbacks = [
        QuantflowEvalCallback(
            eval_env_config = eval_cfg,
            eval_freq       = sac_cfg.eval_freq,
            n_eval_episodes = sac_cfg.n_eval_episodes,
            save_path       = run_dir,
            as_gamma        = sac_cfg.as_baseline_gamma,
            as_kappa_offset = sac_cfg.as_baseline_kappa_offset,
            use_wandb       = wandb_active,
        )
    ]

    model.learn(
        total_timesteps     = sac_cfg.total_timesteps,
        callback            = CallbackList(callbacks),
        reset_num_timesteps = True,
    )

    model.save(run_dir / "final_model")
    print(f"Final model saved → {run_dir / 'final_model.zip'}")

    if wandb_active:
        _wandb.finish()

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train SAC market-making agent")
    p.add_argument("--timesteps",      type=int,  default=1_000_000,
                   help="Total environment steps (default 1M)")
    p.add_argument("--final",          action="store_true",
                   help="Override --timesteps to 2M (final training run)")
    p.add_argument("--run-dir",        type=str,  default="runs/sac",
                   help="Output directory for logs and checkpoints")
    p.add_argument("--wandb",          action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project",  type=str,  default="quantflow-mm")
    p.add_argument("--wandb-name",     type=str,  default=None)
    args = p.parse_args()

    cfg = SACConfig(
        total_timesteps = 2_000_000 if args.final else args.timesteps
    )
    train(
        cfg,
        run_dir       = args.run_dir,
        use_wandb     = args.wandb,
        wandb_project = args.wandb_project,
        wandb_name    = args.wandb_name,
    )


if __name__ == "__main__":
    main()
