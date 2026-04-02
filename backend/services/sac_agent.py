"""
SAC policy wrapper for live dashboard inference.

Replicates the observation normalisation from MarketMakingEnv._build_obs()
(obs_version="v1", the default training configuration) so that the model
receives the same feature distribution it saw during training.

Training constants (must stay in sync with _DEFAULT_CONFIG in market_making.py):
  lob_levels    = 5       → lob_state shape (20,)
  max_qty_scale = 500.0
  inventory_limit = 50
  initial_mid   = 100.0
  base_kappa    = 1.5     → kappa = base_kappa × (1 + kappa_offset)
  t_max         = 3600.0  → time_remaining normalisation
  vol_clip      = 0.05    → realised vol normalisation factor
"""
from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path

import numpy as np

# ── Training constants (must match MarketMakingEnv _DEFAULT_CONFIG) ───────────

_LOB_LEVELS    = 5
_MAX_QTY_SCALE = 500.0
_INV_LIMIT     = 50
_INITIAL_MID   = 100.0
_PNL_SCALE     = _INITIAL_MID * _INV_LIMIT   # 5000.0
_T_MAX_ENV     = 3600.0
_VOL_CLIP      = 0.05
BASE_KAPPA     = 50.0   # kappa = BASE_KAPPA × (1 + kappa_offset)


def _locate_model() -> Path | None:
    env_path = os.environ.get("SAC_MODEL_PATH", "")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    root = Path(__file__).parents[2]
    for rel in ("runs/sac/best_model.zip", "runs/sac/final_model.zip"):
        p = root / rel
        if p.exists():
            return p
    return None


class SACAgent:
    """
    Wraps a trained SB3 SAC model for deterministic inference.

    Usage
    -----
    agent = SACAgent()
    agent.update_mid(mid)           # call once per Hawkes event
    gamma, kappa_off = agent.get_action(sim, mid, inventory, cash, t)
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        from stable_baselines3 import SAC  # type: ignore[import]
        # LobFeatureExtractor must be importable for SAC.load() to reconstruct policy
        import quantflow.training.feature_extractor  # noqa: F401

        path = Path(model_path) if model_path else _locate_model()
        if path is None or not path.exists():
            raise FileNotFoundError(
                "SAC model not found. "
                "Train first: uv run python -m quantflow.training.train\n"
                f"Or set SAC_MODEL_PATH=/path/to/best_model.zip"
            )
        self._model = SAC.load(str(path))
        self._mid_hist: deque[float] = deque(maxlen=50)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_mid(self, mid: float) -> None:
        """Append latest mid price for return / vol computation."""
        self._mid_hist.append(mid)

    def get_action(
        self,
        sim,
        mid: float,
        inventory: int,
        cash: float,
        t: float,
    ) -> tuple[float, float]:
        """
        Run deterministic SAC inference.

        Returns
        -------
        (gamma, kappa_offset)
            gamma        ∈ [0.01, 1.0]
            kappa_offset ∈ [−0.5, 0.5]
        """
        obs = self._build_obs(sim, mid, inventory, cash, t)
        action, _ = self._model.predict(obs, deterministic=True)
        gamma        = float(np.clip(action[0], 0.01, 1.0))
        kappa_offset = float(np.clip(action[1], -0.5, 0.5))
        return gamma, kappa_offset

    # ── Observation construction ───────────────────────────────────────────────

    def _build_obs(
        self,
        sim,
        mid: float,
        inventory: int,
        cash: float,
        t: float,
    ) -> dict[str, np.ndarray]:
        """Mirror of MarketMakingEnv._build_obs() for obs_version='v1'."""
        book = sim.get_book()
        rb   = book.snapshot(_LOB_LEVELS)

        bid_prices = rb.column("bid_price").to_pylist()
        bid_qtys   = rb.column("bid_qty").to_pylist()
        ask_prices = rb.column("ask_price").to_pylist()
        ask_qtys   = rb.column("ask_qty").to_pylist()

        lob_state = np.zeros(_LOB_LEVELS * 4, dtype=np.float32)
        v_bid = v_ask = 0.0
        offset = _LOB_LEVELS * 2

        for i in range(_LOB_LEVELS):
            bp = _safe(bid_prices[i], mid)
            bq = bid_qtys[i] or 0
            lob_state[i * 2]     = float(np.clip((bp - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[i * 2 + 1] = float(np.clip(bq / _MAX_QTY_SCALE, 0.0, 1.0))
            v_bid += bq

            ap = _safe(ask_prices[i], mid)
            aq = ask_qtys[i] or 0
            lob_state[offset + i * 2]     = float(np.clip((ap - mid) / (mid + 1e-9), -1.0, 1.0))
            lob_state[offset + i * 2 + 1] = float(np.clip(aq / _MAX_QTY_SCALE, 0.0, 1.0))
            v_ask += aq

        vol_imbalance = (v_bid - v_ask) / (v_bid + v_ask + 1e-9)
        spread_raw    = book.spread() or 0.0
        spread_norm   = float(np.clip(spread_raw / (mid + 1e-9), 0.0, 1.0))

        hist = list(self._mid_hist)
        r_last = math.log(hist[-1] / hist[-2] + 1e-9) if len(hist) >= 2 else 0.0
        r_clip = float(np.clip(r_last / 0.1, -1.0, 1.0))

        vol = 0.0
        if len(hist) >= 10:
            window = hist[-20:]
            rets   = [math.log(window[j] / window[j - 1] + 1e-9) for j in range(1, len(window))]
            vol    = float(np.clip(
                math.sqrt(sum(r ** 2 for r in rets) / len(rets)) / _VOL_CLIP,
                0.0, 1.0,
            ))

        inv   = float(np.clip(inventory / (_INV_LIMIT + 1e-9), -1.0, 1.0))
        pnl   = cash + inventory * mid
        pnl_n = float(np.clip(pnl / (_PNL_SCALE + 1e-9), -1.0, 1.0))
        t_rem = float(np.clip(1.0 - t / (_T_MAX_ENV + 1e-9), 0.0, 1.0))

        return {
            "lob_state":        lob_state,
            "volume_imbalance": np.array([vol_imbalance], dtype=np.float32),
            "spread":           np.array([spread_norm],   dtype=np.float32),
            "mid_price_return": np.array([r_clip],        dtype=np.float32),
            "volatility":       np.array([vol],           dtype=np.float32),
            "inventory":        np.array([inv],           dtype=np.float32),
            "pnl":              np.array([pnl_n],         dtype=np.float32),
            "time_remaining":   np.array([t_rem],         dtype=np.float32),
        }


def _safe(v: float | None, fallback: float) -> float:
    """Return v if it is a finite number, else fallback."""
    return v if (v is not None and v == v) else fallback
