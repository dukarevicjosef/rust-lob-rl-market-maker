"""
LOB-aware feature extractor for SB3 SAC.

Architecture
------------
lob_state (20,)
    reshape → (batch, 10 levels, 2 features)   [Δprice, qty per level]
    permute → (batch, 2 channels, 10 length)    for Conv1d
    Conv1d(2→16, k=3, padding=0) → ReLU        → (batch, 16, 8)
    Conv1d(16→32, k=3, padding=0) → ReLU       → (batch, 32, 6)
    Flatten                                     → (batch, 192)

LOB-state layout (from market_making.py):
    indices  0–9  : bid side, interleaved [Δprice_i, qty_i] for levels 0–4
    indices 10–19 : ask side, interleaved [Δprice_i, qty_i] for levels 0–4

scalar features (all other obs keys, concatenated)
    Linear(N→64) → ReLU                        → (batch, 64)

combined
    cat([lob_features, scalar_features])        → (batch, 256)
    Linear(256→128) → ReLU                     → (batch, 128)  ← features_dim
"""
from __future__ import annotations

import gymnasium
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_CNN_OUT = 32 * 6   # 32 filters × 6 remaining length after 2× Conv1d(k=3)
_SCALAR_OUT = 64
_COMBINED_IN = _CNN_OUT + _SCALAR_OUT   # 192 + 64 = 256
_FEATURES_DIM = 128


class LobFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes the LOB depth tensor with a 1-D CNN to capture spatial
    patterns across price levels (liquidity gaps, volume clusters), then
    combines with scalar features through a small MLP.

    Compatible with obs_version="v1" (7 scalar keys) and
    obs_version="v2" (14 scalar keys) — scalar_dim is computed
    dynamically from the observation space.
    """

    def __init__(self, observation_space: gymnasium.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=_FEATURES_DIM)

        # ── LOB branch: 1-D CNN over 10 price levels ──────────────────────────
        self.lob_cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # ── Scalar branch: all keys except lob_state ──────────────────────────
        self._scalar_keys: list[str] = sorted(
            k for k in observation_space.spaces if k != "lob_state"
        )
        scalar_dim: int = sum(
            observation_space.spaces[k].shape[0] for k in self._scalar_keys
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, _SCALAR_OUT),
            nn.ReLU(),
        )

        # ── Combined head ─────────────────────────────────────────────────────
        self.combined = nn.Sequential(
            nn.Linear(_COMBINED_IN, _FEATURES_DIM),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # LOB branch
        # lob_state: (batch, 20)
        # Layout: [bid_Δp0,bid_q0,...,bid_Δp4,bid_q4, ask_Δp0,ask_q0,...,ask_Δp4,ask_q4]
        lob = observations["lob_state"]             # (batch, 20)
        lob = lob.view(-1, 10, 2)                   # (batch, 10 levels, 2 features)
        lob = lob.permute(0, 2, 1)                  # (batch, 2 channels, 10 length)
        lob_features = self.lob_cnn(lob)            # (batch, 192)

        # Scalar branch
        scalars = torch.cat(
            [observations[k] for k in self._scalar_keys], dim=-1
        )                                           # (batch, scalar_dim)
        scalar_features = self.scalar_mlp(scalars)  # (batch, 64)

        # Combine
        combined = torch.cat([lob_features, scalar_features], dim=-1)  # (batch, 256)
        return self.combined(combined)              # (batch, 128)
