"""
Paper trading configuration.

Testnet defaults are used unless explicitly set.  Mainnet requires setting
``testnet=False`` AND reducing risk limits manually.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv(path: str | Path | None = None) -> None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables are never overwritten.
    Lines starting with '#' and blank lines are ignored.
    Inline comments (# ...) are stripped.  Quoted values are unquoted.
    """
    candidates = [path] if path else [
        Path.cwd() / ".env",
        Path(__file__).parents[4] / ".env",   # repo root
    ]
    for p in candidates:
        if p and Path(p).is_file():
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    # strip inline comment and surrounding quotes
                    val = val.split("#")[0].strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break   # stop at first file found


@dataclass
class PaperTradingConfig:
    # ── Exchange ──────────────────────────────────────────────────────────────
    api_key:    str  = ""
    api_secret: str  = ""
    symbol:     str  = "BTCUSDT"
    testnet:    bool = True   # ALWAYS True unless you explicitly want real money

    # ── Agent ─────────────────────────────────────────────────────────────────
    model_path: str = "runs/sac_1M_btcusdt/best_model.zip"
    # How many training-side "inventory lots" map to max_position BTC.
    # Training env used inventory_limit=50.  Live max_position=0.01 BTC.
    # ratio = 50 / 0.01 = 5000 → multiply position_btc × 5000 to get "lots".
    training_inventory_limit: int = 50

    # ── Quoting ───────────────────────────────────────────────────────────────
    quote_interval:   float = 5.0     # seconds between quote refreshes
    base_kappa:       float = 1.5     # must match training SACConfig
    quote_qty_btc:    float = 0.001   # BTC per resting quote
    t_max_session:    float = 86_400.0  # session horizon for AS time decay (1 day)
    sigma_fixed:      float = 0.02    # σ passed to AS formula (matches training)

    # ── Observation normalization ─────────────────────────────────────────────
    # Qty normalization for LOB state: qty_norm = qty_btc / qty_scale_btc.
    # Simulation trained with max_qty_scale=500 lots; typical real top-of-book
    # BTC depth is 0.1–5 BTC, so qty_scale_btc=2.0 keeps values in [0, 1].
    qty_scale_btc: float = 2.0

    # ── Risk ─────────────────────────────────────────────────────────────────
    risk_config: dict[str, Any] = field(default_factory=lambda: {
        "max_position":          0.01,   # 0.01 BTC
        "max_daily_loss":        50.0,   # 50 USDT
        "max_drawdown":          100.0,  # 100 USDT
        "max_orders_per_second": 5,
        "max_order_size":        0.005,  # 0.005 BTC
        "max_open_orders":       4,      # 2 bids + 2 asks
        "max_notional":          5_000.0,
    })

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb:      bool  = False
    wandb_project:  str   = "quantflow-paper"
    log_interval:   float = 30.0     # console log every N seconds
    output_dir:     str   = "runs/paper_trading"

    # ── Duration ─────────────────────────────────────────────────────────────
    max_duration_sec: float | None = None   # None = run until Ctrl+C

    # ── Safety rules passthrough ──────────────────────────────────────────────
    inventory_soft_limit_btc:  float = 0.007  # ≈ 70 % of max_position
    inventory_hard_limit_btc:  float = 0.009  # ≈ 90 % of max_position
    tick_size:                 float = 0.10   # BTC/USDT minimum tick ($0.10)
    vol_spread_threshold:      float = 2.0
    vol_spread_multiplier:     float = 2.0

    # ── Internal convenience ──────────────────────────────────────────────────
    @property
    def max_position_btc(self) -> float:
        return float(self.risk_config.get("max_position", 0.01))

    @property
    def pnl_scale(self) -> float:
        """Match training: pnl_scale ≈ mid × max_position ≈ 67000 × 0.01 = 670."""
        return 70_000.0 * self.max_position_btc  # conservative mid estimate

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "PaperTradingConfig":
        """Load .env file (if present) then read credentials from environment."""
        _load_dotenv(dotenv_path)
        return cls(
            api_key    = os.environ.get("BINANCE_API_KEY",    ""),
            api_secret = os.environ.get("BINANCE_API_SECRET", ""),
            symbol     = os.environ.get("BINANCE_SYMBOL",     "BTCUSDT"),
            testnet    = os.environ.get("BINANCE_TESTNET",    "true").lower() != "false",
            model_path = os.environ.get("MODEL_PATH", "runs/sac_1M_btcusdt/best_model.zip"),
        )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing or unsafe."""
        if not self.api_key or not self.api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        if not self.testnet:
            raise ValueError(
                "testnet=False detected — set explicitly via config.testnet=False "
                "after reviewing risk limits"
            )
        if self.quote_qty_btc > self.risk_config.get("max_order_size", 1.0):
            raise ValueError(
                f"quote_qty_btc={self.quote_qty_btc} exceeds "
                f"max_order_size={self.risk_config['max_order_size']}"
            )
