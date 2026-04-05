"""
CLI entry point for the paper trading runner.

Usage
-----
    python -m quantflow.paper_trading \\
        --model runs/sac_1M_btcusdt/best_model.zip \\
        --symbol BTCUSDT \\
        --duration 3600 \\
        --wandb

Environment variables
---------------------
    BINANCE_API_KEY      Binance Futures API key (testnet or mainnet)
    BINANCE_API_SECRET   Corresponding secret
    BINANCE_TESTNET      "true" (default) or "false"
    BINANCE_SYMBOL       Trading pair, e.g. BTCUSDT
    MODEL_PATH           Path to the trained SB3 model archive
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantflow.paper_trading",
        description="Run the SAC market-making agent against Binance Futures Testnet.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "runs/sac_1M_btcusdt/best_model.zip"),
        help="Path to the SB3 SAC model archive (default: $MODEL_PATH or runs/sac_1M_btcusdt/best_model.zip)",
    )
    p.add_argument(
        "--symbol",
        default=os.environ.get("BINANCE_SYMBOL", "BTCUSDT"),
        help="Trading pair (default: BTCUSDT)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl+C)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    p.add_argument(
        "--wandb-project",
        default="quantflow-paper",
        help="W&B project name (default: quantflow-paper)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )
    p.add_argument(
        "--testnet",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Force testnet on/off (default: reads BINANCE_TESTNET env var)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level  = getattr(logging, args.log_level),
        format = "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt= "%H:%M:%S",
    )

    from quantflow.paper_trading.config import PaperTradingConfig

    cfg = PaperTradingConfig.from_env()

    # CLI overrides
    if args.model:
        cfg.model_path = args.model
    if args.symbol:
        cfg.symbol = args.symbol
    if args.duration is not None:
        cfg.max_duration_sec = args.duration
    if args.wandb:
        cfg.use_wandb = True
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.testnet is not None:
        cfg.testnet = args.testnet

    try:
        cfg.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, config={
                "symbol":        cfg.symbol,
                "testnet":       cfg.testnet,
                "model_path":    cfg.model_path,
                "quote_interval": cfg.quote_interval,
                "quote_qty_btc": cfg.quote_qty_btc,
                "max_position":  cfg.max_position_btc,
            })
        except Exception as exc:
            logging.getLogger(__name__).warning("W&B init failed: %s", exc)
            cfg.use_wandb = False

    from quantflow.paper_trading.runner import PaperTradingRunner
    runner = PaperTradingRunner(cfg)

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
