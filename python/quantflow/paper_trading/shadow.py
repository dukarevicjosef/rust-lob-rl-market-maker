"""
CLI entry point for shadow mode evaluation.

Usage
-----
    uv run python -m quantflow.paper_trading.shadow \\
        --model runs/sac_2M_cloud/best_model.zip \\
        --exchange dydx --symbol BTC-USD --duration 3600

    uv run python -m quantflow.paper_trading.shadow \\
        --model runs/sac_2M_cloud/best_model.zip \\
        --exchange dydx --symbol BTC-USD --duration 60 \\
        --maker-fee-bps 0.0 --min-spread-multiplier 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantflow.paper_trading.shadow",
        description="Shadow mode: observe mainnet, simulate agent fills.",
    )
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", "runs/sac_2M_cloud/best_model.zip"))
    p.add_argument("--exchange", default="dydx", choices=["dydx", "binance"])
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--duration", type=float, default=3600.0, help="Seconds to run")
    p.add_argument("--quote-interval", type=float, default=5.0)
    p.add_argument("--quote-qty-btc", type=float, default=0.001)
    p.add_argument("--maker-fee-bps", type=float, default=None,
                   help="Maker fee in bps (default: 0 for dydx, 2 for binance)")
    p.add_argument("--min-spread-multiplier", type=float, default=None,
                   help="Min spread over RT fees (default: 1.0 for dydx, 1.5 for binance)")
    p.add_argument("--max-position-btc", type=float, default=0.01)
    p.add_argument("--log-interval", type=float, default=30.0)
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Exchange-specific defaults
    if args.maker_fee_bps is None:
        args.maker_fee_bps = 0.0 if args.exchange == "dydx" else 2.0
    if args.min_spread_multiplier is None:
        args.min_spread_multiplier = 1.0 if args.exchange == "dydx" else 1.5

    from quantflow.paper_trading.shadow_runner import ShadowConfig, ShadowRunner

    cfg = ShadowConfig(
        model_path=args.model,
        symbol=args.symbol,
        exchange=args.exchange,
        testnet=args.testnet,
        quote_interval=args.quote_interval,
        quote_qty_btc=args.quote_qty_btc,
        maker_fee_bps=args.maker_fee_bps,
        min_spread_multiplier=args.min_spread_multiplier,
        max_position_btc=args.max_position_btc,
        duration=args.duration,
        log_interval=args.log_interval,
    )

    # Build connector
    if args.exchange == "dydx":
        from quantflow.connectors.dydx_market_data import DydxMarketDataConnector
        connector = DydxMarketDataConnector(symbol=args.symbol, testnet=args.testnet)
    else:
        raise NotImplementedError("Binance shadow connector not yet implemented")

    async def _run() -> None:
        await connector.connect()
        try:
            runner = ShadowRunner(connector=connector, config=cfg)
            await runner.run()
        finally:
            await connector.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
