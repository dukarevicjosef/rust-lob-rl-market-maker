"""
Test dYdX v4 market data feed.

Connects to the dYdX mainnet indexer, subscribes to orderbook + trades
for a given market, and logs statistics every 5 seconds for 60 seconds.

Usage
-----
    uv run python scripts/test_dydx_feed.py
    uv run python scripts/test_dydx_feed.py --symbol ETH-USD --duration 120
    uv run python scripts/test_dydx_feed.py --testnet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from quantflow.connectors.dydx_market_data import DydxMarketDataConnector


async def run(symbol: str, duration: float, testnet: bool) -> None:
    connector = DydxMarketDataConnector(symbol=symbol, testnet=testnet)
    await connector.connect()

    start = time.monotonic()
    prev_trade_count = 0
    prev_time = start

    spreads: list[float] = []
    mids: list[float] = []
    tick = 0

    print(f"\n{'='*70}")
    print(f"  dYdX v4 Market Data — {symbol}")
    print(f"  {'Testnet' if testnet else 'Mainnet'} | Duration: {duration:.0f}s")
    print(f"{'='*70}\n")

    try:
        while (elapsed := time.monotonic() - start) < duration:
            await asyncio.sleep(5.0)
            tick += 1

            mid = connector.get_mid_price()
            spread = connector.get_spread()
            bb_price, bb_qty = connector.get_best_bid()
            ba_price, ba_qty = connector.get_best_ask()
            snap = connector.get_orderbook_snapshot(levels=10)
            trades = connector.get_recent_trades(1000)

            dt = time.monotonic() - prev_time
            new_trades = len(trades) - prev_trade_count
            tps = new_trades / dt if dt > 0 else 0.0
            prev_trade_count = len(trades)
            prev_time = time.monotonic()

            if mid > 0:
                spreads.append(spread)
                mids.append(mid)

            bid_depth = sum(q for _, q in snap["bids"])
            ask_depth = sum(q for _, q in snap["asks"])

            spread_bps = spread / mid * 10_000 if mid > 0 else 0.0

            print(
                f"  [{tick:3d}] mid={mid:,.1f} | spread=${spread:.1f} "
                f"({spread_bps:.1f}bps) | "
                f"bid={bb_price:,.1f}×{bb_qty:.4f} "
                f"ask={ba_price:,.1f}×{ba_qty:.4f} | "
                f"tps={tps:.1f} | depth={bid_depth:.3f}/{ask_depth:.3f}"
            )

    except KeyboardInterrupt:
        pass
    finally:
        await connector.close()

    # Summary
    elapsed = time.monotonic() - start
    all_trades = connector.get_recent_trades(10_000)

    print(f"\n{'='*70}")
    print(f"  SESSION SUMMARY — {symbol}")
    print(f"{'='*70}")
    print(f"  Duration:          {elapsed:.0f}s")
    print(f"  Total trades:      {len(all_trades)}")
    if elapsed > 0:
        print(f"  Trade rate:        {len(all_trades)/elapsed:.1f} trades/sec")

    if spreads:
        avg_spread = sum(spreads) / len(spreads)
        avg_mid = sum(mids) / len(mids)
        avg_spread_bps = avg_spread / avg_mid * 10_000 if avg_mid > 0 else 0.0
        print(f"  Avg spread:        ${avg_spread:.1f} ({avg_spread_bps:.1f} bps)")
        print(f"  Price range:       {min(mids):,.1f} — {max(mids):,.1f}")

    snap = connector.get_orderbook_snapshot(10)
    bid_depth = sum(q for _, q in snap["bids"])
    ask_depth = sum(q for _, q in snap["asks"])
    print(f"  Book depth (10L):  {bid_depth:.3f} BTC bid / {ask_depth:.3f} BTC ask")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Test dYdX v4 market data feed")
    p.add_argument("--symbol", default="BTC-USD", help="Market (default: BTC-USD)")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (default: 60)")
    p.add_argument("--testnet", action="store_true", help="Use testnet indexer")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run(args.symbol, args.duration, args.testnet))


if __name__ == "__main__":
    main()
