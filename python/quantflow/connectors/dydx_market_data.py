"""
dYdX v4 Indexer market data connector.

Maintains an in-memory order book and recent trade buffer via the dYdX v4
Indexer WebSocket API.  The initial book state is seeded from a REST snapshot
before the WebSocket delta stream takes over.

Usage
-----
    connector = DydxMarketDataConnector(symbol="BTC-USD")
    await connector.connect()
    print(connector.get_mid_price())
    await connector.close()
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

import aiohttp
import websockets

from quantflow.connectors.base import MarketDataConnector

log = logging.getLogger(__name__)

_MAINNET_REST = "https://indexer.dydx.trade"
_TESTNET_REST = "https://indexer.v4testnet.dydx.exchange"
_MAINNET_WS   = "wss://indexer.dydx.trade/v4/ws"
_TESTNET_WS   = "wss://indexer.v4testnet.dydx.exchange/v4/ws"


class DydxMarketDataConnector(MarketDataConnector):
    """
    Live market data connector for dYdX v4 perpetual markets.

    Parameters
    ----------
    symbol : str
        Market identifier, e.g. ``"BTC-USD"``.
    testnet : bool
        Use the v4 testnet indexer instead of mainnet.
    max_trades : int
        Maximum number of recent trades to keep in memory.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        testnet: bool = False,
        max_trades: int = 1000,
    ) -> None:
        self._symbol = symbol
        self._testnet = testnet
        self._rest_url = _TESTNET_REST if testnet else _MAINNET_REST
        self._ws_url = _TESTNET_WS if testnet else _MAINNET_WS
        self._max_trades = max_trades

        # Order book state: price → size
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

        # Recent trades ring buffer
        self._trades: deque[dict[str, Any]] = deque(maxlen=max_trades)

        # WebSocket state
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._connected = asyncio.Event()

    # ── connect / close ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Fetch REST snapshot, then open WS and subscribe to deltas + trades."""
        await self._fetch_rest_snapshot()
        await self._open_ws()
        self._connected.set()

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # ── REST snapshot ────────────────────────────────────────────────────────

    async def _fetch_rest_snapshot(self) -> None:
        url = f"{self._rest_url}/v4/orderbooks/perpetualMarket/{self._symbol}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()

        self._bids = {
            float(level["price"]): float(level["size"])
            for level in data.get("bids", [])
            if float(level["size"]) > 0.0
        }
        self._asks = {
            float(level["price"]): float(level["size"])
            for level in data.get("asks", [])
            if float(level["size"]) > 0.0
        }
        log.info(
            "REST snapshot: %d bids, %d asks, mid=%.1f",
            len(self._bids), len(self._asks), self.get_mid_price(),
        )

    # ── WebSocket ────────────────────────────────────────────────────────────

    async def _open_ws(self) -> None:
        self._ws = await websockets.connect(self._ws_url)

        # Subscribe to orderbook deltas
        await self._ws.send(json.dumps({
            "type": "subscribe",
            "channel": "v4_orderbook",
            "id": self._symbol,
            "batched": True,
        }))

        # Subscribe to trades
        await self._ws.send(json.dumps({
            "type": "subscribe",
            "channel": "v4_trades",
            "id": self._symbol,
            "batched": True,
        }))

        self._recv_task = asyncio.create_task(self._recv_loop(), name="dydx-ws")

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                channel = msg.get("channel", "")

                if msg_type == "connected":
                    log.debug("WS connected: %s", msg.get("connection_id", ""))
                    continue

                if msg_type == "subscribed":
                    contents = msg.get("contents", {})
                    if channel == "v4_orderbook":
                        self._apply_book_snapshot(contents)
                    elif channel == "v4_trades":
                        self._apply_trades(contents.get("trades", []))
                    continue

                if msg_type in ("channel_data", "channel_batch_data"):
                    contents = msg.get("contents", {})
                    # Batched messages wrap contents in a list
                    items = contents if isinstance(contents, list) else [contents]
                    for item in items:
                        if channel == "v4_orderbook":
                            self._apply_book_delta(item)
                        elif channel == "v4_trades":
                            self._apply_trades(item.get("trades", []))
                    continue

        except websockets.ConnectionClosed:
            log.warning("dYdX WebSocket connection closed")
        except asyncio.CancelledError:
            pass

    # ── book updates ─────────────────────────────────────────────────────────

    def _apply_book_snapshot(self, contents: dict) -> None:
        """Process initial orderbook snapshot from WS subscription response."""
        self._bids = {
            float(level["price"]): float(level["size"])
            for level in contents.get("bids", [])
            if float(level["size"]) > 0.0
        }
        self._asks = {
            float(level["price"]): float(level["size"])
            for level in contents.get("asks", [])
            if float(level["size"]) > 0.0
        }
        log.debug(
            "WS book snapshot: %d bids, %d asks", len(self._bids), len(self._asks),
        )

    def _apply_book_delta(self, contents: dict) -> None:
        """Apply incremental orderbook update. Size=0 means level removed."""
        for level in contents.get("bids", []):
            price = float(level[0]) if isinstance(level, list) else float(level["price"])
            size = float(level[1]) if isinstance(level, list) else float(level["size"])
            if size > 0.0:
                self._bids[price] = size
            else:
                self._bids.pop(price, None)

        for level in contents.get("asks", []):
            price = float(level[0]) if isinstance(level, list) else float(level["price"])
            size = float(level[1]) if isinstance(level, list) else float(level["size"])
            if size > 0.0:
                self._asks[price] = size
            else:
                self._asks.pop(price, None)

    # ── trade updates ────────────────────────────────────────────────────────

    def _apply_trades(self, trades: list[dict]) -> None:
        for t in trades:
            self._trades.append({
                "price": float(t["price"]),
                "qty": float(t["size"]),
                "side": t.get("side", "BUY").upper(),
                "timestamp": t.get("createdAt", ""),
            })

    # ── MarketDataConnector interface ────────────────────────────────────────

    def get_mid_price(self) -> float:
        bb = max(self._bids) if self._bids else 0.0
        ba = min(self._asks) if self._asks else 0.0
        if bb == 0.0 or ba == 0.0:
            return 0.0
        return (bb + ba) / 2.0

    def get_best_bid(self) -> tuple[float, float]:
        if not self._bids:
            return (0.0, 0.0)
        price = max(self._bids)
        return (price, self._bids[price])

    def get_best_ask(self) -> tuple[float, float]:
        if not self._asks:
            return (0.0, 0.0)
        price = min(self._asks)
        return (price, self._asks[price])

    def get_spread(self) -> float:
        bb = max(self._bids) if self._bids else 0.0
        ba = min(self._asks) if self._asks else 0.0
        if bb == 0.0 or ba == 0.0:
            return 0.0
        return ba - bb

    def get_orderbook_snapshot(self, levels: int = 10) -> dict:
        bid_prices = sorted(self._bids, reverse=True)[:levels]
        ask_prices = sorted(self._asks)[:levels]

        bids = [(p, self._bids[p]) for p in bid_prices]
        asks = [(p, self._asks[p]) for p in ask_prices]

        bb = bids[0] if bids else (0.0, 0.0)
        ba = asks[0] if asks else (0.0, 0.0)
        mid = (bb[0] + ba[0]) / 2.0 if bb[0] > 0 and ba[0] > 0 else 0.0
        spread = ba[0] - bb[0] if bb[0] > 0 and ba[0] > 0 else 0.0

        return {
            "bids": bids,
            "asks": asks,
            "mid_price": mid,
            "spread": spread,
            "best_bid": bb,
            "best_ask": ba,
        }

    def get_recent_trades(self, n: int = 100) -> list[dict]:
        trades = list(self._trades)
        return trades[-n:]
