"""
Async Binance Futures exchange client for paper trading.

Uses stdlib urllib for REST (via asyncio.to_thread) and the websockets
library for the User Data Stream.  No additional runtime dependencies.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import websockets

_TESTNET_REST = "https://testnet.binancefuture.com"
_MAINNET_REST = "https://fapi.binance.com"
_TESTNET_WS   = "wss://stream.binancefuture.com"
_MAINNET_WS   = "wss://fapi.binance.com"


class BinanceClientError(Exception):
    """Raised on non-2xx responses or API error codes."""
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(f"Binance error {code}: {msg}")
        self.code = code
        self.msg  = msg


class ExchangeClient:
    """
    Async wrapper around Binance Futures REST + User Data Stream.

    REST calls are executed synchronously in a thread pool via
    asyncio.to_thread so the event loop is never blocked.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self._key    = api_key
        self._secret = api_secret.encode()
        self._rest   = _TESTNET_REST if testnet else _MAINNET_REST
        self._ws_base = _TESTNET_WS  if testnet else _MAINNET_WS

    # ── signing ───────────────────────────────────────────────────────────────

    def _sign(self, params: dict[str, Any]) -> str:
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    # ── raw REST helper ───────────────────────────────────────────────────────

    def _request_sync(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        params = dict(params or {})
        if signed:
            query = self._sign(params)
        else:
            query = urllib.parse.urlencode(params) if params else ""

        url = f"{self._rest}{path}"
        if method == "GET" and query:
            url = f"{url}?{query}"
            data = None
        else:
            data = query.encode() if query else None

        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "X-MBX-APIKEY": self._key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                err = json.loads(body)
                raise BinanceClientError(err.get("code", exc.code), err.get("msg", str(exc))) from exc
            except (json.JSONDecodeError, KeyError):
                raise BinanceClientError(exc.code, body.decode(errors="replace")) from exc

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        return await asyncio.to_thread(
            self._request_sync, method, path, params, signed
        )

    # ── market data ───────────────────────────────────────────────────────────

    async def get_ticker_price(self, symbol: str) -> float:
        data = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        return float(data["price"])

    async def get_depth(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return await self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})

    # ── order management ─────────────────────────────────────────────────────

    async def place_limit_order(
        self,
        symbol:    str,
        side:      str,    # "BUY" or "SELL"
        price:     float,
        qty:       float,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol":      symbol,
            "side":        side.upper(),
            "type":        "LIMIT",
            "price":       f"{price:.2f}",
            "quantity":    f"{qty:.3f}",
            "timeInForce": time_in_force,
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

    async def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return await self._request(
            "DELETE", "/fapi/v1/order",
            {"symbol": symbol, "orderId": order_id},
            signed=True,
        )

    async def cancel_all_orders(self, symbol: str) -> dict[str, Any]:
        try:
            return await self._request(
                "DELETE", "/fapi/v1/allOpenOrders",
                {"symbol": symbol},
                signed=True,
            )
        except BinanceClientError as exc:
            if exc.code == -2011:   # no orders to cancel — idempotent
                return {}
            raise

    async def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return await self._request(
            "GET", "/fapi/v1/openOrders",
            {"symbol": symbol},
            signed=True,
        )

    async def get_account(self) -> dict[str, Any]:
        return await self._request("GET", "/fapi/v2/account", {}, signed=True)

    async def get_position(self, symbol: str) -> float:
        """Return current net position in base asset (positive = long)."""
        account = await self.get_account()
        for pos in account.get("positions", []):
            if pos.get("symbol") == symbol:
                return float(pos.get("positionAmt", 0.0))
        return 0.0

    # ── User Data Stream (listen-key lifecycle) ───────────────────────────────

    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", {}, signed=False)
        return data["listenKey"]

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key}, signed=False)

    async def close_listen_key(self, listen_key: str) -> None:
        await self._request(
            "DELETE", "/fapi/v1/listenKey",
            {"listenKey": listen_key},
            signed=False,
        )

    # ── User Data Stream WebSocket ─────────────────────────────────────────────

    @asynccontextmanager
    async def user_data_stream(
        self,
        listen_key: str,
        reconnect_attempts: int = 5,
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """
        Async context manager yielding an async iterator of user-data events.

        Usage::
            async with client.user_data_stream(lk) as events:
                async for event in events:
                    ...
        """
        url = f"{self._ws_base}/ws/{listen_key}"

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            attempts = 0
            delay    = 1.0
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20) as ws:
                        attempts = 0
                        delay    = 1.0
                        async for raw in ws:
                            try:
                                yield json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                except (websockets.ConnectionClosed,
                        websockets.WebSocketException,
                        OSError) as exc:
                    attempts += 1
                    if attempts > reconnect_attempts:
                        raise RuntimeError(
                            f"User data stream failed after {attempts} reconnect attempts"
                        ) from exc
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        yield _iter()

    # ── Market data stream ────────────────────────────────────────────────────

    @asynccontextmanager
    async def market_data_stream(
        self,
        symbol: str,
        reconnect_attempts: int = 10,
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """
        Combined depth20@100ms + aggTrade stream for a symbol.

        Yields raw Binance stream messages.  Each message has a ``stream``
        field identifying "btcusdt@depth20@100ms" or "btcusdt@aggTrade".
        """
        sym = symbol.lower()
        streams = f"{sym}@depth20@100ms/{sym}@aggTrade"
        url = f"{self._ws_base}/stream?streams={streams}"

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            attempts = 0
            delay    = 1.0
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20) as ws:
                        attempts = 0
                        delay    = 1.0
                        async for raw in ws:
                            try:
                                yield json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                except (websockets.ConnectionClosed,
                        websockets.WebSocketException,
                        OSError) as exc:
                    attempts += 1
                    if attempts > reconnect_attempts:
                        raise RuntimeError(
                            f"Market data stream failed after {attempts} reconnect attempts"
                        ) from exc
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        yield _iter()
