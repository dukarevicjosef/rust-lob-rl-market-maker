"""
Record a Binance Futures BTCUSDT WebSocket session to Parquet files.

Captures depth@100ms, aggTrade, and bookTicker streams from the public
mainnet endpoint (no API keys required).  Events are buffered in memory
and flushed to Parquet periodically and on shutdown.

Usage
-----
    uv run python -m quantflow.data.record_session \
        --symbol BTCUSDT --duration 4h --output data/btcusdt/sessions/

    # With session label
    uv run python -m quantflow.data.record_session \
        --symbol BTCUSDT --duration 4h --label asia_quiet \
        --output data/btcusdt/sessions/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import websockets

log = logging.getLogger(__name__)

_WS_BASE = "wss://fstream.binance.com"
_FLUSH_THRESHOLD = 10_000  # events per buffer before flush


def _parse_duration(s: str) -> float:
    """Parse duration string like '4h', '30m', '1.5h' to seconds."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


class SessionRecorder:
    """Buffered WebSocket recorder with periodic Parquet flushes."""

    def __init__(
        self,
        symbol: str,
        output_dir: Path,
        duration_sec: float,
        label: str | None = None,
    ) -> None:
        self._symbol = symbol.upper()
        self._duration = duration_sec
        self._label = label
        self._stop = asyncio.Event()

        # Session naming
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M")
        self._session_id = f"{self._symbol}_{ts}"
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Buffers
        self._trades: list[dict[str, Any]] = []
        self._depth: list[dict[str, Any]] = []
        self._bookticker: list[dict[str, Any]] = []

        # Counters
        self._total_trades = 0
        self._total_depth = 0
        self._total_bookticker = 0
        self._start_time = now
        self._session_start_mono = 0.0

        # Price tracking for metadata
        self._mids: list[float] = []
        self._spreads: list[float] = []
        self._best_bid = 0.0
        self._best_ask = 0.0

    async def run(self) -> None:
        self._session_start_mono = time.monotonic()
        log.info(
            "Recording %s for %.1fh → %s/",
            self._symbol, self._duration / 3600, self._output_dir,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._stop.set())

        tasks = [
            asyncio.create_task(self._ws_loop(), name="ws"),
            asyncio.create_task(self._status_loop(), name="status"),
            asyncio.create_task(self._duration_watchdog(), name="watchdog"),
        ]

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION,
            )
            for t in done:
                if exc := t.exception():
                    log.error("Task %s failed: %s", t.get_name(), exc)
        finally:
            self._stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        self._flush_all()
        self._write_metadata()
        log.info("Session complete: %d trades, %d depth, %d bookticker",
                 self._total_trades, self._total_depth, self._total_bookticker)

    async def _ws_loop(self) -> None:
        sym = self._symbol.lower()
        streams = f"{sym}@depth20@100ms/{sym}@aggTrade/{sym}@bookTicker"
        url = f"{_WS_BASE}/stream?streams={streams}"

        attempts = 0
        delay = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    attempts = 0
                    delay = 1.0
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self._dispatch(msg)
            except (websockets.ConnectionClosed,
                    websockets.WebSocketException,
                    OSError) as exc:
                if self._stop.is_set():
                    break
                attempts += 1
                if attempts > 20:
                    raise RuntimeError(
                        f"WebSocket failed after {attempts} reconnect attempts"
                    ) from exc
                log.warning("WS disconnected (%s), reconnect in %.0fs...", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        recv_ms = int(time.time() * 1000)

        if "aggTrade" in stream:
            self._trades.append({
                "event_time":    int(data.get("E", 0)),
                "receive_time":  recv_ms,
                "agg_trade_id":  int(data.get("a", 0)),
                "price":         float(data.get("p", 0)),
                "quantity":      float(data.get("q", 0)),
                "is_buy":        not data.get("m", False),
                "first_trade_id": int(data.get("f", 0)),
                "last_trade_id": int(data.get("l", 0)),
            })
            self._total_trades += 1
            if len(self._trades) >= _FLUSH_THRESHOLD:
                self._flush_trades()

        elif "depth" in stream:
            self._depth.append({
                "event_time":   int(data.get("E", 0)),
                "receive_time": recv_ms,
                "bids_json":    json.dumps(data.get("b", [])),
                "asks_json":    json.dumps(data.get("a", [])),
            })
            self._total_depth += 1
            if len(self._depth) >= _FLUSH_THRESHOLD:
                self._flush_depth()

        elif "bookTicker" in stream:
            bid = float(data.get("b", 0))
            ask = float(data.get("a", 0))
            self._best_bid = bid
            self._best_ask = ask
            if bid > 0 and ask > 0:
                self._mids.append((bid + ask) / 2)
                self._spreads.append(ask - bid)

            self._bookticker.append({
                "event_time":   int(data.get("E", 0)),
                "receive_time": recv_ms,
                "bid_price":    bid,
                "bid_qty":      float(data.get("B", 0)),
                "ask_price":    ask,
                "ask_qty":      float(data.get("A", 0)),
            })
            self._total_bookticker += 1
            if len(self._bookticker) >= _FLUSH_THRESHOLD:
                self._flush_bookticker()

    # ── flush helpers ────────────────────────────────────────────────────────

    def _parquet_path(self, suffix: str) -> Path:
        return self._output_dir / f"{self._session_id}_{suffix}.parquet"

    def _flush_trades(self) -> None:
        if not self._trades:
            return
        df = pl.DataFrame(self._trades)
        path = self._parquet_path("trades")
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df])
        df.write_parquet(path)
        self._trades.clear()

    def _flush_depth(self) -> None:
        if not self._depth:
            return
        df = pl.DataFrame(self._depth)
        path = self._parquet_path("depth")
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df])
        df.write_parquet(path)
        self._depth.clear()

    def _flush_bookticker(self) -> None:
        if not self._bookticker:
            return
        df = pl.DataFrame(self._bookticker)
        path = self._parquet_path("bookticker")
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df])
        df.write_parquet(path)
        self._bookticker.clear()

    def _flush_all(self) -> None:
        self._flush_trades()
        self._flush_depth()
        self._flush_bookticker()

    # ── metadata ─────────────────────────────────────────────────────────────

    def _write_metadata(self) -> None:
        end_time = datetime.now(timezone.utc)
        elapsed = time.monotonic() - self._session_start_mono

        mids = np.array(self._mids) if self._mids else np.array([0.0])
        spreads = np.array(self._spreads) if self._spreads else np.array([0.0])

        # Volatility: annualised from log-returns of mid samples
        if len(mids) > 100:
            # Subsample to ~1 per second for vol calc
            step = max(1, len(mids) // int(elapsed))
            sub = mids[::step]
            if len(sub) > 1:
                rets = np.diff(np.log(sub))
                vol_est = float(np.std(rets) * np.sqrt(len(rets) / elapsed))
            else:
                vol_est = 0.0
        else:
            vol_est = 0.0

        meta = {
            "symbol":              self._symbol,
            "session_id":          self._session_id,
            "start_time":          self._start_time.isoformat(),
            "end_time":            end_time.isoformat(),
            "duration_hours":      round(elapsed / 3600, 2),
            "total_trades":        self._total_trades,
            "total_depth_updates": self._total_depth,
            "total_bookticker":    self._total_bookticker,
            "avg_spread_usd":      round(float(np.mean(spreads)), 4),
            "avg_mid_price":       round(float(np.mean(mids)), 2),
            "price_range":         [round(float(np.min(mids)), 2),
                                    round(float(np.max(mids)), 2)],
            "volatility_estimate": round(vol_est, 6),
            "session_label":       self._label or "unlabeled",
        }

        path = self._output_dir / f"{self._session_id}_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Metadata → %s", path)

    # ── periodic loops ───────────────────────────────────────────────────────

    async def _status_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(60)
            if self._stop.is_set():
                break
            elapsed = time.monotonic() - self._session_start_mono
            total = self._total_trades + self._total_depth + self._total_bookticker
            rate = total / max(elapsed, 1)
            mid = self._mids[-1] if self._mids else 0.0
            spread = self._spreads[-1] if self._spreads else 0.0
            log.info(
                "t=%dm | events=%d | trades=%d | depth=%d | "
                "rate=%.0f/s | mid=%.2f | spread=%.4f",
                int(elapsed / 60), total, self._total_trades,
                self._total_depth, rate, mid, spread,
            )

    async def _duration_watchdog(self) -> None:
        await asyncio.sleep(self._duration)
        log.info("Duration %.1fh reached — stopping", self._duration / 3600)
        self._stop.set()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Record Binance Futures WebSocket session to Parquet",
    )
    p.add_argument("--symbol",   type=str, default="BTCUSDT")
    p.add_argument("--duration", type=str, default="4h",
                   help="Recording duration (e.g. 4h, 30m, 7200s)")
    p.add_argument("--output",   type=str, default="data/btcusdt/sessions/",
                   help="Output directory for Parquet files")
    p.add_argument("--label",    type=str, default=None,
                   help="Session label (e.g. asia_quiet, us_open)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    recorder = SessionRecorder(
        symbol=args.symbol,
        output_dir=Path(args.output),
        duration_sec=_parse_duration(args.duration),
        label=args.label,
    )
    asyncio.run(recorder.run())


if __name__ == "__main__":
    main()
