"""
In-memory LOB tracker for live Binance Futures data.

Maintains bid and ask ladders as sorted dicts, updated via the Binance
incremental depth stream (depth20@100ms).  Provides the same snapshot
interface used by the training environment's `_build_obs()`.
"""
from __future__ import annotations

from typing import Any


class LOBTracker:
    """
    Lightweight order-book tracker backed by two plain dicts.

    Binance depth20 snapshots replace the top 20 levels wholesale on each
    update — no partial-update delta logic is required for that stream.
    """

    def __init__(self, levels: int = 5) -> None:
        self._levels = levels
        # price (float) → qty (float); maintained sorted lazily
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

    # ── ingestion ─────────────────────────────────────────────────────────────

    def apply_depth_snapshot(self, data: dict[str, Any]) -> None:
        """
        Replace book state from a Binance depth20@100ms message.

        ``data`` is the ``"data"`` payload of a combined stream message, or
        the direct response from ``GET /fapi/v1/depth``.  Both have the same
        shape: ``{"bids": [["price", "qty"], ...], "asks": [...]}``.
        """
        self._bids = {
            float(p): float(q)
            for p, q in data.get("b", data.get("bids", []))
            if float(q) > 0.0
        }
        self._asks = {
            float(p): float(q)
            for p, q in data.get("a", data.get("asks", []))
            if float(q) > 0.0
        }

    # ── derived quantities ────────────────────────────────────────────────────

    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    def mid(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def snapshot(self, levels: int | None = None) -> dict[str, list]:
        """
        Return sorted top-N bid and ask levels as parallel lists.

        Returns
        -------
        {
            "bid_price": [float, ...],   # descending
            "bid_qty":   [float, ...],
            "ask_price": [float, ...],   # ascending
            "ask_qty":   [float, ...],
        }
        Levels with no quote are filled with ``None``.
        """
        n = levels or self._levels

        bid_prices = sorted(self._bids, reverse=True)[:n]
        ask_prices = sorted(self._asks)[:n]

        def _pad(prices: list[float], side: dict[float, float]) -> tuple[list, list]:
            ps = list(prices)
            qs = [side[p] for p in ps]
            # pad to n if fewer levels exist
            while len(ps) < n:
                ps.append(None)   # type: ignore[arg-type]
                qs.append(None)   # type: ignore[arg-type]
            return ps, qs

        bp, bq = _pad(bid_prices, self._bids)
        ap, aq = _pad(ask_prices, self._asks)

        return {
            "bid_price": bp,
            "bid_qty":   bq,
            "ask_price": ap,
            "ask_qty":   aq,
        }

    def volume_at_levels(self, levels: int | None = None) -> tuple[float, float]:
        """Return (v_bid, v_ask) summed over top-N levels."""
        n  = levels or self._levels
        bp = sorted(self._bids, reverse=True)[:n]
        ap = sorted(self._asks)[:n]
        return sum(self._bids[p] for p in bp), sum(self._asks[p] for p in ap)
