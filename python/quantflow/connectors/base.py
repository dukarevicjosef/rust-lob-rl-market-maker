"""Abstract base class for exchange market data connectors."""
from __future__ import annotations

from abc import ABC, abstractmethod


class MarketDataConnector(ABC):
    """
    Unified interface for live market data feeds.

    Implementations maintain an in-memory order book and recent trade buffer,
    updated via WebSocket streams.  The paper trading runner and any future
    exchange integrations program against this interface.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish WebSocket connection and subscribe to market data."""

    @abstractmethod
    async def close(self) -> None:
        """Gracefully disconnect."""

    @abstractmethod
    def get_mid_price(self) -> float:
        """Current mid-price, or 0.0 if book is empty."""

    @abstractmethod
    def get_best_bid(self) -> tuple[float, float]:
        """Best bid as (price, qty). Returns (0.0, 0.0) if empty."""

    @abstractmethod
    def get_best_ask(self) -> tuple[float, float]:
        """Best ask as (price, qty). Returns (0.0, 0.0) if empty."""

    @abstractmethod
    def get_spread(self) -> float:
        """Current bid-ask spread in quote currency."""

    @abstractmethod
    def get_orderbook_snapshot(self, levels: int = 10) -> dict:
        """
        Return top-N levels in normalised format.

        Returns
        -------
        {
            "bids": [(price, qty), ...],   # descending by price
            "asks": [(price, qty), ...],   # ascending by price
            "mid_price": float,
            "spread": float,
            "best_bid": (price, qty),
            "best_ask": (price, qty),
        }
        """

    @abstractmethod
    def get_recent_trades(self, n: int = 100) -> list[dict]:
        """
        Last *n* trades in normalised format.

        Each dict: {"price": float, "qty": float, "side": "BUY"|"SELL",
                     "timestamp": str}
        """
