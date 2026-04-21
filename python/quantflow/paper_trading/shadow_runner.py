"""
Shadow mode runner for mainnet evaluation without real orders.

Observes a live orderbook via a MarketDataConnector, runs the SAC agent
to compute quotes, and simulates fills against real trades. No orders are
placed on the exchange — all accounting is local.

Known limitations (standard for shadow/backtest evaluations):
- Queue position ignored: any trade touching our price triggers a fill.
  Real fill rate would be lower due to queue priority.
- Market impact ignored: our orders would alter the book in reality.
- Partial fills simplified: always fills full quote qty.

Usage
-----
    uv run python -m quantflow.paper_trading.shadow \\
        --model runs/sac_2M_cloud/best_model.zip \\
        --exchange dydx --symbol BTC-USD --duration 3600
"""
from __future__ import annotations

import asyncio
import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import quantflow
from quantflow.connectors.base import MarketDataConnector
from quantflow.envs.safety_rules import apply_safety_rules
from quantflow.paper_trading.lob_tracker import LOBTracker
from quantflow.paper_trading.obs_builder import ObservationBuilder
from quantflow.paper_trading.config import PaperTradingConfig

log = logging.getLogger(__name__)


@dataclass
class ShadowQuote:
    price: float
    qty: float


@dataclass
class ShadowConfig:
    """Configuration for shadow mode evaluation."""
    model_path: str = "runs/sac_2M_cloud/best_model.zip"
    symbol: str = "BTC-USD"
    exchange: str = "dydx"
    testnet: bool = False

    # Quoting
    quote_interval: float = 5.0
    quote_qty_btc: float = 0.001
    base_kappa: float = 1.5
    sigma_fixed: float = 0.02

    # Fees
    maker_fee_bps: float = 0.0

    # Safety
    min_spread_multiplier: float = 1.0
    max_position_btc: float = 0.01
    inventory_soft_pct: float = 0.7
    inventory_hard_pct: float = 0.9

    # Session
    duration: float = 3600.0
    log_interval: float = 30.0
    t_max_session: float = 86_400.0

    # Training params (for obs normalization)
    training_inventory_limit: int = 50
    qty_scale_btc: float = 2.0

    def to_paper_config(self) -> PaperTradingConfig:
        """Build a PaperTradingConfig for the ObservationBuilder."""
        return PaperTradingConfig(
            symbol=self.symbol,
            testnet=self.testnet,
            model_path=self.model_path,
            quote_interval=self.quote_interval,
            quote_qty_btc=self.quote_qty_btc,
            base_kappa=self.base_kappa,
            sigma_fixed=self.sigma_fixed,
            t_max_session=self.t_max_session,
            training_inventory_limit=self.training_inventory_limit,
            qty_scale_btc=self.qty_scale_btc,
            maker_fee_bps=self.maker_fee_bps,
            min_spread_multiplier=self.min_spread_multiplier,
            risk_config={"max_position": self.max_position_btc},
        )


class ShadowRunner:
    """
    Observes a live market, computes agent quotes, and simulates fills
    against real trades without placing any orders.
    """

    def __init__(
        self,
        connector: MarketDataConnector,
        config: ShadowConfig,
    ) -> None:
        self._connector = connector
        self._cfg = config

        # Agent
        from stable_baselines3 import SAC
        self._agent = SAC.load(config.model_path)

        # Observation builder (reuses existing paper trading obs builder)
        self._paper_cfg = config.to_paper_config()
        self._lob = LOBTracker(levels=5)
        self._obs_bld = ObservationBuilder(self._paper_cfg)

        # Simulated state
        self._position_btc: float = 0.0
        self._cash: float = 0.0
        self._total_fees: float = 0.0
        self._total_fills: int = 0
        self._quote_cycles: int = 0
        self._spread_floor_hits: int = 0

        # Active quotes (None = no resting quote)
        self._active_bid: ShadowQuote | None = None
        self._active_ask: ShadowQuote | None = None

        # Tracking
        self._peak_pnl: float = 0.0
        self._pnl_history: list[float] = []
        self._inventory_history: list[float] = []
        self._spreads_offered: list[float] = []
        self._market_spreads: list[float] = []

        # Safety rule state
        self._last_quote_mid: float | None = None
        self._vol_ema: float = 0.0

        # Trade processing state
        self._last_trade_idx: int = 0

        # Shutdown
        self._stop = asyncio.Event()

    # ── main entry ───────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Run shadow evaluation. Returns session summary dict."""
        log.info(
            "Shadow mode started | exchange=%s symbol=%s model=%s",
            self._cfg.exchange, self._cfg.symbol, self._cfg.model_path,
        )

        start = time.monotonic()

        tasks = [
            asyncio.create_task(self._market_feed_loop(), name="market-feed"),
            asyncio.create_task(self._quoting_loop(start), name="quoting"),
            asyncio.create_task(self._fill_check_loop(), name="fill-check"),
            asyncio.create_task(self._logging_loop(start), name="logging"),
            asyncio.create_task(self._duration_watchdog(), name="watchdog"),
        ]

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if exc := task.exception():
                    log.error("Task %s raised: %s", task.get_name(), exc)
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        return self._print_summary(time.monotonic() - start)

    # ── market feed ──────────────────────────────────────────────────────────

    async def _market_feed_loop(self) -> None:
        """Continuously sync LOBTracker from the connector's book state."""
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
            snap = self._connector.get_orderbook_snapshot(levels=20)
            # Convert connector snapshot into LOBTracker-compatible format
            bids_data = [[str(p), str(q)] for p, q in snap["bids"]]
            asks_data = [[str(p), str(q)] for p, q in snap["asks"]]
            self._lob.apply_depth_snapshot({"bids": bids_data, "asks": asks_data})

            mid = self._lob.mid()
            elapsed = time.monotonic() - self._session_start if hasattr(self, "_session_start") else 0.0
            if mid is not None:
                self._obs_bld.update_market(self._lob, elapsed)

            # Feed trades to obs builder
            trades = self._connector.get_recent_trades(100)
            for t in trades[self._last_trade_idx:]:
                side_sign = 1 if t["side"] == "BUY" else -1
                self._obs_bld.record_trade(elapsed, side_sign, t["qty"])
            self._last_trade_idx = len(trades)

    # ── quoting ──────────────────────────────────────────────────────────────

    async def _quoting_loop(self, start: float) -> None:
        self._session_start = start
        # Wait for initial market data
        await asyncio.sleep(2.0)

        while not self._stop.is_set():
            await asyncio.sleep(self._cfg.quote_interval)
            if self._stop.is_set():
                break
            try:
                self._refresh_quotes()
            except Exception as exc:
                log.error("Quote refresh failed: %s", exc)

    def _refresh_quotes(self) -> None:
        cfg = self._cfg
        mid = self._connector.get_mid_price()
        if mid <= 0.0:
            return

        elapsed = time.monotonic() - self._session_start

        # Build observation
        obs = self._obs_bld.build_obs(
            lob=self._lob,
            position_btc=self._position_btc,
            cash=self._cash,
            session_time=elapsed,
        )

        # Agent inference
        action, _ = self._agent.predict(obs, deterministic=True)
        gamma = float(np.clip(action[0], 0.01, 1.0))
        kappa_off = float(np.clip(action[1], -0.5, 0.5))
        kappa = cfg.base_kappa * (1.0 + kappa_off)

        # AS quotes
        inv_lots = self._obs_bld.inventory_for_as(self._position_btc)
        t = min(elapsed, cfg.t_max_session)
        strat = quantflow.AvellanedaStoikov(
            gamma=gamma, kappa=kappa, t_end=cfg.t_max_session, sigma=cfg.sigma_fixed,
        )
        bid_p, ask_p = strat.compute_quotes(mid=mid, inventory=inv_lots, t=t)

        # Vol EMA
        step_vol = self._obs_bld._realized_vol()
        if step_vol > 0.0:
            self._vol_ema = 0.99 * self._vol_ema + 0.01 * step_vol

        # Safety rules
        soft_limit = int(round(cfg.inventory_soft_pct * cfg.training_inventory_limit))
        hard_limit = int(round(cfg.inventory_hard_pct * cfg.training_inventory_limit))

        safe_bid, safe_ask, rules = apply_safety_rules(
            bid_p, ask_p, mid, inv_lots, step_vol, self._vol_ema,
            self._last_quote_mid,
            inventory_soft_limit=soft_limit,
            inventory_hard_limit=hard_limit,
            tick_size=1.0,  # dYdX tick = $1
            vol_spread_threshold=2.0,
            vol_spread_multiplier=2.0,
        )

        self._last_quote_mid = mid if (safe_bid is not None or safe_ask is not None) else None

        # Spread floor
        self._quote_cycles += 1
        if safe_bid is not None and safe_ask is not None:
            rt_fee_rate = 2.0 * cfg.maker_fee_bps / 10_000.0
            min_spread = mid * rt_fee_rate * cfg.min_spread_multiplier
            agent_spread = safe_ask - safe_bid

            if min_spread > 0.0 and agent_spread < min_spread:
                half_min = min_spread / 2.0
                safe_bid = mid - half_min
                safe_ask = mid + half_min
                self._spread_floor_hits += 1

        # Record spreads
        mkt_spread = self._connector.get_spread()
        self._market_spreads.append(mkt_spread)
        if safe_bid is not None and safe_ask is not None:
            self._spreads_offered.append(safe_ask - safe_bid)

        # Set active quotes
        qty = cfg.quote_qty_btc
        self._active_bid = ShadowQuote(safe_bid, qty) if safe_bid is not None else None
        self._active_ask = ShadowQuote(safe_ask, qty) if safe_ask is not None else None

    # ── fill simulation ──────────────────────────────────────────────────────

    async def _fill_check_loop(self) -> None:
        """Check for simulated fills against real trades every 0.5s."""
        seen = 0
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
            trades = self._connector.get_recent_trades(1000)
            new_trades = trades[seen:]
            seen = len(trades)

            for t in new_trades:
                self._check_fill(t)

    def _check_fill(self, trade: dict) -> None:
        cfg = self._cfg
        price = trade["price"]
        qty = trade["qty"]
        fee_rate = cfg.maker_fee_bps / 10_000.0

        # Bid fill: trade at or below our bid price
        if self._active_bid is not None and price <= self._active_bid.price:
            fill_qty = min(qty, self._active_bid.qty)
            fill_price = self._active_bid.price
            fee = fill_price * fill_qty * fee_rate

            # Check position limit
            if abs(self._position_btc + fill_qty) > cfg.max_position_btc:
                return

            self._position_btc += fill_qty
            self._cash -= fill_price * fill_qty + fee
            self._total_fees += fee
            self._total_fills += 1
            self._active_bid = None

            elapsed = time.monotonic() - self._session_start
            self._obs_bld.record_fill(elapsed, "BUY", fill_price)
            log.info("SHADOW FILL | BUY %.4f @ %.1f (fee=%.4f)", fill_qty, fill_price, fee)

        # Ask fill: trade at or above our ask price
        if self._active_ask is not None and price >= self._active_ask.price:
            fill_qty = min(qty, self._active_ask.qty)
            fill_price = self._active_ask.price
            fee = fill_price * fill_qty * fee_rate

            if abs(self._position_btc - fill_qty) > cfg.max_position_btc:
                return

            self._position_btc -= fill_qty
            self._cash += fill_price * fill_qty - fee
            self._total_fees += fee
            self._total_fills += 1
            self._active_ask = None

            elapsed = time.monotonic() - self._session_start
            self._obs_bld.record_fill(elapsed, "SELL", fill_price)
            log.info("SHADOW FILL | SELL %.4f @ %.1f (fee=%.4f)", fill_qty, fill_price, fee)

    # ── logging ──────────────────────────────────────────────────────────────

    async def _logging_loop(self, start: float) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._cfg.log_interval)
            elapsed = time.monotonic() - start
            mid = self._connector.get_mid_price()

            net_pnl = self._cash + self._position_btc * mid
            gross_pnl = net_pnl + self._total_fees
            if net_pnl > self._peak_pnl:
                self._peak_pnl = net_pnl

            self._pnl_history.append(net_pnl)
            self._inventory_history.append(self._position_btc)

            mkt_spread = self._connector.get_spread()
            offered = (
                self._active_ask.price - self._active_bid.price
                if self._active_bid and self._active_ask else 0.0
            )

            log.info(
                "t=%ds | mid=%.1f | pos=%.4f BTC | grossPnL=%.2f | "
                "fees=%.2f | netPnL=%.2f | fills=%d | "
                "offered=%.1f | mkt_spread=%.1f | shadow=True",
                int(elapsed), mid, self._position_btc,
                gross_pnl, self._total_fees, net_pnl,
                self._total_fills, offered, mkt_spread,
            )

    async def _duration_watchdog(self) -> None:
        await asyncio.sleep(self._cfg.duration)
        log.info("Duration %.0fs reached — stopping", self._cfg.duration)
        self._stop.set()

    # ── summary ──────────────────────────────────────────────────────────────

    def _print_summary(self, elapsed: float) -> dict[str, Any]:
        mid = self._connector.get_mid_price()
        net_pnl = self._cash + self._position_btc * mid
        gross_pnl = net_pnl + self._total_fees
        drawdown = self._peak_pnl - net_pnl

        avg_spread_offered = (
            sum(self._spreads_offered) / len(self._spreads_offered)
            if self._spreads_offered else 0.0
        )
        avg_market_spread = (
            sum(self._market_spreads) / len(self._market_spreads)
            if self._market_spreads else 0.0
        )
        fill_rate = (
            self._total_fills / max(1, self._quote_cycles) * 100.0
        )

        inv_arr = np.array(self._inventory_history) if self._inventory_history else np.zeros(1)
        pnl_arr = np.array(self._pnl_history) if self._pnl_history else np.zeros(1)
        pnl_returns = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.zeros(1)
        sharpe = (
            float(np.mean(pnl_returns) / (np.std(pnl_returns) + 1e-9) * np.sqrt(len(pnl_returns)))
            if len(pnl_returns) > 1 else 0.0
        )

        summary = {
            "exchange": self._cfg.exchange,
            "symbol": self._cfg.symbol,
            "duration_s": elapsed,
            "maker_fee_bps": self._cfg.maker_fee_bps,
            "fills": self._total_fills,
            "gross_pnl": gross_pnl,
            "total_fees": self._total_fees,
            "net_pnl": net_pnl,
            "avg_spread_offered": avg_spread_offered,
            "avg_market_spread": avg_market_spread,
            "fill_rate_pct": fill_rate,
            "max_inventory": float(np.max(np.abs(inv_arr))) if len(inv_arr) > 0 else 0.0,
            "inventory_std": float(np.std(inv_arr)),
            "max_drawdown": float(np.max(np.abs(np.diff(pnl_arr)))) if len(pnl_arr) > 1 else 0.0,
            "peak_to_trough": drawdown,
            "sharpe": sharpe,
        }

        print()
        print("=" * 60)
        print("  SHADOW TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"  Exchange:          {self._cfg.exchange.upper()} {'Testnet' if self._cfg.testnet else 'Mainnet'}")
        print(f"  Symbol:            {self._cfg.symbol}")
        print(f"  Duration:          {elapsed:.0f}s")
        print(f"  Maker Fee:         {self._cfg.maker_fee_bps} bps")
        print()
        print(f"  Fills:             {self._total_fills}")
        print(f"  Gross PnL:         {gross_pnl:+.2f} USD")
        print(f"  Total Fees:        {self._total_fees:.2f} USD")
        print(f"  Net PnL:           {net_pnl:+.2f} USD")
        print()
        print(f"  Avg Spread Offered: {avg_spread_offered:.1f} USD")
        print(f"  Avg Market Spread:  {avg_market_spread:.1f} USD")
        print(f"  Fill Rate:          {fill_rate:.1f}% ({self._total_fills}/{self._quote_cycles})")
        print()
        print(f"  Max Inventory:     {float(np.max(np.abs(inv_arr))):.4f} BTC")
        print(f"  Inventory Std:     {float(np.std(inv_arr)):.4f} BTC")
        print(f"  Max Drawdown:      {drawdown:.2f} USD")
        print(f"  Sharpe Ratio:      {sharpe:.2f}")
        print()

        return summary
