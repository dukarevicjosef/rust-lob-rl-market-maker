"""
Paper trading runner.

Connects five concurrent async loops:
  1. _market_data_loop  — depth20 + aggTrade → LOB tracker + rolling buffers
  2. _quoting_loop      — obs → agent → AS quotes → safety rules → orders
  3. _fill_monitor_loop — User Data Stream → fill callbacks → risk manager
  4. _risk_monitor_loop — 1-second PnL / kill-switch watchdog
  5. _logging_loop      — periodic console + W&B metrics
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import numpy as np

import quantflow
from quantflow.envs.safety_rules import apply_safety_rules
from quantflow.paper_trading.config import PaperTradingConfig
from quantflow.paper_trading.exchange_client import ExchangeClient
from quantflow.paper_trading.lob_tracker import LOBTracker
from quantflow.paper_trading.obs_builder import ObservationBuilder

log = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Orchestrates all live paper-trading loops.

    Usage::

        cfg = PaperTradingConfig.from_env()
        cfg.validate()
        runner = PaperTradingRunner(cfg)
        asyncio.run(runner.run())
    """

    def __init__(self, cfg: PaperTradingConfig) -> None:
        self._cfg = cfg

        self._client = ExchangeClient(
            api_key    = cfg.api_key,
            api_secret = cfg.api_secret,
            testnet    = cfg.testnet,
        )
        self._lob     = LOBTracker(levels=5)
        self._obs_bld = ObservationBuilder(cfg)

        # Risk manager — loaded from risk_config dict
        self._risk = quantflow.RiskManager(cfg.risk_config, testnet=cfg.testnet)

        # SAC agent loaded from disk
        from stable_baselines3 import SAC
        self._agent = SAC.load(cfg.model_path)

        # Session state
        self._position_btc: float       = 0.0
        self._cash:         float       = 0.0
        self._session_start: float      = 0.0
        self._last_mid:     float       = 0.0
        self._bid_id:       int | None  = None
        self._ask_id:       int | None  = None
        self._last_quote_mid: float | None = None
        self._vol_ema:      float       = 0.0

        # Metrics for logging
        self._total_fills:   int   = 0
        self._total_pnl:     float = 0.0
        self._peak_pnl:      float = 0.0

        # Shutdown flag
        self._stop = asyncio.Event()

    # ── public entry point ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all loops and wait until stopped or max_duration reached."""
        self._session_start = time.monotonic()
        log.info(
            "Paper trading started | symbol=%s testnet=%s model=%s",
            self._cfg.symbol, self._cfg.testnet, self._cfg.model_path,
        )

        # Warm up: fetch current position and mid from REST
        try:
            self._position_btc = await self._client.get_position(self._cfg.symbol)
            self._last_mid = await self._client.get_ticker_price(self._cfg.symbol)
            log.info("Initial position=%.4f BTC mid=%.2f", self._position_btc, self._last_mid)
        except Exception as exc:
            log.warning("Warm-up REST failed: %s — continuing with defaults", exc)

        # Create user-data listen key
        listen_key = await self._client.create_listen_key()
        log.debug("Listen key acquired: %s…", listen_key[:8])

        tasks = [
            asyncio.create_task(self._market_data_loop(), name="market-data"),
            asyncio.create_task(self._quoting_loop(), name="quoting"),
            asyncio.create_task(self._fill_monitor_loop(listen_key), name="fill-monitor"),
            asyncio.create_task(self._risk_monitor_loop(), name="risk-monitor"),
            asyncio.create_task(self._logging_loop(), name="logging"),
            asyncio.create_task(self._keepalive_loop(listen_key), name="keepalive"),
        ]

        if self._cfg.max_duration_sec is not None:
            tasks.append(
                asyncio.create_task(
                    self._duration_watchdog(self._cfg.max_duration_sec),
                    name="watchdog",
                )
            )

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                if exc := task.exception():
                    log.error("Task %s raised: %s", task.get_name(), exc)
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown(listen_key)

    # ── loop 1: market data ───────────────────────────────────────────────────

    async def _market_data_loop(self) -> None:
        async with self._client.market_data_stream(self._cfg.symbol) as events:
            async for msg in events:
                if self._stop.is_set():
                    break
                stream = msg.get("stream", "")
                data   = msg.get("data", msg)

                if "depth20" in stream:
                    self._lob.apply_depth_snapshot(data)
                    mid = self._lob.mid()
                    if mid is not None:
                        self._last_mid = mid
                    elapsed = time.monotonic() - self._session_start
                    self._obs_bld.update_market(self._lob, elapsed)

                elif "aggTrade" in stream:
                    # m=True means the buyer is the market maker → sell aggressor
                    is_buyer_mm = data.get("m", False)
                    side_sign   = -1 if is_buyer_mm else 1
                    qty_btc     = float(data.get("q", 0.0))
                    elapsed     = time.monotonic() - self._session_start
                    self._obs_bld.record_trade(elapsed, side_sign, qty_btc)

    # ── loop 2: quoting ───────────────────────────────────────────────────────

    async def _quoting_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._cfg.quote_interval)
            if self._stop.is_set():
                break
            if self._risk.is_killed:
                log.warning("Kill switch active — quoting suspended")
                continue
            try:
                await self._refresh_quotes()
            except Exception as exc:
                log.error("Quote refresh failed: %s", exc)

    async def _refresh_quotes(self) -> None:
        cfg     = self._cfg
        mid     = self._last_mid
        if mid <= 0.0:
            return

        elapsed = time.monotonic() - self._session_start

        # Build observation
        obs = self._obs_bld.build_obs(
            lob          = self._lob,
            position_btc = self._position_btc,
            cash         = self._cash,
            session_time = elapsed,
        )

        # Agent inference
        action, _ = self._agent.predict(obs, deterministic=True)
        gamma      = float(np.clip(action[0], 0.01, 1.0))
        kappa_off  = float(np.clip(action[1], -0.5, 0.5))
        kappa      = cfg.base_kappa * (1.0 + kappa_off)

        # Map live position to training-space inventory for AS formula
        inv_lots = self._obs_bld.inventory_for_as(self._position_btc)
        t        = min(elapsed, cfg.t_max_session)

        # Avellaneda-Stoikov quotes
        strat  = quantflow.AvellanedaStoikov(
            gamma=gamma, kappa=kappa, t_end=cfg.t_max_session, sigma=cfg.sigma_fixed
        )
        bid_p, ask_p = strat.compute_quotes(mid=mid, inventory=inv_lots, t=t)

        # Safety rules — limits are in BTC (live); convert to lots for comparison
        pos_abs = abs(self._position_btc)
        _step_vol = self._obs_bld._realized_vol()
        if _step_vol > 0.0:
            self._vol_ema = 0.99 * self._vol_ema + 0.01 * _step_vol

        safe_bid, safe_ask, rules = apply_safety_rules(
            bid_p, ask_p, mid, inv_lots, _step_vol, self._vol_ema,
            self._last_quote_mid,
            inventory_soft_limit  = int(round(cfg.inventory_soft_limit_btc / (cfg.max_position_btc + 1e-9) * cfg.training_inventory_limit)),
            inventory_hard_limit  = int(round(cfg.inventory_hard_limit_btc / (cfg.max_position_btc + 1e-9) * cfg.training_inventory_limit)),
            tick_size             = cfg.tick_size,
            vol_spread_threshold  = cfg.vol_spread_threshold,
            vol_spread_multiplier = cfg.vol_spread_multiplier,
        )

        self._last_quote_mid = mid if (safe_bid is not None or safe_ask is not None) else None

        if rules.vol_regime:
            log.debug("Vol-regime rule fired — spread widened")
        if rules.quote_pull:
            log.debug("Quote-pull rule fired — quotes suppressed")
        if rules.inventory_hard:
            log.debug("Hard inventory limit hit — one-sided quoting")
        if rules.inventory_soft:
            log.debug("Soft inventory limit hit — one-sided quoting")

        # Cancel existing resting quotes
        await self._cancel_live_quotes()

        # Place new quotes
        await self._place_live_quotes(safe_bid, safe_ask)

    async def _cancel_live_quotes(self) -> None:
        tasks = []
        if self._bid_id is not None:
            tasks.append(self._client.cancel_order(self._cfg.symbol, self._bid_id))
            self._bid_id = None
        if self._ask_id is not None:
            tasks.append(self._client.cancel_order(self._cfg.symbol, self._ask_id))
            self._ask_id = None
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.debug("Cancel order failed (likely already filled): %s", r)
            for _ in tasks:
                self._risk.on_order_cancelled()

    @staticmethod
    def _min_qty(price: float, min_notional: float = 105.0) -> float:
        """Minimum BTC qty so that price × qty ≥ min_notional, rounded up to 3 dp."""
        return math.ceil(min_notional / price * 1000) / 1000

    async def _place_live_quotes(self, bid_p: float | None, ask_p: float | None) -> None:
        cfg = self._cfg

        if bid_p is not None and ask_p is not None and ask_p <= bid_p:
            return  # crossed — skip

        if bid_p is not None and bid_p > 0.0:
            bid_p_rounded = round(bid_p / cfg.tick_size) * cfg.tick_size
            qty = max(cfg.quote_qty_btc, self._min_qty(bid_p_rounded))
            try:
                self._risk.check_order("buy", qty, bid_p_rounded)
                resp = await self._client.place_limit_order(
                    cfg.symbol, "BUY", bid_p_rounded, qty
                )
                self._bid_id = int(resp["orderId"])
                self._risk.on_order_placed()
                log.debug("Placed BID %.2f × %.4f BTC (id=%d)", bid_p_rounded, qty, self._bid_id)
            except ValueError as exc:
                log.warning("Risk rejected BID: %s", exc)
            except Exception as exc:
                log.error("Failed to place BID: %s", exc)

        if ask_p is not None and ask_p > 0.0:
            ask_p_rounded = round(ask_p / cfg.tick_size) * cfg.tick_size
            qty = max(cfg.quote_qty_btc, self._min_qty(ask_p_rounded))
            try:
                self._risk.check_order("sell", qty, ask_p_rounded)
                resp = await self._client.place_limit_order(
                    cfg.symbol, "SELL", ask_p_rounded, qty
                )
                self._ask_id = int(resp["orderId"])
                self._risk.on_order_placed()
                log.debug("Placed ASK %.2f × %.4f BTC (id=%d)", ask_p_rounded, qty, self._ask_id)
            except ValueError as exc:
                log.warning("Risk rejected ASK: %s", exc)
            except Exception as exc:
                log.error("Failed to place ASK: %s", exc)

    # ── loop 3: fill monitor ──────────────────────────────────────────────────

    async def _fill_monitor_loop(self, listen_key: str) -> None:
        async with self._client.user_data_stream(listen_key) as events:
            async for event in events:
                if self._stop.is_set():
                    break
                etype = event.get("e", "")
                if etype == "ORDER_TRADE_UPDATE":
                    await self._handle_order_update(event.get("o", {}))
                elif etype == "listenKeyExpired":
                    log.warning("Listen key expired — attempting renewal")
                    try:
                        listen_key = await self._client.create_listen_key()
                    except Exception as exc:
                        log.error("Could not renew listen key: %s", exc)

    async def _handle_order_update(self, order: dict[str, Any]) -> None:
        status   = order.get("X", "")
        order_id = int(order.get("i", 0))
        side     = order.get("S", "")
        price    = float(order.get("L", 0.0))   # last filled price
        qty      = float(order.get("l", 0.0))   # last filled qty

        if status not in ("FILLED", "PARTIALLY_FILLED"):
            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                # Decrement open orders if we still track this order
                if order_id in (self._bid_id, self._ask_id):
                    self._risk.on_order_cancelled()
                    if order_id == self._bid_id:
                        self._bid_id = None
                    if order_id == self._ask_id:
                        self._ask_id = None
            return

        if qty <= 0.0:
            return

        log.info("FILL | %s %.4f BTC @ %.2f", side, qty, price)

        elapsed = time.monotonic() - self._session_start

        if side.upper() == "BUY":
            self._position_btc += qty
            self._cash         -= price * qty
            if order_id == self._bid_id:
                self._bid_id = None
        else:
            self._position_btc -= qty
            self._cash         += price * qty
            if order_id == self._ask_id:
                self._ask_id = None

        self._risk.on_fill(side.lower(), qty, price)
        self._obs_bld.record_fill(elapsed, side, price)
        self._total_fills += 1

    # ── loop 4: risk monitor ──────────────────────────────────────────────────

    async def _risk_monitor_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            mid = self._last_mid
            if mid <= 0.0:
                continue

            mtm_pnl = self._cash + self._position_btc * mid
            self._total_pnl = mtm_pnl
            if mtm_pnl > self._peak_pnl:
                self._peak_pnl = mtm_pnl

            self._risk.update_pnl(mtm_pnl, self._peak_pnl)

            if self._risk.is_killed:
                log.critical("KILL SWITCH ACTIVE — flattening position")
                await self._emergency_flatten()
                break

    async def _emergency_flatten(self) -> None:
        """Cancel all resting orders and send a market order to flatten."""
        try:
            await self._client.cancel_all_orders(self._cfg.symbol)
        except Exception as exc:
            log.error("Emergency cancel_all failed: %s", exc)

        self._bid_id = None
        self._ask_id = None

        pos = self._position_btc
        if abs(pos) < 1e-6:
            return

        side = "SELL" if pos > 0 else "BUY"
        qty  = round(abs(pos), 3)
        try:
            # Market order — bypass risk manager (kill switch override)
            await asyncio.to_thread(
                self._client._request_sync,
                "POST", "/fapi/v1/order",
                {
                    "symbol":   self._cfg.symbol,
                    "side":     side,
                    "type":     "MARKET",
                    "quantity": f"{qty:.3f}",
                    "timestamp": int(time.time() * 1000),
                },
                True,
            )
            log.info("Emergency flatten: %s %.4f BTC", side, qty)
        except Exception as exc:
            log.error("Emergency flatten order failed: %s", exc)

    # ── loop 5: logging ───────────────────────────────────────────────────────

    async def _logging_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._cfg.log_interval)
            status  = self._risk.status()
            elapsed = time.monotonic() - self._session_start
            mid     = self._last_mid
            mtm     = self._cash + self._position_btc * mid

            log.info(
                "t=%ds | mid=%.2f | pos=%.4f BTC | PnL=%.2f USDT | fills=%d | "
                "open_orders=%d | kill=%s",
                int(elapsed),
                mid,
                self._position_btc,
                mtm,
                self._total_fills,
                status["open_orders"],
                status["is_killed"],
            )

            if self._cfg.use_wandb:
                self._log_wandb(elapsed, mid, mtm, status)

    def _log_wandb(
        self,
        elapsed:  float,
        mid:      float,
        mtm:      float,
        status:   dict,
    ) -> None:
        try:
            import wandb
            wandb.log({
                "time_s":         elapsed,
                "mid_price":      mid,
                "position_btc":   self._position_btc,
                "pnl_usdt":       mtm,
                "peak_pnl":       self._peak_pnl,
                "drawdown_usdt":  self._peak_pnl - mtm,
                "total_fills":    self._total_fills,
                "open_orders":    status["open_orders"],
                "is_killed":      int(status["is_killed"]),
                "vol_ema":        self._vol_ema,
            })
        except Exception as exc:
            log.debug("W&B log failed: %s", exc)

    # ── keepalive loop ────────────────────────────────────────────────────────

    async def _keepalive_loop(self, listen_key: str) -> None:
        """Ping the User Data Stream listen key every 25 minutes."""
        while not self._stop.is_set():
            await asyncio.sleep(25 * 60)
            if self._stop.is_set():
                break
            try:
                await self._client.keepalive_listen_key(listen_key)
                log.debug("Listen key keepalive sent")
            except Exception as exc:
                log.warning("Listen key keepalive failed: %s", exc)

    # ── duration watchdog ─────────────────────────────────────────────────────

    async def _duration_watchdog(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        log.info("Max duration %.0fs reached — stopping", seconds)
        self._stop.set()

    # ── shutdown ──────────────────────────────────────────────────────────────

    async def _shutdown(self, listen_key: str) -> None:
        log.info("Shutting down — cancelling all open orders")
        try:
            await self._client.cancel_all_orders(self._cfg.symbol)
        except Exception as exc:
            log.error("Shutdown cancel_all failed: %s", exc)
        try:
            await self._client.close_listen_key(listen_key)
        except Exception as exc:
            log.debug("Listen key close failed: %s", exc)

        elapsed = time.monotonic() - self._session_start
        mid     = self._last_mid
        final   = self._cash + self._position_btc * mid
        log.info(
            "Session ended | duration=%.0fs | final_pos=%.4f BTC | final_PnL=%.2f USDT",
            elapsed, self._position_btc, final,
        )

        if self._cfg.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
