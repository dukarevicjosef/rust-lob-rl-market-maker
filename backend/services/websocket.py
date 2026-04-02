from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from fastapi import WebSocket

from quantflow import AvellanedaStoikov, HawkesSimulator

_SAC_BASE_KAPPA = 1.5   # must stay in sync with sac_agent.BASE_KAPPA

# ── Session config ────────────────────────────────────────────────────────────

_T_MAX          = 900.0   # simulated session length (seconds)
_TICK_SIZE      = 0.01    # price tick (matches Hawkes tick_size_f)
_INITIAL_MID    = 100.0
_EVENTS_PER_FRAME = 50    # Hawkes events per 100 ms at 1× speed
_QUOTE_INTERVAL = 1.0     # sim-seconds between quote refreshes
_QUOTE_QTY      = 10      # lots per resting quote
_INV_LIMIT      = 50      # hard inventory cap (must match AvellanedaStoikov param)
_DUMP_OFFSET    = 2 * _TICK_SIZE   # aggressive dump: 2 ticks through mid


# ── Simulation state ──────────────────────────────────────────────────────────

class SimState:
    """
    Drives the Rust Hawkes LOB simulator with an Avellaneda-Stoikov agent.

    The agent places resting limit orders; the Hawkes engine generates
    background order flow that fills them.  Fill detection is exact:
    we track order IDs returned by ``place_limit_order`` and match them
    against ``maker_id`` fields in incoming trade events.
    """

    def __init__(self, seed: int, strategy: str = "as") -> None:
        self.seed     = seed
        self.strategy = strategy
        # Load SAC model once; reused across sessions within the same runner.
        self._sac = None
        if strategy == "sac":
            try:
                from backend.services.sac_agent import SACAgent
                self._sac = SACAgent()
                print("SAC model loaded.")
            except (ImportError, FileNotFoundError) as exc:
                print(f"[warn] {exc} — falling back to AS strategy.")
                self.strategy = "as"
        self._init_session(seed)

    def _init_session(self, seed: int) -> None:
        self.sim = HawkesSimulator.new({
            "t_max":           _T_MAX,
            "initial_mid":     _INITIAL_MID,
            "tick_size_f":     _TICK_SIZE,
            "snapshot_interval": 50,
        })
        self.sim.reset(seed)

        # Avellaneda-Stoikov (2008) — fixed σ calibrated to Hawkes regime.
        # κ=50 gives half-spread ≈ 2 ticks at mid-session, narrowing toward T.
        self.strat = AvellanedaStoikov(
            gamma=0.1,
            kappa=50.0,
            t_end=_T_MAX,
            inventory_limit=_INV_LIMIT,
            sigma=0.01,
            spread_floor=_TICK_SIZE,
        )

        self.cash:       float    = 0.0
        self.inventory:  int      = 0
        self.fills:      int      = 0
        self.bid_id:     int|None = None
        self.ask_id:     int|None = None
        self.last_quote_t: float  = -_QUOTE_INTERVAL
        self.t:          float    = 0.0

        # Actual placed order prices for display (None = no resting order).
        self._last_bid_q:      float | None = None
        self._last_ask_q:      float | None = None
        # SAC-derived parameters for display.
        self._last_gamma:      float = self.strat.gamma
        self._last_kappa_off:  float = 0.0

        self._pending_trades: list[dict] = []
        self._pnl_hist:       list[float] = [0.0]

    # ── Public entry point ────────────────────────────────────────────────────

    def run_frame(self, n_events: int) -> dict[str, Any] | None:
        """
        Advance the simulator by *n_events* Hawkes events.
        Returns a tick dict suitable for JSON broadcast, or None if the
        book is still empty.
        """
        for _ in range(n_events):
            event = self.sim.step()
            if event is None:
                # Session exhausted — carry over cash and inventory into the
                # next seed so PnL and position accumulate across sessions.
                carry_cash      = self.cash
                carry_inventory = self.inventory
                carry_fills     = self.fills
                self.seed += 1
                self._init_session(self.seed)
                self.cash      = carry_cash
                self.inventory = carry_inventory
                self.fills     = carry_fills
                break
            self._process_event(event)

        mid = self.sim.mid_price()
        if mid is None:
            return None
        return self._build_tick(mid)

    # ── Event processing ──────────────────────────────────────────────────────

    def _process_event(self, event: dict) -> None:
        self.t = event["sim_time"]

        # Keep SAC rolling mid-price buffer up to date.
        if self._sac is not None:
            mid = self.sim.mid_price()
            if mid is not None:
                self._sac.update_mid(mid)

        for trade in event["trades"]:
            maker = trade["maker_id"]
            price = trade["price"]
            qty   = trade["qty"]

            if self.bid_id is not None and maker == self.bid_id:
                # Our resting bid was filled — we bought.
                self.inventory += qty
                self.cash      -= qty * price
                self.fills     += 1
                self._pending_trades.append({
                    "price": price, "quantity": qty,
                    "side": "buy", "is_agent": True,
                })
                self.bid_id = None

            elif self.ask_id is not None and maker == self.ask_id:
                # Our resting ask was filled — we sold.
                self.inventory -= qty
                self.cash      += qty * price
                self.fills     += 1
                self._pending_trades.append({
                    "price": price, "quantity": qty,
                    "side": "sell", "is_agent": True,
                })
                self.ask_id = None

            else:
                # Background market trade (Hawkes order flow).
                side = "buy" if event["event_type"] % 2 == 0 else "sell"
                self._pending_trades.append({
                    "price": price, "quantity": qty,
                    "side": side, "is_agent": False,
                })

        if self.t - self.last_quote_t >= _QUOTE_INTERVAL:
            self._refresh_quotes()

    # ── Quote management ──────────────────────────────────────────────────────

    def _refresh_quotes(self) -> None:
        """Cancel stale quotes and place fresh AS quotes."""
        mid = self.sim.mid_price()
        if mid is None:
            return

        # Cancel existing resting orders and clear stored display prices.
        if self.bid_id is not None:
            self.sim.cancel_agent_order(self.bid_id)
            self.bid_id    = None
            self._last_bid_q = None
        if self.ask_id is not None:
            self.sim.cancel_agent_order(self.ask_id)
            self.ask_id    = None
            self._last_ask_q = None

        book     = self.sim.get_book()
        best_bid = book.best_bid()
        best_ask = book.best_ask()

        inv_ratio = abs(self.inventory) / _INV_LIMIT

        if inv_ratio >= 1.0:
            # Inventory dump: aggressive limit order 2 ticks through mid.
            if self.inventory > 0:
                dump_p = round(mid - _DUMP_OFFSET, 4)
                # Safety: must still be > best_bid to avoid crossing bid side.
                if best_bid is None or dump_p > best_bid[0]:
                    try:
                        self.ask_id = self.sim.place_limit_order("ask", dump_p, _QUOTE_QTY)
                    except Exception:
                        pass
            else:
                dump_p = round(mid + _DUMP_OFFSET, 4)
                if best_ask is None or dump_p < best_ask[0]:
                    try:
                        self.bid_id = self.sim.place_limit_order("bid", dump_p, _QUOTE_QTY)
                    except Exception:
                        pass
        else:
            if self._sac is not None:
                # SAC agent: derive gamma and kappa dynamically each quote refresh.
                gamma, kappa_off = self._sac.get_action(
                    self.sim, mid, self.inventory, self.cash, self.t
                )
                kappa = _SAC_BASE_KAPPA * (1.0 + kappa_off)
                # AvellanedaStoikov is a PyO3 object — attributes are not
                # writable. Recreate with updated params instead.
                self.strat = AvellanedaStoikov(
                    gamma=gamma,
                    kappa=kappa,
                    t_end=_T_MAX,
                    inventory_limit=_INV_LIMIT,
                    sigma=0.01,
                    spread_floor=_TICK_SIZE,
                )
                self._last_gamma     = gamma
                self._last_kappa_off = kappa_off

            (bid_p, ask_p), mode = self.strat.compute_quotes_skewed(
                mid, self.inventory, self.t
            )
            bid_p = round(bid_p, 4)
            ask_p = round(ask_p, 4)

            if bid_p <= 0 or ask_p <= bid_p:
                self.last_quote_t = self.t
                return

            # Clamp to the live spread so quotes never cross the book.
            # When AS skewing pushes an ask below best_bid (or bid above
            # best_ask) we pin it one tick inside the market spread.  This
            # preserves the directional intent while staying valid.
            if best_ask is not None and bid_p >= best_ask[0]:
                bid_p = round(best_ask[0] - _TICK_SIZE, 4)
            if best_bid is not None and ask_p <= best_bid[0]:
                ask_p = round(best_bid[0] + _TICK_SIZE, 4)

            if bid_p <= 0 or ask_p <= bid_p:
                self.last_quote_t = self.t
                return

            place_bid = (mode != "suppress" or self.inventory < 0)
            place_ask = (mode != "suppress" or self.inventory > 0)

            if place_bid and abs(self.inventory + _QUOTE_QTY) <= _INV_LIMIT:
                try:
                    self.bid_id = self.sim.place_limit_order("bid", bid_p, _QUOTE_QTY)
                    self._last_bid_q = bid_p
                except Exception:
                    pass

            if place_ask and abs(self.inventory - _QUOTE_QTY) <= _INV_LIMIT:
                try:
                    self.ask_id = self.sim.place_limit_order("ask", ask_p, _QUOTE_QTY)
                    self._last_ask_q = ask_p
                except Exception:
                    pass

        self.last_quote_t = self.t

    # ── Tick assembly ─────────────────────────────────────────────────────────

    def _build_tick(self, mid: float) -> dict[str, Any]:
        # Use actual placed order prices; None when no resting order exists.
        bid_q = self._last_bid_q
        ask_q = self._last_ask_q

        # skew_mode is purely for display — compute from current strat params.
        _, skew_m = self.strat.compute_quotes_skewed(mid, self.inventory, self.t)
        inv_ratio = abs(self.inventory) / _INV_LIMIT
        if inv_ratio >= 1.0:
            skew_m = "dump"

        lob = self._lob_snapshot()

        pnl        = round(self.cash + self.inventory * mid, 2)
        unrealized = round(
            self.inventory * (mid - ask_q if (self.inventory > 0 and ask_q is not None)
                              else (bid_q - mid) if (self.inventory < 0 and bid_q is not None)
                              else 0.0),
            2,
        )

        self._pnl_hist.append(pnl)
        if len(self._pnl_hist) > 120:
            self._pnl_hist.pop(0)

        sharpe = 0.0
        if len(self._pnl_hist) > 10:
            rets = [b - a for a, b in zip(self._pnl_hist[:-1], self._pnl_hist[1:])]
            m    = sum(rets) / len(rets)
            v    = sum((r - m) ** 2 for r in rets) / len(rets)
            if v > 1e-10:
                sharpe = round(m / math.sqrt(v), 3)

        trades = list(self._pending_trades)
        self._pending_trades.clear()

        best_bid_p = lob["bids"][0]["price"] if lob["bids"] else round(mid - _TICK_SIZE, 4)
        best_ask_p = lob["asks"][0]["price"] if lob["asks"] else round(mid + _TICK_SIZE, 4)

        return {
            "type":      "tick",
            "timestamp": round(self.t, 3),
            "mid_price": round(mid, 4),
            "spread":    round(ask_q - bid_q, 4) if (ask_q is not None and bid_q is not None) else None,
            "best_bid":  best_bid_p,
            "best_ask":  best_ask_p,
            "lob":       lob,
            "agent": {
                "inventory":      self.inventory,
                "pnl":            pnl,
                "unrealized_pnl": unrealized,
                "bid_quote":      round(bid_q, 4) if bid_q is not None else None,
                "ask_quote":      round(ask_q, 4) if ask_q is not None else None,
                "gamma":          round(self._last_gamma, 3),
                "kappa_offset":   round(self._last_kappa_off, 4),
                "fills_total":    self.fills,
                "sharpe":         sharpe,
                "skew_mode":      skew_m,
            },
            "trades": trades,
        }

    def _lob_snapshot(self) -> dict[str, list[dict]]:
        try:
            rb         = self.sim.get_book().snapshot(10)
            bid_prices = rb.column("bid_price").to_pylist()
            bid_qtys   = rb.column("bid_qty").to_pylist()
            ask_prices = rb.column("ask_price").to_pylist()
            ask_qtys   = rb.column("ask_qty").to_pylist()
        except Exception:
            return {"bids": [], "asks": []}

        bids: list[dict] = []
        asks: list[dict] = []
        cum_b = cum_a = 0

        for p, q in zip(bid_prices, bid_qtys):
            if p is None or math.isnan(p) or not q:
                continue
            cum_b += q
            bids.append({"price": round(p, 4), "quantity": int(q), "cumulative": cum_b})

        for p, q in zip(ask_prices, ask_qtys):
            if p is None or math.isnan(p) or not q:
                continue
            cum_a += q
            asks.append({"price": round(p, 4), "quantity": int(q), "cumulative": cum_a})

        return {"bids": bids, "asks": asks}


# ── Simulation runner (asyncio task) ─────────────────────────────────────────

class SimulationRunner:
    _EVENTS_PER_FRAME = _EVENTS_PER_FRAME

    def __init__(self) -> None:
        self._state:   SimState | None     = None
        self._conns:   set[WebSocket]      = set()
        self._running: bool                = False
        self._speed:   float               = 1.0
        self._task:    asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def elapsed(self) -> float:
        return self._state.t if self._state else 0.0

    @property
    def n_connections(self) -> int:
        return len(self._conns)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._conns.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._conns.discard(ws)

    async def start(self, seed: int = 42, speed: float = 1.0, strategy: str = "as") -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.sleep(0)
        self._state   = SimState(seed=seed, strategy=strategy)
        self._speed   = max(0.25, min(20.0, speed))
        self._running = True
        self._task    = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.25, min(20.0, speed))

    async def reset(self, seed: int = 42) -> None:
        self.stop()
        self._state = None
        await asyncio.sleep(0.05)

    async def _broadcast(self, payload: str) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._conns):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._conns -= dead

    async def _loop(self) -> None:
        while self._running:
            if self._state and self._conns:
                n    = max(1, int(self._EVENTS_PER_FRAME * self._speed))
                tick = self._state.run_frame(n)
                if tick:
                    await self._broadcast(json.dumps(tick))
            await asyncio.sleep(0.1)


# ── Singleton ─────────────────────────────────────────────────────────────────

runner  = SimulationRunner()
manager = runner          # backward-compat alias
