"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// ── Types ────────────────────────────────────────────────────────────────────

export interface LobLevel {
  price: number;
  quantity: number;
  cumulative: number;
}

export interface AgentState {
  inventory:      number;
  pnl:            number;
  unrealized_pnl: number;
  bid_quote:      number;
  ask_quote:      number;
  gamma:          number;
  kappa_offset:   number;
  fills_total:    number;
  sharpe:         number;
  skew_mode:      "normal" | "skew" | "suppress" | "dump";
}

export interface Trade {
  price:    number;
  quantity: number;
  side:     "buy" | "sell";
  is_agent: boolean;
}

export interface TickData {
  type:       string;
  timestamp:  number;
  mid_price:  number;
  spread:     number;
  best_bid:   number;
  best_ask:   number;
  lob:        { bids: LobLevel[]; asks: LobLevel[] };
  agent:      AgentState;
  trades:     Trade[];
}

export interface PricePoint {
  time: number;  // ms since epoch
  mid:  number;
  bid:  number;
  ask:  number;
}

export interface TradeRecord extends Trade {
  timestamp: number;  // ms since epoch
}

export interface SimConfig {
  seed:     number;
  speed:    number;
  strategy: string;
}

interface SimState {
  isConnected:     boolean;
  isRunning:       boolean;
  isPaused:        boolean;
  lob:             { bids: LobLevel[]; asks: LobLevel[] } | null;
  agent:           AgentState | null;
  priceHistory:    PricePoint[];
  tradeHistory:    TradeRecord[];
  eventsProcessed: number;
  elapsedTime:     number;
}

const INITIAL: SimState = {
  isConnected:     false,
  isRunning:       false,
  isPaused:        false,
  lob:             null,
  agent:           null,
  priceHistory:    [],
  tradeHistory:    [],
  eventsProcessed: 0,
  elapsedTime:     0,
};

const MAX_PRICE_PTS  = 500;
const MAX_TRADE_PTS  = 80;
const WS_URL         = "ws://localhost:8000/ws/live";

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useSimulation() {
  const wsRef  = useRef<WebSocket | null>(null);
  const [state, setState] = useState<SimState>(INITIAL);

  // Pending config for resume
  const lastCfgRef = useRef<SimConfig>({ seed: 42, speed: 1.0, strategy: "as" });

  // ── Send helper ────────────────────────────────────────────────────────────

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  // ── WebSocket lifecycle ────────────────────────────────────────────────────

  useEffect(() => {
    let ws: WebSocket;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen  = () => setState((s) => ({ ...s, isConnected: true }));
      ws.onclose = () => {
        setState((s) => ({ ...s, isConnected: false, isRunning: false }));
        // Reconnect after 2s
        if (!cancelled) setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        try {
          const tick: TickData = JSON.parse(ev.data as string);
          if (tick.type !== "tick") return;

          const now = Date.now();
          const newPt: PricePoint = {
            time: now,
            mid:  tick.mid_price,
            bid:  tick.agent.bid_quote,
            ask:  tick.agent.ask_quote,
          };
          const newTrades: TradeRecord[] = tick.trades.map((t) => ({
            ...t,
            timestamp: now,
          }));

          setState((prev) => ({
            ...prev,
            isRunning:       true,
            lob:             tick.lob,
            agent:           tick.agent,
            elapsedTime:     tick.timestamp,
            eventsProcessed: prev.eventsProcessed + 1,
            priceHistory: [
              ...prev.priceHistory.slice(-(MAX_PRICE_PTS - 1)),
              newPt,
            ],
            tradeHistory: [
              ...newTrades,
              ...prev.tradeHistory.slice(0, MAX_TRADE_PTS - newTrades.length),
            ],
          }));
        } catch {
          // ignore
        }
      };
    };

    connect();
    return () => {
      cancelled = true;
      ws?.close();
      wsRef.current = null;
    };
  }, []);

  // ── Controls ───────────────────────────────────────────────────────────────

  const start = useCallback(
    (cfg: SimConfig) => {
      lastCfgRef.current = cfg;
      send({ action: "start", config: cfg });
      setState((s) => ({ ...s, isRunning: true, isPaused: false }));
    },
    [send],
  );

  const stop = useCallback(() => {
    send({ action: "stop" });
    setState((s) => ({ ...s, isRunning: false, isPaused: false }));
  }, [send]);

  const pause = useCallback(() => {
    send({ action: "stop" });
    setState((s) => ({ ...s, isRunning: false, isPaused: true }));
  }, [send]);

  const resume = useCallback(() => {
    send({ action: "start", config: lastCfgRef.current });
    setState((s) => ({ ...s, isRunning: true, isPaused: false }));
  }, [send]);

  const setSpeed = useCallback(
    (speed: number) => {
      lastCfgRef.current = { ...lastCfgRef.current, speed };
      send({ action: "set_speed", speed });
    },
    [send],
  );

  const reset = useCallback(
    (seed: number) => {
      send({ action: "reset", seed });
      setState((s) => ({ ...INITIAL, isConnected: s.isConnected }));
    },
    [send],
  );

  return { ...state, start, stop, pause, resume, setSpeed, reset };
}
