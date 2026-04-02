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
  bid_quote:      number | null;
  ask_quote:      number | null;
  gamma:          number;
  kappa_offset:   number;
  fills_total:    number;
  sharpe:         number;
  skew_mode:      "normal" | "skew" | "suppress" | "dump" | "replay";
}

export interface Trade {
  price:    number;
  quantity: number;
  side:     "buy" | "sell";
  is_agent: boolean;
}

export interface TickData {
  type:             string;
  timestamp:        number;
  mid_price:        number;
  spread:           number;
  best_bid:         number;
  best_ask:         number;
  lob:              { bids: LobLevel[]; asks: LobLevel[] };
  agent:            AgentState;
  trades:           Trade[];
  replay_progress?: number;
}

export interface PricePoint {
  time: number;  // ms since epoch
  mid:  number;
  bid:  number;
  ask:  number;
}

export interface TradeRecord extends Trade {
  timestamp: number;  // ms since epoch
  sim_time:  number;  // simulation seconds (0–900)
}

export interface MidPoint {
  sim_time: number;
  mid:      number;
  bid:      number;
  ask:      number;
}

export interface SimConfig {
  seed:        number;
  speed:       number;
  strategy:    string;
  mode:        "simulate" | "replay";
  replayPath?: string;
}

interface SimState {
  isConnected:     boolean;
  isRunning:       boolean;
  isPaused:        boolean;
  lob:             { bids: LobLevel[]; asks: LobLevel[] } | null;
  agent:           AgentState | null;
  priceHistory:    PricePoint[];
  tradeHistory:    TradeRecord[];
  midHistory:      MidPoint[];
  eventsProcessed: number;
  elapsedTime:     number;
  mode:            "simulate" | "replay";
  replayProgress:  number;
  lastError:       string | null;
}

const INITIAL: SimState = {
  isConnected:     false,
  isRunning:       false,
  isPaused:        false,
  lob:             null,
  agent:           null,
  priceHistory:    [],
  tradeHistory:    [],
  midHistory:      [],
  eventsProcessed: 0,
  elapsedTime:     0,
  mode:            "simulate",
  replayProgress:  0,
  lastError:       null,
};

const MAX_PRICE_PTS  = 500;
const MAX_TRADE_PTS  = 3000;
const MAX_MID_PTS    = 1000;
const SESSION_RESET_THRESHOLD = 5; // seconds — if sim_time drops by this much, new session
const WS_URL         = "ws://localhost:8000/ws/live";

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useSimulation() {
  const wsRef          = useRef<WebSocket | null>(null);
  const simOffsetRef   = useRef<number>(0);
  const lastSimTimeRef = useRef<number>(0);
  const [state, setState] = useState<SimState>(INITIAL);

  const lastCfgRef = useRef<SimConfig>({ seed: 42, speed: 1.0, strategy: "as", mode: "simulate" });

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

      ws.onopen  = () => setState(() => ({ ...INITIAL, isConnected: true }));
      ws.onclose = () => {
        setState((s) => ({ ...s, isConnected: false, isRunning: false }));
        if (!cancelled) setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as { type: string; message?: string } & TickData;
          if (msg.type === "error") {
            setState((s) => ({ ...s, isRunning: false, isPaused: false, lastError: msg.message ?? "Unknown error" }));
            return;
          }
          const tick = msg as TickData;
          if (tick.type !== "tick") return;

          const now     = Date.now();
          const rawTime = tick.timestamp;

          if (rawTime < lastSimTimeRef.current - SESSION_RESET_THRESHOLD) {
            simOffsetRef.current += lastSimTimeRef.current;
          }
          lastSimTimeRef.current = rawTime;
          const simTime = simOffsetRef.current + rawTime;

          const newPt: PricePoint = {
            time: now,
            mid:  tick.mid_price,
            bid:  tick.agent.bid_quote ?? tick.mid_price,
            ask:  tick.agent.ask_quote ?? tick.mid_price,
          };
          const newMid: MidPoint = {
            sim_time: simTime,
            mid:      tick.mid_price,
            bid:      tick.agent.bid_quote ?? tick.mid_price,
            ask:      tick.agent.ask_quote ?? tick.mid_price,
          };
          const newTrades: TradeRecord[] = tick.trades.map((t) => ({
            ...t,
            timestamp: now,
            sim_time:  simTime,
          }));

          const progress    = tick.replay_progress ?? 0;
          const replayDone  = progress >= 1.0;

          setState((prev) => ({
            ...prev,
            isRunning:      replayDone ? false : prev.isRunning,
            lob:            tick.lob,
            agent:          tick.agent,
            elapsedTime:    simTime,
            eventsProcessed: prev.eventsProcessed + 1,
            replayProgress: tick.replay_progress !== undefined ? progress : prev.replayProgress,
            priceHistory: [
              ...prev.priceHistory.slice(-(MAX_PRICE_PTS - 1)),
              newPt,
            ],
            midHistory: [
              ...prev.midHistory.slice(-(MAX_MID_PTS - 1)),
              newMid,
            ],
            tradeHistory: [
              ...prev.tradeHistory.slice(-(MAX_TRADE_PTS - newTrades.length)),
              ...newTrades,
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
      send({
        action: "start",
        config: {
          mode:        cfg.mode,
          seed:        cfg.seed,
          speed:       cfg.speed,
          strategy:    cfg.strategy,
          replay_path: cfg.replayPath,
        },
      });
      setState((s) => ({
        ...s,
        isRunning:      true,
        isPaused:       false,
        mode:           cfg.mode,
        replayProgress: cfg.mode === "replay" ? 0 : s.replayProgress,
      }));
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
    send({ action: "start", config: {
      mode:        lastCfgRef.current.mode,
      seed:        lastCfgRef.current.seed,
      speed:       lastCfgRef.current.speed,
      strategy:    lastCfgRef.current.strategy,
      replay_path: lastCfgRef.current.replayPath,
    }});
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
      simOffsetRef.current   = 0;
      lastSimTimeRef.current = 0;
      setState((s) => ({
        ...INITIAL,
        isConnected:    s.isConnected,
        mode:           s.mode,
        replayProgress: 0,
      }));
    },
    [send],
  );

  return { ...state, start, stop, pause, resume, setSpeed, reset };
}
