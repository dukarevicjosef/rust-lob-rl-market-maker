"use client";

import { useEffect, useRef } from "react";
import type { TradeRecord } from "@/hooks/useSimulation";

interface TimesSalesProps {
  trades:   TradeRecord[];
  isReplay?: boolean;
}

function fmtTime(simTime: number): string {
  if (simTime > 86400) {
    // absolute unix timestamp (seconds) — show wall-clock HH:MM:SS UTC
    return new Date(simTime * 1000).toISOString().slice(11, 19);
  }
  const m = Math.floor(simTime / 60);
  const s = (simTime % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function fmtPrice(price: number): string {
  return price >= 1000 ? price.toFixed(2) : price.toFixed(4);
}

function fmtQty(qty: number, isReplay: boolean): string {
  if (isReplay) return qty.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return String(qty);
}

export default function TimesSales({ trades, isReplay = false }: TimesSalesProps) {
  const mktTrades = trades.filter((t) => !t.is_agent);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [mktTrades.length]);

  return (
    <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
      {/* Column labels */}
      <div className="grid grid-cols-[4rem_3rem_5.5rem_4rem] gap-1 px-2 py-0.5 border-b border-[#111] text-[#333] uppercase shrink-0">
        <span>TIME</span>
        <span>SIDE</span>
        <span className="text-right">PRICE</span>
        <span className="text-right">QTY{isReplay ? " BTC" : ""}</span>
      </div>

      {mktTrades.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-[#333] tracking-widest">
          AWAITING DATA
        </div>
      ) : (
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {mktTrades.map((t, i) => {
            const isBuy      = t.side === "buy";
            const sideColor  = isBuy ? "#00d26a" : "#ff3b3b";
            return (
              <div
                key={i}
                className="grid grid-cols-[4rem_3rem_5.5rem_4rem] gap-1 px-2 py-0.5 items-center border-b border-[#0a0a0a]"
              >
                <span className="text-[#333] tabular-nums">{fmtTime(t.sim_time)}</span>
                <span style={{ color: sideColor }} className="font-bold uppercase tabular-nums">
                  {isBuy ? "BUY" : "SELL"}
                </span>
                <span style={{ color: sideColor }} className="text-right tabular-nums">
                  {fmtPrice(t.price)}
                </span>
                <span className="text-right text-[#555] tabular-nums">
                  {fmtQty(t.quantity, isReplay)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
