"use client";

import { useEffect, useRef } from "react";
import type { TradeRecord } from "@/hooks/useSimulation";

interface TimesSalesProps {
  trades: TradeRecord[];
}

function fmtTime(simTime: number): string {
  const m = Math.floor(simTime / 60);
  const s = (simTime % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

export default function TimesSales({ trades }: TimesSalesProps) {
  const mktTrades = trades.filter((t) => !t.is_agent);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever new market trades arrive
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [mktTrades.length]);

  return (
    <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
      {/* Column labels */}
      <div className="grid grid-cols-[2.5rem_3rem_5rem_2.5rem] gap-1 px-2 py-0.5 border-b border-[#111] text-[#333] uppercase shrink-0">
        <span>TIME</span>
        <span>SIDE</span>
        <span className="text-right">PRICE</span>
        <span className="text-right">QTY</span>
      </div>

      {mktTrades.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-[#333] tracking-widest">
          AWAITING DATA
        </div>
      ) : (
        /* Rows — oldest first, scroll to bottom = newest */
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {mktTrades.map((t, i) => {
            const sideColor = t.side === "buy" ? "rgba(0,210,106,0.7)" : "rgba(255,59,59,0.7)";
            return (
              <div
                key={i}
                className="grid grid-cols-[2.5rem_3rem_5rem_2.5rem] gap-1 px-2 py-0.5 items-center border-b border-[#0a0a0a]"
              >
                <span className="text-[#333] tabular-nums">{fmtTime(t.sim_time)}</span>
                <span style={{ color: sideColor }} className="uppercase tabular-nums">{t.side === "buy" ? "B" : "S"}</span>
                <span style={{ color: sideColor }} className="text-right tabular-nums">{t.price.toFixed(4)}</span>
                <span className="text-right text-[#555] tabular-nums">{t.quantity}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
