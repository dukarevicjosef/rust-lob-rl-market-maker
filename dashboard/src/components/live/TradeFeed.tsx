"use client";

import { useEffect, useRef } from "react";
import type { TradeRecord } from "@/hooks/useSimulation";

interface TradeFeedProps {
  trades: TradeRecord[];
}

function fmtTime(simTime: number): string {
  const m = Math.floor(simTime / 60);
  const s = (simTime % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

export default function TradeFeed({ trades }: TradeFeedProps) {
  const agentTrades = trades.filter((t) => t.is_agent);
  const scrollRef   = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever new fills arrive
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [agentTrades.length]);

  if (agentTrades.length === 0) {
    return (
      <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
        <div className="grid grid-cols-[2.5rem_3.5rem_5rem_2.5rem] gap-1 px-2 py-0.5 border-b border-[#111] text-[#333] uppercase shrink-0">
          <span>TIME</span>
          <span>SIDE</span>
          <span className="text-right">PRICE</span>
          <span className="text-right">QTY</span>
        </div>
        <div className="flex-1 flex items-center justify-center text-[#333] tracking-widest">
          NO FILLS YET
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
      {/* Column labels */}
      <div className="grid grid-cols-[2.5rem_3.5rem_5rem_2.5rem] gap-1 px-2 py-0.5 border-b border-[#111] text-[#333] uppercase shrink-0">
        <span>TIME</span>
        <span>SIDE</span>
        <span className="text-right">PRICE</span>
        <span className="text-right">QTY</span>
      </div>

      {/* Rows — oldest first, scroll to bottom = newest */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {agentTrades.map((t, i) => {
          const sideColor = t.side === "buy" ? "#00d26a" : "#ff3b3b";
          return (
            <div
              key={i}
              className="grid grid-cols-[2.5rem_3.5rem_5rem_2.5rem] gap-1 px-2 py-0.5 items-center border-b border-[#0f0f0f] border-l-2 border-l-[#3b82f6]"
            >
              <span className="text-[#444] tabular-nums">{fmtTime(t.sim_time)}</span>
              <span style={{ color: sideColor }} className="font-bold uppercase">{t.side}</span>
              <span style={{ color: sideColor }} className="text-right tabular-nums">{t.price.toFixed(4)}</span>
              <span className="text-right text-[#888] tabular-nums">{t.quantity}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
