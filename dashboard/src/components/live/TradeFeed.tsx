"use client";

import type { TradeRecord } from "@/hooks/useSimulation";

interface TradeFeedProps {
  trades: TradeRecord[];
}

export default function TradeFeed({ trades }: TradeFeedProps) {
  const agentTrades = trades.filter((t) => t.is_agent);
  const mktCount    = trades.filter((t) => !t.is_agent).length;

  if (agentTrades.length === 0) {
    return (
      <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
        <div className="px-2 py-1 border-b border-[#1e1e1e] text-[#444] uppercase tracking-widest shrink-0">
          AGENT FILLS
          <span className="ml-2 text-[#333]">MKT VOL {mktCount}</span>
        </div>
        <div className="flex-1 flex items-center justify-center text-[#333] tracking-widest">
          NO FILLS YET
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden font-mono text-[0.6rem]">
      {/* Header */}
      <div className="flex justify-between px-2 py-1 border-b border-[#1e1e1e] text-[#444] uppercase tracking-widest shrink-0">
        <span>AGENT FILLS</span>
        <span className="text-[#333]">MKT VOL {mktCount}</span>
      </div>

      {/* Column labels */}
      <div className="grid grid-cols-[3rem_5rem_3rem] gap-1 px-2 py-0.5 border-b border-[#111] text-[#333] uppercase shrink-0">
        <span>SIDE</span>
        <span className="text-right">PRICE</span>
        <span className="text-right">QTY</span>
      </div>

      {/* Rows — newest first */}
      <div className="flex-1 overflow-y-auto">
        {agentTrades.map((t, i) => {
          const sideColor = t.side === "buy" ? "#00d26a" : "#ff3b3b";
          return (
            <div
              key={i}
              className="grid grid-cols-[3rem_5rem_3rem] gap-1 px-2 py-0.5 items-center border-b border-[#0f0f0f] border-l-2 border-l-[#3b82f6]"
            >
              <span style={{ color: sideColor }} className="font-bold uppercase">
                {t.side}
              </span>
              <span style={{ color: sideColor }} className="text-right tabular-nums">
                {t.price.toFixed(4)}
              </span>
              <span className="text-right text-[#888] tabular-nums">{t.quantity}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
