"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { simInfoStore } from "@/lib/sim-info-store";

export default function TopBar() {
  const [now, setNow] = useState("");
  const simInfo = useSyncExternalStore(
    simInfoStore.subscribe,
    simInfoStore.get,
    simInfoStore.get,
  );

  useEffect(() => {
    const fmt = () => {
      const d = new Date();
      setNow(
        d.toISOString().replace("T", " ").slice(0, 19) + " UTC",
      );
    };
    fmt();
    const id = setInterval(fmt, 1000);
    return () => clearInterval(id);
  }, []);

  const fmtEvents = (n: number) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000)     return `${Math.round(n / 1_000)}K`;
    return String(n);
  };

  return (
    <div className="flex items-center h-7 bg-[#111111] border-b border-[#1e1e1e] px-3 shrink-0 select-none">
      {/* Logo */}
      <span className="text-[#ff8c00] font-bold uppercase tracking-widest text-[0.75rem] mr-6">
        QUANTFLOW
      </span>

      {/* Sim info — switches based on active mode */}
      <span className="text-[0.65rem] text-[#666666] tracking-wide flex-1">
        {simInfo.mode === "replay" ? (
          <>
            <span className="text-[#0055cc] font-bold">REPLAY</span>
            {"  "}:{" "}
            <span className="text-[#cccccc]">BTCUSDT</span>
            {"  "}|{"  "}
            {simInfo.replayDate && (
              <>
                <span className="text-[#cccccc]">{simInfo.replayDate}</span>
                {"  "}|{"  "}
              </>
            )}
            {simInfo.replayEvents != null && (
              <span className="text-[#cccccc]">{fmtEvents(simInfo.replayEvents)} EVENTS</span>
            )}
          </>
        ) : (
          <>
            SIM:{" "}
            <span className="text-[#cccccc]">HAWKES‑12D</span>
            {"  "}|{"  "}
            σ={" "}
            <span className="text-[#cccccc]">0.061</span>
            {"  "}|{"  "}
            SEED{" "}
            <span className="text-[#cccccc]">42</span>
            {"  "}|{"  "}
            <span className="text-[#cccccc]">82K EVENTS</span>
          </>
        )}
      </span>

      {/* Timestamp + connection */}
      <div className="flex items-center gap-3 text-[0.65rem]">
        <span className="text-[#444444] font-mono">{now}</span>
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00d26a] glow-green inline-block" />
          <span className="text-[#00d26a] text-[0.6rem]">LIVE</span>
        </div>
      </div>
    </div>
  );
}
