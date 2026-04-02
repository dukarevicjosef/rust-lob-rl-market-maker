"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import Panel        from "@/components/terminal/Panel";
import ControlBar   from "@/components/live/ControlBar";
import LobDepthChart from "@/components/live/LobDepthChart";
import StatsPanel   from "@/components/live/StatsPanel";
import TradeFeed    from "@/components/live/TradeFeed";
import TimesSales   from "@/components/live/TimesSales";
import { useSimulation } from "@/hooks/useSimulation";
import { simInfoStore } from "@/lib/sim-info-store";

// lazy-load the chart so it never runs on the server
const TradeFlowChart = dynamic(() => import("@/components/live/TradeFlowChart"), { ssr: false });

export default function LivePage() {
  const sim = useSimulation();

  const isReplay = sim.mode === "replay";

  // Keep the TopBar in sync with the active mode
  useEffect(() => {
    simInfoStore.set({
      mode:         sim.mode,
      replayDate:   sim.replayDate,
      replayEvents: sim.replayEvents,
    });
  }, [sim.mode, sim.replayDate, sim.replayEvents]);

  const bids = sim.lob?.bids ?? [];
  const asks = sim.lob?.asks ?? [];
  const mid  = sim.agent?.bid_quote != null && sim.agent?.ask_quote != null
    ? (sim.agent.bid_quote + sim.agent.ask_quote) / 2
    : sim.priceHistory[sim.priceHistory.length - 1]?.mid ?? 100;

  const priceSubtitle = isReplay
    ? `MID ${mid.toFixed(2)} — market replay, no agent`
    : sim.agent?.bid_quote != null && sim.agent?.ask_quote != null
      ? `BID ${sim.agent.bid_quote.toFixed(3)} / ASK ${sim.agent.ask_quote.toFixed(3)} | SPREAD ${(sim.agent.ask_quote - sim.agent.bid_quote).toFixed(4)}`
      : "AWAITING DATA";

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* Control bar */}
      <ControlBar
        isConnected={sim.isConnected}
        isRunning={sim.isRunning}
        isPaused={sim.isPaused}
        eventsProcessed={sim.eventsProcessed}
        elapsedTime={sim.elapsedTime}
        replayProgress={sim.replayProgress}
        lastError={sim.lastError}
        onStart={sim.start}
        onStop={sim.stop}
        onPause={sim.pause}
        onResume={sim.resume}
        onSpeedChange={sim.setSpeed}
        onReset={sim.reset}
      />

      {/* Main grid */}
      <div
        className="flex-1 grid overflow-hidden"
        style={{
          gridTemplateColumns: "minmax(0,3fr) minmax(0,9fr)",
          gridTemplateRows:    "minmax(0,4fr) minmax(0,2fr)",
        }}
      >

        {/* ── LOB DEPTH — col 1, full height ─────────────────────────────── */}
        <div className="row-span-2 border-r border-[#1e1e1e] overflow-hidden">
          <Panel
            title="LOB DEPTH"
            subtitle={isReplay ? "REPLAY" : "AGENT QUOTES"}
            className="border-0 h-full"
          >
            <div className="h-full">
              <LobDepthChart
                bids={bids}
                asks={asks}
                midPrice={mid}
                agentBid={sim.agent?.bid_quote ?? undefined}
                agentAsk={sim.agent?.ask_quote ?? undefined}
              />
            </div>
          </Panel>
        </div>

        {/* ── TRADE FLOW — col 2, row 1 ─────────────────────────────────── */}
        <div className="border-b border-[#1e1e1e] overflow-hidden">
          <Panel title="TRADE FLOW" subtitle={priceSubtitle} className="border-0 h-full">
            <div className="h-full">
              <TradeFlowChart
                trades={sim.tradeHistory}
                midHistory={sim.midHistory}
                simTime={sim.elapsedTime}
              />
            </div>
          </Panel>
        </div>

        {/* ── BOTTOM ROW: stats + trades + T&S ────────────────────────────── */}
        <div
          className="grid overflow-hidden"
          style={{ gridTemplateColumns: "1fr 1fr 1fr" }}
        >
          {/* Left: Replay status or Agent state */}
          <div className="border-r border-[#1e1e1e] overflow-hidden">
            <Panel
              title={isReplay ? "REPLAY STATUS" : "AGENT STATE"}
              className="border-0 h-full"
            >
              {isReplay ? (
                <div className="flex flex-col gap-3 p-3 font-mono text-[0.6rem]">
                  <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                    <span className="text-[#444] uppercase tracking-widest">MODE</span>
                    <span className="text-[#0055cc] font-bold tracking-widest">MARKET REPLAY</span>
                  </div>
                  {sim.replayDate && (
                    <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                      <span className="text-[#444] uppercase tracking-widest">DATE</span>
                      <span className="text-[#cccccc]">{sim.replayDate}</span>
                    </div>
                  )}
                  {sim.replayEvents != null && (
                    <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                      <span className="text-[#444] uppercase tracking-widest">EVENTS</span>
                      <span className="text-[#cccccc]">
                        {sim.replayEvents >= 1_000_000
                          ? `${(sim.replayEvents / 1_000_000).toFixed(2)}M`
                          : `${Math.round(sim.replayEvents / 1000)}K`}
                      </span>
                    </div>
                  )}
                  <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                    <span className="text-[#444] uppercase tracking-widest">PROGRESS</span>
                    <span className="text-[#0088ff]">{(sim.replayProgress * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                    <span className="text-[#444] uppercase tracking-widest">PROCESSED</span>
                    <span className="text-[#cccccc]">{sim.eventsProcessed.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center border border-[#1e1e1e] bg-[#0d0d0d] px-2 py-1.5">
                    <span className="text-[#444] uppercase tracking-widest">AGENT</span>
                    <span className="text-[#444]">NONE</span>
                  </div>
                </div>
              ) : (
                <StatsPanel
                  agent={sim.agent}
                  priceHistory={sim.priceHistory}
                />
              )}
            </Panel>
          </div>

          {/* Middle: Market trades (replay) or Agent fills (simulate) */}
          <div className="border-r border-[#1e1e1e] overflow-hidden">
            <Panel
              title={isReplay ? "MARKET TRADES" : "AGENT FILLS"}
              className="border-0 h-full"
            >
              <TradeFeed trades={sim.tradeHistory} isReplay={isReplay} />
            </Panel>
          </div>

          {/* Right: Times & Sales */}
          <div className="overflow-hidden">
            <Panel title="TIMES & SALES" className="border-0 h-full">
              <TimesSales trades={sim.tradeHistory} isReplay={isReplay} />
            </Panel>
          </div>
        </div>

      </div>
    </div>
  );
}
