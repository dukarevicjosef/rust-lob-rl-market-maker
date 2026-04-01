"use client";

import dynamic from "next/dynamic";
import Panel        from "@/components/terminal/Panel";
import ControlBar   from "@/components/live/ControlBar";
import LobDepthChart from "@/components/live/LobDepthChart";
import StatsPanel   from "@/components/live/StatsPanel";
import TradeFeed    from "@/components/live/TradeFeed";
import { useSimulation } from "@/hooks/useSimulation";

// lazy-load the chart so it never runs on the server
const TradeFlowChart = dynamic(() => import("@/components/live/TradeFlowChart"), { ssr: false });

export default function LivePage() {
  const sim = useSimulation();

  const bids = sim.lob?.bids ?? [];
  const asks = sim.lob?.asks ?? [];
  const mid  = sim.agent?.bid_quote !== undefined && sim.agent?.ask_quote !== undefined
    ? (sim.agent.bid_quote + sim.agent.ask_quote) / 2
    : sim.priceHistory[sim.priceHistory.length - 1]?.mid ?? 100;

  const spread = sim.agent
    ? (sim.agent.ask_quote - sim.agent.bid_quote).toFixed(4)
    : "—";
  const priceSubtitle = sim.agent
    ? `BID ${sim.agent.bid_quote.toFixed(3)} / ASK ${sim.agent.ask_quote.toFixed(3)} | SPREAD ${spread}`
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
        <div
          className="row-span-2 border-r border-[#1e1e1e] overflow-hidden"
        >
          <Panel title="LOB DEPTH" subtitle="±12 TICKS" className="border-0 h-full">
            <div className="h-full">
              <LobDepthChart
                bids={bids}
                asks={asks}
                midPrice={mid}
                agentBid={sim.agent?.bid_quote}
                agentAsk={sim.agent?.ask_quote}
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
                agent={sim.agent}
                simTime={sim.elapsedTime}
              />
            </div>
          </Panel>
        </div>

        {/* ── BOTTOM ROW: stats + trades ──────────────────────────────────── */}
        <div
          className="grid overflow-hidden"
          style={{ gridTemplateColumns: "1fr 1fr" }}
        >
          {/* Stats panel */}
          <div className="border-r border-[#1e1e1e] overflow-hidden">
            <Panel title="AGENT STATE" className="border-0 h-full">
              <StatsPanel
                agent={sim.agent}
                priceHistory={sim.priceHistory}
              />
            </Panel>
          </div>

          {/* Trade feed */}
          <div className="overflow-hidden">
            <Panel title="TRADE FEED" className="border-0 h-full">
              <TradeFeed trades={sim.tradeHistory} />
            </Panel>
          </div>
        </div>

      </div>
    </div>
  );
}
