"use client";

import type { AgentState, PricePoint } from "@/hooks/useSimulation";

interface StatsPanelProps {
  agent:        AgentState | null;
  priceHistory: PricePoint[];
  INV_LIMIT?:   number;
}

const INV_MAX = 50;

function Sparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const W = 200;
  const H = 36;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((v - min) / range) * H;
    return `${x},${y}`;
  });
  const last = data[data.length - 1];
  const color = last >= 0 ? "#00d26a" : "#ff3b3b";
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1}
        strokeOpacity={0.8}
      />
      {/* Zero line */}
      {min < 0 && max > 0 && (
        <line
          x1={0} x2={W}
          y1={H - ((0 - min) / range) * H}
          y2={H - ((0 - min) / range) * H}
          stroke="#333"
          strokeWidth={0.5}
          strokeDasharray="2 2"
        />
      )}
    </svg>
  );
}

function InventoryGauge({ inventory, limit }: { inventory: number; limit: number }) {
  const pct     = Math.max(-1, Math.min(1, inventory / limit));
  const W       = 200;
  const H       = 10;
  const center  = W / 2;
  const fillW   = Math.abs(pct) * (W / 2);
  const fillX   = pct >= 0 ? center : center - fillW;
  const color   = inventory > 0 ? "#00d26a" : inventory < 0 ? "#ff3b3b" : "#333";
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <rect x={0}      y={0} width={W}     height={H} fill="#111" />
      <rect x={fillX}  y={0} width={fillW} height={H} fill={color} fillOpacity={0.7} />
      <line x1={center} x2={center} y1={0} y2={H} stroke="#333" strokeWidth={1} />
    </svg>
  );
}

export default function StatsPanel({ agent, priceHistory, INV_LIMIT = INV_MAX }: StatsPanelProps) {
  const pnlHistory = priceHistory.map(() => 0); // placeholder
  const realPnl    = agent?.pnl ?? 0;
  const inv        = agent?.inventory ?? 0;
  const sharpe     = agent?.sharpe ?? 0;
  const fills      = agent?.fills_total ?? 0;
  const upnl       = agent?.unrealized_pnl ?? 0;

  // Build a simple PnL sparkline from priceHistory (using agent pnl would be ideal,
  // but we only have the current value — use mid price change as proxy for visual)
  const sparkData: number[] = (() => {
    if (priceHistory.length === 0) return [];
    const base = priceHistory[0].mid;
    return priceHistory.map((p) => (p.mid - base) * 100);
  })();

  const pnlColor   = realPnl >= 0 ? "text-[#00d26a]" : "text-[#ff3b3b]";
  const upnlColor  = upnl  >= 0 ? "text-[#00d26a]" : "text-[#ff3b3b]";
  const invColor   = inv > 0 ? "text-[#00d26a]" : inv < 0 ? "text-[#ff3b3b]" : "text-[#cccccc]";
  const sharpeColor = sharpe >= 1 ? "text-[#00d26a]" : sharpe >= 0 ? "text-[#cccccc]" : "text-[#ff3b3b]";

  return (
    <div className="flex flex-col gap-2 h-full p-2 overflow-hidden font-mono text-[0.6rem]">

      {/* PnL row */}
      <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-[#444] uppercase tracking-widest">PnL</span>
          <span className={`text-sm font-bold ${pnlColor}`}>
            {realPnl >= 0 ? "+" : ""}{realPnl.toFixed(2)}
          </span>
        </div>
        <div className="h-9">
          <Sparkline data={sparkData} />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[#444]">UNREALIZED</span>
          <span className={upnlColor}>{upnl >= 0 ? "+" : ""}{upnl.toFixed(2)}</span>
        </div>
      </div>

      {/* Inventory */}
      <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-[#444] uppercase tracking-widest">INVENTORY</span>
          <span className={`text-sm font-bold ${invColor}`}>{inv > 0 ? "+" : ""}{inv}</span>
        </div>
        <InventoryGauge inventory={inv} limit={INV_LIMIT} />
        <div className="flex justify-between mt-1">
          <span className="text-[#555]">-{INV_LIMIT}</span>
          <span className="text-[#555]">0</span>
          <span className="text-[#555]">+{INV_LIMIT}</span>
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-1">
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">SHARPE</div>
          <div className={`text-base font-bold ${sharpeColor}`}>{sharpe.toFixed(3)}</div>
        </div>
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">FILLS</div>
          <div className="text-base font-bold text-[#cccccc]">{fills}</div>
        </div>
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">BID Q</div>
          <div className="text-base font-bold text-[#00d26a]">
            {agent?.bid_quote?.toFixed(3) ?? "—"}
          </div>
        </div>
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">ASK Q</div>
          <div className="text-base font-bold text-[#ff3b3b]">
            {agent?.ask_quote?.toFixed(3) ?? "—"}
          </div>
        </div>
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">HALF SPREAD</div>
          <div className="text-base font-bold text-[#ff8c00]">
            {agent?.kappa_offset !== undefined
              ? (agent.kappa_offset * 10000).toFixed(1) + " bp"
              : "—"}
          </div>
        </div>
        <div className="border border-[#1e1e1e] bg-[#0d0d0d] p-2">
          <div className="text-[#444] uppercase tracking-widest mb-0.5">INV MODE</div>
          <div className={[
            "text-base font-bold",
            agent?.skew_mode === "normal"   ? "text-[#444]"    :
            agent?.skew_mode === "skew"     ? "text-[#f59e0b]" :
            agent?.skew_mode === "suppress" ? "text-[#ff3b3b]" :
            agent?.skew_mode === "dump"     ? "text-[#ff3b3b]" : "text-[#444]",
          ].join(" ")}>
            {agent?.skew_mode?.toUpperCase() ?? "—"}
          </div>
        </div>
      </div>
    </div>
  );
}
