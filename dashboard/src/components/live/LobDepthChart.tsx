"use client";

import type { LobLevel } from "@/hooks/useSimulation";

interface LobDepthChartProps {
  bids:       LobLevel[];
  asks:       LobLevel[];
  midPrice:   number;
  agentBid?:  number;
  agentAsk?:  number;
}

export default function LobDepthChart({
  bids, asks, midPrice, agentBid, agentAsk,
}: LobDepthChartProps) {
  if (bids.length === 0 && asks.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-[#333] text-[0.6rem] tracking-widest">
        AWAITING DATA
      </div>
    );
  }

  // SVG virtual dimensions
  const VW  = 320;
  const VH  = 600;
  const PL  = 54;   // price label area
  const PR  = 6;
  const PT  = 16;
  const PB  = 16;
  const CW  = VW - PL - PR;   // chart width
  const CH  = VH - PT - PB;   // chart height
  const CX  = PL + CW / 2;    // centre x (split bids/asks)

  // Y-axis: always centre on midPrice ± 0.5%, regardless of where LOB levels are.
  // This keeps the chart stable when only a few sparse levels exist and prevents
  // stale far-out levels from collapsing the visible range to a single pixel row.
  const HALF_RANGE = Math.max(midPrice * 0.005, 0.05);   // 0.5% of mid, min 0.05
  const pMin  = midPrice - HALF_RANGE;
  const pMax  = midPrice + HALF_RANGE;
  const pRange = pMax - pMin;

  // Adaptive tick: targets ~10 grid lines within the fixed range
  const TICK  = Math.pow(10, Math.floor(Math.log10(pRange / 10)));

  const yS = (price: number) => PT + ((pMax - price) / pRange) * CH;

  // Normalise against the actual data — not a fixed floor of 1.
  // BTC quantities (0.001–5.0) would otherwise all render as sub-pixel bars.
  const maxQty = Math.max(
    1e-9,
    ...bids.map((l) => l.quantity),
    ...asks.map((l) => l.quantity),
  );

  const halfW = CW / 2 - 4;
  const qW = (qty: number) => (qty / maxQty) * halfW;

  const rowH = Math.min(CH / (bids.length + asks.length + 1), 22);

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${VW} ${VH}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Grid lines at each tick within the dynamic range */}
      {(() => {
        const firstTick = Math.ceil(pMin / TICK) * TICK;
        const lines = [];
        for (let p = firstTick; p <= pMax + TICK * 0.5; p = Math.round((p + TICK) * 1e8) / 1e8) {
          const y     = yS(p);
          const atMid = Math.abs(p - midPrice) < TICK * 0.5;
          lines.push(
            <line
              key={p}
              x1={PL} x2={PL + CW}
              y1={y}  y2={y}
              stroke={atMid ? "#2a2a2a" : "#131313"}
              strokeWidth={atMid ? 0.8 : 0.4}
              strokeDasharray={atMid ? "3 3" : undefined}
            />
          );
        }
        return lines;
      })()}

      {/* Bids — green, from centre going left */}
      {/* Guard against stale persisted levels that crossed mid after a price move */}
      {bids.filter((lvl) => lvl.price <= midPrice).map((lvl, i) => {
        const y  = yS(lvl.price);
        const w  = qW(lvl.quantity);
        return (
          <g key={`bid-${i}`}>
            <rect
              x={CX - w}
              y={y - rowH / 2}
              width={w}
              height={rowH}
              fill="#00d26a"
              fillOpacity={0.55 - i * 0.04}
            />
            <line
              x1={CX - w} x2={CX}
              y1={y}       y2={y}
              stroke="#00d26a"
              strokeOpacity={0.8}
              strokeWidth={0.8}
            />
          </g>
        );
      })}

      {/* Asks — red, from centre going right */}
      {asks.filter((lvl) => lvl.price >= midPrice).map((lvl, i) => {
        const y  = yS(lvl.price);
        const w  = qW(lvl.quantity);
        return (
          <g key={`ask-${i}`}>
            <rect
              x={CX}
              y={y - rowH / 2}
              width={w}
              height={rowH}
              fill="#ff3b3b"
              fillOpacity={0.55 - i * 0.04}
            />
            <line
              x1={CX} x2={CX + w}
              y1={y}  y2={y}
              stroke="#ff3b3b"
              strokeOpacity={0.8}
              strokeWidth={0.8}
            />
          </g>
        );
      })}

      {/* Centre divider */}
      <line
        x1={CX} x2={CX}
        y1={PT}  y2={PT + CH}
        stroke="#1e1e1e"
        strokeWidth={1}
      />

      {/* Mid-price line */}
      <line
        x1={PL}    x2={PL + CW}
        y1={yS(midPrice)} y2={yS(midPrice)}
        stroke="#ff8c00"
        strokeWidth={0.8}
        strokeDasharray="4 3"
      />

      {/* Agent bid quote */}
      {agentBid !== undefined && agentBid >= pMin && agentBid <= pMax && (
        <>
          <line
            x1={PL}    x2={PL + CW}
            y1={yS(agentBid)} y2={yS(agentBid)}
            stroke="#3b82f6"
            strokeWidth={0.8}
            strokeDasharray="3 3"
          />
          <text
            x={PL - 2}
            y={yS(agentBid) + 3}
            textAnchor="end"
            fill="#3b82f6"
            fontSize={6.5}
            fontFamily="monospace"
          >
            B {agentBid.toFixed(3)}
          </text>
        </>
      )}

      {/* Agent ask quote */}
      {agentAsk !== undefined && agentAsk >= pMin && agentAsk <= pMax && (
        <>
          <line
            x1={PL}    x2={PL + CW}
            y1={yS(agentAsk)} y2={yS(agentAsk)}
            stroke="#3b82f6"
            strokeWidth={0.8}
            strokeDasharray="3 3"
          />
          <text
            x={PL - 2}
            y={yS(agentAsk) + 3}
            textAnchor="end"
            fill="#3b82f6"
            fontSize={6.5}
            fontFamily="monospace"
          >
            A {agentAsk.toFixed(3)}
          </text>
        </>
      )}

      {/* Price labels — every 2 ticks */}
      {(() => {
        const firstTick = Math.ceil(pMin / TICK) * TICK;
        // Decimal places: enough to show TICK resolution without trailing zeros
        const decimals  = Math.max(0, -Math.floor(Math.log10(TICK)));
        const labels = [];
        let idx = 0;
        for (let p = firstTick; p <= pMax + TICK * 0.5; p = Math.round((p + TICK) * 1e8) / 1e8, idx++) {
          if (idx % 2 !== 0) continue;
          const y     = yS(p);
          const atMid = Math.abs(p - midPrice) < TICK * 0.5;
          labels.push(
            <text
              key={`lbl-${p}`}
              x={PL - 3}
              y={y + 3}
              textAnchor="end"
              fill={atMid ? "#ff8c00" : "#444"}
              fontSize={6.5}
              fontFamily="monospace"
            >
              {p.toFixed(decimals)}
            </text>
          );
        }
        return labels;
      })()}

      {/* Spread band */}
      {agentBid && agentAsk && (
        <rect
          x={PL}
          y={yS(agentAsk)}
          width={CW}
          height={Math.abs(yS(agentBid) - yS(agentAsk))}
          fill="#3b82f6"
          fillOpacity={0.04}
        />
      )}

      {/* Column labels */}
      <text x={PL + CW / 4}   y={VH - 4} textAnchor="middle" fill="#333" fontSize={6} fontFamily="monospace">BID VOL</text>
      <text x={PL + CW * 3/4} y={VH - 4} textAnchor="middle" fill="#333" fontSize={6} fontFamily="monospace">ASK VOL</text>
    </svg>
  );
}
