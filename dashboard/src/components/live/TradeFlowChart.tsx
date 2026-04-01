"use client";

import { useEffect, useRef, useCallback } from "react";
import type { TradeRecord, MidPoint, AgentState } from "@/hooks/useSimulation";

interface TradeFlowChartProps {
  trades:     TradeRecord[];
  midHistory: MidPoint[];
  agent:      AgentState | null;
  simTime:    number;
}

const PAD_L = 52;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 28;

export default function TradeFlowChart({
  trades,
  midHistory,
  agent,
  simTime,
}: TradeFlowChartProps) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    if (W === 0 || H === 0) return;

    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;

    // Background
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, W, H);

    if (trades.length === 0 && midHistory.length === 0) return;

    // ── Price range ───────────────────────────────────────────────────────────
    const prices: number[] = [];
    for (const t of trades)     prices.push(t.price);
    for (const m of midHistory) prices.push(m.mid);
    if (agent) { prices.push(agent.bid_quote, agent.ask_quote); }

    let minP = Math.min(...prices);
    let maxP = Math.max(...prices);
    const pad = (maxP - minP) * 0.15 || 0.05;
    minP -= pad;
    maxP += pad;

    // ── Time range ────────────────────────────────────────────────────────────
    const maxT = Math.max(simTime, 30);

    // ── Coordinate helpers ────────────────────────────────────────────────────
    const tx = (t: number) => PAD_L + (t / maxT) * plotW;
    const py = (p: number) => PAD_T + plotH - ((p - minP) / (maxP - minP)) * plotH;

    // ── Grid ──────────────────────────────────────────────────────────────────
    ctx.strokeStyle = "#111111";
    ctx.lineWidth   = 0.5;
    ctx.font        = "9px 'JetBrains Mono', monospace";

    // Horizontal (price)
    const nH = 5;
    for (let i = 0; i <= nH; i++) {
      const p = minP + (maxP - minP) * (i / nH);
      const y = py(p);
      ctx.beginPath();
      ctx.moveTo(PAD_L, y);
      ctx.lineTo(W - PAD_R, y);
      ctx.stroke();
      ctx.fillStyle  = "#3a3a3a";
      ctx.textAlign  = "right";
      ctx.fillText(p.toFixed(2), PAD_L - 4, y + 3);
    }

    // Vertical (time)
    const nV = Math.min(6, Math.floor(maxT / 30));
    const tStep = maxT / Math.max(nV, 1);
    for (let i = 0; i <= nV; i++) {
      const t = tStep * i;
      const x = tx(t);
      ctx.beginPath();
      ctx.moveTo(x, PAD_T);
      ctx.lineTo(x, H - PAD_B);
      ctx.stroke();
      ctx.fillStyle = "#3a3a3a";
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round(t)}s`, x, H - PAD_B + 14);
    }

    // ── Mid price line ────────────────────────────────────────────────────────
    if (midHistory.length > 1) {
      ctx.strokeStyle = "rgba(255, 140, 0, 0.45)";
      ctx.lineWidth   = 1;
      ctx.beginPath();
      for (let i = 0; i < midHistory.length; i++) {
        const x = tx(midHistory[i].sim_time);
        const y = py(midHistory[i].mid);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // ── Background market trades ──────────────────────────────────────────────
    for (const trade of trades) {
      if (trade.is_agent) continue;
      const x = tx(trade.sim_time);
      const y = py(trade.price);
      const r = Math.max(1.5, Math.min(3.5, 1 + trade.quantity / 30));
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = trade.side === "buy"
        ? "rgba(0, 210, 106, 0.30)"
        : "rgba(255, 59, 59, 0.30)";
      ctx.fill();
    }

    // ── Agent fills (on top, prominent) ──────────────────────────────────────
    for (const trade of trades) {
      if (!trade.is_agent) continue;
      const x     = tx(trade.sim_time);
      const y     = py(trade.price);
      const color = trade.side === "buy" ? "#00d26a" : "#ff3b3b";

      // Glow ring
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.globalAlpha = 0.4;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Solid dot
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    // ── Current agent bid/ask quote lines ─────────────────────────────────────
    if (agent) {
      const drawQuote = (price: number, color: string, label: string) => {
        const y = py(price);
        if (y < PAD_T - 2 || y > H - PAD_B + 2) return;
        ctx.setLineDash([4, 5]);
        ctx.strokeStyle = color;
        ctx.lineWidth   = 0.8;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(W - PAD_R, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font      = "8px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`${label} ${price.toFixed(3)}`, PAD_L + 4, y - 3);
      };
      drawQuote(agent.bid_quote, "rgba(0, 210, 106, 0.75)", "BID");
      drawQuote(agent.ask_quote, "rgba(255, 59, 59, 0.75)",  "ASK");
    }

    // ── Axis border ───────────────────────────────────────────────────────────
    ctx.strokeStyle = "#1e1e1e";
    ctx.lineWidth   = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

  }, [trades, midHistory, agent, simTime]);

  // Redraw when data changes
  useEffect(() => { draw(); }, [draw]);

  // Resize → resize canvas → redraw
  useEffect(() => {
    const container = containerRef.current;
    const canvas    = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver(() => {
      canvas.width  = container.clientWidth;
      canvas.height = container.clientHeight;
      draw();
    });
    ro.observe(container);
    // Initial size
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    draw();
    return () => ro.disconnect();
  }, [draw]);

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <canvas ref={canvasRef} className="block w-full h-full" />
      {trades.length === 0 && midHistory.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-[#333] text-[0.6rem] tracking-widest pointer-events-none">
          AWAITING DATA
        </div>
      )}
    </div>
  );
}
