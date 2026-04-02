"use client";

import { useEffect, useRef, useCallback } from "react";
import type { TradeRecord, MidPoint } from "@/hooks/useSimulation";

interface TradeFlowChartProps {
  trades:     TradeRecord[];
  midHistory: MidPoint[];
  simTime:    number;
}

const PAD_L = 52;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 28;

export default function TradeFlowChart({
  trades,
  midHistory,
  simTime,
}: TradeFlowChartProps) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    void simTime; // used to trigger redraws
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

    // ── Time range — rolling window of last WINDOW_S seconds ─────────────────
    const WINDOW_S = 120;
    const maxT = Math.max(simTime, midHistory.length > 0 ? midHistory[midHistory.length - 1].sim_time : 0, WINDOW_S);
    const minT = maxT - WINDOW_S;

    // ── Price range — only within visible window to avoid Y-axis distortion ───
    const prices: number[] = [];
    for (const t of trades)     if (t.sim_time >= minT) prices.push(t.price);
    for (const m of midHistory) if (m.sim_time >= minT) prices.push(m.mid);

    let minP = Math.min(...prices);
    let maxP = Math.max(...prices);
    const pad = (maxP - minP) * 0.15 || 0.05;
    minP -= pad;
    maxP += pad;

    // ── Coordinate helpers ────────────────────────────────────────────────────
    const tx = (t: number) => PAD_L + ((t - minT) / WINDOW_S) * plotW;
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

    // Vertical (time) — labels show absolute sim-time
    const nV    = 6;
    const tStep = WINDOW_S / nV;
    for (let i = 0; i <= nV; i++) {
      const t = minT + tStep * i;
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

    // ── Axis border ───────────────────────────────────────────────────────────
    ctx.strokeStyle = "#1e1e1e";
    ctx.lineWidth   = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

  }, [trades, midHistory, simTime]);

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
