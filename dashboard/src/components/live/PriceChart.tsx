"use client";

import { useEffect, useRef } from "react";
import { createChart, LineSeries, ColorType, type ISeriesApi, type UTCTimestamp } from "lightweight-charts";
import type { PricePoint, TradeRecord } from "@/hooks/useSimulation";

interface PriceChartProps {
  priceHistory: PricePoint[];
  tradeHistory: TradeRecord[];
}

export default function PriceChart({ priceHistory, tradeHistory }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<ReturnType<typeof createChart> | null>(null);
  const midRef       = useRef<ISeriesApi<"Line"> | null>(null);
  const bidRef       = useRef<ISeriesApi<"Line"> | null>(null);
  const askRef       = useRef<ISeriesApi<"Line"> | null>(null);
  const tickRef      = useRef(0);

  // Init chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0a0a0a" },
        textColor:  "#444444",
        fontFamily: "JetBrains Mono, monospace",
        fontSize:   10,
      },
      grid: {
        vertLines: { color: "#111111" },
        horzLines: { color: "#111111" },
      },
      crosshair: {
        vertLine: { color: "#333333", width: 1, style: 1 },
        horzLine: { color: "#333333", width: 1, style: 1 },
      },
      rightPriceScale: {
        borderColor: "#1e1e1e",
        textColor:   "#444444",
      },
      timeScale: {
        borderColor:     "#1e1e1e",
        timeVisible:     true,
        secondsVisible:  true,
        tickMarkFormatter: (t: number) => String(t),
      },
      handleScroll:  true,
      handleScale:   true,
    });

    const mid = chart.addSeries(LineSeries, {
      color:       "#ff8c00",
      lineWidth:   1,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const bid = chart.addSeries(LineSeries, {
      color:       "#00d26a",
      lineWidth:   1,
      lineStyle:   2,  // dashed
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const ask = chart.addSeries(LineSeries, {
      color:       "#ff3b3b",
      lineWidth:   1,
      lineStyle:   2,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    chartRef.current = chart;
    midRef.current   = mid;
    bidRef.current   = bid;
    askRef.current   = ask;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width:  containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      midRef.current   = null;
      bidRef.current   = null;
      askRef.current   = null;
      tickRef.current  = 0;
    };
  }, []);

  // Update data
  useEffect(() => {
    if (!midRef.current || !bidRef.current || !askRef.current) return;
    if (priceHistory.length === 0) return;

    // Build strictly-increasing integer time series
    const midData = priceHistory.map((p, i) => ({ time: (i + 1) as UTCTimestamp, value: p.mid }));
    const bidData = priceHistory.map((p, i) => ({ time: (i + 1) as UTCTimestamp, value: p.bid }));
    const askData = priceHistory.map((p, i) => ({ time: (i + 1) as UTCTimestamp, value: p.ask }));

    midRef.current.setData(midData);
    bidRef.current.setData(bidData);
    askRef.current.setData(askData);

    chartRef.current?.timeScale().scrollToRealTime();
  }, [priceHistory]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {priceHistory.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-[#333] text-[0.6rem] tracking-widest pointer-events-none">
          AWAITING DATA
        </div>
      )}
    </div>
  );
}
