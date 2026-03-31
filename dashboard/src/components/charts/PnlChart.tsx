"use client";

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { StrategyConfig } from "@/lib/types";

export interface PnlChartPoint {
  time: number;
  [key: string]: number;
}

interface PnlChartProps {
  data:       PnlChartPoint[];
  strategies: StrategyConfig[];
  showBands?: boolean;
  height?:    number;
}

const TICK = { fill: "#555", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" } as const;

function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: { dataKey: string; name: string; value: number; color: string }[];
  label?:   number;
}) {
  if (!active || !payload?.length) return null;
  const primary = payload.filter((p) => p.dataKey.endsWith("_med"));
  return (
    <div style={{
      background:  "#0f0f0f",
      border:      "1px solid #1e1e1e",
      padding:     "8px 12px",
      fontFamily:  "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: "#444", fontSize: "0.6rem", marginBottom: 4 }}>T = {label}s</div>
      {primary.map((p) => (
        <div key={p.dataKey} style={{ color: p.color, fontSize: "0.7rem" }}>
          {p.name}:{" "}
          <span style={{ color: p.value >= 0 ? "#00d26a" : "#ff3b3b" }}>
            {p.value >= 0 ? "+" : ""}{p.value.toFixed(0)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function PnlChart({
  data,
  strategies,
  showBands = true,
  height    = 300,
}: PnlChartProps) {
  const fmt = (v: number) =>
    `${v >= 0 ? "+" : ""}${(v / 1000).toFixed(0)}K`;

  const hasBands =
    showBands && data.length > 0 && `${strategies[0]?.key}_p25` in (data[0] ?? {});

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 46 }}>
        <CartesianGrid
          strokeDasharray="1 4"
          stroke="#161616"
          vertical={false}
        />
        <XAxis
          dataKey="time"
          stroke="#2a2a2a"
          tick={TICK}
          tickLine={false}
          tickFormatter={(v) => `${v}s`}
          interval="preserveStartEnd"
        />
        <YAxis
          stroke="#2a2a2a"
          tick={TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={fmt}
          width={44}
        />
        <ReferenceLine y={0} stroke="#2e2e2e" strokeDasharray="4 4" />
        <Tooltip content={<ChartTooltip />} />
        <Legend
          wrapperStyle={{
            fontSize:   "0.6rem",
            fontFamily: "'JetBrains Mono', monospace",
            color:      "#555",
            paddingTop: 4,
          }}
        />

        {strategies.flatMap((s) => {
          const lines = [
            <Line
              key={`${s.key}_med`}
              type="monotone"
              dataKey={`${s.key}_med`}
              stroke={s.color}
              strokeWidth={1.5}
              dot={false}
              name={s.name}
              isAnimationActive={false}
            />,
          ];
          if (hasBands) {
            lines.push(
              <Line
                key={`${s.key}_p25`}
                type="monotone"
                dataKey={`${s.key}_p25`}
                stroke={s.color}
                strokeWidth={0.5}
                strokeDasharray="2 4"
                strokeOpacity={0.35}
                dot={false}
                legendType="none"
                isAnimationActive={false}
              />,
              <Line
                key={`${s.key}_p75`}
                type="monotone"
                dataKey={`${s.key}_p75`}
                stroke={s.color}
                strokeWidth={0.5}
                strokeDasharray="2 4"
                strokeOpacity={0.35}
                dot={false}
                legendType="none"
                isAnimationActive={false}
              />,
            );
          }
          return lines;
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
