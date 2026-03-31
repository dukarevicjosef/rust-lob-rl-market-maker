"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  LabelList,
} from "recharts";

export interface SharpeBarDatum {
  name:   string;
  key:    string;
  color:  string;
  sharpe: number;
  std:    number;
}

const TICK = { fill: "#555", fontSize: 8, fontFamily: "'JetBrains Mono', monospace" } as const;

function SharpeTooltip({ active, payload }: {
  active?:  boolean;
  payload?: { payload: SharpeBarDatum }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "#0f0f0f",
      border:     "1px solid #1e1e1e",
      padding:    "8px 12px",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: "#444", fontSize: "0.6rem", marginBottom: 2 }}>{d.name}</div>
      <div style={{ color: d.color, fontSize: "0.7rem" }}>
        SHARPE: {d.sharpe >= 0 ? "+" : ""}{d.sharpe.toFixed(2)}{" "}
        <span style={{ color: "#555" }}>± {d.std.toFixed(2)}</span>
      </div>
    </div>
  );
}

export default function SharpeBar({ data }: { data: SharpeBarDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 20, right: 8, bottom: 0, left: 30 }}>
        <CartesianGrid strokeDasharray="1 4" stroke="#161616" vertical={false} />
        <XAxis
          dataKey="name"
          stroke="#2a2a2a"
          tick={TICK}
          tickLine={false}
          tickFormatter={(v: string) => v.split(" ")[0]}
        />
        <YAxis
          stroke="#2a2a2a"
          tick={TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v: number) => v.toFixed(1)}
          width={32}
        />
        <ReferenceLine y={0} stroke="#333" />
        <Tooltip content={<SharpeTooltip />} cursor={{ fill: "rgba(255,140,0,0.03)" }} />
        <Bar dataKey="sharpe" radius={0} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.key} fill={d.color} fillOpacity={0.75} />
          ))}
          <ErrorBar dataKey="std" width={4} strokeWidth={1} stroke="#666" direction="y" />
          <LabelList
            dataKey="sharpe"
            position="top"
            formatter={(v) => {
              const n = Number(v);
              return (n >= 0 ? "+" : "") + n.toFixed(2);
            }}
            style={{ fill: "#888", fontSize: 8, fontFamily: "monospace" }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
