"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export interface DecompDatum {
  name:          string;
  key:           string;
  spread_pnl:    number;
  inventory_pnl: number;
}

const TICK = { fill: "#555", fontSize: 8, fontFamily: "'JetBrains Mono', monospace" } as const;

function DecompTooltip({ active, payload }: {
  active?:  boolean;
  payload?: { name: string; value: number; fill: string }[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0f0f0f",
      border:     "1px solid #1e1e1e",
      padding:    "8px 12px",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.fill, fontSize: "0.7rem" }}>
          {p.name}: {p.value >= 0 ? "+" : ""}{(p.value / 1000).toFixed(1)}K
        </div>
      ))}
    </div>
  );
}

export default function PnlDecomposition({ data }: { data: DecompDatum[] }) {
  const fmt = (v: number) => `${v >= 0 ? "+" : ""}${(v / 1000).toFixed(0)}K`;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 38 }}>
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
          tickFormatter={fmt}
          width={36}
        />
        <ReferenceLine y={0} stroke="#333" />
        <Tooltip content={<DecompTooltip />} cursor={{ fill: "rgba(255,140,0,0.03)" }} />
        <Legend
          wrapperStyle={{
            fontSize:   "0.6rem",
            fontFamily: "'JetBrains Mono', monospace",
            color:      "#555",
          }}
        />
        <Bar
          dataKey="spread_pnl"
          name="Spread PnL"
          stackId="a"
          fill="#00d26a"
          fillOpacity={0.75}
          isAnimationActive={false}
        />
        <Bar
          dataKey="inventory_pnl"
          name="Inventory PnL"
          stackId="a"
          fill="#ff3b3b"
          fillOpacity={0.75}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
