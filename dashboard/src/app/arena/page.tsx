"use client";

import { useState } from "react";
import Panel from "@/components/terminal/Panel";
import DataCell from "@/components/terminal/DataCell";
import BloombergTable from "@/components/terminal/BloombergTable";

const STRATEGIES = [
  { key: "naive",        label: "1  NAIVE SYM" },
  { key: "static_as",   label: "2  STATIC AS" },
  { key: "opt_as",      label: "3  OPT AS" },
  { key: "sac",         label: "4  SAC" },
];

const METRICS = [
  { agent: "SAC",          total_pnl: 274.1,  spread_pnl: 198.3,  inventory_pnl: 75.8,   sharpe: 1.91,  fill_rate: 30.1, inv_std: 4.2  },
  { agent: "Optimized AS", total_pnl: 201.7,  spread_pnl: 180.1,  inventory_pnl: 21.6,   sharpe: 1.44,  fill_rate: 28.4, inv_std: 3.1  },
  { agent: "Static AS",    total_pnl: 142.3,  spread_pnl: 148.2,  inventory_pnl: -5.9,   sharpe: 0.97,  fill_rate: 24.1, inv_std: 5.8  },
  { agent: "Naive",        total_pnl: 88.6,   spread_pnl: 120.4,  inventory_pnl: -31.8,  sharpe: 0.61,  fill_rate: 20.3, inv_std: 8.7  },
];

export default function ArenaPage() {
  const [active, setActive] = useState("sac");

  return (
    <div className="flex flex-col h-full">

      {/* Strategy key bar */}
      <div className="flex h-7 border-b border-[#1e1e1e] bg-[#0a0a0a] select-none shrink-0">
        {STRATEGIES.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActive(key)}
            className={[
              "px-4 h-full text-[0.65rem] tracking-widest border-r border-[#1e1e1e] cursor-pointer transition-colors",
              active === key
                ? "bg-[#ff8c00] text-black font-bold"
                : "text-[#666666] hover:bg-[#1e1e1e] hover:text-[#cccccc]",
            ].join(" ")}
          >
            {"<"}{label}{">"}
          </button>
        ))}
        <div className="flex-1" />
        <span className="self-center pr-3 text-[0.6rem] text-[#444444]">50 EPISODES | SEED 0-49</span>
      </div>

      {/* PnL Chart */}
      <div className="flex-1 border-b border-[#1e1e1e] min-h-0">
        <Panel title="CUMULATIVE PNL" subtitle="MEAN OVER 50 EPISODES" className="border-0 h-full">
          <div className="flex items-center justify-center h-full text-[#333333] text-[0.65rem] tracking-widest">
            [ RECHARTS LINE CHART — WIRED IN PROMPT 2 ]
          </div>
        </Panel>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 border-b border-[#1e1e1e] shrink-0">
        <DataCell label="MEAN PNL"   value={274.1}  colorize precision={1} className="border-r border-[#1e1e1e]" />
        <DataCell label="SHARPE"     value={1.91}   colorize precision={2} className="border-r border-[#1e1e1e]" />
        <DataCell label="FILL RATE"  value="30.1"   unit="%"               className="border-r border-[#1e1e1e]" />
        <DataCell label="INV STD"    value={4.2}    precision={1} />
      </div>

      {/* Comparison table */}
      <div className="shrink-0 p-2">
        <div className="text-[0.6rem] text-[#ff8c00] uppercase tracking-widest mb-1">STRATEGY COMPARISON</div>
        <BloombergTable
          rowKey="agent"
          columns={[
            { key: "agent",        header: "AGENT",       align: "left" },
            { key: "total_pnl",    header: "TOTAL PNL",   colorize: true,  render: (v) => Number(v).toFixed(1) },
            { key: "spread_pnl",   header: "SPREAD PNL",  colorize: true,  render: (v) => Number(v).toFixed(1) },
            { key: "inventory_pnl",header: "INV PNL",     colorize: true,  render: (v) => Number(v).toFixed(1) },
            { key: "sharpe",       header: "SHARPE",      colorize: true,  render: (v) => Number(v).toFixed(2) },
            { key: "fill_rate",    header: "FILL RT%",    render: (v) => Number(v).toFixed(1) },
            { key: "inv_std",      header: "INV STD",     render: (v) => Number(v).toFixed(1) },
          ]}
          rows={METRICS as unknown as Record<string, unknown>[]}
        />
      </div>

    </div>
  );
}
