"use client";

import { useState } from "react";
import Panel from "@/components/terminal/Panel";

interface Param {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}

const PARAMS: Param[] = [
  { key: "baseline",  label: "HAWKES BASELINE λ₀", min: 0.1, max: 2.0, step: 0.1, unit: "evt/s" },
  { key: "alpha",     label: "EXCITATION α",        min: 0.0, max: 0.9, step: 0.05, unit: "" },
  { key: "beta",      label: "DECAY β",             min: 0.1, max: 5.0, step: 0.1, unit: "" },
  { key: "tick",      label: "TICK SIZE",           min: 0.01, max: 0.1, step: 0.01, unit: "USD" },
  { key: "inv_lim",   label: "INV LIMIT",           min: 10, max: 200, step: 5, unit: "lots" },
  { key: "t_max",     label: "EPISODE LENGTH",      min: 500, max: 5000, step: 100, unit: "steps" },
];

const DEFAULTS: Record<string, number> = {
  baseline: 0.5, alpha: 0.3, beta: 1.0, tick: 0.01, inv_lim: 50, t_max: 1000,
};

export default function SimulatorPage() {
  const [params, setParams] = useState<Record<string, number>>(DEFAULTS);

  const set = (key: string, v: number) => setParams((p) => ({ ...p, [key]: v }));

  return (
    <div className="flex h-full">

      {/* ── Parameter Panel ──────────────────────────────────────────────── */}
      <div className="w-72 shrink-0 border-r border-[#1e1e1e] flex flex-col">
        <Panel title="PARAMETERS" className="border-0 flex-1">
          <div className="p-2 space-y-4 overflow-y-auto h-full">
            {PARAMS.map((p) => (
              <div key={p.key}>
                <div className="flex justify-between text-[0.6rem] uppercase tracking-widest mb-1">
                  <span className="text-[#666666]">{p.label}</span>
                  <span className="text-[#ff8c00] font-bold font-mono">
                    {params[p.key].toFixed(p.step < 1 ? 2 : 0)} {p.unit}
                  </span>
                </div>

                {/* Slider row */}
                <div className="flex items-center gap-1">
                  <button
                    className="w-5 h-5 border border-[#333] text-[#666] hover:border-[#ff8c00] hover:text-[#ff8c00] text-xs flex items-center justify-center"
                    onClick={() => set(p.key, Math.max(p.min, +(params[p.key] - p.step).toFixed(10)))}
                  >−</button>
                  <div className="flex-1 relative h-1 bg-[#1e1e1e]">
                    <div
                      className="absolute left-0 top-0 h-full bg-[#ff8c00]"
                      style={{
                        width: `${((params[p.key] - p.min) / (p.max - p.min)) * 100}%`,
                      }}
                    />
                    <input
                      type="range"
                      min={p.min} max={p.max} step={p.step}
                      value={params[p.key]}
                      onChange={(e) => set(p.key, parseFloat(e.target.value))}
                      className="absolute inset-0 w-full opacity-0 cursor-pointer"
                    />
                  </div>
                  <button
                    className="w-5 h-5 border border-[#333] text-[#666] hover:border-[#ff8c00] hover:text-[#ff8c00] text-xs flex items-center justify-center"
                    onClick={() => set(p.key, Math.min(p.max, +(params[p.key] + p.step).toFixed(10)))}
                  >+</button>
                </div>
              </div>
            ))}

            <div className="pt-4 space-y-2">
              <button className="w-full h-8 bg-[#ff8c00] text-black text-[0.7rem] font-bold uppercase tracking-widest hover:bg-[#ffb700] transition-colors">
                RUN SIMULATION
              </button>
              <button className="w-full h-8 border border-[#4a9eff] text-[#4a9eff] text-[0.7rem] uppercase tracking-widest hover:bg-[#4a9eff]/10 transition-colors">
                CALIBRATE
              </button>
            </div>
          </div>
        </Panel>
      </div>

      {/* ── Stylized Facts 2×2 ─────────────────────────────────────────────── */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2">
        {[
          { title: "RETURN DISTRIBUTION",  sub: "vs. NORMAL" },
          { title: "AUTOCORRELATION",       sub: "|rₜ| LAGS 1-50" },
          { title: "SPREAD DISTRIBUTION",   sub: "BID-ASK" },
          { title: "ORDER ARRIVALS",         sub: "HAWKES INTENSITY" },
        ].map(({ title, sub }, i) => (
          <Panel
            key={i}
            title={title}
            subtitle={sub}
            className={[
              "border-0",
              i % 2 === 0 ? "border-r border-[#1e1e1e]" : "",
              i < 2      ? "border-b border-[#1e1e1e]" : "",
            ].join(" ")}
          >
            <div className="flex items-center justify-center h-full text-[#333333] text-[0.6rem] tracking-widest">
              [ RECHARTS — PROMPT 2 ]
            </div>
          </Panel>
        ))}
      </div>

    </div>
  );
}
