"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import Panel from "@/components/terminal/Panel";
import DataCell from "@/components/terminal/DataCell";
import PnlChart, { type PnlChartPoint } from "@/components/charts/PnlChart";
import SharpeBar, { type SharpeBarDatum } from "@/components/charts/SharpeBar";
import PnlDecomposition, { type DecompDatum } from "@/components/charts/PnlDecomposition";
import DistributionBox from "@/components/charts/DistributionBox";
import { STRATEGIES } from "@/lib/types";
import type {
  StrategySummary,
  EpisodeResult,
  AggregatePnlCurves,
  SeedPnlCurves,
  StrategyConfig,
} from "@/lib/types";

// ── Constants ───────────────────────────────────────────────────────────────

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const N_SEEDS = 50;

const TABS = [
  { key: "all",       label: "ALL STRATEGIES",  keys: ["sac_agent", "optimized_as", "static_as", "naive"] },
  { key: "sac",       label: "SAC AGENT",       keys: ["sac_agent"] },
  { key: "opt_as",    label: "OPTIMIZED AS",    keys: ["optimized_as"] },
  { key: "baselines", label: "BASELINES",       keys: ["static_as", "naive"] },
];

const TABLE_COLS: { key: keyof StrategySummary; label: string; fmt: (v: number) => string }[] = [
  { key: "pnl_mean",           label: "PNL MEAN",    fmt: (v) => `${v >= 0 ? "+" : ""}${(v / 1000).toFixed(1)}K` },
  { key: "pnl_std",            label: "± STD",       fmt: (v) => `${(v / 1000).toFixed(1)}K` },
  { key: "sharpe_mean",        label: "SHARPE",      fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}` },
  { key: "max_drawdown_mean",  label: "MAX DD",      fmt: (v) => `${(v / 1000).toFixed(1)}K` },
  { key: "calmar_mean",        label: "CALMAR",      fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}` },
  { key: "fill_rate_mean",     label: "FILL RT",     fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "inventory_std_mean", label: "INV STD",     fmt: (v) => v.toFixed(1) },
  { key: "win_rate",           label: "WIN RATE",    fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: "spread_pnl_mean",    label: "SPREAD PNL",  fmt: (v) => `${(v / 1000).toFixed(1)}K` },
  { key: "inventory_pnl_mean", label: "INV PNL",     fmt: (v) => `${v >= 0 ? "+" : ""}${(v / 1000).toFixed(1)}K` },
];

// "Higher is better" columns
const HIGHER_BETTER = new Set<keyof StrategySummary>([
  "pnl_mean", "sharpe_mean", "calmar_mean", "fill_rate_mean", "win_rate", "spread_pnl_mean",
]);

// ── Page ────────────────────────────────────────────────────────────────────

export default function ArenaPage() {
  const [activeTab,   setActiveTab]   = useState("all");
  const [selectedSeed, setSeed]       = useState<number | null>(null);
  const [sortCol, setSortCol]         = useState<keyof StrategySummary>("pnl_mean");
  const [sortDir, setSortDir]         = useState<"asc" | "desc">("desc");

  const [summary,    setSummary]      = useState<StrategySummary[]>([]);
  const [episodes,   setEpisodes]     = useState<EpisodeResult[]>([]);
  const [aggCurves,  setAggCurves]    = useState<AggregatePnlCurves | null>(null);
  const [seedCurves, setSeedCurves]   = useState<SeedPnlCurves | null>(null);
  const [loading,    setLoading]      = useState(true);
  const [backendOk,  setBackendOk]    = useState(true);

  // ── Data fetching ──────────────────────────────────────────────────────────

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/evaluation/summary`).then((r) => r.json()),
      fetch(`${API}/api/evaluation/results`).then((r) => r.json()),
      fetch(`${API}/api/evaluation/pnl-curves/aggregate`).then((r) => r.json()),
    ])
      .then(([sum, eps, curves]) => {
        setSummary(sum?.strategies ?? []);
        setEpisodes(eps ?? []);
        setAggCurves(curves ?? null);
        setLoading(false);
      })
      .catch(() => {
        setBackendOk(false);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (selectedSeed === null) { setSeedCurves(null); return; }
    fetch(`${API}/api/evaluation/pnl-curves/${selectedSeed}`)
      .then((r) => r.json())
      .then((d) => setSeedCurves(d))
      .catch(() => setSeedCurves(null));
  }, [selectedSeed]);

  // ── Derived data ───────────────────────────────────────────────────────────

  const tab            = TABS.find((t) => t.key === activeTab)!;
  const visibleKeys    = tab.keys;
  const visibleStrategies: StrategyConfig[] = STRATEGIES.filter((s) =>
    visibleKeys.includes(s.key),
  );
  const focused        = summary.find((s) => s.key === visibleKeys[0]);

  // Chart data: reshape API format → recharts format
  const chartData: PnlChartPoint[] = useMemo(() => {
    if (!aggCurves) return [];
    const hasSeed = selectedSeed !== null && seedCurves;
    return aggCurves.timestamps.map((t, i) => {
      const pt: PnlChartPoint = { time: t };
      for (const key of visibleKeys) {
        const agg = aggCurves.strategies[key];
        if (!agg) continue;
        if (hasSeed && Array.isArray(seedCurves![key as string])) {
          pt[`${key}_med`] = (seedCurves![key as string] as number[])[i] ?? 0;
        } else {
          pt[`${key}_med`] = agg.median[i] ?? 0;
          pt[`${key}_p25`] = agg.p25[i]    ?? 0;
          pt[`${key}_p75`] = agg.p75[i]    ?? 0;
        }
      }
      return pt;
    });
  }, [aggCurves, seedCurves, selectedSeed, visibleKeys]);

  const sharpeData: SharpeBarDatum[] = useMemo(
    () =>
      STRATEGIES.map((s) => {
        const m = summary.find((x) => x.key === s.key);
        return { name: s.name, key: s.key, color: s.color, sharpe: m?.sharpe_mean ?? 0, std: m?.sharpe_std ?? 0 };
      }),
    [summary],
  );

  const decompData: DecompDatum[] = useMemo(
    () =>
      STRATEGIES.map((s) => {
        const m = summary.find((x) => x.key === s.key);
        return {
          name: s.name, key: s.key,
          spread_pnl:    m?.spread_pnl_mean    ?? 0,
          inventory_pnl: m?.inventory_pnl_mean ?? 0,
        };
      }),
    [summary],
  );

  const boxData = useMemo(
    () =>
      STRATEGIES.map((s) => ({
        name:   s.name,
        key:    s.key,
        color:  s.color,
        values: episodes.filter((e) => e.strategy === s.name).map((e) => e.pnl),
      })),
    [episodes],
  );

  // Sortable table
  const tableRows = useMemo(() => {
    return [...summary].sort((a, b) => {
      const va = (a[sortCol] as number) ?? 0;
      const vb = (b[sortCol] as number) ?? 0;
      return sortDir === "desc" ? vb - va : va - vb;
    });
  }, [summary, sortCol, sortDir]);

  // Per-column best/worst for highlighting
  const colStats = useMemo(() => {
    const out: Record<string, { best: number; worst: number }> = {};
    for (const col of TABLE_COLS) {
      const vals = tableRows.map((r) => r[col.key] as number);
      out[String(col.key)] = {
        best:  HIGHER_BETTER.has(col.key) ? Math.max(...vals) : Math.min(...vals),
        worst: HIGHER_BETTER.has(col.key) ? Math.min(...vals) : Math.max(...vals),
      };
    }
    return out;
  }, [tableRows]);

  const toggleSort = useCallback(
    (col: keyof StrategySummary) => {
      if (sortCol === col) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
      else { setSortCol(col); setSortDir("desc"); }
    },
    [sortCol],
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  if (!backendOk) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <div className="text-[#ff3b3b] text-[0.7rem] font-mono tracking-widest">
          BACKEND UNAVAILABLE
        </div>
        <div className="text-[#444] text-[0.6rem] font-mono">
          uv run uvicorn backend.main:app --reload --port 8000
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-[#444] text-[0.65rem] font-mono tracking-widest">
        LOADING<span className="blink">_</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e1e1e] shrink-0">
        <div>
          <div className="text-[#ff8c00] font-bold text-[0.75rem] uppercase tracking-widest">
            STRATEGY ARENA
          </div>
          <div className="text-[#444] text-[0.6rem] font-mono">
            {N_SEEDS} EPISODES · SEEDS 0–{N_SEEDS - 1} · 1H SIMULATIONS
          </div>
        </div>

        {/* Episode selector */}
        <div className="flex items-center gap-2 text-[0.6rem] font-mono">
          <span className="text-[#444]">EPISODE</span>
          <select
            value={selectedSeed ?? "agg"}
            onChange={(e) =>
              setSeed(e.target.value === "agg" ? null : parseInt(e.target.value))
            }
            className="bg-[#111] border border-[#1e1e1e] text-[#cccccc] text-[0.6rem] px-2 py-0.5 font-mono appearance-none cursor-pointer hover:border-[#ff8c00] outline-none"
          >
            <option value="agg">ALL (AGGREGATE)</option>
            {Array.from({ length: N_SEEDS }, (_, i) => (
              <option key={i} value={i}>SEED {i}</option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Stat cards ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 border-b border-[#1e1e1e] shrink-0">
        <DataCell
          label={`PNL — ${focused?.name ?? "—"}`}
          value={focused ? focused.pnl_mean : 0}
          colorize
          precision={0}
          unit={focused ? `  ±${(focused.pnl_std / 1000).toFixed(1)}K` : ""}
          className="border-r border-[#1e1e1e]"
        />
        <DataCell
          label="SHARPE (MEAN)"
          value={focused ? focused.sharpe_mean : 0}
          colorize
          precision={3}
          className="border-r border-[#1e1e1e]"
        />
        <DataCell
          label="MAX DRAWDOWN"
          value={focused ? focused.max_drawdown_mean : 0}
          precision={0}
          className="border-r border-[#1e1e1e]"
        />
        <DataCell
          label="FILL RATE"
          value={focused ? `${(focused.fill_rate_mean * 100).toFixed(1)}` : "0.0"}
          unit="%"
        />
      </div>

      {/* ── Strategy tab bar ────────────────────────────────────────────────── */}
      <div className="flex h-7 border-b border-[#1e1e1e] shrink-0">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={[
              "px-4 h-full text-[0.62rem] tracking-widest border-r border-[#1e1e1e] cursor-pointer transition-colors",
              activeTab === key
                ? "bg-[#ff8c00] text-black font-bold"
                : "text-[#555] hover:bg-[#161616] hover:text-[#aaa]",
            ].join(" ")}
          >
            {`<${label}>`}
          </button>
        ))}
        <div className="flex-1" />
        <span className="self-center pr-3 text-[0.6rem] text-[#333] font-mono">
          {selectedSeed !== null ? `SEED ${selectedSeed}` : "MEDIAN + P25/P75"}
        </span>
      </div>

      {/* ── PnL Chart ───────────────────────────────────────────────────────── */}
      <Panel
        title="CUMULATIVE PNL"
        subtitle="TIME (s) → CUMULATIVE PNL ($)"
        className="border-0 border-b border-[#1e1e1e] shrink-0"
      >
        <div style={{ height: 280 }}>
          <PnlChart
            data={chartData}
            strategies={visibleStrategies}
            showBands={selectedSeed === null}
            height={280}
          />
        </div>
      </Panel>

      {/* ── Bottom 3 charts ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-3 border-b border-[#1e1e1e] shrink-0" style={{ height: 220 }}>
        <Panel title="SHARPE RATIO" subtitle="MEAN ± 1σ" className="border-0 border-r border-[#1e1e1e] h-full">
          <SharpeBar data={sharpeData} />
        </Panel>
        <Panel title="PNL DECOMPOSITION" subtitle="SPREAD vs. INVENTORY" className="border-0 border-r border-[#1e1e1e] h-full">
          <PnlDecomposition data={decompData} />
        </Panel>
        <Panel title="PNL DISTRIBUTION" subtitle="Q1 / MEDIAN / Q3" className="border-0 h-full">
          <DistributionBox data={boxData} />
        </Panel>
      </div>

      {/* ── Detail table ────────────────────────────────────────────────────── */}
      <div className="shrink-0 pb-4">
        <div className="px-3 pt-2 pb-1 text-[0.6rem] text-[#ff8c00] uppercase tracking-widest border-b border-[#1e1e1e]">
          STRATEGY COMPARISON
        </div>
        <div className="overflow-x-auto">
          <table className="w-full bb-table border-collapse">
            <thead>
              <tr>
                <th
                  style={{ textAlign: "left", cursor: "pointer" }}
                  onClick={() => toggleSort("name" as keyof StrategySummary)}
                >
                  STRATEGY
                </th>
                {TABLE_COLS.map((col) => (
                  <th
                    key={String(col.key)}
                    style={{ textAlign: "right", cursor: "pointer" }}
                    onClick={() => toggleSort(col.key)}
                  >
                    {col.label}
                    {sortCol === col.key && (
                      <span className="ml-1 text-[#ff8c00]">
                        {sortDir === "desc" ? "▼" : "▲"}
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row) => {
                const strat = STRATEGIES.find((s) => s.key === row.key);
                return (
                  <tr key={row.key}>
                    <td style={{ textAlign: "left" }}>
                      <span
                        style={{
                          display:      "inline-block",
                          width:        6,
                          height:       6,
                          borderRadius: 0,
                          background:   strat?.color ?? "#666",
                          marginRight:  6,
                        }}
                      />
                      <span style={{ color: "#aaa" }}>{row.name}</span>
                    </td>
                    {TABLE_COLS.map((col) => {
                      const raw  = row[col.key] as number;
                      const stat = colStats[String(col.key)];
                      const isBest  = raw === stat?.best;
                      const isWorst = raw === stat?.worst;
                      const isNum   = typeof raw === "number";
                      const isPos   = isNum && raw > 0;
                      const isNeg   = isNum && raw < 0;
                      const color   = isBest
                        ? "#00d26a"
                        : isWorst
                        ? "#ff3b3b"
                        : isPos && HIGHER_BETTER.has(col.key)
                        ? "#cccccc"
                        : isNeg && HIGHER_BETTER.has(col.key)
                        ? "#888"
                        : "#999";
                      return (
                        <td
                          key={String(col.key)}
                          style={{
                            color,
                            fontWeight: isBest ? "bold" : undefined,
                            textAlign: "right",
                          }}
                        >
                          {col.fmt(raw)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
