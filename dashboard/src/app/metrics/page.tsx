import Panel from "@/components/terminal/Panel";
import BloombergTable from "@/components/terminal/BloombergTable";

const BENCHMARKS = [
  { operation: "LOB INSERT",          mean_ns: 778,   std_ns: 12,   throughput: "20.3M" },
  { operation: "LOB CANCEL",          mean_ns: 595,   std_ns: 9,    throughput: "16.8M" },
  { operation: "ORDER MATCH",         mean_ns: 1204,  std_ns: 31,   throughput: "10.5M" },
  { operation: "LOB SNAPSHOT L5",     mean_ns: 2180,  std_ns: 45,   throughput: "5.8M"  },
  { operation: "HAWKES STEP (12D)",   mean_ns: 8340,  std_ns: 120,  throughput: "1.2M"  },
  { operation: "PYO3 ROUND-TRIP",     mean_ns: 18500, std_ns: 280,  throughput: "540K"  },
];

const TRAIN_ROWS = [
  { timestep: "10K",  reward: "−12.4",  pnl: "−8.1",  sharpe: "−0.21" },
  { timestep: "50K",  reward: "42.3",   pnl: "38.7",  sharpe: "0.51"  },
  { timestep: "100K", reward: "118.9",  pnl: "112.4", sharpe: "1.14"  },
  { timestep: "200K", reward: "198.3",  pnl: "187.2", sharpe: "1.62"  },
  { timestep: "500K", reward: "248.7",  pnl: "238.1", sharpe: "1.88"  },
  { timestep: "1M",   reward: "274.1",  pnl: "261.9", sharpe: "1.91"  },
];

export default function MetricsPage() {
  return (
    <div className="flex flex-col h-full">

      {/* ── Benchmark Table ────────────────────────────────────────────────── */}
      <div className="border-b border-[#1e1e1e]">
        <Panel title="LATENCY BENCHMARKS" subtitle="CRITERION — cargo bench" className="border-0">
          <div className="p-2">
            <BloombergTable
              rowKey="operation"
              columns={[
                { key: "operation",   header: "OPERATION",    align: "left" },
                { key: "mean_ns",     header: "MEAN (ns)",    render: (v) => Number(v).toLocaleString() },
                { key: "std_ns",      header: "STD (ns)",     render: (v) => `±${v}` },
                { key: "throughput",  header: "THROUGHPUT",   render: (v) => <span className="text-[#ff8c00] font-bold">{String(v)}</span> },
              ]}
              rows={BENCHMARKS as unknown as Record<string, unknown>[]}
            />
          </div>
        </Panel>
      </div>

      {/* ── Summary stats ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 border-b border-[#1e1e1e] shrink-0">
        {[
          { label: "LOB INSERT",  value: "20.3M", unit: "OPS/S" },
          { label: "LOB CANCEL",  value: "16.8M", unit: "OPS/S" },
          { label: "ORDER MATCH", value: "10.5M", unit: "OPS/S" },
          { label: "PyO3 RTT",    value: "18.5",  unit: "μs" },
        ].map(({ label, value, unit }) => (
          <div
            key={label}
            className="p-3 border-r border-[#1e1e1e] last:border-0"
          >
            <div className="text-[0.6rem] uppercase tracking-widest text-[#666666] mb-0.5">{label}</div>
            <div className="text-lg font-mono font-bold text-[#ff8c00] glow-orange leading-none">
              {value}
              <span className="text-[0.6rem] font-normal text-[#444444] ml-1">{unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* ── Training curves table ──────────────────────────────────────────── */}
      <div className="flex-1">
        <Panel title="TRAINING PROGRESS" subtitle="SAC — 1M TIMESTEPS" className="border-0 h-full">
          <div className="p-2">
            <BloombergTable
              rowKey="timestep"
              columns={[
                { key: "timestep", header: "TIMESTEPS", align: "left" },
                { key: "reward",   header: "MEAN REWARD", colorize: true,
                  render: (v) => {
                    const n = parseFloat(String(v));
                    return <span style={{ color: n >= 0 ? "#00d26a" : "#ff3b3b" }}>{String(v)}</span>;
                  }
                },
                { key: "pnl",    header: "MEAN PNL",    colorize: true,
                  render: (v) => {
                    const n = parseFloat(String(v));
                    return <span style={{ color: n >= 0 ? "#00d26a" : "#ff3b3b" }}>{String(v)}</span>;
                  }
                },
                { key: "sharpe", header: "SHARPE",      colorize: true,
                  render: (v) => {
                    const n = parseFloat(String(v));
                    return <span style={{ color: n >= 0 ? "#00d26a" : "#ff3b3b" }}>{String(v)}</span>;
                  }
                },
              ]}
              rows={TRAIN_ROWS as unknown as Record<string, unknown>[]}
            />
          </div>
        </Panel>
      </div>

    </div>
  );
}
