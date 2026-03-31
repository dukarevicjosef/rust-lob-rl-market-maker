"use client";

interface BoxDatum {
  name:   string;
  key:    string;
  color:  string;
  values: number[];
}

interface BoxStats {
  min:    number;
  q1:     number;
  median: number;
  q3:     number;
  max:    number;
}

function computeBox(values: number[]): BoxStats {
  if (values.length === 0) return { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
  const sv = [...values].sort((a, b) => a - b);
  const n  = sv.length;
  const q  = (p: number) => {
    const k = (n - 1) * p;
    const f = Math.floor(k);
    const c = Math.min(f + 1, n - 1);
    return sv[f] + (k - f) * (sv[c] - sv[f]);
  };
  const q1  = q(0.25);
  const med = q(0.50);
  const q3  = q(0.75);
  const iqr = q3 - q1;
  return {
    min:    Math.max(sv[0],     q1 - 1.5 * iqr),
    q1,
    median: med,
    q3,
    max:    Math.min(sv[n - 1], q3 + 1.5 * iqr),
  };
}

export default function DistributionBox({ data }: { data: BoxDatum[] }) {
  if (data.length === 0 || data.every((d) => d.values.length === 0)) {
    return (
      <div className="flex items-center justify-center h-full text-[#333] text-[0.65rem] tracking-widest">
        NO DATA
      </div>
    );
  }

  const stats = data.map((d) => ({ ...d, box: computeBox(d.values) }));

  // Layout constants (viewBox coordinates)
  const VW = 400;
  const VH = 200;
  const PL = 48;   // left padding for y-axis labels
  const PR = 8;
  const PT = 12;
  const PB = 20;
  const plotW = VW - PL - PR;
  const plotH = VH - PT - PB;

  // Y-axis range
  const allVals = stats.flatMap((s) => [s.box.min, s.box.q1, s.box.q3, s.box.max]);
  const raw_min = Math.min(...allVals);
  const raw_max = Math.max(...allVals);
  const pad     = (raw_max - raw_min) * 0.12 || 1000;
  const yMin    = raw_min - pad;
  const yMax    = raw_max + pad;
  const yRange  = yMax - yMin;

  const yS  = (v: number) => PT + plotH - ((v - yMin) / yRange) * plotH;
  const xC  = (i: number) => PL + ((i + 0.5) / data.length) * plotW;
  const boxW = Math.min(plotW / data.length * 0.38, 32);

  const yTicks = [-20000, -10000, 0, 10000, 20000].filter(
    (v) => v >= yMin && v <= yMax,
  );

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${VW} ${VH}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Y-axis grid + labels */}
      {yTicks.map((v) => {
        const y = yS(v);
        return (
          <g key={v}>
            <line
              x1={PL} x2={PL + plotW} y1={y} y2={y}
              stroke={v === 0 ? "#2e2e2e" : "#161616"}
              strokeWidth={v === 0 ? 0.7 : 0.5}
              strokeDasharray={v === 0 ? "4 4" : undefined}
            />
            <text
              x={PL - 4} y={y + 3}
              textAnchor="end"
              fill="#555"
              fontSize={7}
              fontFamily="'JetBrains Mono', monospace"
            >
              {v >= 0 ? "+" : ""}{(v / 1000).toFixed(0)}K
            </text>
          </g>
        );
      })}

      {/* Box plots */}
      {stats.map((s, i) => {
        const { min, q1, median, q3, max } = s.box;
        const cx = xC(i);
        const bw = boxW;

        return (
          <g key={s.key}>
            {/* Whisker: lower */}
            <line x1={cx} x2={cx} y1={yS(q1)}  y2={yS(min)}
              stroke={s.color} strokeWidth={0.8} />
            <line x1={cx - bw / 3} x2={cx + bw / 3} y1={yS(min)} y2={yS(min)}
              stroke={s.color} strokeWidth={0.8} />

            {/* Whisker: upper */}
            <line x1={cx} x2={cx} y1={yS(max)}  y2={yS(q3)}
              stroke={s.color} strokeWidth={0.8} />
            <line x1={cx - bw / 3} x2={cx + bw / 3} y1={yS(max)} y2={yS(max)}
              stroke={s.color} strokeWidth={0.8} />

            {/* Box */}
            <rect
              x={cx - bw / 2}
              y={yS(q3)}
              width={bw}
              height={Math.abs(yS(q1) - yS(q3))}
              fill={s.color}
              fillOpacity={0.12}
              stroke={s.color}
              strokeWidth={0.8}
            />

            {/* Median */}
            <line
              x1={cx - bw / 2} x2={cx + bw / 2}
              y1={yS(median)} y2={yS(median)}
              stroke={s.color} strokeWidth={1.8}
            />

            {/* Median value label */}
            <text
              x={cx + bw / 2 + 3} y={yS(median) + 3}
              fill={s.color} fontSize={6} fontFamily="monospace"
            >
              {median >= 0 ? "+" : ""}{(median / 1000).toFixed(1)}K
            </text>

            {/* X label */}
            <text
              x={cx} y={VH - 4}
              textAnchor="middle" fill="#555" fontSize={7} fontFamily="monospace"
            >
              {s.name.split(" ")[0]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
