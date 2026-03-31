export interface StrategyConfig {
  name:  string;
  key:   string;
  color: string;
}

export const STRATEGIES: StrategyConfig[] = [
  { name: "SAC Agent",       key: "sac_agent",    color: "#3b82f6" },
  { name: "Optimized AS",    key: "optimized_as", color: "#22c55e" },
  { name: "Static AS",       key: "static_as",    color: "#f59e0b" },
  { name: "Naive Symmetric", key: "naive",        color: "#6b7280" },
];

export interface EpisodeResult {
  seed:          number;
  strategy:      string;
  pnl:           number;
  sharpe:        number;
  max_drawdown:  number;
  fill_rate:     number;
  inventory_std: number;
  spread_pnl:    number;
  inventory_pnl: number;
  calmar:        number;
}

export interface StrategySummary {
  name:                string;
  key:                 string;
  color:               string;
  total_episodes:      number;
  pnl_mean:            number;
  pnl_std:             number;
  sharpe_mean:         number;
  sharpe_std:          number;
  max_drawdown_mean:   number;
  fill_rate_mean:      number;
  inventory_std_mean:  number;
  spread_pnl_mean:     number;
  inventory_pnl_mean:  number;
  calmar_mean:         number;
  win_rate:            number;
}

export interface PnlBand {
  median: number[];
  p25:    number[];
  p75:    number[];
}

export interface AggregatePnlCurves {
  timestamps: number[];
  strategies: Record<string, PnlBand>;
}

export interface SeedPnlCurves {
  seed:       number;
  timestamps: number[];
  [key: string]: number[] | number;
}
