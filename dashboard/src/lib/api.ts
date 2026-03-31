import type {
  EpisodeResult,
  StrategySummary,
  AggregatePnlCurves,
  SeedPnlCurves,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

export async function fetchSummary(): Promise<{ strategies: StrategySummary[] } | null> {
  return get("/api/evaluation/summary");
}

export async function fetchResults(): Promise<EpisodeResult[] | null> {
  return get("/api/evaluation/results");
}

export async function fetchAggregateCurves(): Promise<AggregatePnlCurves | null> {
  return get("/api/evaluation/pnl-curves/aggregate");
}

export async function fetchSeedCurves(seed: number): Promise<SeedPnlCurves | null> {
  return get(`/api/evaluation/pnl-curves/${seed}`);
}
