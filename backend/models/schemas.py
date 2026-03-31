from __future__ import annotations

from pydantic import BaseModel


class SimulationConfig(BaseModel):
    episode_length: int = 1000
    hawkes_baseline: float = 0.5
    hawkes_alpha: float = 0.3
    hawkes_beta: float = 1.0
    tick_size: float = 0.01
    max_inventory: int = 50


class SimulationStatus(BaseModel):
    is_running: bool
    step: int
    elapsed_s: float


class LOBSnapshot(BaseModel):
    bids: list[tuple[float, float]]  # [(price, qty), ...]
    asks: list[tuple[float, float]]
    mid_price: float
    spread: float
    timestamp_ns: int


class AgentAction(BaseModel):
    bid_offset: float
    ask_offset: float
    inventory: float
    pnl: float
    step: int


class LiveEvent(BaseModel):
    type: str  # "snapshot" | "trade" | "action"
    snapshot: LOBSnapshot | None = None
    action: AgentAction | None = None


class EpisodeResult(BaseModel):
    agent: str
    seed: int
    total_pnl: float
    spread_pnl: float
    inventory_pnl: float
    sharpe: float
    inventory_std: float
    fill_rate: float


class BenchmarkResult(BaseModel):
    operation: str
    throughput_per_sec: float
    mean_ns: float
    std_ns: float
