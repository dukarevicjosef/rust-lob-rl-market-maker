from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

RESULTS_PATH = Path("results/evaluation/results.parquet")
TRAJ_PATH = Path("results/evaluation/trajectories.parquet")


def _load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="results.parquet not found — run evaluation first")
    import polars as pl
    return pl.read_parquet(RESULTS_PATH).to_dicts()


@router.get("/results")
async def get_results() -> list[dict]:
    return _load_results()


@router.get("/comparison")
async def get_comparison() -> list[dict]:
    import polars as pl
    rows = _load_results()
    df = pl.DataFrame(rows)
    numeric = ["total_pnl", "spread_pnl", "inventory_pnl", "sharpe", "inventory_std", "fill_rate"]
    existing = [c for c in numeric if c in df.columns]
    agg = (
        df.group_by("agent")
        .agg([pl.col(c).mean().alias(c) for c in existing])
        .sort("total_pnl", descending=True)
    )
    return agg.to_dicts()


@router.get("/episode/{seed}")
async def get_episode(seed: int) -> list[dict]:
    if not TRAJ_PATH.exists():
        raise HTTPException(status_code=404, detail="trajectories.parquet not found")
    import polars as pl
    df = pl.read_parquet(TRAJ_PATH).filter(pl.col("seed") == seed)
    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"No episode with seed={seed}")
    return df.to_dicts()
