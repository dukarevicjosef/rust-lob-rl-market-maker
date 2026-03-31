from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

CRITERION_ROOT = Path("target/criterion")
WANDB_DIR = Path("wandb")


@router.get("/benchmarks")
async def get_benchmarks() -> list[dict]:
    results: list[dict] = []
    if not CRITERION_ROOT.exists():
        return results
    for estimates_path in CRITERION_ROOT.rglob("new/estimates.json"):
        bench_name = estimates_path.parts[-4]  # target/criterion/<name>/new/estimates.json
        try:
            data = json.loads(estimates_path.read_text())
            mean_ns = data["mean"]["point_estimate"]
            std_ns = data["std_dev"]["point_estimate"]
            results.append(
                {
                    "operation": bench_name,
                    "mean_ns": mean_ns,
                    "std_ns": std_ns,
                    "throughput_per_sec": 1e9 / mean_ns if mean_ns > 0 else 0.0,
                }
            )
        except (KeyError, json.JSONDecodeError):
            continue
    return sorted(results, key=lambda r: r["throughput_per_sec"], reverse=True)


@router.get("/training")
async def get_training_curves() -> list[dict]:
    """Return latest W&B run metrics if available, else empty list."""
    if not WANDB_DIR.exists():
        return []
    run_dirs = sorted(WANDB_DIR.glob("run-*"), reverse=True)
    if not run_dirs:
        return []
    history_path = run_dirs[0] / "files" / "wandb-history.jsonl"
    if not history_path.exists():
        return []
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    return rows
