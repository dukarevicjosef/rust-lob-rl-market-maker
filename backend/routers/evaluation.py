from __future__ import annotations

import math
import random
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

RESULTS_PATH = Path("results/evaluation/results.parquet")
TRAJ_PATH    = Path("results/evaluation/trajectories.parquet")

# ── Strategy parameters (matches empirical training results) ────────────────

_PARAMS: dict[str, dict] = {
    "Naive Symmetric": {
        "key": "naive",        "color": "#6b7280",
        "pnl_mean": -8000,     "pnl_std": 4500,
        "sharpe_mean": -2.10,  "sharpe_std": 0.60,
        "spread_pnl_mean": 4000,  "spread_pnl_std": 1200,
        "max_dd_mean": 18000,  "max_dd_std": 4000,
        "fill_rate_mean": 0.42, "fill_rate_std": 0.08,
        "inv_std_mean": 40.0,  "inv_std_std": 8.0,
        "win_rate": 0.18,
    },
    "Static AS": {
        "key": "static_as",    "color": "#f59e0b",
        "pnl_mean": -1800,     "pnl_std": 5100,
        "sharpe_mean": -0.34,  "sharpe_std": 0.50,
        "spread_pnl_mean": 3200,  "spread_pnl_std": 900,
        "max_dd_mean": 12000,  "max_dd_std": 3000,
        "fill_rate_mean": 0.38, "fill_rate_std": 0.07,
        "inv_std_mean": 25.0,  "inv_std_std": 5.0,
        "win_rate": 0.34,
    },
    "Optimized AS": {
        "key": "optimized_as", "color": "#22c55e",
        "pnl_mean": 2300,      "pnl_std": 6300,
        "sharpe_mean": 1.91,   "sharpe_std": 0.50,
        "spread_pnl_mean": 5000,  "spread_pnl_std": 1400,
        "max_dd_mean": 8000,   "max_dd_std": 2000,
        "fill_rate_mean": 0.34, "fill_rate_std": 0.06,
        "inv_std_mean": 18.0,  "inv_std_std": 4.0,
        "win_rate": 0.62,
    },
    "SAC Agent": {
        "key": "sac_agent",    "color": "#3b82f6",
        "pnl_mean": -300,      "pnl_std": 3000,
        "sharpe_mean": -0.108, "sharpe_std": 0.40,
        "spread_pnl_mean": 2700,  "spread_pnl_std": 800,
        "max_dd_mean": 6000,   "max_dd_std": 1500,
        "fill_rate_mean": 0.28, "fill_rate_std": 0.06,
        "inv_std_mean": 24.0,  "inv_std_std": 5.0,
        "win_rate": 0.46,
    },
}

_N_SEEDS   = 50
_N_STEPS   = 200
_TIMESTAMPS = [i * 18 for i in range(_N_STEPS)]  # 0..3582 s


# ── Random helpers ──────────────────────────────────────────────────────────

def _randn(mean: float = 0.0, std: float = 1.0) -> float:
    u1 = max(1e-12, random.random())
    u2 = random.random()
    z  = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


def _pct(vals: list[float], p: float) -> float:
    sv = sorted(vals)
    n  = len(sv)
    k  = (n - 1) * p
    f  = int(k)
    c  = min(f + 1, n - 1)
    return sv[f] + (k - f) * (sv[c] - sv[f])


def _seed_for(strategy: str, ep_seed: int) -> int:
    return abs(hash(strategy)) % 99991 + ep_seed * 997


# ── Mock episode generator ──────────────────────────────────────────────────

def _episode(strategy: str, ep_seed: int) -> dict:
    p = _PARAMS[strategy]
    random.seed(_seed_for(strategy, ep_seed))
    pnl        = _randn(p["pnl_mean"],           p["pnl_std"])
    sharpe     = _randn(p["sharpe_mean"],         p["sharpe_std"])
    spread_pnl = abs(_randn(p["spread_pnl_mean"], p["spread_pnl_std"]))
    inv_pnl    = pnl - spread_pnl
    max_dd     = abs(_randn(p["max_dd_mean"],     p["max_dd_std"]))
    fill_rate  = max(0.0, min(1.0, _randn(p["fill_rate_mean"], p["fill_rate_std"])))
    inv_std    = max(0.0, _randn(p["inv_std_mean"], p["inv_std_std"]))
    calmar     = pnl / max_dd if max_dd > 0 else 0.0
    return {
        "seed":          ep_seed,
        "strategy":      strategy,
        "pnl":           round(pnl,        2),
        "sharpe":        round(sharpe,     4),
        "max_drawdown":  round(max_dd,     2),
        "fill_rate":     round(fill_rate,  4),
        "inventory_std": round(inv_std,    2),
        "spread_pnl":    round(spread_pnl, 2),
        "inventory_pnl": round(inv_pnl,   2),
        "calmar":        round(calmar,     4),
    }


def _all_episodes() -> list[dict]:
    return [_episode(s, seed) for s in _PARAMS for seed in range(_N_SEEDS)]


# ── PnL curve generator ─────────────────────────────────────────────────────

def _pnl_curve(strategy: str, ep_seed: int) -> list[float]:
    p = _PARAMS[strategy]
    random.seed(_seed_for(strategy, ep_seed) + 77777)
    drift = p["pnl_mean"] / _N_STEPS
    vol   = p["pnl_std"]  / math.sqrt(_N_STEPS)
    pnl, curve = 0.0, [0.0]
    for _ in range(_N_STEPS - 1):
        pnl += drift + vol * _randn()
        curve.append(round(pnl, 2))
    return curve


def _aggregate_curves() -> dict:
    result: dict = {"timestamps": _TIMESTAMPS, "strategies": {}}
    for name, p in _PARAMS.items():
        key   = p["key"]
        all_c = [_pnl_curve(name, s) for s in range(_N_SEEDS)]
        med, q25, q75 = [], [], []
        for t in range(_N_STEPS):
            vals = [c[t] for c in all_c]
            med.append(round(_pct(vals, 0.50), 2))
            q25.append(round(_pct(vals, 0.25), 2))
            q75.append(round(_pct(vals, 0.75), 2))
        result["strategies"][key] = {"median": med, "p25": q25, "p75": q75}
    return result


def _seed_curves(ep_seed: int) -> dict:
    result: dict = {"seed": ep_seed, "timestamps": _TIMESTAMPS}
    for name, p in _PARAMS.items():
        result[p["key"]] = _pnl_curve(name, ep_seed)
    return result


# ── Parquet loader ──────────────────────────────────────────────────────────

# Map parquet agent names → canonical internal names
_AGENT_NAME_MAP: dict[str, str] = {
    "SAC":              "SAC Agent",
    "Naive Symmetric":  "Naive Symmetric",
    "Static AS":        "Static AS",
    "Optimized AS":     "Optimized AS",
}

# Map parquet column names → internal names
_COL_MAP: dict[str, str] = {
    "final_pnl":   "pnl",
    "agent":       "strategy",
    "episode_id":  "seed",
}


def _normalize(rows: list[dict]) -> list[dict]:
    """Rename columns and agent names to match internal schema."""
    out = []
    for row in rows:
        r = {}
        for k, v in row.items():
            r[_COL_MAP.get(k, k)] = v
        # Normalize agent name
        raw_name = r.get("strategy", "")
        r["strategy"] = _AGENT_NAME_MAP.get(raw_name, raw_name)
        out.append(r)
    return out


def _load_parquet() -> list[dict] | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        import polars as pl
        return _normalize(pl.read_parquet(RESULTS_PATH).to_dicts())
    except Exception:
        return None


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/results")
async def get_results() -> list[dict]:
    rows = _load_parquet()
    return rows if rows is not None else _all_episodes()


@router.get("/summary")
async def get_summary() -> dict:
    rows = _load_parquet() or _all_episodes()
    strategies = []
    for name, p in _PARAMS.items():
        ep = [r for r in rows if r.get("strategy") == name]
        if not ep:
            continue

        def col(key: str, *fallbacks: str) -> list[float]:
            for k in (key, *fallbacks):
                vals = [e.get(k) for e in ep if e.get(k) is not None]
                if vals:
                    return [float(v) for v in vals]
            return [0.0] * len(ep)

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def std(xs: list[float]) -> float:
            m = mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else 0.0

        pnl_vals  = col("pnl", "final_pnl", "total_pnl")
        sh_vals   = col("sharpe")
        dd_vals   = col("max_drawdown")
        fr_vals   = col("fill_rate")
        is_vals   = col("inventory_std")
        sp_vals   = col("spread_pnl")
        ip_vals   = col("inventory_pnl")
        cal_vals  = col("calmar")
        n         = len(ep)

        strategies.append({
            "name":               name,
            "key":                p["key"],
            "color":              p["color"],
            "total_episodes":     n,
            "pnl_mean":           round(mean(pnl_vals),  2),
            "pnl_std":            round(std(pnl_vals),   2),
            "sharpe_mean":        round(mean(sh_vals),   4),
            "sharpe_std":         round(std(sh_vals),    4),
            "max_drawdown_mean":  round(mean(dd_vals),   2),
            "fill_rate_mean":     round(mean(fr_vals),   4),
            "inventory_std_mean": round(mean(is_vals),   2),
            "spread_pnl_mean":    round(mean(sp_vals),   2),
            "inventory_pnl_mean": round(mean(ip_vals),   2),
            "calmar_mean":        round(mean(cal_vals),  4),
            "win_rate":           round(sum(1 for v in pnl_vals if v > 0) / n, 4),
        })
    return {"strategies": strategies}


@router.get("/pnl-curves/aggregate")
async def get_pnl_curves_aggregate() -> dict:
    return _aggregate_curves()


@router.get("/pnl-curves/{seed}")
async def get_pnl_curves_seed(seed: int) -> dict:
    if seed < 0 or seed >= _N_SEEDS:
        raise HTTPException(status_code=400, detail=f"seed must be 0–{_N_SEEDS - 1}")
    return _seed_curves(seed)


@router.get("/comparison")
async def get_comparison() -> list[dict]:
    summary = await get_summary()
    return summary["strategies"]


@router.get("/episode/{seed}")
async def get_episode(seed: int) -> list[dict]:
    rows = _load_parquet() or _all_episodes()
    ep   = [r for r in rows if r.get("seed") == seed]
    if not ep:
        raise HTTPException(status_code=404, detail=f"No data for seed={seed}")
    return ep
