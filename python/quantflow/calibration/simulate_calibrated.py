"""
Simulate trading sessions using calibrated multivariate Hawkes parameters.

Price path is modelled as a tick-level random walk driven by market-order events.
Spread evolves via a simple mean-reverting noise model.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from .hawkes_mle import CalibrationResult

# ── LOB event-type constants (must match EventClassifier.DIM_NAMES) ──────────
DIM_MARKET_BUY     = 0
DIM_MARKET_SELL    = 1
DIM_LIM_BUY_BEST   = 2
DIM_LIM_SELL_BEST  = 3
DIM_CXL_BUY_BEST   = 6   # best-level cancel → widens spread
DIM_CXL_SELL_BEST  = 7   # best-level cancel → widens spread


def simulate_from_calibration(
    params_path:  str | Path,
    n_sessions:   int = 10,
    t_max:        float = 600.0,
    sigma_tick:   float | None = None,
    P0:           float = 65_000.0,
    spread_init:  float = 0.20,
    seed_base:    int = 42,
) -> list[dict]:
    """
    Simulate N independent trading sessions with calibrated Hawkes parameters.

    Uses Ogata's thinning algorithm for exact multivariate Hawkes simulation.
    Price dynamics: random walk driven by market-order events.

    Parameters
    ----------
    params_path  : path to hawkes_params.json (output of HawkesMLE.calibrate)
    n_sessions   : number of independent sessions to simulate
    t_max        : length of each session in seconds
    sigma_tick   : price move per market event (auto-calibrated if None)
    P0           : starting mid price in USD
    spread_init  : initial bid-ask spread in USD
    seed_base    : base seed; session i uses seed_base + i

    Returns
    -------
    list of session dicts, each with keys:
        "events"            : list[(time, dim)]
        "mid_prices"        : ndarray (K, 2)  — [(time, price)]
        "spreads"           : ndarray (K, 2)  — [(time, spread)]
        "inter_event_times" : dict[dim -> ndarray]
        "t_max"             : float
    """
    params = CalibrationResult.load(params_path)

    # Auto-calibrate σ_tick from typical BTC intraday vol:
    # daily vol ≈ 2 %, ~6 market events/s  →  σ per event ≈ P0·0.02/√(86400·6)
    if sigma_tick is None:
        sigma_tick = P0 * 0.02 / np.sqrt(86_400 * 6)

    return [
        _simulate_session(params, t_max, seed_base + i, P0, sigma_tick, spread_init)
        for i in range(n_sessions)
    ]


# ── Simulation core ───────────────────────────────────────────────────────────


def _simulate_session(
    params:       CalibrationResult,
    t_max:        float,
    seed:         int,
    P0:           float,
    sigma_tick:   float,
    spread_init:  float,
) -> dict:
    """
    Ogata thinning for multivariate Hawkes + simplified price model.

    Intensity for dim i:
        λ_i(t) = μ_i + Σ_j α_{ij} · R_{ij}(t)
        R_{ij}(t) = Σ_{s<t, dim=j} exp(-β_{ij}·(t−s))

    Upper bound for thinning: λ*(t) = Σ_i λ_i(t), monotonically decreasing
    between events (Hawkes excitation decays exponentially).
    """
    rng = np.random.default_rng(seed)
    D   = params.n_dims

    # Build parameter matrices (D×D)
    mu    = np.zeros(D)
    alpha = np.zeros((D, D))   # alpha[i, j] = α_{ij}: excitation from j to i
    beta  = np.zeros((D, D))   # beta[i, j]  = β_{ij}

    for p in params.dim_params:
        i         = p.dim
        mu[i]     = p.mu
        alpha[i]  = p.alpha   # HawkesParams.alpha[j] = α_{i,j}
        beta[i]   = p.beta

    # Running sums R[i, j] = Σ_{s<t, dim=j} exp(-β_{ij}·(t−s))
    R = np.zeros((D, D))

    events: list[tuple[float, int]] = []
    t = 0.0

    # Price state
    mid         = P0
    half_spread = spread_init / 2.0
    half_spread_min = max(spread_init / 4.0, 0.05)   # minimum half-spread (≥ 0.05 USD)
    half_spread_max = spread_init * 5.0               # cap at 5× initial
    mid_recs:    list[tuple[float, float]] = [(0.0, mid)]
    spread_recs: list[tuple[float, float]] = [(0.0, spread_init)]

    while t < t_max:
        # Intensity vector at current t (shape D)
        lambda_i    = mu + np.einsum("ij,ij->i", alpha, R)
        lambda_star = float(np.sum(lambda_i))

        if lambda_star <= 0.0:
            break

        # Draw inter-arrival from Exp(λ*)
        dt    = rng.exponential(1.0 / lambda_star)
        t_new = t + dt
        if t_new > t_max:
            break

        # Decay R over dt
        exp_decay = np.exp(-beta * dt)
        R_new     = R * exp_decay

        # Actual intensity at t_new
        lambda_i_new  = mu + np.einsum("ij,ij->i", alpha, R_new)
        lambda_new    = float(np.sum(lambda_i_new))

        # Accept with probability λ_new / λ*
        if rng.uniform() <= lambda_new / lambda_star:
            # Determine which dimension
            u_dim = rng.uniform() * lambda_new
            dim   = int(np.searchsorted(np.cumsum(lambda_i_new), u_dim))
            dim   = min(dim, D - 1)

            events.append((t_new, dim))
            R = R_new.copy()
            R[:, dim] += 1.0   # New event from dim j contributes to all rows

            # Update price state
            if dim == DIM_MARKET_BUY:
                mid        += abs(rng.normal(0.0, sigma_tick))
                half_spread = max(half_spread * rng.uniform(0.95, 1.05), half_spread_min)
            elif dim == DIM_MARKET_SELL:
                mid        -= abs(rng.normal(0.0, sigma_tick))
                half_spread = max(half_spread * rng.uniform(0.95, 1.05), half_spread_min)
            elif dim in (DIM_LIM_BUY_BEST, DIM_LIM_SELL_BEST):
                # New limit quote at best: spread narrows slightly
                half_spread = max(half_spread * rng.uniform(0.92, 1.0), half_spread_min)
            elif dim in (DIM_CXL_BUY_BEST, DIM_CXL_SELL_BEST):
                # Best-level cancel: spread widens slightly
                half_spread = min(half_spread * rng.uniform(1.0, 1.03), half_spread_max)
            # Deep limit / deep cancel events don't affect spread

            mid_recs.append((t_new, mid))
            spread_recs.append((t_new, 2.0 * half_spread))
        else:
            R = R_new.copy()

        t = t_new

    # Per-dim inter-event times
    dim_times: dict[int, list[float]] = {j: [] for j in range(D)}
    for ev_t, ev_d in events:
        dim_times[ev_d].append(ev_t)

    iet: dict[int, np.ndarray] = {
        j: np.diff(np.array(ts)) if len(ts) > 1 else np.array([])
        for j, ts in dim_times.items()
    }

    return {
        "events":            events,
        "mid_prices":        np.array(mid_recs,    dtype=np.float64),
        "spreads":           np.array(spread_recs, dtype=np.float64),
        "inter_event_times": iet,
        "t_max":             t_max,
    }
