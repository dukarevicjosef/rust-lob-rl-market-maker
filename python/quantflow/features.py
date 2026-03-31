"""
LOB microstructure feature engineering.

All feature functions are pure (no side effects) and work directly on the
types produced by the Rust extension:

- ``snapshot`` — ``pyarrow.RecordBatch`` from ``PyOrderBook.snapshot(n)``
  columns: bid_price (f64), bid_qty (u64), ask_price (f64), ask_qty (u64)
  rows:    best level first, NaN price = empty level

- ``trades`` — ``list[dict]`` from ``HawkesSimulator.step()["trades"]``
  keys:    price (float), qty (int), maker_id (int), taker_id (int)
  optionally extended with ``sim_time`` (float) and ``is_buy`` (bool)

RunningNormalizer
-----------------
Online mean/variance via Welford (1962).  Each feature dimension is tracked
independently.  ``normalize(x)`` returns (x - μ) / (σ + ε) element-wise.
Parameters can be serialised to/from JSON for inference deployment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

Snapshot = pa.RecordBatch
Trade    = dict          # {"price": float, "qty": int, ...}


# ─────────────────────────────────────────────────────────────────────────────
# LOB snapshot features
# ─────────────────────────────────────────────────────────────────────────────

def volume_imbalance(snapshot: Snapshot, level: int = 1) -> float:
    """
    Signed volume imbalance at the first *level* price levels.

    .. math::
        \\text{VI} = \\frac{V_{\\text{bid}} - V_{\\text{ask}}}
                          {V_{\\text{bid}} + V_{\\text{ask}} + \\varepsilon}

    Returns a value in (−1, +1).  +1 = only bid volume; −1 = only ask volume.

    Parameters
    ----------
    snapshot : pyarrow.RecordBatch
        Output of ``PyOrderBook.snapshot(n)``.
    level : int
        Number of price levels to aggregate (1 = touch only).
    """
    n    = min(level, snapshot.num_rows)
    bids = snapshot["bid_qty"][:n].to_pylist()
    asks = snapshot["ask_qty"][:n].to_pylist()
    v_bid = sum(q for q in bids if q)
    v_ask = sum(q for q in asks if q)
    denom = v_bid + v_ask
    return (v_bid - v_ask) / denom if denom > 0 else 0.0


def weighted_mid_price(snapshot: Snapshot) -> float:
    """
    Microprice — liquidity-weighted mid adjusted for order book pressure.

    .. math::
        P^* = \\frac{\\sum_i P^a_i V^b_i + P^b_i V^a_i}
                   {\\sum_i V^b_i + V^a_i}

    Weighting ask prices by bid volume (and vice versa) shifts the estimate
    toward the side with greater execution pressure (Stoikov, 2018).
    """
    bid_prices = np.array(_to_float_list(snapshot["bid_price"]), dtype=np.float64)
    bid_qtys   = np.array(snapshot["bid_qty"].to_pylist(),   dtype=np.float64)
    ask_prices = np.array(_to_float_list(snapshot["ask_price"]), dtype=np.float64)
    ask_qtys   = np.array(snapshot["ask_qty"].to_pylist(),   dtype=np.float64)

    # Mask empty levels (price = NaN or qty = 0)
    valid = (
        np.isfinite(bid_prices) & (bid_qtys > 0) &
        np.isfinite(ask_prices) & (ask_qtys > 0)
    )
    if not np.any(valid):
        # Fall back to simple mid of best available prices
        bp = bid_prices[np.isfinite(bid_prices)]
        ap = ask_prices[np.isfinite(ask_prices)]
        if len(bp) and len(ap):
            return float((bp[0] + ap[0]) / 2)
        return 0.0

    bp, bq, ap, aq = bid_prices[valid], bid_qtys[valid], ask_prices[valid], ask_qtys[valid]
    numerator   = np.sum(ap * bq + bp * aq)
    denominator = np.sum(bq + aq)
    return float(numerator / denominator)


def depth_ratio(snapshot: Snapshot, levels: int = 5) -> float:
    """
    Ratio of total bid depth to total ask depth at the top *levels* levels.

    .. math::
        D = \\frac{\\sum_{i=1}^{L} V^b_i}{\\sum_{i=1}^{L} V^a_i + \\varepsilon}

    Values > 1 indicate a heavier bid side; < 1 a heavier ask side.
    """
    n    = min(levels, snapshot.num_rows)
    bids = sum(q for q in snapshot["bid_qty"][:n].to_pylist() if q)
    asks = sum(q for q in snapshot["ask_qty"][:n].to_pylist() if q)
    return bids / (asks + 1e-9)


def spread_bps(snapshot: Snapshot) -> float:
    """
    Bid-ask spread in basis points relative to mid.

    .. math::
        S_{\\text{bps}} = \\frac{P^a_0 - P^b_0}{\\text{mid}} \\times 10{,}000

    Returns 0.0 if the book is empty on either side.
    """
    bid_p = snapshot["bid_price"][0].as_py()
    ask_p = snapshot["ask_price"][0].as_py()
    if bid_p is None or ask_p is None:
        return 0.0
    if not (math.isfinite(bid_p) and math.isfinite(ask_p)):
        return 0.0
    mid = (bid_p + ask_p) / 2.0
    return (ask_p - bid_p) / (mid + 1e-9) * 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Trade-flow features
# ─────────────────────────────────────────────────────────────────────────────

def order_flow_imbalance(
    trades:  Sequence[Trade],
    window:  int   = 100,
    mid:     float | None = None,
) -> float:
    """
    Signed order-flow imbalance over the last *window* trades.

    .. math::
        \\text{OFI} = \\frac{V_{\\text{buy}} - V_{\\text{sell}}}
                           {V_{\\text{buy}} + V_{\\text{sell}} + \\varepsilon}

    Trade direction is determined by the ``is_buy`` key when present.
    Without it, trades at or above *mid* are classified as buys (tick rule).
    If *mid* is also absent, the median price of the window is used as proxy.

    Returns a value in (−1, +1).
    """
    recent = list(trades)[-window:]
    if not recent:
        return 0.0

    if mid is None:
        prices = [t["price"] for t in recent]
        mid = float(np.median(prices))

    buy_vol = sell_vol = 0.0
    for t in recent:
        qty = float(t.get("qty", 0))
        if "is_buy" in t:
            is_buy: bool = t["is_buy"]
        else:
            is_buy = t["price"] >= mid

        if is_buy:
            buy_vol += qty
        else:
            sell_vol += qty

    denom = buy_vol + sell_vol
    return (buy_vol - sell_vol) / denom if denom > 0 else 0.0


def realized_volatility(
    trades:     Sequence[Trade],
    window:     int   = 500,
    ann_factor: float = 23_400.0,
) -> float:
    """
    Realized volatility from log-returns of consecutive trade prices.

    .. math::
        \\hat{\\sigma} = \\sqrt{\\sum_{k} r_k^2} \\cdot \\sqrt{N_{\\text{day}}}

    Uses quadratic variation (Andersen & Benzoni, 2008): variance is estimated
    as the sum of squared log-returns (not sample variance), then annualised to
    one trading day by multiplying by ``sqrt(ann_factor / n_window)``.

    Parameters
    ----------
    trades : sequence of trade dicts
        Most-recent trades used; only the last *window* are considered.
    window : int
        Number of trades to use (at most).
    ann_factor : float
        Number of trades in a full trading day for annualisation.
        Default 23 400 ≈ one trade per second over 6.5 h.
    """
    recent = list(trades)[-window:]
    if len(recent) < 2:
        return 0.0

    prices = np.array([t["price"] for t in recent], dtype=np.float64)
    pos    = prices > 0
    if pos.sum() < 2:
        return 0.0
    prices  = prices[pos]
    returns = np.log(prices[1:] / prices[:-1])
    qv      = float(np.sum(returns ** 2))
    n       = len(returns)
    return math.sqrt(qv * ann_factor / n)


def trade_arrival_rate(
    trades:      Sequence[Trade],
    timestamps:  Sequence[float],
    window_s:    float = 60.0,
) -> float:
    """
    Number of trades per second within the trailing *window_s* second window.

    .. math::
        \\lambda = \\frac{N_{[t-W, t]}}{W}

    Parameters
    ----------
    trades : sequence of trade dicts
        Must be aligned 1-to-1 with *timestamps*.
    timestamps : sequence of floats
        Simulation time in seconds for each trade.
    window_s : float
        Length of the trailing window in seconds.
    """
    if not timestamps:
        return 0.0
    t_end   = float(timestamps[-1])
    t_start = t_end - window_s
    count   = sum(1 for ts in timestamps if ts >= t_start)
    return count / max(window_s, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all features into a flat numpy array
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "volume_imbalance_1",
    "volume_imbalance_5",
    "weighted_mid_price",
    "depth_ratio_5",
    "spread_bps",
    "order_flow_imbalance",
    "realized_volatility",
    "trade_arrival_rate",
]


def compute_all(
    snapshot:    Snapshot,
    trades:      Sequence[Trade],
    timestamps:  Sequence[float] | None = None,
    mid:         float | None = None,
) -> np.ndarray:
    """
    Compute all eight features and return a float32 vector of shape (8,).

    The order matches ``FEATURE_NAMES``.

    Parameters
    ----------
    snapshot : pyarrow.RecordBatch
        Current order book snapshot.
    trades : sequence of trade dicts
        Recent trades (e.g. last N events aggregated).
    timestamps : sequence of floats, optional
        Sim-time per trade for arrival-rate estimation.
        If omitted, arrival rate is returned as 0.
    mid : float, optional
        Current mid-price.  Inferred from snapshot if not given.
    """
    if mid is None:
        mid = weighted_mid_price(snapshot)

    rate = (
        trade_arrival_rate(trades, timestamps, window_s=60.0)
        if timestamps
        else 0.0
    )

    return np.array([
        volume_imbalance(snapshot, level=1),
        volume_imbalance(snapshot, level=5),
        mid,
        depth_ratio(snapshot, levels=5),
        spread_bps(snapshot),
        order_flow_imbalance(trades, window=100, mid=mid),
        realized_volatility(trades, window=500),
        rate,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Running normalizer (Welford, 1962)
# ─────────────────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """
    Online mean and variance tracking using Welford's one-pass algorithm.

    Each call to :meth:`update` incorporates one observation vector.
    :meth:`normalize` maps a vector to approximate N(0, 1) using the
    running statistics.

    The algorithm is numerically stable (Knuth, TAOCP Vol. 2 §4.2.2).

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature vectors.
    eps : float
        Added to the denominator to avoid division by zero.  Acts as a
        minimum standard deviation — features that never vary will be
        mapped to 0.
    """

    def __init__(self, n_features: int, eps: float = 1e-6) -> None:
        self.n_features = n_features
        self.eps        = eps
        self._count  = 0
        self._mean   = np.zeros(n_features, dtype=np.float64)
        self._m2     = np.zeros(n_features, dtype=np.float64)  # Σ(x − mean)²

    # ── online update ─────────────────────────────────────────────────────────

    def update(self, x: np.ndarray) -> None:
        """
        Incorporate one observation *x* of shape (n_features,).

        Uses Welford's recurrence:
            δ₁ = x − mean_{n−1}
            mean_n = mean_{n−1} + δ₁ / n
            δ₂ = x − mean_n
            M2_n = M2_{n−1} + δ₁ · δ₂
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_features,):
            raise ValueError(
                f"Expected shape ({self.n_features},), got {x.shape}"
            )
        self._count += 1
        delta  = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2   += delta * delta2

    def update_batch(self, xs: np.ndarray) -> None:
        """Update with a 2-D array of shape (N, n_features)."""
        for row in xs:
            self.update(row)

    # ── statistics ───────────────────────────────────────────────────────────

    @property
    def mean(self) -> np.ndarray:
        """Current running mean, shape (n_features,)."""
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """
        Population variance (not sample variance).

        Returns zero for dimensions with fewer than 2 observations.
        """
        if self._count < 2:
            return np.zeros(self.n_features, dtype=np.float64)
        return self._m2 / self._count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation (square root of population variance)."""
        return np.sqrt(self.var)

    # ── normalization ─────────────────────────────────────────────────────────

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize *x* using the running statistics.

        .. math::
            z = \\frac{x - \\mu}{\\sigma + \\varepsilon}

        Returns float32 array of the same shape as *x*.
        """
        x = np.asarray(x, dtype=np.float64)
        z = (x - self._mean) / (self.std + self.eps)
        return z.astype(np.float32)

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """Convenience: update statistics then return the normalized value."""
        self.update(x)
        return self.normalize(x)

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Write normalization parameters to a JSON file.

        Saved keys: ``n_features``, ``eps``, ``count``, ``mean``, ``m2``.
        """
        payload = {
            "n_features": self.n_features,
            "eps":        self.eps,
            "count":      self._count,
            "mean":       self._mean.tolist(),
            "m2":         self._m2.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RunningNormalizer":
        """
        Restore a normalizer from a previously saved JSON file.

        The loaded instance continues accumulating statistics from where
        it was saved — it is not frozen.
        """
        payload = json.loads(Path(path).read_text())
        inst = cls(n_features=payload["n_features"], eps=payload["eps"])
        inst._count = payload["count"]
        inst._mean  = np.array(payload["mean"],  dtype=np.float64)
        inst._m2    = np.array(payload["m2"],    dtype=np.float64)
        return inst

    def __repr__(self) -> str:
        return (
            f"RunningNormalizer(n_features={self.n_features}, "
            f"count={self._count}, eps={self.eps})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float_list(col: pa.Array) -> list[float]:
    """Convert a pyarrow array to a list of floats, replacing null/NaN with 0."""
    result = col.to_pylist()
    return [v if (v is not None and math.isfinite(v)) else 0.0 for v in result]
