"""
Observation-space feature functions for the extended (v2) observation.

All functions are pure, operate on Python collections.deque / list inputs,
and return a single float already normalised to the range stated in the
docstring — no further normalisation layer required.

References
----------
- Cont, Kukanov & Stoikov (2013) — Order-flow imbalance as a price predictor
- Ogata (1981) — Hawkes process intensity estimation
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence


def compute_order_flow_imbalance(
    trade_sides:   deque | Sequence[int],
    trade_volumes: deque | Sequence[int],
    window:        int,
) -> float:
    """
    Signed buy/sell volume imbalance over the last *window* market trades.

    .. math::
        \\text{OFI} = \\frac{V_{\\text{buy}} - V_{\\text{sell}}}
                           {V_{\\text{buy}} + V_{\\text{sell}} + \\varepsilon}

    Parameters
    ----------
    trade_sides : deque of int
        +1 = buy-side event, −1 = sell-side event.
    trade_volumes : deque of int
        Trade quantity aligned with *trade_sides*.
    window : int
        Number of most-recent trades to consider.

    Returns
    -------
    float in [−1, +1].  0 = perfectly balanced.
    """
    sides = list(trade_sides)[-window:]
    vols  = list(trade_volumes)[-window:]
    if not sides:
        return 0.0
    buy_vol  = sum(v for s, v in zip(sides, vols) if s > 0)
    sell_vol = sum(v for s, v in zip(sides, vols) if s < 0)
    total    = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return (buy_vol - sell_vol) / total


def compute_trade_arrival_rate(
    trade_times:  deque | Sequence[float],
    current_time: float,
    window_sec:   float,
    expected_rate: float = 5.0,
) -> float:
    """
    Trades per second in the trailing *window_sec* of simulated time,
    normalised via tanh so the output stays in [0, 1).

    Parameters
    ----------
    trade_times : deque of float
        Simulation timestamp (seconds) for each recorded trade.
    current_time : float
        Current simulation time in seconds.
    window_sec : float
        Length of the trailing window in seconds.
    expected_rate : float
        Baseline trades-per-second used as the tanh scale parameter.
        Default 5.0 ≈ mid-session rate for the 12-d Hawkes calibration.

    Returns
    -------
    float in [0, 1).
    """
    if not trade_times:
        return 0.0
    threshold = current_time - window_sec
    count = sum(1 for t in trade_times if t >= threshold)
    rate  = count / max(window_sec, 1e-9)
    return float(math.tanh(rate / max(expected_rate, 1e-9)))


def compute_realized_vol(
    mid_prices:    deque | Sequence[float],
    window:        int,
    baseline_sigma: float = 0.001,
) -> float:
    """
    Realized volatility from log-returns of mid-price observations,
    normalised via tanh so the output stays in [0, 1).

    Parameters
    ----------
    mid_prices : deque of float
        Sequence of mid-price observations (most recent last).
    window : int
        Number of most-recent observations to use.
    baseline_sigma : float
        Scale for tanh normalisation; values near this map to ≈ 0.76.

    Returns
    -------
    float in [0, 1).
    """
    prices = list(mid_prices)[-window:]
    if len(prices) < 2:
        return 0.0
    log_rets = [
        math.log(b / a)
        for a, b in zip(prices[:-1], prices[1:])
        if a > 0.0 and b > 0.0
    ]
    if len(log_rets) < 2:
        return 0.0
    n    = len(log_rets)
    mean = sum(log_rets) / n
    var  = sum((r - mean) ** 2 for r in log_rets) / n
    sigma = math.sqrt(var)
    return float(math.tanh(sigma / max(baseline_sigma, 1e-12)))


def compute_spread_percentile(
    current_spread: float,
    spread_history: deque | Sequence[float],
) -> float:
    """
    Rank of the current spread within its recent history (empirical CDF).

    Parameters
    ----------
    current_spread : float
        The spread observed at the current step.
    spread_history : deque of float
        Up to the last 200 spread observations.

    Returns
    -------
    float in [0, 1].  0.5 = median historical spread.
    """
    history = list(spread_history)
    if not history:
        return 0.5
    below = sum(1 for s in history if s <= current_spread)
    return below / len(history)


def compute_agent_fill_imbalance(
    fill_history: deque | Sequence[tuple],
) -> float:
    """
    Signed imbalance between the agent's buy-fills and sell-fills.

    A strongly positive value means the agent is being filled predominantly
    on the bid — a potential adverse-selection signal.

    Parameters
    ----------
    fill_history : deque of (sim_time, side, price)
        side = +1 for buy fill, −1 for sell fill.
        At most the last 20 fills are considered.

    Returns
    -------
    float in [−1, +1].  0 = equal fills on both sides.
    """
    if not fill_history:
        return 0.0
    buys  = sum(1 for _, side, _ in fill_history if side > 0)
    sells = sum(1 for _, side, _ in fill_history if side < 0)
    total = buys + sells
    if total == 0:
        return 0.0
    return (buys - sells) / total


def compute_depth_ratio(
    bid_qtys: Sequence[int | None],
    ask_qtys: Sequence[int | None],
    levels:   int = 5,
) -> float:
    """
    Fraction of total visible liquidity sitting on the bid side.

    Parameters
    ----------
    bid_qtys, ask_qtys : sequences of int | None
        Quantity at each price level; None = empty level.
    levels : int
        Number of levels to aggregate.

    Returns
    -------
    float in [0, 1].  0.5 = balanced book.
    """
    bid_total = sum(q for q in list(bid_qtys)[:levels] if q)
    ask_total = sum(q for q in list(ask_qtys)[:levels] if q)
    total = bid_total + ask_total
    if total == 0:
        return 0.5
    return bid_total / total
