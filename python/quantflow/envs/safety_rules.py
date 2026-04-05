"""
Rule-based safety overrides for the market-making agent.

These rules post-process AS quotes and can:
  - Suppress one or both sides (returns None for that side)
  - Widen the spread via a vol-regime multiplier
  - Force an aggressive crossing quote for inventory liquidation

Rules are pure functions — no state, no side effects.  State (vol_ema,
last_quote_mid) is owned by the caller and passed in on every call.

Rule application order (highest priority first):
  1. Vol-regime spread widening  — widens half-spread before limit checks
  2. Quote pull on mid move      — suppresses both sides if mid jumped
  3. Inventory hard limit        — overrides both sides with aggressive dump
  4. Inventory soft limit        — suppresses the inventory-increasing side
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RulesTriggered:
    vol_regime:     bool = False
    quote_pull:     bool = False
    inventory_hard: bool = False
    inventory_soft: bool = False


def apply_safety_rules(
    bid_p: float,
    ask_p: float,
    mid: float,
    inventory: int,
    current_vol: float,
    vol_ema: float,
    last_quote_mid: float | None,
    *,
    inventory_soft_limit: int,
    inventory_hard_limit: int,
    tick_size: float,
    vol_spread_threshold: float,
    vol_spread_multiplier: float,
) -> tuple[float | None, float | None, RulesTriggered]:
    """
    Apply rule-based safety overrides to AS quotes.

    Parameters
    ----------
    bid_p, ask_p : float
        Agent-computed bid/ask prices.
    mid : float
        Current mid-price.
    inventory : int
        Signed inventory (positive = long).
    current_vol : float
        Realized vol for the current step.
    vol_ema : float
        Long-run vol EMA (updated by caller; 0.0 at episode start means
        vol-regime rule is disabled until the EMA has warmed up).
    last_quote_mid : float | None
        Mid-price at the time of the last quote placement; None = first step.
    inventory_soft_limit : int
        One-sided quoting threshold (default 30).
    inventory_hard_limit : int
        Forced liquidation threshold (default 40).
    tick_size : float
        Price grid tick (default 0.01).
    vol_spread_threshold : float
        Vol multiple above EMA that triggers spread widening (default 2.0).
    vol_spread_multiplier : float
        Half-spread scale factor when vol regime fires (default 2.0).

    Returns
    -------
    (bid_p | None, ask_p | None, RulesTriggered)
        None for a side means: do not place that quote.
    """
    rules = RulesTriggered()

    # ── 1. Vol regime: widen half-spread ────────────────────────────────────
    # Only fires once the EMA has accumulated signal (vol_ema > 0).
    if vol_ema > 0.0 and current_vol > vol_spread_threshold * vol_ema:
        rules.vol_regime = True
        half = (ask_p - bid_p) / 2.0
        bid_p = mid - half * vol_spread_multiplier
        ask_p = mid + half * vol_spread_multiplier

    # ── 2. Quote pull: mid moved > 1.5 ticks since last placement ───────────
    # Suppresses both sides so stale quotes don't rest against an adverse move.
    if last_quote_mid is not None:
        if abs(mid - last_quote_mid) > 1.5 * tick_size:
            rules.quote_pull = True
            return None, None, rules

    # ── 3. Inventory hard limit: forced liquidation (Avellaneda & Stoikov §3) ─
    # Place an aggressive order one tick through mid to guarantee execution.
    if abs(inventory) >= inventory_hard_limit:
        rules.inventory_hard = True
        if inventory > 0:
            # Long: aggressive ask — price below mid crosses resting bids
            return None, round(mid - tick_size, 8), rules
        else:
            # Short: aggressive bid — price above mid crosses resting asks
            return round(mid + tick_size, 8), None, rules

    # ── 4. Inventory soft limit: one-sided quoting ───────────────────────────
    # Suppress the side that would increase inventory exposure.
    if abs(inventory) >= inventory_soft_limit:
        rules.inventory_soft = True
        if inventory > 0:
            return None, ask_p, rules   # long → only offer to sell
        else:
            return bid_p, None, rules   # short → only offer to buy

    return bid_p, ask_p, rules
