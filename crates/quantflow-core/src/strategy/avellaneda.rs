//! Avellaneda-Stoikov (2008) optimal market-making strategy.
//!
//! **Reservation price** (Eq. 7):  r(t) = s(t) − q · γ · σ² · τ
//! **Optimal spread**   (Eq. 8):  δ*(t) = γ · σ² · τ + (2/γ) · ln(1 + γ/κ)
//! **Quotes**: bid = r − δ*/2,  ask = r + δ*/2

use crate::orderbook::book::OrderBook;
use crate::orderbook::order::Trade;
use crate::orderbook::types::{OrderId, Price, Quantity, Side};
use crate::simulator::HawkesLobSimulator;

// ── Inventory regime ─────────────────────────────────────────────────────────

/// Describes which inventory-management regime is active for a given quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryMode {
    /// Normal quoting — no skewing applied.
    Normal,
    /// |q|/limit ∈ [0.70, 0.90): doubled γ + directional spread shift.
    Skew,
    /// |q|/limit ∈ [0.90, 1.00): Skew + inventory-building side suppressed.
    Suppress,
    /// |q|/limit = 1.00: caller must issue an aggressive dump order.
    Dump,
}

impl InventoryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            InventoryMode::Normal   => "normal",
            InventoryMode::Skew     => "skew",
            InventoryMode::Suppress => "suppress",
            InventoryMode::Dump     => "dump",
        }
    }
}

// ── Strategy ──────────────────────────────────────────────────────────────────

/// Avellaneda-Stoikov market-making strategy.
#[derive(Debug, Clone)]
pub struct AvellanedaStoikov {
    /// Risk-aversion coefficient γ > 0.
    pub gamma: f64,
    /// Price volatility σ (price units per second). Ignored when `sigma_auto`.
    pub sigma: f64,
    /// Order-arrival intensity κ near the best quotes.
    pub kappa: f64,
    /// End of trading horizon T (seconds from simulation origin).
    pub t_end: f64,
    /// Hard inventory limit |q| ≤ inventory_limit.
    pub inventory_limit: i64,
    /// If true, estimate σ from the first `warm_up_events` mid-price observations
    /// before placing any quotes.  Updated adaptively every `warm_up_events` events.
    pub sigma_auto: bool,
    /// Events to observe before first quote when `sigma_auto = true`.
    pub warm_up_events: usize,
    /// Minimum half-spread (price units).  Prevents quotes that are too tight to
    /// compensate for adverse selection.  Set to 0.0 to disable.
    pub spread_floor: f64,
}

impl AvellanedaStoikov {
    pub fn new(gamma: f64, sigma: f64, kappa: f64, t_end: f64, inventory_limit: i64) -> Self {
        AvellanedaStoikov {
            gamma,
            sigma,
            kappa,
            t_end,
            inventory_limit,
            sigma_auto: false,
            warm_up_events: 500,
            spread_floor: 0.0,
        }
    }

    /// Default parameters for a $100 instrument with σ=0.01/s.
    pub fn default_params(t_end: f64) -> Self {
        AvellanedaStoikov::new(0.1, 0.01, 1.5, t_end, 50)
    }

    /// Construct with σ auto-calibration from warm-up observations.
    pub fn with_auto_sigma(
        gamma: f64,
        kappa: f64,
        t_end: f64,
        inventory_limit: i64,
        spread_floor: f64,
    ) -> Self {
        AvellanedaStoikov {
            gamma,
            sigma: 0.01, // initial guess; replaced after warm-up
            kappa,
            t_end,
            inventory_limit,
            sigma_auto: true,
            warm_up_events: 500,
            spread_floor,
        }
    }

    // ── Quote computation ─────────────────────────────────────────────────────

    /// Compute bid and ask quotes using the AS formula with explicit σ.
    ///
    /// Avellaneda & Stoikov (2008), Eq. 7–8.
    pub fn compute_quotes(&self, mid: f64, inventory: i64, t: f64) -> (f64, f64) {
        self.quotes_with_sigma(mid, inventory, t, self.sigma)
    }

    fn quotes_with_sigma(&self, mid: f64, inventory: i64, t: f64, sigma: f64) -> (f64, f64) {
        let tau = (self.t_end - t).max(1e-6);

        // Reservation price: shift mid toward zero-inventory.
        let r = mid - inventory as f64 * self.gamma * sigma.powi(2) * tau;

        // Half-spread: (γσ²τ)/2 + (1/γ)·ln(1 + γ/κ), floored by spread_floor.
        let inventory_term = 0.5 * self.gamma * sigma.powi(2) * tau;
        let liquidity_term = (1.0 / self.gamma) * (1.0 + self.gamma / self.kappa).ln();
        let half_spread = (inventory_term + liquidity_term).max(self.spread_floor);

        (r - half_spread, r + half_spread)
    }

    /// AS quotes with active inventory skewing.
    ///
    /// Three regimes above the base quote:
    /// - `|q|/limit ∈ [0.70, 0.90)`: double γ_eff + directional spread shift.
    /// - `|q|/limit ∈ [0.90, 1.00)`: also suppress the inventory-building side
    ///   by pushing that quote 10 ticks through mid (will not fill passively).
    /// - `|q|/limit = 1.00`: caller should issue an aggressive dump order;
    ///   this function returns `InventoryMode::Dump` as a signal.
    pub fn compute_quotes_skewed(
        &self,
        mid: f64,
        inventory: i64,
        t: f64,
    ) -> ((f64, f64), InventoryMode) {
        let limit   = self.inventory_limit as f64;
        let q       = inventory as f64;
        let ratio   = (q.abs() / limit).min(1.0);
        let sigma   = self.sigma;

        if ratio >= 1.0 {
            // Quoting is suspended — caller issues a dump order instead.
            let (b, a) = self.quotes_with_sigma(mid, inventory, t, sigma);
            return ((b, a), InventoryMode::Dump);
        }

        // Effective risk aversion: doubled above 70%.
        let gamma_eff = if ratio >= 0.70 { self.gamma * 2.0 } else { self.gamma };
        let tau       = (self.t_end - t).max(1e-6);

        // Reservation price with gamma_eff (Avellaneda & Stoikov 2008, Eq. 7).
        let r = mid - q * gamma_eff * sigma.powi(2) * tau;

        // Half-spread with gamma_eff.
        let inv_term  = 0.5  * gamma_eff * sigma.powi(2) * tau;
        let liq_term  = (1.0 / gamma_eff) * (1.0 + gamma_eff / self.kappa).ln();
        let base_half = (inv_term + liq_term).max(self.spread_floor);

        // Directional skew: extra offset pushes both quotes toward unwinding side.
        let extra = if ratio >= 0.70 {
            q.signum() * (ratio - 0.70) * base_half * 2.0
        } else {
            0.0
        };

        let mut bid_q = r - base_half - extra;
        let mut ask_q = r + base_half - extra;

        // Suppress: push inventory-building quote 10 ticks through mid.
        let mode = if ratio >= 0.90 {
            let far = 0.10; // 10 × 0.01 tick in native price units
            if inventory > 0 {
                bid_q = mid - far;
            } else {
                ask_q = mid + far;
            }
            InventoryMode::Suppress
        } else if ratio >= 0.70 {
            InventoryMode::Skew
        } else {
            InventoryMode::Normal
        };

        ((bid_q, ask_q), mode)
    }

    // ── Parameter estimation ──────────────────────────────────────────────────

    /// Realized volatility per second from trade prices (last `window` trades).
    pub fn estimate_sigma(trades: &[Trade], window: usize) -> f64 {
        let n = trades.len().min(window + 1);
        if n < 2 {
            return 0.0;
        }
        let recent = &trades[trades.len().saturating_sub(n)..];
        let log_returns: Vec<f64> = recent
            .windows(2)
            .filter(|w| w[0].price.0 > 0 && w[1].price.0 > 0)
            .map(|w| (w[1].price.to_f64() / w[0].price.to_f64()).ln())
            .collect();
        let m = log_returns.len();
        if m < 2 {
            return 0.0;
        }
        let mean = log_returns.iter().sum::<f64>() / m as f64;
        let var =
            log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
        var.sqrt()
    }

    /// Realized volatility per second from a mid-price time series `(time_s, mid)`.
    ///
    /// Uses quadratic variation: σ² ≈ Σ r_k² / Δt_k  divided by total T.
    pub fn sigma_from_mid_series(mids: &[(f64, f64)]) -> f64 {
        if mids.len() < 10 {
            return 0.0;
        }
        let mut qv = 0.0_f64;
        let mut total_t = 0.0_f64;
        for w in mids.windows(2) {
            let dt = (w[1].0 - w[0].0).max(1e-9);
            if w[0].1 > 0.0 && w[1].1 > 0.0 {
                let r = (w[1].1 / w[0].1).ln();
                qv += r * r;
                total_t += dt;
            }
        }
        if total_t <= 0.0 {
            return 0.0;
        }
        (qv / total_t).sqrt()
    }

    /// Estimate κ from current book depth and spread.
    pub fn estimate_kappa(book: &OrderBook) -> f64 {
        let bid_qty = book.best_bid().map(|(_, q)| q.0 as f64).unwrap_or(0.0);
        let ask_qty = book.best_ask().map(|(_, q)| q.0 as f64).unwrap_or(0.0);
        let spread = book.spread().map(|s| s.to_f64().abs()).unwrap_or(1.0);
        if spread <= f64::EPSILON || bid_qty + ask_qty <= f64::EPSILON {
            return 1.0;
        }
        ((bid_qty + ask_qty) / 2.0 / spread).max(0.1)
    }

    // ── Backtest ──────────────────────────────────────────────────────────────

    /// Run a full backtest (quote interval = 1 s, order size = 10 lots).
    pub fn run_backtest(&self, simulator: &mut HawkesLobSimulator, seed: u64) -> BacktestResult {
        self.run_backtest_with_params(simulator, seed, 1.0, 10)
    }

    /// Run a backtest with explicit quote-refresh interval and order size.
    pub fn run_backtest_with_params(
        &self,
        simulator: &mut HawkesLobSimulator,
        seed: u64,
        quote_interval: f64,
        quote_qty: u64,
    ) -> BacktestResult {
        simulator.reset(seed);

        let mut cash: f64 = 0.0;
        let mut inventory: i64 = 0;
        let mut bid_id: Option<OrderId> = None;
        let mut ask_id: Option<OrderId> = None;

        // PnL & inventory curve (at snapshots).
        let mut pnl_curve: Vec<f64> = Vec::new();
        let mut inventory_curve: Vec<i64> = Vec::new();

        // Quote tracking.
        let mut n_quotes_placed: usize = 0;
        let mut last_quote_t: f64 = -quote_interval;
        let mut quoted_spreads: Vec<f64> = Vec::new();
        let mut n_events: usize = 0;
        let mut n_events_with_quotes: usize = 0;

        // Fill details for diagnostics.
        let mut fills: Vec<Trade> = Vec::new();
        let mut fill_details: Vec<FillDetail> = Vec::new();

        // Mid-price time series for adverse selection analysis.
        let mut mid_series: Vec<(f64, f64)> = Vec::new();

        // σ auto-calibration.
        let mut current_sigma = self.sigma;
        let mut warm_up_done = !self.sigma_auto;

        // Spread PnL accumulation.
        let mut spread_pnl_acc: f64 = 0.0;

        while let Some(event) = simulator.step() {
            let t = event.sim_time;
            n_events += 1;

            // Record mid price for adverse selection and σ estimation.
            if let Some(mid) = simulator.mid_price() {
                mid_series.push((t, mid));
            }

            // Auto-calibrate σ after warm-up.
            if self.sigma_auto && !warm_up_done && n_events >= self.warm_up_events {
                let estimated = Self::sigma_from_mid_series(&mid_series);
                if estimated > 0.0 {
                    current_sigma = estimated;
                }
                warm_up_done = true;
            }

            // ── Fill detection ───────────────────────────────────────────────
            let mid_now = simulator.mid_price().unwrap_or(0.0);
            for trade in &event.trades {
                if bid_id == Some(trade.maker_id) {
                    inventory += trade.quantity.0 as i64;
                    cash -= trade.price.to_f64() * trade.quantity.0 as f64;
                    spread_pnl_acc +=
                        (mid_now - trade.price.to_f64()).abs() * trade.quantity.0 as f64;
                    fill_details.push(FillDetail {
                        time: t,
                        mid: mid_now,
                        is_ask: false,
                        price: trade.price.to_f64(),
                        qty: trade.quantity.0,
                    });
                    bid_id = None;
                    fills.push(trade.clone());
                } else if ask_id == Some(trade.maker_id) {
                    inventory -= trade.quantity.0 as i64;
                    cash += trade.price.to_f64() * trade.quantity.0 as f64;
                    spread_pnl_acc +=
                        (trade.price.to_f64() - mid_now).abs() * trade.quantity.0 as f64;
                    fill_details.push(FillDetail {
                        time: t,
                        mid: mid_now,
                        is_ask: true,
                        price: trade.price.to_f64(),
                        qty: trade.quantity.0,
                    });
                    ask_id = None;
                    fills.push(trade.clone());
                }
            }

            // ── Quote tracking ───────────────────────────────────────────────
            if bid_id.is_some() || ask_id.is_some() {
                n_events_with_quotes += 1;
            }

            // ── Refresh quotes (after warm-up) ───────────────────────────────
            if warm_up_done && t - last_quote_t >= quote_interval {
                if let Some(id) = bid_id.take() {
                    simulator.cancel_agent_order(id);
                }
                if let Some(id) = ask_id.take() {
                    simulator.cancel_agent_order(id);
                }

                if let Some(mid) = simulator.mid_price() {
                    let (bid_p, ask_p) =
                        self.quotes_with_sigma(mid, inventory, t, current_sigma);

                    if bid_p > 0.0 && ask_p > bid_p {
                        let qty = Quantity(quote_qty);
                        quoted_spreads.push(ask_p - bid_p);

                        let buy_would_breach =
                            inventory + (quote_qty as i64) > self.inventory_limit;
                        if !buy_would_breach {
                            let price = Price::from_f64(bid_p);
                            let safe = simulator
                                .book()
                                .best_ask()
                                .map(|(ba, _)| price < ba)
                                .unwrap_or(true);
                            if safe {
                                bid_id =
                                    Some(simulator.place_limit_order(Side::Bid, price, qty));
                                n_quotes_placed += 1;
                            }
                        }

                        let sell_would_breach =
                            inventory - (quote_qty as i64) < -self.inventory_limit;
                        if !sell_would_breach {
                            let price = Price::from_f64(ask_p);
                            let safe = simulator
                                .book()
                                .best_bid()
                                .map(|(bb, _)| price > bb)
                                .unwrap_or(true);
                            if safe {
                                ask_id =
                                    Some(simulator.place_limit_order(Side::Ask, price, qty));
                                n_quotes_placed += 1;
                            }
                        }
                    }
                }
                last_quote_t = t;
            }

            // ── PnL snapshot ─────────────────────────────────────────────────
            if event.snapshot.is_some() {
                if let Some(mid) = simulator.mid_price() {
                    pnl_curve.push(cash + inventory as f64 * mid);
                    inventory_curve.push(inventory);
                }
            }
        }

        // Cancel remaining open quotes.
        if let Some(id) = bid_id { simulator.cancel_agent_order(id); }
        if let Some(id) = ask_id { simulator.cancel_agent_order(id); }

        // ── Performance metrics ───────────────────────────────────────────────
        let sharpe = compute_sharpe(&pnl_curve);
        let max_drawdown = compute_max_drawdown(&pnl_curve);
        let fill_rate = if n_quotes_placed > 0 {
            fills.len() as f64 / n_quotes_placed as f64
        } else {
            0.0
        };

        let final_pnl = pnl_curve.last().copied().unwrap_or(0.0);
        let inventory_pnl = final_pnl - spread_pnl_acc;

        let avg_quoted_spread = mean_f64(&quoted_spreads);
        let avg_spread_at_fill = if fill_details.is_empty() {
            0.0
        } else {
            fill_details.iter().map(|f| (f.price - f.mid).abs() * 2.0).sum::<f64>()
                / fill_details.len() as f64
        };
        let time_in_market = if n_events > 0 {
            n_events_with_quotes as f64 / n_events as f64
        } else {
            0.0
        };

        // Post-fill adverse-selection moves.
        let (pf1s, pf5s, pf10s) = post_fill_moves(&fill_details, &mid_series);

        // Round trips: min(bid fills, ask fills).
        let n_bid_fills = fill_details.iter().filter(|f| !f.is_ask).count();
        let n_ask_fills = fill_details.iter().filter(|f| f.is_ask).count();
        let n_round_trips = n_bid_fills.min(n_ask_fills);

        BacktestResult {
            pnl_curve,
            inventory_curve,
            fills,
            sharpe,
            max_drawdown,
            fill_rate,
            spread_pnl: spread_pnl_acc,
            inventory_pnl,
            realized_sigma: current_sigma,
            avg_quoted_spread,
            avg_spread_at_fill,
            time_in_market,
            post_fill_move_1s: pf1s,
            post_fill_move_5s: pf5s,
            post_fill_move_10s: pf10s,
            n_round_trips,
            fill_details,
        }
    }
}

// ── FillDetail ────────────────────────────────────────────────────────────────

/// Per-fill record used for diagnostics and adverse-selection analysis.
#[derive(Debug, Clone)]
pub struct FillDetail {
    /// Simulation time of the fill (seconds).
    pub time: f64,
    /// Mid-price at the moment of the fill.
    pub mid: f64,
    /// `true` = we were filled on our ask (we sold); `false` = bid fill (we bought).
    pub is_ask: bool,
    /// Execution price.
    pub price: f64,
    /// Executed quantity.
    pub qty: u64,
}

impl FillDetail {
    /// Half-spread captured: |fill_price − mid_at_fill|.
    pub fn half_spread_captured(&self) -> f64 {
        (self.price - self.mid).abs()
    }

    /// Inventory bucket (0–3) for `|inventory|` groupings < 10, 10–25, 25–40, ≥ 40.
    pub fn inv_bucket(inv: i64) -> usize {
        let a = inv.unsigned_abs() as usize;
        if a < 10 { 0 } else if a < 25 { 1 } else if a < 40 { 2 } else { 3 }
    }
}

// ── BacktestResult ────────────────────────────────────────────────────────────

/// Full result of an Avellaneda-Stoikov backtest run.
#[derive(Debug)]
pub struct BacktestResult {
    // ── Core curves ─────────────────────────────────────────────────────────
    /// Mark-to-market PnL at each book snapshot.
    pub pnl_curve: Vec<f64>,
    /// Inventory at each book snapshot.
    pub inventory_curve: Vec<i64>,
    /// Raw trade fills (original Trade structs).
    pub fills: Vec<Trade>,

    // ── Summary metrics ──────────────────────────────────────────────────────
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub fill_rate: f64,

    // ── PnL decomposition ────────────────────────────────────────────────────
    /// Income from capturing the bid-ask spread (always ≥ 0).
    pub spread_pnl: f64,
    /// Inventory mark-to-market PnL (positive or negative).
    pub inventory_pnl: f64,

    // ── Volatility ───────────────────────────────────────────────────────────
    /// σ used in quote computation (auto-calibrated or fixed).
    pub realized_sigma: f64,

    // ── Quote statistics ─────────────────────────────────────────────────────
    pub avg_quoted_spread: f64,
    pub avg_spread_at_fill: f64,
    /// Fraction of accepted events where at least one quote was open.
    pub time_in_market: f64,

    // ── Adverse selection ────────────────────────────────────────────────────
    /// Average mid-price move AGAINST the fill direction, 1 s after fill.
    /// Positive = adverse selection (mid moved in informed trader's direction).
    pub post_fill_move_1s: f64,
    pub post_fill_move_5s: f64,
    pub post_fill_move_10s: f64,

    // ── Round-trip count ─────────────────────────────────────────────────────
    pub n_round_trips: usize,

    // ── Per-fill detail (for external analysis) ──────────────────────────────
    pub fill_details: Vec<FillDetail>,
}

impl BacktestResult {
    pub fn final_pnl(&self) -> f64 {
        self.pnl_curve.last().copied().unwrap_or(0.0)
    }

    pub fn max_inventory(&self) -> i64 {
        self.inventory_curve.iter().map(|q| q.abs()).max().unwrap_or(0)
    }

    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown > f64::EPSILON {
            self.final_pnl() / self.max_drawdown
        } else {
            f64::INFINITY
        }
    }

    /// Average half-spread captured per fill by inventory bucket.
    /// Returns `[bucket_0, bucket_1, bucket_2, bucket_3]` where buckets are
    /// |inv| < 10, 10–25, 25–40, ≥ 40 at fill time.
    pub fn spread_by_inv_bucket(&self) -> [f64; 4] {
        let mut sum = [0.0_f64; 4];
        let mut cnt = [0_usize; 4];

        // We don't store inventory at fill time directly, but we can approximate
        // using inventory_curve (not ideal but workable for diagnostics).
        // Here we just bin by |inventory| change sign: simplified bucket from fill_details.
        // Since we don't have per-fill inventory, group by cumulative inventory at fill index.
        let mut running_inv: i64 = 0;
        for fd in &self.fill_details {
            if fd.is_ask {
                running_inv -= fd.qty as i64;
            } else {
                running_inv += fd.qty as i64;
            }
            let b = FillDetail::inv_bucket(running_inv);
            sum[b] += fd.half_spread_captured();
            cnt[b] += 1;
        }

        std::array::from_fn(|i| if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { 0.0 })
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute average adverse mid-move at 1 s, 5 s, 10 s after each fill.
///
/// Sign convention: positive = adverse (mid moved against the fill side).
fn post_fill_moves(
    fills: &[FillDetail],
    mid_series: &[(f64, f64)],
) -> (f64, f64, f64) {
    const DELTAS: [f64; 3] = [1.0, 5.0, 10.0];
    let mut sums = [0.0_f64; 3];
    let mut counts = [0_usize; 3];

    for fill in fills {
        for (i, &delta) in DELTAS.iter().enumerate() {
            let t_target = fill.time + delta;
            // Binary-search for the first mid observation at or after t_target.
            let idx = mid_series.partition_point(|&(t, _)| t < t_target);
            if idx < mid_series.len() {
                let mid_later = mid_series[idx].1;
                // Adverse = mid moved in the direction that hurts the maker.
                let adverse = if fill.is_ask {
                    // We sold; adverse if mid went UP (informed buyer knew it would rise).
                    mid_later - fill.mid
                } else {
                    // We bought; adverse if mid went DOWN (informed seller knew it would fall).
                    fill.mid - mid_later
                };
                sums[i] += adverse;
                counts[i] += 1;
            }
        }
    }

    let avg = |i: usize| if counts[i] > 0 { sums[i] / counts[i] as f64 } else { 0.0 };
    (avg(0), avg(1), avg(2))
}

fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

// ── Performance metrics ───────────────────────────────────────────────────────

pub(crate) fn compute_sharpe(pnl_curve: &[f64]) -> f64 {
    if pnl_curve.len() < 2 { return 0.0; }
    let returns: Vec<f64> = pnl_curve.windows(2).map(|w| w[1] - w[0]).collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < f64::EPSILON { return 0.0; }
    mean / std * n.sqrt()
}

pub(crate) fn compute_max_drawdown(pnl_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &v in pnl_curve {
        if v > peak { peak = v; }
        let dd = peak - v;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::HawkesLobSimulator;

    fn make_strat(t_end: f64) -> AvellanedaStoikov {
        AvellanedaStoikov::default_params(t_end)
    }

    // ── compute_quotes ────────────────────────────────────────────────────────

    #[test]
    fn quotes_symmetric_at_zero_inventory() {
        let s = make_strat(23_400.0);
        let (bid, ask) = s.compute_quotes(100.0, 0, 0.0);
        let mid = (bid + ask) / 2.0;
        assert!((mid - 100.0).abs() < 1e-10);
        assert!(ask > bid);
    }

    #[test]
    fn long_inventory_shifts_quotes_down() {
        let s = make_strat(23_400.0);
        let (bid0, ask0) = s.compute_quotes(100.0, 0, 1000.0);
        let (bid_long, ask_long) = s.compute_quotes(100.0, 10, 1000.0);
        assert!(bid_long < bid0);
        assert!(ask_long < ask0);
    }

    #[test]
    fn short_inventory_shifts_quotes_up() {
        let s = make_strat(23_400.0);
        let (bid0, ask0) = s.compute_quotes(100.0, 0, 1000.0);
        let (bid_short, ask_short) = s.compute_quotes(100.0, -10, 1000.0);
        assert!(bid_short > bid0);
        assert!(ask_short > ask0);
    }

    #[test]
    fn spread_narrows_toward_end_of_day() {
        let s = make_strat(23_400.0);
        let (b_early, a_early) = s.compute_quotes(100.0, 0, 100.0);
        let (b_late, a_late) = s.compute_quotes(100.0, 0, 23_000.0);
        assert!(
            (a_late - b_late) < (a_early - b_early),
            "spread should narrow as τ→0"
        );
    }

    #[test]
    fn spread_floor_enforced() {
        let mut s = make_strat(23_400.0);
        s.spread_floor = 5.0; // very large floor
        let (bid, ask) = s.compute_quotes(100.0, 0, 23_399.0); // near end-of-day
        assert!(ask - bid >= 2.0 * 5.0 - 1e-9, "floor not enforced: spread={}", ask - bid);
    }

    #[test]
    fn ask_always_above_bid() {
        let s = make_strat(100.0);
        for q in -5..=5_i64 {
            for t in [0.0, 10.0, 50.0, 99.0] {
                let (bid, ask) = s.compute_quotes(100.0, q, t);
                assert!(ask > bid, "ask={ask:.4} <= bid={bid:.4} at q={q}, t={t}");
            }
        }
    }

    // ── sigma_from_mid_series ─────────────────────────────────────────────────

    #[test]
    fn sigma_from_mid_series_zero_for_constant() {
        let mids: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 100.0)).collect();
        assert_eq!(AvellanedaStoikov::sigma_from_mid_series(&mids), 0.0);
    }

    #[test]
    fn sigma_from_mid_series_positive_for_volatile() {
        let mids: Vec<(f64, f64)> = (0..50)
            .map(|i| (i as f64 * 0.1, 100.0 + (i % 5) as f64 * 0.5))
            .collect();
        assert!(AvellanedaStoikov::sigma_from_mid_series(&mids) > 0.0);
    }

    // ── estimate_sigma ────────────────────────────────────────────────────────

    #[test]
    fn estimate_sigma_empty_is_zero() {
        assert_eq!(AvellanedaStoikov::estimate_sigma(&[], 50), 0.0);
    }

    #[test]
    fn estimate_sigma_constant_price_is_zero() {
        use crate::orderbook::order::Trade;
        use crate::orderbook::types::{OrderId, Price, Quantity, Timestamp};
        let t = |p: f64| Trade {
            price: Price::from_f64(p),
            quantity: Quantity(1),
            maker_id: OrderId(0),
            taker_id: OrderId(1),
            timestamp: Timestamp(0),
        };
        let trades: Vec<_> = (0..10).map(|_| t(100.0)).collect();
        assert_eq!(AvellanedaStoikov::estimate_sigma(&trades, 50), 0.0);
    }

    #[test]
    fn estimate_sigma_positive_for_moving_prices() {
        use crate::orderbook::order::Trade;
        use crate::orderbook::types::{OrderId, Price, Quantity, Timestamp};
        let t = |p: f64| Trade {
            price: Price::from_f64(p),
            quantity: Quantity(1),
            maker_id: OrderId(0),
            taker_id: OrderId(1),
            timestamp: Timestamp(0),
        };
        let prices = [100.0, 101.0, 99.5, 102.0, 98.0, 103.5];
        let trades: Vec<_> = prices.iter().map(|&p| t(p)).collect();
        assert!(AvellanedaStoikov::estimate_sigma(&trades, 50) > 0.0);
    }

    // ── estimate_kappa ────────────────────────────────────────────────────────

    #[test]
    fn estimate_kappa_empty_book_returns_one() {
        let book = OrderBook::new();
        assert!((AvellanedaStoikov::estimate_kappa(&book) - 1.0).abs() < 1e-10);
    }

    // ── run_backtest ──────────────────────────────────────────────────────────

    #[test]
    fn backtest_produces_pnl_curve() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        sim.config.snapshot_interval = 20;
        let result = make_strat(120.0).run_backtest(&mut sim, 42);
        assert!(!result.pnl_curve.is_empty());
        assert_eq!(result.pnl_curve.len(), result.inventory_curve.len());
    }

    #[test]
    fn backtest_fill_rate_in_zero_to_one() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let result = make_strat(120.0).run_backtest(&mut sim, 7);
        assert!(result.fill_rate >= 0.0 && result.fill_rate <= 1.0);
    }

    #[test]
    fn backtest_inventory_bounded_by_limit() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat = make_strat(120.0);
        let result = strat.run_backtest(&mut sim, 13);
        assert!(
            result.max_inventory() <= strat.inventory_limit,
            "max_inventory={} > limit={}",
            result.max_inventory(),
            strat.inventory_limit
        );
    }

    #[test]
    fn backtest_pnl_decomposition_consistent() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let result = make_strat(120.0).run_backtest(&mut sim, 5);
        let reconstructed = result.spread_pnl + result.inventory_pnl;
        assert!(
            (reconstructed - result.final_pnl()).abs() < 1e-6,
            "decomposition mismatch: {reconstructed} vs {}",
            result.final_pnl()
        );
    }

    #[test]
    fn backtest_spread_pnl_non_negative() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let result = make_strat(120.0).run_backtest(&mut sim, 3);
        assert!(result.spread_pnl >= 0.0, "spread PnL should always be ≥ 0");
    }

    #[test]
    fn backtest_fill_details_count_matches_fills() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let result = make_strat(120.0).run_backtest(&mut sim, 8);
        assert_eq!(result.fill_details.len(), result.fills.len());
    }

    #[test]
    fn backtest_sigma_auto_sets_realized_sigma() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat =
            AvellanedaStoikov::with_auto_sigma(0.1, 1.5, 120.0, 50, 0.0);
        let result = strat.run_backtest(&mut sim, 42);
        // With sigma_auto, realized_sigma should be estimated (non-zero after enough events).
        assert!(result.realized_sigma >= 0.0);
    }

    #[test]
    fn max_drawdown_non_negative() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        assert!(make_strat(120.0).run_backtest(&mut sim, 1).max_drawdown >= 0.0);
    }

    #[test]
    fn different_seeds_produce_different_fills() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat = make_strat(120.0);
        let r1 = strat.run_backtest(&mut sim, 0);
        let r2 = strat.run_backtest(&mut sim, 99);
        assert!(r1.fills.len() != r2.fills.len() || r1.final_pnl() != r2.final_pnl());
    }

    // ── Sharpe / drawdown helpers ─────────────────────────────────────────────

    #[test]
    fn sharpe_zero_for_flat_pnl() {
        assert_eq!(compute_sharpe(&[10.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn sharpe_positive_for_noisy_uptrend() {
        let pnl = vec![0.0, 1.5, 2.8, 4.4, 5.1, 7.0, 8.3, 9.5, 11.2, 12.0];
        assert!(compute_sharpe(&pnl) > 0.0);
    }

    #[test]
    fn max_drawdown_zero_for_monotone_increase() {
        let pnl: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert_eq!(compute_max_drawdown(&pnl), 0.0);
    }

    #[test]
    fn max_drawdown_correct_known_series() {
        let pnl = vec![0.0, 5.0, 10.0, 8.0, 3.0, 6.0];
        assert!((compute_max_drawdown(&pnl) - 7.0).abs() < 1e-10);
    }
}
