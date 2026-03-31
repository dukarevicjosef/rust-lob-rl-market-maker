//! Avellaneda-Stoikov (2008) optimal market-making strategy.
//!
//! ## Model
//!
//! The agent continuously quotes a bid and an ask around a *reservation price*
//! that accounts for inventory risk.  Let:
//!
//!   s(t)  — mid-price at time t
//!   q     — current inventory (positive = long, negative = short)
//!   τ     — time remaining: T − t
//!   γ     — risk-aversion parameter
//!   σ     — price volatility (per second)
//!   κ     — order-arrival intensity near the quotes
//!
//! **Reservation price** (Avellaneda & Stoikov 2008, Eq. 7):
//!
//!   r(t) = s(t) − q · γ · σ² · τ
//!
//! **Optimal symmetric spread** (Eq. 8):
//!
//!   δ*(t) = γ · σ² · τ + (2/γ) · ln(1 + γ/κ)
//!
//! **Quotes**:
//!
//!   bid(t) = r(t) − δ*(t)/2
//!   ask(t) = r(t) + δ*(t)/2
//!
//! The inventory penalty γ·σ²·τ·q shifts the reservation price away from
//! mid: a long agent quotes lower to encourage sells, a short agent quotes
//! higher to encourage buys.

use crate::orderbook::book::OrderBook;
use crate::orderbook::order::Trade;
use crate::orderbook::types::{OrderId, Price, Quantity, Side};
use crate::simulator::HawkesLobSimulator;

// ── Strategy ──────────────────────────────────────────────────────────────────

/// Avellaneda-Stoikov market-making strategy.
#[derive(Debug, Clone)]
pub struct AvellanedaStoikov {
    /// Risk-aversion coefficient γ > 0.  Higher γ → wider spread, faster
    /// inventory mean-reversion.
    pub gamma: f64,
    /// Price volatility σ (in price units per second).
    pub sigma: f64,
    /// Order-arrival intensity κ near the best quotes.
    pub kappa: f64,
    /// End of trading horizon T (seconds from simulation origin).
    pub t_end: f64,
    /// Hard inventory limit |q| ≤ inventory_limit.  No new quotes are placed
    /// when this is breached.
    pub inventory_limit: i64,
}

impl AvellanedaStoikov {
    pub fn new(
        gamma: f64,
        sigma: f64,
        kappa: f64,
        t_end: f64,
        inventory_limit: i64,
    ) -> Self {
        AvellanedaStoikov { gamma, sigma, kappa, t_end, inventory_limit }
    }

    /// Conservative defaults appropriate for a $100 instrument with σ=0.01/s.
    pub fn default_params(t_end: f64) -> Self {
        AvellanedaStoikov {
            gamma: 0.1,
            sigma: 0.01,
            kappa: 1.5,
            t_end,
            inventory_limit: 50,
        }
    }

    /// Compute bid and ask quotes.
    ///
    /// Returns `(bid, ask)` in floating-point price units.
    /// The spread widens as τ grows (more risk when far from close) and
    /// shrinks as κ grows (more liquid book → tighter quotes profitable).
    ///
    /// Avellaneda & Stoikov (2008), Eq. 7–8.
    pub fn compute_quotes(&self, mid: f64, inventory: i64, t: f64) -> (f64, f64) {
        let tau = (self.t_end - t).max(1e-6);

        // Reservation price: shift mid toward zero inventory.
        let r = mid - inventory as f64 * self.gamma * self.sigma.powi(2) * tau;

        // Half-spread: δ*/2 = (γ·σ²·τ)/2 + (1/γ)·ln(1 + γ/κ)
        let inventory_term = 0.5 * self.gamma * self.sigma.powi(2) * tau;
        let liquidity_term = (1.0 / self.gamma) * (1.0 + self.gamma / self.kappa).ln();
        let half_spread = inventory_term + liquidity_term;

        (r - half_spread, r + half_spread)
    }

    /// Realized volatility from the last `window` trade-to-trade log-returns.
    ///
    /// Returns the sample standard deviation of log(p_k / p_{k-1}).
    /// Interpreting trades as arriving roughly 1 second apart gives a
    /// per-second volatility estimate.
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
        let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
        var.sqrt()
    }

    /// Estimate κ from the current book state.
    ///
    /// κ is proportional to depth at the touch divided by the spread:
    ///   κ ≈ (Q_bid + Q_ask) / (2 · spread)
    ///
    /// This is a heuristic; in production κ is calibrated from historical
    /// order-flow data.
    pub fn estimate_kappa(book: &OrderBook) -> f64 {
        let bid_qty = book.best_bid().map(|(_, q)| q.0 as f64).unwrap_or(0.0);
        let ask_qty = book.best_ask().map(|(_, q)| q.0 as f64).unwrap_or(0.0);
        let spread = book.spread().map(|s| s.to_f64().abs()).unwrap_or(1.0);
        if spread <= f64::EPSILON || bid_qty + ask_qty <= f64::EPSILON {
            return 1.0;
        }
        ((bid_qty + ask_qty) / 2.0 / spread).max(0.1)
    }

    /// Run a full backtest against the simulator.
    ///
    /// The agent quotes every `quote_interval` seconds of simulation time.
    /// Both quotes are placed with a fixed size of `quote_qty` lots.
    pub fn run_backtest(
        &self,
        simulator: &mut HawkesLobSimulator,
        seed: u64,
    ) -> BacktestResult {
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

        let mut pnl_curve: Vec<f64> = Vec::new();
        let mut inventory_curve: Vec<i64> = Vec::new();
        let mut fills: Vec<Trade> = Vec::new();
        let mut n_quotes_placed: usize = 0;
        let mut last_quote_t: f64 = -quote_interval; // force immediate first quote

        while let Some(event) = simulator.step() {
            let t = event.sim_time;

            // ── Check for fills ─────────────────────────────────────────────
            // Our resting orders appear as maker when market events cross them.
            for trade in &event.trades {
                if bid_id == Some(trade.maker_id) {
                    // Our bid was hit: we bought at trade.price.
                    inventory += trade.quantity.0 as i64;
                    cash -= trade.price.to_f64() * trade.quantity.0 as f64;
                    bid_id = None;
                    fills.push(trade.clone());
                } else if ask_id == Some(trade.maker_id) {
                    // Our ask was hit: we sold at trade.price.
                    inventory -= trade.quantity.0 as i64;
                    cash += trade.price.to_f64() * trade.quantity.0 as f64;
                    ask_id = None;
                    fills.push(trade.clone());
                }
            }

            // ── Refresh quotes ──────────────────────────────────────────────
            if t - last_quote_t >= quote_interval {
                // Cancel stale open quotes.
                if let Some(id) = bid_id.take() {
                    simulator.cancel_agent_order(id);
                }
                if let Some(id) = ask_id.take() {
                    simulator.cancel_agent_order(id);
                }

                if let Some(mid) = simulator.mid_price() {
                    let _inv_abs = inventory.abs();
                    let (bid_p, ask_p) = self.compute_quotes(mid, inventory, t);

                    // Validate quotes: positive prices, ask > bid.
                    if bid_p > 0.0 && ask_p > bid_p {
                        let qty = Quantity(quote_qty);

                        // Bid: only if buying would not breach the long limit.
                        let buy_would_breach =
                            inventory + (quote_qty as i64) > self.inventory_limit;
                        if !buy_would_breach {
                            let price = Price::from_f64(bid_p);
                            // Ensure the bid is strictly below the best ask.
                            let safe = simulator
                                .book()
                                .best_ask()
                                .map(|(ba, _)| price < ba)
                                .unwrap_or(true);
                            if safe {
                                bid_id = Some(simulator.place_limit_order(Side::Bid, price, qty));
                                n_quotes_placed += 1;
                            }
                        }

                        // Ask: only if selling would not breach the short limit.
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
                                ask_id = Some(simulator.place_limit_order(Side::Ask, price, qty));
                                n_quotes_placed += 1;
                            }
                        }
                    }
                }
                last_quote_t = t;
            }

            // ── Record mark-to-market PnL at each snapshot ──────────────────
            if event.snapshot.is_some() {
                if let Some(mid) = simulator.mid_price() {
                    pnl_curve.push(cash + inventory as f64 * mid);
                    inventory_curve.push(inventory);
                }
            }
        }

        // Cancel any remaining open quotes at end of day.
        if let Some(id) = bid_id {
            simulator.cancel_agent_order(id);
        }
        if let Some(id) = ask_id {
            simulator.cancel_agent_order(id);
        }

        // ── Compute performance metrics ──────────────────────────────────────
        let sharpe = compute_sharpe(&pnl_curve);
        let max_drawdown = compute_max_drawdown(&pnl_curve);
        let fill_rate = if n_quotes_placed > 0 {
            fills.len() as f64 / n_quotes_placed as f64
        } else {
            0.0
        };

        BacktestResult { pnl_curve, inventory_curve, fills, sharpe, max_drawdown, fill_rate }
    }
}

// ── BacktestResult ────────────────────────────────────────────────────────────

/// Aggregated results of an Avellaneda-Stoikov backtest run.
#[derive(Debug)]
pub struct BacktestResult {
    /// Mark-to-market PnL at each book snapshot (`cash + inventory * mid`).
    pub pnl_curve: Vec<f64>,
    /// Inventory at each book snapshot.
    pub inventory_curve: Vec<i64>,
    /// All fills received by the strategy (trades where we were the maker).
    pub fills: Vec<Trade>,
    /// Sharpe ratio of the PnL increments.
    pub sharpe: f64,
    /// Maximum peak-to-trough drawdown of the PnL curve.
    pub max_drawdown: f64,
    /// Fraction of placed quotes that resulted in a fill.
    pub fill_rate: f64,
}

impl BacktestResult {
    /// Final mark-to-market PnL (last entry of `pnl_curve`).
    pub fn final_pnl(&self) -> f64 {
        self.pnl_curve.last().copied().unwrap_or(0.0)
    }

    /// Maximum absolute inventory reached during the backtest.
    pub fn max_inventory(&self) -> i64 {
        self.inventory_curve.iter().map(|q| q.abs()).max().unwrap_or(0)
    }
}

// ── Performance metrics ───────────────────────────────────────────────────────

/// Sharpe ratio: mean(ΔPnL) / std(ΔPnL) · √n.
fn compute_sharpe(pnl_curve: &[f64]) -> f64 {
    if pnl_curve.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = pnl_curve.windows(2).map(|w| w[1] - w[0]).collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < f64::EPSILON {
        return 0.0;
    }
    mean / std * n.sqrt()
}

/// Maximum peak-to-trough drawdown of a PnL curve.
fn compute_max_drawdown(pnl_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &v in pnl_curve {
        if v > peak {
            peak = v;
        }
        let dd = peak - v;
        if dd > max_dd {
            max_dd = dd;
        }
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
        let spread = ask - bid;
        let mid = (bid + ask) / 2.0;
        assert!((mid - 100.0).abs() < 1e-10, "reservation price should equal mid at q=0");
        assert!(spread > 0.0, "spread must be positive");
    }

    #[test]
    fn long_inventory_shifts_quotes_down() {
        let s = make_strat(23_400.0);
        let (bid0, ask0) = s.compute_quotes(100.0, 0, 1000.0);
        let (bid_long, ask_long) = s.compute_quotes(100.0, 10, 1000.0);
        assert!(bid_long < bid0, "long inventory should lower the bid");
        assert!(ask_long < ask0, "long inventory should lower the ask");
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
        let spread_early = a_early - b_early;
        let spread_late = a_late - b_late;
        assert!(
            spread_late < spread_early,
            "spread should narrow as τ → 0: early={spread_early:.4}, late={spread_late:.4}"
        );
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

    // ── estimate_sigma ────────────────────────────────────────────────────────

    #[test]
    fn estimate_sigma_empty_is_zero() {
        assert_eq!(AvellanedaStoikov::estimate_sigma(&[], 50), 0.0);
    }

    #[test]
    fn estimate_sigma_constant_price_is_zero() {
        use crate::orderbook::order::Trade;
        use crate::orderbook::types::{OrderId, Price, Quantity, Timestamp};
        let trade = |p: f64| Trade {
            price: Price::from_f64(p),
            quantity: Quantity(1),
            maker_id: OrderId(0),
            taker_id: OrderId(1),
            timestamp: Timestamp(0),
        };
        let trades: Vec<_> = (0..10).map(|_| trade(100.0)).collect();
        assert_eq!(AvellanedaStoikov::estimate_sigma(&trades, 50), 0.0);
    }

    #[test]
    fn estimate_sigma_positive_for_moving_prices() {
        use crate::orderbook::order::Trade;
        use crate::orderbook::types::{OrderId, Price, Quantity, Timestamp};
        let trade = |p: f64| Trade {
            price: Price::from_f64(p),
            quantity: Quantity(1),
            maker_id: OrderId(0),
            taker_id: OrderId(1),
            timestamp: Timestamp(0),
        };
        let prices = [100.0, 101.0, 99.5, 102.0, 98.0, 103.5];
        let trades: Vec<_> = prices.iter().map(|&p| trade(p)).collect();
        let sigma = AvellanedaStoikov::estimate_sigma(&trades, 50);
        assert!(sigma > 0.0, "sigma should be positive for moving prices");
    }

    // ── estimate_kappa ────────────────────────────────────────────────────────

    #[test]
    fn estimate_kappa_empty_book_returns_one() {
        let book = OrderBook::new();
        let kappa = AvellanedaStoikov::estimate_kappa(&book);
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    // ── run_backtest ──────────────────────────────────────────────────────────

    #[test]
    fn backtest_produces_pnl_curve() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        sim.config.snapshot_interval = 20;
        let strat = make_strat(120.0);
        let result = strat.run_backtest(&mut sim, 42);
        assert!(!result.pnl_curve.is_empty(), "no PnL data");
        assert_eq!(result.pnl_curve.len(), result.inventory_curve.len());
    }

    #[test]
    fn backtest_fill_rate_in_zero_to_one() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat = make_strat(120.0);
        let result = strat.run_backtest(&mut sim, 7);
        assert!(
            result.fill_rate >= 0.0 && result.fill_rate <= 1.0,
            "fill_rate={}", result.fill_rate
        );
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
    fn max_drawdown_non_negative() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat = make_strat(120.0);
        let result = strat.run_backtest(&mut sim, 1);
        assert!(result.max_drawdown >= 0.0);
    }

    #[test]
    fn different_seeds_produce_different_fills() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max = 120.0;
        let strat = make_strat(120.0);
        let r1 = strat.run_backtest(&mut sim, 0);
        let r2 = strat.run_backtest(&mut sim, 99);
        // Two runs almost certainly differ in number of fills.
        assert!(
            r1.fills.len() != r2.fills.len()
                || r1.final_pnl() != r2.final_pnl(),
            "seeds 0 and 99 produced identical results"
        );
    }

    // ── Sharpe / drawdown helpers ─────────────────────────────────────────────

    #[test]
    fn sharpe_zero_for_flat_pnl() {
        assert_eq!(compute_sharpe(&[10.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn sharpe_positive_for_noisy_uptrend() {
        // Mean return = 1.0, std > 0 due to alternating noise.
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
        // Peak = 10, then drops to 3: drawdown = 7.
        let pnl = vec![0.0, 5.0, 10.0, 8.0, 3.0, 6.0];
        assert!((compute_max_drawdown(&pnl) - 7.0).abs() < 1e-10);
    }
}
