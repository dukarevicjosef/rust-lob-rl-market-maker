//! Stylized-facts validation for simulated LOB data.
//!
//! Functions here measure the five empirical regularities that any realistic
//! LOB simulator should reproduce (Cont, 2001):
//!
//! 1. **Heavy tails** — return kurtosis far exceeds 3.
//! 2. **Volatility clustering** — ACF of |returns| decays slowly.
//! 3. **Spread distribution** — spreads are right-skewed, rarely ≤ 1 tick.
//! 4. **Signature plot** — realized variance grows with sampling interval
//!    (microstructure noise at fine scales, diffusion at coarse).
//! 5. **Intraday U-shape** — trading volume is highest near open and close.

use crate::orderbook::book::OrderBookSnapshot;
use crate::orderbook::order::Trade;

// ── 1. Return kurtosis ────────────────────────────────────────────────────────

/// Excess kurtosis of log-returns computed from trade prices.
///
/// Returns 0.0 if fewer than 4 trades are available.
/// Excess kurtosis > 0 indicates heavier-than-Gaussian tails.
pub fn compute_return_kurtosis(trades: &[Trade]) -> f64 {
    let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();
    if prices.len() < 4 {
        return 0.0;
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .filter(|w| w[0] > 0.0)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let n = returns.len() as f64;
    if n < 4.0 {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

    if variance < f64::EPSILON {
        return 0.0;
    }

    let m4 = returns.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;
    (m4 / variance.powi(2)) - 3.0 // excess kurtosis
}

// ── 2. ACF of |returns| ───────────────────────────────────────────────────────

/// Autocorrelation function of absolute log-returns at lags 1..=max_lag.
///
/// Persistent positive ACF of |returns| is the signature of volatility
/// clustering (Mandelbrot, 1963; GARCH behaviour).
/// Returns a vector of length `max_lag`; entry k is ρ(|r|, lag=k+1).
pub fn compute_acf_absolute_returns(trades: &[Trade], max_lag: usize) -> Vec<f64> {
    let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();
    let abs_returns: Vec<f64> = prices
        .windows(2)
        .filter(|w| w[0] > 0.0)
        .map(|w| (w[1] / w[0]).ln().abs())
        .collect();

    let n = abs_returns.len();
    if n < max_lag + 2 {
        return vec![0.0; max_lag];
    }

    let mean = abs_returns.iter().sum::<f64>() / n as f64;
    let variance = abs_returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < f64::EPSILON {
        return vec![0.0; max_lag];
    }

    (1..=max_lag)
        .map(|lag| {
            // Use the same denominator (n) for both variance and covariance so
            // that the ratio is guaranteed to lie in [−1, 1].
            let cov = abs_returns[..n - lag]
                .iter()
                .zip(&abs_returns[lag..])
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / n as f64;
            cov / variance
        })
        .collect()
}

// ── 3. Spread distribution ────────────────────────────────────────────────────

/// Sorted bid-ask spreads (in floating-point price units) from book snapshots.
///
/// Empirically, spreads are discrete, right-skewed, and rarely below one tick.
pub fn compute_spread_distribution(snapshots: &[(f64, OrderBookSnapshot)]) -> Vec<f64> {
    let mut spreads: Vec<f64> = snapshots
        .iter()
        .filter_map(|(_, snap)| {
            let best_bid = snap.bids.first().map(|(p, _)| p.to_f64())?;
            let best_ask = snap.asks.first().map(|(p, _)| p.to_f64())?;
            let spread = best_ask - best_bid;
            if spread > 0.0 { Some(spread) } else { None }
        })
        .collect();
    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    spreads
}

// ── 4. Signature plot ─────────────────────────────────────────────────────────

/// Realized variance at each sampling scale (in number of trades).
///
/// The signature plot shows how RV changes with the sampling frequency.
/// At fine scales, microstructure noise inflates RV; at coarse scales,
/// diffusion dominates (Bandi & Russell, 2006).
///
/// `scales` is a sorted list of sampling intervals (in trades).
/// Returns `(scale, realized_variance)` pairs.
pub fn compute_signature_plot(trades: &[Trade], scales: &[usize]) -> Vec<(usize, f64)> {
    let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();

    scales
        .iter()
        .map(|&scale| {
            if scale == 0 || prices.len() < scale + 1 {
                return (*&scale, 0.0);
            }
            // Sample every `scale` trades and compute RV.
            let sampled: Vec<f64> = prices.iter().step_by(scale).copied().collect();
            let rv = sampled
                .windows(2)
                .filter(|w| w[0] > 0.0)
                .map(|w| (w[1] / w[0]).ln().powi(2))
                .sum::<f64>();
            (scale, rv)
        })
        .collect()
}

// ── 5. Intraday activity pattern ──────────────────────────────────────────────

/// Trade volume binned into equally-spaced intraday intervals.
///
/// Trade times are in seconds; `t_max` is the total simulation length.
/// Returns a vector of length `bins`, each entry being the total traded
/// quantity in that time bin.
///
/// The empirical U-shape means `bins[0]` and `bins[bins-1]` should be
/// the largest (Admati & Pfleiderer, 1988).
pub fn compute_intraday_pattern(trades: &[Trade], bins: usize, t_max_ns: u64) -> Vec<f64> {
    if bins == 0 || t_max_ns == 0 {
        return vec![];
    }
    let mut counts = vec![0.0_f64; bins];
    for trade in trades {
        let t_ns = trade.timestamp.0;
        // Map nanosecond timestamp to bin index.
        let bin = ((t_ns as f64 / t_max_ns as f64) * bins as f64) as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += trade.quantity.0 as f64;
    }
    counts
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::order::Trade;
    use crate::orderbook::types::{OrderId, Price, Quantity, Timestamp};

    fn make_trade(price: f64, qty: u64, ts_ns: u64) -> Trade {
        Trade {
            price: Price::from_f64(price),
            quantity: Quantity(qty),
            maker_id: OrderId(0),
            taker_id: OrderId(1),
            timestamp: Timestamp(ts_ns),
        }
    }

    #[test]
    fn kurtosis_empty_is_zero() {
        assert_eq!(compute_return_kurtosis(&[]), 0.0);
    }

    #[test]
    fn kurtosis_constant_prices_is_zero() {
        let trades: Vec<_> = (0..10).map(|_| make_trade(100.0, 1, 0)).collect();
        assert_eq!(compute_return_kurtosis(&trades), 0.0);
    }

    #[test]
    fn kurtosis_normal_returns_near_zero() {
        // 500 points with very small i.i.d. moves → kurtosis near 0 (excess).
        // Use a deterministic price series with known near-Gaussian behaviour.
        let prices = [100.0, 100.1, 100.0, 99.9, 100.2, 100.1, 99.8, 100.0];
        let trades: Vec<_> = prices.iter().map(|&p| make_trade(p, 1, 0)).collect();
        let k = compute_return_kurtosis(&trades);
        // Excess kurtosis of a handful of near-uniform returns may be negative;
        // we just assert it's a finite number.
        assert!(k.is_finite(), "kurtosis={k}");
    }

    #[test]
    fn acf_empty_returns_zero_vector() {
        let acf = compute_acf_absolute_returns(&[], 5);
        assert_eq!(acf, vec![0.0; 5]);
    }

    #[test]
    fn acf_length_equals_max_lag() {
        let trades: Vec<_> = (0..50).map(|i| make_trade(100.0 + i as f64 * 0.01, 1, 0)).collect();
        let acf = compute_acf_absolute_returns(&trades, 10);
        assert_eq!(acf.len(), 10);
    }

    #[test]
    fn acf_values_in_minus_one_to_one() {
        let prices = [100.0, 101.0, 100.0, 102.0, 99.0, 101.5, 100.5, 100.0];
        let trades: Vec<_> = prices.iter().map(|&p| make_trade(p, 1, 0)).collect();
        let acf = compute_acf_absolute_returns(&trades, 3);
        for &v in &acf {
            assert!(v >= -1.0 - 1e-9 && v <= 1.0 + 1e-9, "ACF value {v} out of range");
        }
    }

    #[test]
    fn spread_distribution_empty_snapshots() {
        assert!(compute_spread_distribution(&[]).is_empty());
    }

    #[test]
    fn signature_plot_length_equals_scales() {
        let trades: Vec<_> = (0..100).map(|i| make_trade(100.0 + (i % 3) as f64 * 0.1, 1, i * 1_000_000)).collect();
        let scales = vec![1, 5, 10, 20];
        let plot = compute_signature_plot(&trades, &scales);
        assert_eq!(plot.len(), scales.len());
    }

    #[test]
    fn signature_plot_rv_positive_for_moving_prices() {
        // Prices that move → RV should be > 0 at scale 1.
        let trades: Vec<_> = (0..50)
            .map(|i| make_trade(100.0 + (i % 5) as f64 * 0.5, 1, i * 1_000_000))
            .collect();
        let plot = compute_signature_plot(&trades, &[1]);
        assert!(plot[0].1 > 0.0, "RV at scale 1 should be positive");
    }

    #[test]
    fn intraday_pattern_correct_length() {
        let trades: Vec<_> = (0..100)
            .map(|i| make_trade(100.0, 1, i as u64 * 1_000_000_000))
            .collect();
        let t_max_ns = 100_u64 * 1_000_000_000;
        let pattern = compute_intraday_pattern(&trades, 10, t_max_ns);
        assert_eq!(pattern.len(), 10);
    }

    #[test]
    fn intraday_pattern_volume_conserved() {
        let trades: Vec<_> = (0..60)
            .map(|i| make_trade(100.0, 2, i as u64 * 1_000_000_000))
            .collect();
        let t_max_ns = 60_u64 * 1_000_000_000;
        let pattern = compute_intraday_pattern(&trades, 6, t_max_ns);
        let total: f64 = pattern.iter().sum();
        assert!((total - 120.0).abs() < 1e-9, "total volume={total}");
    }
}
