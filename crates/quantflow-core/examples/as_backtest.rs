// Avellaneda-Stoikov backtest v2 — full diagnostics.
//
// Usage:
//   cargo run -p quantflow-core --example as_backtest
//   cargo run -p quantflow-core --example as_backtest -- --seed 99 --gamma 0.3 --kappa 1.0
//   cargo run -p quantflow-core --example as_backtest -- --auto-sigma

use std::env;

use quantflow_core::simulator::HawkesLobSimulator;
use quantflow_core::strategy::AvellanedaStoikov;

fn main() {
    // ── Parse CLI args ────────────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let get = |flag: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };

    let seed:       u64 = get("--seed",        "42").parse().unwrap_or(42);
    let t_max:      f64 = get("--t-max",        "3600").parse().unwrap_or(3600.0);
    let gamma:      f64 = get("--gamma",        "0.1").parse().unwrap_or(0.1);
    let sigma:      f64 = get("--sigma",        "0.01").parse().unwrap_or(0.01);
    let kappa:      f64 = get("--kappa",        "1.5").parse().unwrap_or(1.5);
    let inv_lim:    i64 = get("--inv-limit",    "50").parse().unwrap_or(50);
    let snap_int: usize = get("--snap-interval","50").parse().unwrap_or(50);
    let auto_sigma: bool = flag("--auto-sigma");
    let spread_floor: f64 = get("--spread-floor", "0.0").parse().unwrap_or(0.0);

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║       Avellaneda-Stoikov Backtest  v2                ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Parameters");
    eprintln!("  seed              = {seed}");
    eprintln!("  t_max             = {t_max:.0} s  ({:.1} h)", t_max / 3600.0);
    eprintln!("  γ (risk aversion) = {gamma}");
    eprintln!("  σ mode            = {}", if auto_sigma { "auto (quadratic variation)".to_string() } else { format!("{sigma}") });
    eprintln!("  κ (fill intensity)= {kappa}");
    eprintln!("  inventory limit   = ±{inv_lim}");
    eprintln!("  spread floor      = {spread_floor}");
    eprintln!();

    // ── Build simulator ───────────────────────────────────────────────────────
    let mut sim = HawkesLobSimulator::default_12d().unwrap();
    sim.config.t_max = t_max;
    sim.config.snapshot_interval = snap_int;

    // ── Build strategy ────────────────────────────────────────────────────────
    let strat = if auto_sigma {
        AvellanedaStoikov::with_auto_sigma(gamma, kappa, t_max, inv_lim, spread_floor)
    } else {
        let mut s = AvellanedaStoikov::new(gamma, sigma, kappa, t_max, inv_lim);
        s.spread_floor = spread_floor;
        s
    };

    eprintln!("Simulating {t_max:.0} s of trading (seed {seed})…");
    let result = strat.run_backtest(&mut sim, seed);
    let n_events = result.pnl_curve.len() * snap_int;
    eprintln!("Done. ~{n_events} events processed.\n");

    // ── Summary ───────────────────────────────────────────────────────────────
    let final_pnl = result.final_pnl();
    let max_inv   = result.max_inventory();
    let n_fills   = result.fills.len();
    let calmar    = result.calmar_ratio();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  PERFORMANCE SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Final PnL        {:>+12.4}", final_pnl);
    println!("  Sharpe ratio     {:>12.4}", result.sharpe);
    println!("  Max drawdown     {:>12.4}", result.max_drawdown);
    println!("  Calmar ratio     {:>12.4}", if calmar.is_finite() { format!("{calmar:>12.4}") } else { "         ∞".to_string() });
    println!("  Fill rate        {:>11.1}%", result.fill_rate * 100.0);
    println!("  Total fills      {:>12}", n_fills);
    println!("  Round trips      {:>12}", result.n_round_trips);
    println!("  Max |inventory|  {:>12}", max_inv);
    println!();

    // ── PnL decomposition ─────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  PnL DECOMPOSITION                                   ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Spread PnL       {:>+12.4}  (bid-ask capture)", result.spread_pnl);
    println!("  Inventory PnL    {:>+12.4}  (mark-to-market drift)", result.inventory_pnl);
    println!("  Net PnL          {:>+12.4}", final_pnl);
    let ratio = if result.spread_pnl.abs() > f64::EPSILON {
        result.inventory_pnl.abs() / result.spread_pnl * 100.0
    } else {
        0.0
    };
    println!("  |Inv PnL| / Spread PnL = {ratio:.1}%  (lower is better)");
    println!();

    // ── Volatility section ────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  VOLATILITY                                          ║");
    println!("╚══════════════════════════════════════════════════════╝");
    if auto_sigma {
        println!("  σ mode           auto-calibrated (quadratic variation)");
    } else {
        println!("  σ configured     {:>12.5}", sigma);
    }
    println!("  σ used in quotes {:>12.5}", result.realized_sigma);
    if !auto_sigma {
        let ratio_sig = result.realized_sigma / sigma;
        println!("  realized / configured = {ratio_sig:.2}×");
    }
    println!();

    // ── Quote statistics ──────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  QUOTE STATISTICS                                    ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Avg quoted spread     {:>10.5}", result.avg_quoted_spread);
    println!("  Avg spread at fill    {:>10.5}", result.avg_spread_at_fill);
    println!("  Time-in-market        {:>9.1}%", result.time_in_market * 100.0);
    if result.avg_quoted_spread > f64::EPSILON && result.avg_spread_at_fill > f64::EPSILON {
        let capture = result.avg_spread_at_fill / result.avg_quoted_spread * 100.0;
        println!("  Fill spread / quoted  {:>9.1}%  (>100% = adverse fill tightening)", capture);
    }
    println!();

    // ── Adverse selection ─────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  ADVERSE SELECTION                                   ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Post-fill mid move (1 s)  {:>+10.5}  (+ = adverse)", result.post_fill_move_1s);
    println!("  Post-fill mid move (5 s)  {:>+10.5}", result.post_fill_move_5s);
    println!("  Post-fill mid move (10 s) {:>+10.5}", result.post_fill_move_10s);
    let half_s = result.avg_spread_at_fill / 2.0;
    if half_s > f64::EPSILON && result.post_fill_move_1s.abs() > f64::EPSILON {
        let as_ratio = result.post_fill_move_1s / half_s * 100.0;
        println!("  Adverse / half-spread (1s) {:>+8.1}%  (<50% = manageable)", as_ratio);
    }
    println!();

    // ── Realized spread by inventory bucket ──────────────────────────────────
    let buckets = result.spread_by_inv_bucket();
    let bucket_labels = ["|inv|<10", "10-25", "25-40", "≥40"];
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  REALIZED HALF-SPREAD BY INVENTORY BUCKET            ║");
    println!("╚══════════════════════════════════════════════════════╝");
    for (label, &val) in bucket_labels.iter().zip(buckets.iter()) {
        if val > f64::EPSILON {
            println!("  {:>10}  {:>10.5}", label, val);
        } else {
            println!("  {:>10}  {:>10}  (no fills)", label, "—");
        }
    }
    println!();

    // ── PnL curve ─────────────────────────────────────────────────────────────
    let curve = &result.pnl_curve;
    if curve.len() >= 2 {
        let step = (curve.len() / 20).max(1);
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║  PnL CURVE  (every ~{:.0} s)                          ", t_max / 20.0);
        println!("╚══════════════════════════════════════════════════════╝");
        println!("  {:<10} {:>10}  {:>10}", "t (s)", "PnL", "inventory");
        println!("  {}", "─".repeat(34));
        for (i, (&pnl, &inv)) in curve.iter()
            .zip(result.inventory_curve.iter())
            .enumerate()
            .filter(|(i, _)| i % step == 0 || *i == curve.len() - 1)
        {
            let t = (i as f64 / curve.len() as f64) * t_max;
            println!("  {:<10.1} {:>+10.4}  {:>10}", t, pnl, inv);
        }
        println!();
    }

    // ── Inventory distribution ────────────────────────────────────────────────
    if !result.inventory_curve.is_empty() {
        let min_inv     = *result.inventory_curve.iter().min().unwrap();
        let max_inv_val = *result.inventory_curve.iter().max().unwrap();

        let n_bins = 11_i64;
        let range = (max_inv_val - min_inv).max(1);
        let mut bins = vec![0usize; n_bins as usize];
        for &q in &result.inventory_curve {
            let b = ((q - min_inv) * (n_bins - 1) / range) as usize;
            bins[b.min(n_bins as usize - 1)] += 1;
        }
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_width = 28;

        println!("╔══════════════════════════════════════════════════════╗");
        println!("║  INVENTORY DISTRIBUTION  [{min_inv}, {max_inv_val}]");
        println!("╚══════════════════════════════════════════════════════╝");
        for (b, &count) in bins.iter().enumerate() {
            let inv_val = min_inv + b as i64 * range / (n_bins - 1);
            let bar_len = if max_count > 0 { count * bar_width / max_count } else { 0 };
            let bar: String = "█".repeat(bar_len);
            println!("  {:>5} │{:<28} {}", inv_val, bar, count);
        }
        println!();
    }

    // ── Fill price range ──────────────────────────────────────────────────────
    if !result.fills.is_empty() {
        let prices: Vec<f64> = result.fills.iter().map(|t| t.price.to_f64()).collect();
        let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  Fill prices: [{min_p:.4}, {max_p:.4}]  ({n_fills} fills)");
        println!();
    }

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  END OF REPORT                                       ║");
    println!("╚══════════════════════════════════════════════════════╝");
}
