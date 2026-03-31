// Avellaneda-Stoikov backtest over a simulated trading day.
//
// Usage:
//   cargo run -p quantflow-core --example as_backtest
//   cargo run -p quantflow-core --example as_backtest -- --seed 99 --gamma 0.05

use std::env;

use quantflow_core::simulator::HawkesLobSimulator;
use quantflow_core::strategy::AvellanedaStoikov;

fn main() {
    // ── Parse simple CLI args ─────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };

    let seed:     u64 = get("--seed",     "42").parse().unwrap_or(42);
    let t_max:    f64 = get("--t-max",    "3600").parse().unwrap_or(3600.0);
    let gamma:    f64 = get("--gamma",    "0.1").parse().unwrap_or(0.1);
    let sigma:    f64 = get("--sigma",    "0.01").parse().unwrap_or(0.01);
    let kappa:    f64 = get("--kappa",    "1.5").parse().unwrap_or(1.5);
    let inv_lim:  i64 = get("--inv-limit","50").parse().unwrap_or(50);
    let snap_int: usize = get("--snap-interval", "50").parse().unwrap_or(50);

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║         Avellaneda-Stoikov Backtest                  ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Parameters");
    eprintln!("  seed            = {seed}");
    eprintln!("  t_max           = {t_max:.0} s  ({:.1} h)", t_max / 3600.0);
    eprintln!("  γ (risk aversion) = {gamma}");
    eprintln!("  σ (volatility)    = {sigma}");
    eprintln!("  κ (fill intensity)= {kappa}");
    eprintln!("  inventory limit = ±{inv_lim}");
    eprintln!();

    // ── Build simulator ───────────────────────────────────────────────────────
    let mut sim = HawkesLobSimulator::default_12d().unwrap();
    sim.config.t_max = t_max;
    sim.config.snapshot_interval = snap_int;

    // ── Build strategy ────────────────────────────────────────────────────────
    let strat = AvellanedaStoikov::new(gamma, sigma, kappa, t_max, inv_lim);

    eprintln!("Simulating {t_max:.0} s of trading (seed {seed})…");
    let result = strat.run_backtest(&mut sim, seed);
    eprintln!("Done. {} events processed.\n", result.pnl_curve.len() * snap_int);

    // ── Summary ───────────────────────────────────────────────────────────────
    let final_pnl  = result.final_pnl();
    let max_inv    = result.max_inventory();
    let n_fills    = result.fills.len();

    println!("══════════════════════════════════════════════════════");
    println!("  PERFORMANCE SUMMARY");
    println!("══════════════════════════════════════════════════════");
    println!("  Final PnL        {:>+12.4}", final_pnl);
    println!("  Sharpe ratio     {:>12.4}", result.sharpe);
    println!("  Max drawdown     {:>12.4}", result.max_drawdown);
    println!("  Fill rate        {:>11.1}%", result.fill_rate * 100.0);
    println!("  Total fills      {:>12}",    n_fills);
    println!("  Max |inventory|  {:>12}",    max_inv);
    println!("──────────────────────────────────────────────────────");

    // ── PnL curve (sampled at 20 points) ─────────────────────────────────────
    let curve = &result.pnl_curve;
    if curve.len() >= 2 {
        let step = (curve.len() / 20).max(1);
        println!();
        println!("  PnL curve (every ~{:.0} s)", t_max / 20.0);
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
    }

    // ── Inventory histogram ───────────────────────────────────────────────────
    if !result.inventory_curve.is_empty() {
        let min_inv = *result.inventory_curve.iter().min().unwrap();
        let max_inv_val = *result.inventory_curve.iter().max().unwrap();
        println!();
        println!("  Inventory range: [{min_inv}, {max_inv_val}]");

        // Simple ASCII bar chart: bin inventories into 11 buckets
        let n_bins = 11_i64;
        let range = (max_inv_val - min_inv).max(1);
        let mut bins = vec![0usize; n_bins as usize];
        for &q in &result.inventory_curve {
            let b = ((q - min_inv) * (n_bins - 1) / range) as usize;
            bins[b.min(n_bins as usize - 1)] += 1;
        }
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_width = 30;
        println!();
        println!("  Inventory distribution");
        for (b, &count) in bins.iter().enumerate() {
            let inv_val = min_inv + b as i64 * range / (n_bins - 1);
            let bar_len = count * bar_width / max_count;
            let bar: String = "█".repeat(bar_len);
            println!("  {:>5} │{:<30} {}", inv_val, bar, count);
        }
    }

    // ── Fill price histogram ──────────────────────────────────────────────────
    if !result.fills.is_empty() {
        let prices: Vec<f64> = result.fills.iter().map(|t| t.price.to_f64()).collect();
        let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!();
        println!("  Fill prices: [{min_p:.4}, {max_p:.4}]  ({n_fills} fills)");
    }

    println!("══════════════════════════════════════════════════════");
}
