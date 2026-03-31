// Avellaneda-Stoikov parameter grid search.
//
// Evaluates all (γ, κ, inv_limit) combinations with σ auto-calibrated from
// realized mid-price data (quadratic variation warm-up).  Each combination is
// run across 5 independent seeds; results are averaged.  Top 20 configurations
// are printed sorted by Sharpe ratio, followed by marginal Sharpe bar charts.
//
// Usage:
//   cargo run -p quantflow-core --example as_grid_search --release
//   cargo run -p quantflow-core --example as_grid_search --release -- --t-max 7200 --top 30

use std::env;

use rayon::prelude::*;

use quantflow_core::simulator::HawkesLobSimulator;
use quantflow_core::strategy::AvellanedaStoikov;

// ── Grid axes ─────────────────────────────────────────────────────────────────

const SEEDS: [u64; 5] = [42, 123, 456, 789, 1337];

const GAMMAS:     &[f64] = &[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
const KAPPAS:     &[f64] = &[0.5, 1.0, 1.5, 2.0, 3.0];
const INV_LIMITS: &[i64] = &[10, 20, 30, 50];

// ── Per-combo result ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ComboResult {
    gamma:     f64,
    kappa:     f64,
    inv_limit: i64,
    // mean across seeds
    sharpe:    f64,
    pnl_mean:  f64,
    pnl_std:   f64,
    max_dd:    f64,
    fill_rate: f64,
    // per-seed sharpe (for stability bar)
    sharpe_std: f64,
}

fn run_combo(gamma: f64, kappa: f64, inv_limit: i64, t_max: f64) -> ComboResult {
    let strat = AvellanedaStoikov::with_auto_sigma(gamma, kappa, t_max, inv_limit, 0.0);

    let mut sharpe_buf = [0.0_f64; 5];
    let mut pnl_buf    = [0.0_f64; 5];
    let mut dd_buf     = [0.0_f64; 5];
    let mut fr_buf     = [0.0_f64; 5];

    for (i, &seed) in SEEDS.iter().enumerate() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max              = t_max;
        sim.config.snapshot_interval  = 50;

        let r = strat.run_backtest(&mut sim, seed);
        sharpe_buf[i] = r.sharpe;
        pnl_buf[i]    = r.final_pnl();
        dd_buf[i]     = r.max_drawdown;
        fr_buf[i]     = r.fill_rate;
    }

    let n = SEEDS.len() as f64;
    let mean = |buf: &[f64; 5]| buf.iter().sum::<f64>() / n;
    let std  = |buf: &[f64; 5], m: f64| {
        (buf.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n).sqrt()
    };

    let sharpe_mean = mean(&sharpe_buf);
    let pnl_mean    = mean(&pnl_buf);

    ComboResult {
        gamma,
        kappa,
        inv_limit,
        sharpe:    sharpe_mean,
        pnl_mean,
        pnl_std:   std(&pnl_buf, pnl_mean),
        max_dd:    mean(&dd_buf),
        fill_rate: mean(&fr_buf),
        sharpe_std: std(&sharpe_buf, sharpe_mean),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };
    let t_max: f64   = get("--t-max", "3600").parse().unwrap_or(3600.0);
    let top_n: usize = get("--top",   "20").parse().unwrap_or(20);

    let total = GAMMAS.len() * KAPPAS.len() * INV_LIMITS.len();

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║       Avellaneda-Stoikov Grid Search                 ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  γ values     : {:?}", GAMMAS);
    eprintln!("  κ values     : {:?}", KAPPAS);
    eprintln!("  inv_limits   : {:?}", INV_LIMITS);
    eprintln!("  σ mode       : auto  (quadratic variation warm-up)");
    eprintln!("  Seeds        : {:?}", SEEDS);
    eprintln!("  t_max        : {t_max:.0} s  ({:.1} h)", t_max / 3600.0);
    eprintln!();
    eprintln!("  Total combos : {} × {} × {} = {total}  ({} runs)",
        GAMMAS.len(), KAPPAS.len(), INV_LIMITS.len(), total * SEEDS.len());
    eprintln!("  Parallelised via rayon — running...");
    eprintln!();

    // Build combo list and run in parallel.
    let combos: Vec<(f64, f64, i64)> = GAMMAS
        .iter()
        .flat_map(|&g| KAPPAS.iter().flat_map(move |&k| INV_LIMITS.iter().map(move |&il| (g, k, il))))
        .collect();

    let mut results: Vec<ComboResult> = combos
        .into_par_iter()
        .map(|(g, k, il)| run_combo(g, k, il, t_max))
        .collect();

    // Sort by mean Sharpe descending.
    results.sort_by(|a, b| b.sharpe.partial_cmp(&a.sharpe).unwrap_or(std::cmp::Ordering::Equal));

    // ── Top-N table ───────────────────────────────────────────────────────────
    let display_n = top_n.min(results.len());

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  TOP {display_n:<2} CONFIGURATIONS  (σ = auto, averaged over {} seeds)                   ║", SEEDS.len());
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Header
    println!("  {:<4} │ {:<5} │ {:<4} │ {:<3} │ {:<7} │ {:<6} │ {:<17} │ {:<7} │ {:<8}",
        "Rank", "γ", "σ", "κ", "inv_lim", "Sharpe", "PnL (mean±std)", "MaxDD", "FillRate");
    println!("  {}┼{}┼{}┼{}┼{}┼{}┼{}┼{}┼{}",
        "─".repeat(5), "─".repeat(7), "─".repeat(6), "─".repeat(5),
        "─".repeat(9), "─".repeat(8), "─".repeat(19),
        "─".repeat(9), "─".repeat(9));

    for (rank, r) in results.iter().take(display_n).enumerate() {
        let pnl_fmt = format!("{:+.0} ± {:.0}", r.pnl_mean, r.pnl_std);
        println!("  {:<4} │ {:<5.2} │ {:<4} │ {:<3.1} │ {:<7} │ {:>+6.2} │ {:<17} │ {:>7.0} │ {:>7.1}%",
            rank + 1,
            r.gamma,
            "auto",
            r.kappa,
            r.inv_limit,
            r.sharpe,
            pnl_fmt,
            r.max_dd,
            r.fill_rate * 100.0,
        );
    }

    println!();
    println!("  {total} combos evaluated  ·  {} seeds each  ·  {} total runs",
        SEEDS.len(), total * SEEDS.len());

    // ── Marginal Sharpe analysis ──────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  MARGINAL SHARPE BY PARAMETER                        ║");
    println!("╚══════════════════════════════════════════════════════╝");

    let marginal = |label: &str, header: &str, vals_fn: &dyn Fn(&ComboResult) -> f64, keys: Vec<String>| {
        println!();
        println!("  {label}");
        for key in &keys {
            // We'll handle this below per-parameter
            let _ = (header, vals_fn, key);
        }
    };
    // Suppress unused warning — inline the loop directly for each axis.
    let _ = marginal;

    // γ axis
    println!();
    println!("  γ  (avg Sharpe across all κ, inv_limit)");
    for &g in GAMMAS {
        let vals: Vec<f64> = results.iter().filter(|r| r.gamma == g).map(|r| r.sharpe).collect();
        print_marginal_bar(&format!("γ={:.2}", g), &vals);
    }

    // κ axis
    println!();
    println!("  κ  (avg Sharpe across all γ, inv_limit)");
    for &k in KAPPAS {
        let vals: Vec<f64> = results.iter().filter(|r| r.kappa == k).map(|r| r.sharpe).collect();
        print_marginal_bar(&format!("κ={:.1}", k), &vals);
    }

    // inv_limit axis
    println!();
    println!("  inv_limit  (avg Sharpe across all γ, κ)");
    for &il in INV_LIMITS {
        let vals: Vec<f64> = results.iter().filter(|r| r.inv_limit == il).map(|r| r.sharpe).collect();
        print_marginal_bar(&format!("il={:<3}", il), &vals);
    }

    // ── Best config ───────────────────────────────────────────────────────────
    let best = &results[0];
    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  BEST CONFIGURATION                                  ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!("  γ = {:.2}   κ = {:.1}   inv_limit = {}",
        best.gamma, best.kappa, best.inv_limit);
    println!("  Sharpe = {:+.4} ± {:.4}", best.sharpe, best.sharpe_std);
    println!("  PnL    = {:+.2} ± {:.2}", best.pnl_mean, best.pnl_std);
    println!("  MaxDD  = {:.2}", best.max_dd);
    println!();
    println!("  Reproduce with:");
    println!("    cargo run -p quantflow-core --example as_backtest --release -- \\");
    println!("      --gamma {:.2} --kappa {:.1} --inv-limit {} --auto-sigma",
        best.gamma, best.kappa, best.inv_limit);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_marginal_bar(label: &str, vals: &[f64]) {
    if vals.is_empty() { return; }
    let avg = vals.iter().sum::<f64>() / vals.len() as f64;
    // Map avg Sharpe [-2, +2] onto bar width [0, 40].
    let bar_len = (((avg + 2.0) / 4.0) * 40.0).clamp(0.0, 40.0) as usize;
    let bar = "▓".repeat(bar_len);
    println!("  {:<8}  {:>+7.4}  │{:<40}│", label, avg, bar);
}
