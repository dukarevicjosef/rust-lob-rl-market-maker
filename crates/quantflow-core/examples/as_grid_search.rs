// Avellaneda-Stoikov parameter grid search.
//
// Evaluates all (γ, κ, inv_limit) combinations with σ auto-calibrated from
// realized mid-price data.  Each combination is run across 5 seeds and the
// results are averaged.  Top 20 configurations by Sharpe are printed.
//
// Usage:
//   cargo run -p quantflow-core --example as_grid_search --release
//   cargo run -p quantflow-core --example as_grid_search --release -- --t-max 3600

use std::env;

use rayon::prelude::*;

use quantflow_core::simulator::HawkesLobSimulator;
use quantflow_core::strategy::AvellanedaStoikov;

const SEEDS: [u64; 5] = [42, 123, 456, 789, 1337];

const GAMMAS: &[f64] = &[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
const KAPPAS: &[f64] = &[0.5, 1.0, 1.5, 2.0, 3.0];
const INV_LIMITS: &[i64] = &[10, 20, 30, 50];
// Spread floor expressed as fraction of initial mid (100.0).  0.5% and 1% in price units.
const SPREAD_FLOORS: &[f64] = &[0.0, 0.5, 1.0];

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct RunResult {
    gamma:        f64,
    kappa:        f64,
    inv_limit:    i64,
    spread_floor: f64,
    // averages across seeds
    sharpe:       f64,
    final_pnl:    f64,
    max_drawdown: f64,
    calmar:       f64,
    fill_rate:    f64,
    spread_pnl:   f64,
    inv_pnl:      f64,
    as_1s:        f64,
    realized_sigma: f64,
    // raw per-seed values (for std-dev)
    sharpe_vals:  [f64; 5],
}

fn run_combo(gamma: f64, kappa: f64, inv_limit: i64, spread_floor: f64, t_max: f64) -> RunResult {
    let mut sharpe_vals = [0.0_f64; 5];
    let mut sum_pnl       = 0.0_f64;
    let mut sum_dd        = 0.0_f64;
    let mut sum_calmar    = 0.0_f64;
    let mut sum_fill_rate = 0.0_f64;
    let mut sum_spread    = 0.0_f64;
    let mut sum_inv_pnl   = 0.0_f64;
    let mut sum_as1s      = 0.0_f64;
    let mut sum_sigma     = 0.0_f64;

    let strat = AvellanedaStoikov::with_auto_sigma(gamma, kappa, t_max, inv_limit, spread_floor);

    for (i, &seed) in SEEDS.iter().enumerate() {
        let mut sim = HawkesLobSimulator::default_12d().unwrap();
        sim.config.t_max       = t_max;
        sim.config.snapshot_interval = 50;

        let r = strat.run_backtest(&mut sim, seed);

        sharpe_vals[i] = r.sharpe;
        sum_pnl       += r.final_pnl();
        sum_dd        += r.max_drawdown;
        sum_calmar    += r.calmar_ratio().min(100.0);   // cap infinite calmar
        sum_fill_rate += r.fill_rate;
        sum_spread    += r.spread_pnl;
        sum_inv_pnl   += r.inventory_pnl;
        sum_as1s      += r.post_fill_move_1s;
        sum_sigma     += r.realized_sigma;
    }

    let n = SEEDS.len() as f64;
    RunResult {
        gamma,
        kappa,
        inv_limit,
        spread_floor,
        sharpe:         sharpe_vals.iter().sum::<f64>() / n,
        final_pnl:      sum_pnl       / n,
        max_drawdown:   sum_dd        / n,
        calmar:         sum_calmar    / n,
        fill_rate:      sum_fill_rate / n,
        spread_pnl:     sum_spread    / n,
        inv_pnl:        sum_inv_pnl   / n,
        as_1s:          sum_as1s      / n,
        realized_sigma: sum_sigma     / n,
        sharpe_vals,
    }
}

fn sharpe_std(r: &RunResult) -> f64 {
    let mean = r.sharpe;
    let var  = r.sharpe_vals.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / SEEDS.len() as f64;
    var.sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };
    let t_max: f64 = get("--t-max", "3600").parse().unwrap_or(3600.0);
    let top_n: usize = get("--top", "20").parse().unwrap_or(20);

    // Build all parameter combos.
    let combos: Vec<(f64, f64, i64, f64)> = GAMMAS
        .iter()
        .flat_map(|&g| {
            KAPPAS.iter().flat_map(move |&k| {
                INV_LIMITS.iter().flat_map(move |&il| {
                    SPREAD_FLOORS.iter().map(move |&sf| (g, k, il, sf))
                })
            })
        })
        .collect();

    let total = combos.len();
    eprintln!(
        "Grid search: {} γ × {} κ × {} inv_limits × {} spread_floors = {} combos, {} seeds each",
        GAMMAS.len(), KAPPAS.len(), INV_LIMITS.len(), SPREAD_FLOORS.len(),
        total, SEEDS.len()
    );
    eprintln!("σ mode: auto (quadratic variation warm-up)");
    eprintln!("t_max: {t_max:.0} s  ({:.1} h)", t_max / 3600.0);
    eprintln!("Parallelized with rayon — running...\n");

    // Parallel evaluation.
    let mut results: Vec<RunResult> = combos
        .into_par_iter()
        .map(|(g, k, il, sf)| run_combo(g, k, il, sf, t_max))
        .collect();

    // Sort by Sharpe descending.
    results.sort_by(|a, b| b.sharpe.partial_cmp(&a.sharpe).unwrap_or(std::cmp::Ordering::Equal));

    // Print summary table.
    let display_n = top_n.min(results.len());

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  TOP {display_n} CONFIGURATIONS  (sorted by Sharpe, averaged over {} seeds)                          ║", SEEDS.len());
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {:>6}  {:>5}  {:>8}  {:>6}  {:>8}  {:>9}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}  {:>8}",
        "γ", "κ", "inv_lim", "sf", "Sharpe", "Sharpe±σ", "PnL", "Drawdown", "Calmar",
        "FillRate", "Adv1s", "σ_real");
    println!("  {}", "─".repeat(106));

    for r in results.iter().take(display_n) {
        let std = sharpe_std(r);
        println!(
            "  {:>6.2}  {:>5.1}  {:>8}  {:>6.1}  {:>+8.4}  {:>+8.4}±{:<4.3}  {:>+8.2}  {:>8.4}  {:>8.4}  {:>7.1}%  {:>+7.5}  {:>8.5}",
            r.gamma, r.kappa, r.inv_limit, r.spread_floor,
            r.sharpe, r.sharpe, std,
            r.final_pnl, r.max_drawdown, r.calmar,
            r.fill_rate * 100.0,
            r.as_1s,
            r.realized_sigma,
        );
    }

    println!();
    println!("  Total combos evaluated: {}", results.len());

    // ── Marginal analysis ────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  MARGINAL SHARPE BY PARAMETER                        ║");
    println!("╚══════════════════════════════════════════════════════╝");

    // γ marginals
    println!();
    println!("  γ (avg Sharpe across all κ, inv_limit, spread_floor):");
    for &g in GAMMAS {
        let vals: Vec<f64> = results.iter().filter(|r| r.gamma == g).map(|r| r.sharpe).collect();
        if !vals.is_empty() {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            let bar_len = ((avg + 1.0) * 15.0).max(0.0) as usize;
            let bar = "▓".repeat(bar_len.min(40));
            println!("  γ={:.2}  {:>+7.4}  {}", g, avg, bar);
        }
    }

    // κ marginals
    println!();
    println!("  κ (avg Sharpe across all γ, inv_limit, spread_floor):");
    for &k in KAPPAS {
        let vals: Vec<f64> = results.iter().filter(|r| r.kappa == k).map(|r| r.sharpe).collect();
        if !vals.is_empty() {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            let bar_len = ((avg + 1.0) * 15.0).max(0.0) as usize;
            let bar = "▓".repeat(bar_len.min(40));
            println!("  κ={:.1}   {:>+7.4}  {}", k, avg, bar);
        }
    }

    // inv_limit marginals
    println!();
    println!("  inv_limit (avg Sharpe):");
    for &il in INV_LIMITS {
        let vals: Vec<f64> = results.iter().filter(|r| r.inv_limit == il).map(|r| r.sharpe).collect();
        if !vals.is_empty() {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            let bar_len = ((avg + 1.0) * 15.0).max(0.0) as usize;
            let bar = "▓".repeat(bar_len.min(40));
            println!("  il={:<3}   {:>+7.4}  {}", il, avg, bar);
        }
    }

    // spread_floor marginals
    println!();
    println!("  spread_floor (avg Sharpe):");
    for &sf in SPREAD_FLOORS {
        let vals: Vec<f64> = results.iter().filter(|r| (r.spread_floor - sf).abs() < 1e-9).map(|r| r.sharpe).collect();
        if !vals.is_empty() {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            let bar_len = ((avg + 1.0) * 15.0).max(0.0) as usize;
            let bar = "▓".repeat(bar_len.min(40));
            println!("  sf={:.1}   {:>+7.4}  {}", sf, avg, bar);
        }
    }

    println!();
    println!("  Best config: γ={:.2}  κ={:.1}  inv_limit={}  spread_floor={:.1}  Sharpe={:+.4}",
        results[0].gamma, results[0].kappa, results[0].inv_limit, results[0].spread_floor, results[0].sharpe);
    println!();
    println!("  Run the best config with:");
    println!("    cargo run -p quantflow-core --example as_backtest --release -- \\");
    println!("      --gamma {:.2} --kappa {:.1} --inv-limit {} --spread-floor {:.1} --auto-sigma",
        results[0].gamma, results[0].kappa, results[0].inv_limit, results[0].spread_floor);
}
