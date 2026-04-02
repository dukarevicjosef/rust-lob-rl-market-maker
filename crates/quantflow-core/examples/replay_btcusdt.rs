/// Replays recorded BTCUSDT market data through the LOB engine.
///
/// Usage:
///   cargo run --example replay_btcusdt --release -- \
///     --input data/btcusdt/processed/2026-04-02_events.parquet
use clap::Parser;

use quantflow_core::data::replay::ReplayEngine;

#[derive(Parser)]
#[command(name = "replay_btcusdt", about = "Replay normalised events through the LOB engine")]
struct Args {
    /// Path to the _events.parquet file
    #[arg(long)]
    input: String,
    /// Log mid-price every N events
    #[arg(long, default_value_t = 1000usize)]
    log_interval: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let path = std::path::Path::new(&args.input);

    if !path.exists() {
        eprintln!("File not found: {}", path.display());
        eprintln!("Run normalize_btcusdt first.");
        return Ok(());
    }

    println!("Loading events from {}…", path.display());
    let mut replay = ReplayEngine::from_parquet(path)?;
    let total = replay.len();
    println!("Loaded {} events.\n", total);

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Replaying through LOB engine                        ║");
    println!("╚══════════════════════════════════════════════════════╝");

    let mut spreads = Vec::with_capacity(total / args.log_interval + 1);
    let mut mid_prices = Vec::with_capacity(total / args.log_interval + 1);
    let mut i = 0usize;

    loop {
        let event_ts = {
            let e = replay.next_event();
            match e { None => break, Some(ev) => ev.timestamp }
        };
        i += 1;
        if i % args.log_interval == 0 {
            let mid    = replay.mid_price();
            let bid    = replay.best_bid();
            let ask    = replay.best_ask();
            let spread = bid.zip(ask).map(|(b, a)| a - b);
            let snap   = replay.snapshot(1);

            if let Some(m) = mid {
                println!(
                    "  [{:>7}] t={:>8.1}s  mid={:>10.2}  spread={:>6.4}  depth_bid={}  depth_ask={}",
                    i, event_ts, m,
                    spread.unwrap_or(0.0),
                    snap.bids.len(),
                    snap.asks.len(),
                );
                mid_prices.push(m);
                if let Some(s) = spread { spreads.push(s); }
            }
        }
    }

    // Final statistics
    let snap = replay.snapshot(10);
    let final_mid = replay.mid_price();

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  REPLAY COMPLETE                                     ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Events replayed:  {}", total);
    println!("  Final mid-price:  {}", final_mid.map(|m| format!("{:.4}", m)).unwrap_or("N/A".to_string()));
    println!("  Final bid levels: {}", snap.bids.len());
    println!("  Final ask levels: {}", snap.asks.len());

    if !spreads.is_empty() {
        let avg_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let min_spread = spreads.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_spread = spreads.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("\n  Spread statistics (sampled every {} events):", args.log_interval);
        println!("    avg: {:.6}", avg_spread);
        println!("    min: {:.6}", min_spread);
        println!("    max: {:.6}", max_spread);
    }

    if !mid_prices.is_empty() {
        let price_min = mid_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let price_max = mid_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("\n  Price range:  {:.2} – {:.2}", price_min, price_max);
    }

    println!("\n  Final LOB (top 5):");
    let snap5 = replay.snapshot(5);
    for (ask, bid) in snap5.asks.iter().rev().zip(snap5.bids.iter()) {
        println!("    {:>12.4} | {:<12.4}", ask.0.to_f64(), bid.0.to_f64());
    }

    Ok(())
}
