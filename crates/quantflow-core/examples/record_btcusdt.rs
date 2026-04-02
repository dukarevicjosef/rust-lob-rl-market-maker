/// Records Binance BTCUSDT market data to Parquet files.
///
/// Usage:
///   cargo run --example record_btcusdt --release
///   cargo run --example record_btcusdt --release -- --duration 6h --output data/btcusdt
///   cargo run --example record_btcusdt --release -- --duration 30m --symbol ethusdt --no-futures

use std::path::Path;

use clap::Parser;
use quantflow_core::data::binance_ws::{BinanceClient, BinanceMessage};
use quantflow_core::data::recorder::ParquetRecorder;

#[derive(Parser)]
#[command(name = "record_btcusdt", about = "Record Binance market data to Parquet")]
struct Args {
    /// Recording duration: "30m", "1h", "6h", "24h", or raw seconds
    #[arg(long, default_value = "1h")]
    duration: String,

    /// Output directory root
    #[arg(long, default_value = "data")]
    output: String,

    /// Symbol to record (lowercased automatically)
    #[arg(long, default_value = "btcusdt")]
    symbol: String,

    /// Use Futures/Perp stream instead of Spot
    #[arg(long, default_value_t = true)]
    futures: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let duration = parse_duration(&args.duration)?;
    let symbol = args.symbol.to_lowercase();

    let mut recorder = ParquetRecorder::new(Path::new(&args.output), &symbol)?;

    let client = BinanceClient::new(&symbol, args.futures);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<BinanceMessage>(50_000);

    // WebSocket reader with auto-reconnect (Binance disconnects after 24h).
    let ws_handle = tokio::spawn(async move {
        loop {
            if let Err(e) = client.stream(tx.clone()).await {
                tracing::warn!("WebSocket disconnected: {}. Reconnecting in 5s...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    });

    // Graceful Ctrl+C shutdown.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        shutdown_tx.send(()).ok();
    });

    // Check connectivity: require first message within 10s.
    let first = match tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        rx.recv(),
    )
    .await
    {
        Err(_) => {
            ws_handle.abort();
            eprintln!(
                "Could not connect to Binance WebSocket. Check your internet connection \
                 and ensure wss://fstream.binance.com is reachable."
            );
            return Ok(());
        }
        Ok(None) => {
            ws_handle.abort();
            return Ok(());
        }
        Ok(Some(msg)) => msg,
    };

    let start = tokio::time::Instant::now();
    let deadline = start + duration;
    let sleep = tokio::time::sleep_until(deadline);
    tokio::pin!(sleep);

    let stream_type = if args.futures { "FUTURES" } else { "SPOT" };
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Recording Binance {} {} ({})", symbol.to_uppercase(), stream_type, args.duration);
    println!("║  Output: {}/{}/raw/", args.output, symbol);
    println!("║  Press Ctrl+C to stop early                          ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Process the first message received before the main loop.
    recorder.record(first)?;

    let mut last_progress = tokio::time::Instant::now();

    loop {
        tokio::select! {
            _ = &mut sleep => {
                println!("\n  Duration reached. Stopping...");
                break;
            }
            _ = &mut shutdown_rx => {
                println!("\n  Ctrl+C received. Stopping...");
                break;
            }
            Some(msg) = rx.recv() => {
                recorder.record(msg)?;

                if last_progress.elapsed() > tokio::time::Duration::from_secs(60) {
                    let elapsed_s = start.elapsed().as_secs().max(1);
                    let depth = recorder.stats.depth_count;
                    let trades = recorder.stats.trade_count;
                    let tickers = recorder.stats.ticker_count;
                    println!(
                        "  [{:>4}m] depth: {} ({}/s)  trades: {} ({}/s)  tickers: {}",
                        elapsed_s / 60,
                        depth, depth / elapsed_s,
                        trades, trades / elapsed_s,
                        tickers,
                    );
                    last_progress = tokio::time::Instant::now();
                }
            }
        }
    }

    ws_handle.abort();
    let stats = recorder.finish()?;

    let output_dir = Path::new(&args.output).join(&symbol).join("raw");
    let elapsed = start.elapsed();

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  RECORDING COMPLETE                                  ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Duration:        {}m {}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60);
    println!("  Depth updates:   {}", stats.depth_count);
    println!("  Trades:          {}", stats.trade_count);
    println!("  Book tickers:    {}", stats.ticker_count);
    println!("  Total written:   {:.1} MB", stats.bytes_written as f64 / 1_000_000.0);
    println!("  Output dir:      {}", output_dir.display());

    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            println!(
                "    {} ({:.1} MB)",
                entry.file_name().to_string_lossy(),
                size as f64 / 1_000_000.0
            );
        }
    }

    Ok(())
}

fn parse_duration(s: &str) -> Result<tokio::time::Duration, Box<dyn std::error::Error>> {
    let s = s.trim();
    if let Some(mins) = s.strip_suffix('m') {
        Ok(tokio::time::Duration::from_secs(mins.parse::<u64>()? * 60))
    } else if let Some(hours) = s.strip_suffix('h') {
        Ok(tokio::time::Duration::from_secs(hours.parse::<u64>()? * 3600))
    } else {
        Ok(tokio::time::Duration::from_secs(s.parse()?))
    }
}
