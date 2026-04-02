/// Connects to Binance, streams for 60 seconds, logs statistics.
///
/// Usage: cargo run --example binance_stream_test --release
use quantflow_core::data::binance_ws::{BinanceClient, BinanceMessage};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let client = BinanceClient::new("btcusdt", true); // Futures/Perps
    let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);

    let handle = tokio::spawn(async move {
        if let Err(e) = client.stream(tx).await {
            tracing::error!("WebSocket error: {}", e);
        }
    });

    // Wait up to 10 seconds for the first message — fail fast if unreachable.
    let connection_timeout = tokio::time::Duration::from_secs(10);
    let first_msg = match tokio::time::timeout(connection_timeout, rx.recv()).await {
        Err(_) => {
            handle.abort();
            eprintln!(
                "Could not connect to Binance WebSocket. Check your internet connection \
                 and ensure wss://fstream.binance.com is reachable."
            );
            return;
        }
        Ok(None) => {
            handle.abort();
            return;
        }
        Ok(Some(msg)) => msg,
    };

    let duration = tokio::time::Duration::from_secs(60);
    let start = tokio::time::Instant::now();

    let mut depth_count = 0u64;
    let mut trade_count = 0u64;
    let mut ticker_count = 0u64;
    let mut last_trade_price = 0.0f64;
    let mut last_best_bid = 0.0f64;
    let mut last_best_ask = 0.0f64;

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Binance BTCUSDT Stream Test (60s)           ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("Streaming...\n");

    // Process the first message already received, then loop.
    let mut pending = Some(first_msg);

    while start.elapsed() < duration {
        let msg = if let Some(m) = pending.take() {
            m
        } else {
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(5),
                rx.recv(),
            )
            .await
            {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(_) => {
                    println!("  [TIMEOUT] No message for 5s");
                    continue;
                }
            }
        };

        match msg {
            BinanceMessage::Depth(d) => {
                depth_count += 1;
                if depth_count % 10 == 0 {
                    let bids = d.parsed_bids();
                    let asks = d.parsed_asks();
                    println!(
                        "  [DEPTH #{:>5}] bids: {} levels, asks: {} levels",
                        depth_count,
                        bids.len(),
                        asks.len()
                    );
                }
            }
            BinanceMessage::Trade(t) => {
                trade_count += 1;
                last_trade_price = t.parsed_price();
                let side = if t.is_buy() { "BUY " } else { "SELL" };
                println!(
                    "  [TRADE #{:>5}] {} {:.4} BTC @ {:.2}",
                    trade_count,
                    side,
                    t.parsed_quantity(),
                    last_trade_price
                );
            }
            BinanceMessage::BookTicker(b) => {
                ticker_count += 1;
                last_best_bid = b.best_bid_price.parse().unwrap_or(0.0);
                last_best_ask = b.best_ask_price.parse().unwrap_or(0.0);
            }
        }
    }

    handle.abort();

    let spread = last_best_ask - last_best_bid;
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  RESULTS                                     ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Duration:       {:.1}s", elapsed);
    println!(
        "  Depth updates:  {} ({:.1}/s)",
        depth_count,
        depth_count as f64 / elapsed
    );
    println!(
        "  Trades:         {} ({:.1}/s)",
        trade_count,
        trade_count as f64 / elapsed
    );
    println!(
        "  Book tickers:   {} ({:.1}/s)",
        ticker_count,
        ticker_count as f64 / elapsed
    );
    println!("  Last price:     {:.2}", last_trade_price);
    println!("  Best bid:       {:.2}", last_best_bid);
    println!("  Best ask:       {:.2}", last_best_ask);
    if last_trade_price > 0.0 {
        println!(
            "  Spread:         {:.2} ({:.4}%)",
            spread,
            spread / last_trade_price * 100.0
        );
    }
}
