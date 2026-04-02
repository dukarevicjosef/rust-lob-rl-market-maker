/// Converts raw Binance Parquet files (from record_btcusdt) into normalised
/// MarketEvents and LOB snapshots.
///
/// Usage:
///   cargo run --example normalize_btcusdt --release -- \
///     --input data/btcusdt/raw/ \
///     --output data/btcusdt/processed/ \
///     --date 2026-04-02
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, BooleanArray, Float64Array, ListArray, UInt64Array};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use quantflow_core::data::binance::{AggTrade, DepthUpdate, PriceLevel};
use quantflow_core::data::market_event::{MarketEvent, events_to_parquet};
use quantflow_core::data::normalizer::BinanceNormalizer;
use quantflow_core::data::replay::ReplayEngine;
use quantflow_core::data::arrow_export::snapshots_to_parquet;

#[derive(Parser)]
#[command(name = "normalize_btcusdt", about = "Normalise raw Binance Parquet to MarketEvents")]
struct Args {
    /// Directory containing raw Parquet files
    #[arg(long, default_value = "data/btcusdt/raw")]
    input: String,
    /// Output directory for processed files
    #[arg(long, default_value = "data/btcusdt/processed")]
    output: String,
    /// Date string matching the filename prefix (YYYY-MM-DD)
    #[arg(long, default_value_t = chrono::Utc::now().format("%Y-%m-%d").to_string())]
    date: String,
    /// LOB snapshot interval in events
    #[arg(long, default_value_t = 1000usize)]
    snapshot_interval: usize,
}

// ── Raw row types ─────────────────────────────────────────────────────────────

struct RawTrade {
    event_time: u64,
    price:      f64,
    quantity:   f64,
    is_buy:     bool,
}

struct RawDepth {
    event_time: u64,
    bids:       Vec<PriceLevel>,
    asks:       Vec<PriceLevel>,
}

enum RawEvent {
    Trade(RawTrade),
    Depth(RawDepth),
}

impl RawEvent {
    fn event_time(&self) -> u64 {
        match self { RawEvent::Trade(t) => t.event_time, RawEvent::Depth(d) => d.event_time }
    }
}

// ── Parquet readers ───────────────────────────────────────────────────────────

fn read_trades(path: &Path) -> anyhow::Result<Vec<RawTrade>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut trades = Vec::new();

    for batch in reader {
        let batch = batch?;
        let event_times = batch.column(0).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow::anyhow!("expected UInt64 for event_time"))?;
        let prices      = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("expected Float64 for price"))?;
        let quantities  = batch.column(4).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("expected Float64 for quantity"))?;
        let is_buys     = batch.column(5).as_any().downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow::anyhow!("expected Boolean for is_buy"))?;

        for i in 0..batch.num_rows() {
            trades.push(RawTrade {
                event_time: event_times.value(i),
                price:      prices.value(i),
                quantity:   quantities.value(i),
                is_buy:     is_buys.value(i),
            });
        }
    }
    Ok(trades)
}

fn read_depth(path: &Path) -> anyhow::Result<Vec<RawDepth>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut depths = Vec::new();

    for batch in reader {
        let batch = batch?;
        let event_times   = batch.column(0).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow::anyhow!("expected UInt64 for event_time"))?;
        let bid_prices    = batch.column(4).as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow::anyhow!("expected ListArray for bid_prices"))?;
        let bid_quantities = batch.column(5).as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow::anyhow!("expected ListArray for bid_quantities"))?;
        let ask_prices    = batch.column(6).as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow::anyhow!("expected ListArray for ask_prices"))?;
        let ask_quantities = batch.column(7).as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow::anyhow!("expected ListArray for ask_quantities"))?;

        for i in 0..batch.num_rows() {
            let bp = read_f64_list(bid_prices, i);
            let bq = read_f64_list(bid_quantities, i);
            let ap = read_f64_list(ask_prices, i);
            let aq = read_f64_list(ask_quantities, i);

            let bids: Vec<PriceLevel> = bp.iter().zip(bq.iter())
                .map(|(&p, &q)| PriceLevel { price: p, quantity: q })
                .collect();
            let asks: Vec<PriceLevel> = ap.iter().zip(aq.iter())
                .map(|(&p, &q)| PriceLevel { price: p, quantity: q })
                .collect();

            depths.push(RawDepth { event_time: event_times.value(i), bids, asks });
        }
    }
    Ok(depths)
}

fn read_f64_list(array: &ListArray, row: usize) -> Vec<f64> {
    let values = array.value(row);
    if let Some(fa) = values.as_any().downcast_ref::<Float64Array>() {
        (0..fa.len()).map(|j| fa.value(j)).collect()
    } else {
        Vec::new()
    }
}

// ── Fake AggTrade/DepthUpdate constructors ────────────────────────────────────

fn raw_trade_to_event(t: &RawTrade, norm: &mut BinanceNormalizer) -> MarketEvent {
    // Build a minimal AggTrade from the raw row
    let trade: AggTrade = serde_json::from_value(serde_json::json!({
        "e": "aggTrade", "E": t.event_time, "s": "BTCUSDT",
        "a": 0_u64, "p": t.price.to_string(), "q": t.quantity.to_string(),
        "f": 0_u64, "l": 0_u64, "T": t.event_time,
        "m": !t.is_buy  // is_buyer_maker = !is_buy
    }))
    .expect("valid AggTrade JSON");
    norm.normalize_trade(&trade)
}

fn raw_depth_to_events(d: &RawDepth, norm: &mut BinanceNormalizer) -> Vec<MarketEvent> {
    let bid_pairs: Vec<[String; 2]> = d.bids.iter()
        .map(|l| [l.price.to_string(), l.quantity.to_string()])
        .collect();
    let ask_pairs: Vec<[String; 2]> = d.asks.iter()
        .map(|l| [l.price.to_string(), l.quantity.to_string()])
        .collect();
    let depth: DepthUpdate = serde_json::from_value(serde_json::json!({
        "e": "depthUpdate", "E": d.event_time, "s": "BTCUSDT",
        "U": 0_u64, "u": 0_u64, "b": bid_pairs, "a": ask_pairs
    }))
    .expect("valid DepthUpdate JSON");
    norm.normalize_depth(&depth)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input = Path::new(&args.input);
    let output = Path::new(&args.output);
    std::fs::create_dir_all(output)?;

    let trades_path: PathBuf = input.join(format!("{}_trades.parquet",  args.date));
    let depth_path:  PathBuf = input.join(format!("{}_depth.parquet",   args.date));

    if !trades_path.exists() && !depth_path.exists() {
        eprintln!("No input files found. Run record_btcusdt first.");
        eprintln!("  Expected: {}", trades_path.display());
        return Ok(());
    }

    let raw_trades = if trades_path.exists() {
        println!("Reading {}…", trades_path.display());
        read_trades(&trades_path)?
    } else { Vec::new() };

    let raw_depths = if depth_path.exists() {
        println!("Reading {}…", depth_path.display());
        read_depth(&depth_path)?
    } else { Vec::new() };

    // Merge by event_time (stable sort, trades first on tie)
    let mut raw_events: Vec<RawEvent> = raw_trades.into_iter().map(RawEvent::Trade)
        .chain(raw_depths.into_iter().map(RawEvent::Depth))
        .collect();
    raw_events.sort_by_key(|e| e.event_time());

    println!("Processing {} raw events…", raw_events.len());

    // Normalize
    let mut norm = BinanceNormalizer::new();
    let mut market_events: Vec<MarketEvent> = Vec::with_capacity(raw_events.len() * 3);
    let mut trade_in = 0u64;
    let mut depth_in = 0u64;

    for raw in &raw_events {
        match raw {
            RawEvent::Trade(t) => {
                trade_in += 1;
                market_events.push(raw_trade_to_event(t, &mut norm));
            }
            RawEvent::Depth(d) => {
                depth_in += 1;
                let evs = raw_depth_to_events(d, &mut norm);
                market_events.extend(evs);
            }
        }
    }

    // Write events parquet
    let events_path = output.join(format!("{}_events.parquet", args.date));
    println!("Writing {} events to {}…", market_events.len(), events_path.display());
    events_to_parquet(&market_events, &events_path)?;

    // Generate LOB snapshots by replaying events through the LOB engine
    let mut replay = ReplayEngine::from_events(market_events.clone());
    let mut snapshots = Vec::new();
    let mut i = 0usize;
    while let Some(_event) = replay.next_event() {
        i += 1;
        if i % args.snapshot_interval == 0 {
            snapshots.push(replay.snapshot(10));
        }
    }
    let snap_count = snapshots.len();
    let snap_path = output.join(format!("{}_snapshots.parquet", args.date));
    if !snapshots.is_empty() {
        snapshots_to_parquet(&snapshots, &snap_path)?;
        println!("Wrote {} snapshots to {}.", snap_count, snap_path.display());
    }

    // Statistics
    let mut counts = [0u64; 12];
    for e in &market_events { counts[e.event_type.hawkes_dim()] += 1; }
    let total = market_events.len() as f64;
    let time_span = market_events.last().map(|e| e.timestamp).unwrap_or(0.0);

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  NORMALIZATION COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Input:");
    println!("    Trades:        {:>8}", trade_in);
    println!("    Depth updates: {:>8}", depth_in);
    println!("\n  Output:");
    println!("    Market events: {:>8}", total as u64);
    use quantflow_core::data::market_event::MarketEventType::*;
    for (et, label) in [
        (MarketBuy, "Market Buy"), (MarketSell, "Market Sell"),
        (LimitBuyBest, "Limit Buy Best"), (LimitSellBest, "Limit Sell Best"),
        (LimitBuyDeep, "Limit Buy Deep"), (LimitSellDeep, "Limit Sell Deep"),
        (CancelBuyBest, "Cancel Buy Best"), (CancelSellBest, "Cancel Sell Best"),
        (CancelBuyDeep, "Cancel Buy Deep"), (CancelSellDeep, "Cancel Sell Deep"),
        (ModifyBuy, "Modify Buy"), (ModifySell, "Modify Sell"),
    ] {
        let c = counts[et.hawkes_dim()];
        println!("      {:20} {:>8}  ({:.1}%)", label, c, c as f64 / total * 100.0);
    }
    println!("\n    LOB Snapshots: {:>8}", snap_count);
    println!("    Time span:     {:.1}s", time_span);
    if time_span > 0.0 { println!("    Events/sec:    {:.1}", total / time_span); }

    println!("\n  Files:");
    for path in [&events_path, &snap_path] {
        if path.exists() {
            let size = std::fs::metadata(path)?.len();
            println!("    {} ({:.1} MB)", path.display(), size as f64 / 1_000_000.0);
        }
    }

    Ok(())
}
