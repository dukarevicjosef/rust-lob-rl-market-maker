//! Arrow and Parquet serialization for order book data.
//!
//! The snapshot schema is **row-per-level** (denormalized): one `RecordBatch`
//! row = one price level within one snapshot. This layout is optimal for
//! columnar analytics over depth-at-level queries.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::errors::ParquetError;
use thiserror::Error;

use crate::orderbook::{OrderBookSnapshot, Trade};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("Parquet error: {0}")]
    Parquet(#[from] ParquetError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Schemas ───────────────────────────────────────────────────────────────────

/// Schema for the order book snapshot batch.
///
/// ```text
/// timestamp  UInt64   nanoseconds since epoch (from OrderBookSnapshot)
/// level      Int32    1-based depth level
/// bid_price  Int64    integer ticks (PRICE_SCALE = 10^8)
/// bid_qty    UInt64   lots
/// ask_price  Int64    integer ticks
/// ask_qty    UInt64   lots
/// ```
pub fn snapshot_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("level", DataType::Int32, false),
        Field::new("bid_price", DataType::Int64, false),
        Field::new("bid_qty", DataType::UInt64, false),
        Field::new("ask_price", DataType::Int64, false),
        Field::new("ask_qty", DataType::UInt64, false),
    ]))
}

/// Schema for the trade history batch.
///
/// ```text
/// timestamp  UInt64   nanoseconds
/// price      Int64    ticks
/// quantity   UInt64   lots
/// maker_id   UInt64
/// taker_id   UInt64
/// ```
pub fn trade_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("price", DataType::Int64, false),
        Field::new("quantity", DataType::UInt64, false),
        Field::new("maker_id", DataType::UInt64, false),
        Field::new("taker_id", DataType::UInt64, false),
    ]))
}

// ── RecordBatch builders ──────────────────────────────────────────────────────

/// Convert a slice of `OrderBookSnapshot`s into one Arrow `RecordBatch`.
///
/// Each snapshot contributes `min(bids.len(), asks.len())` rows. Levels
/// beyond the shorter side are dropped so the row is always fully populated.
/// The inner price/qty data is copied from the snapshot vecs — the primitive
/// buffers are Arrow-native `i64`/`u64`, so no further transformation is
/// needed after extraction.
pub fn snapshots_to_record_batch(
    snapshots: &[OrderBookSnapshot],
) -> Result<RecordBatch, ArrowError> {
    let schema = snapshot_schema();
    let n_rows: usize = snapshots.iter().map(|s| s.bids.len().min(s.asks.len())).sum();

    let mut timestamps = Vec::with_capacity(n_rows);
    let mut levels = Vec::<i32>::with_capacity(n_rows);
    let mut bid_prices = Vec::<i64>::with_capacity(n_rows);
    let mut bid_qtys = Vec::<u64>::with_capacity(n_rows);
    let mut ask_prices = Vec::<i64>::with_capacity(n_rows);
    let mut ask_qtys = Vec::<u64>::with_capacity(n_rows);

    for snap in snapshots {
        let depth = snap.bids.len().min(snap.asks.len());
        for i in 0..depth {
            timestamps.push(snap.timestamp.0);
            levels.push((i + 1) as i32);
            bid_prices.push(snap.bids[i].0 .0);
            bid_qtys.push(snap.bids[i].1 .0);
            ask_prices.push(snap.asks[i].0 .0);
            ask_qtys.push(snap.asks[i].1 .0);
        }
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(Int32Array::from(levels)),
            Arc::new(Int64Array::from(bid_prices)),
            Arc::new(UInt64Array::from(bid_qtys)),
            Arc::new(Int64Array::from(ask_prices)),
            Arc::new(UInt64Array::from(ask_qtys)),
        ],
    )
}

/// Convert a slice of `Trade`s into one Arrow `RecordBatch`.
pub fn trades_to_record_batch(trades: &[Trade]) -> Result<RecordBatch, ArrowError> {
    let schema = trade_schema();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.timestamp.0))),
            Arc::new(Int64Array::from_iter_values(trades.iter().map(|t| t.price.0))),
            Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.quantity.0))),
            Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.maker_id.0))),
            Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.taker_id.0))),
        ],
    )
}

// ── Parquet export ────────────────────────────────────────────────────────────

/// Write snapshots to a Parquet file at `path`.
pub fn snapshots_to_parquet(
    snapshots: &[OrderBookSnapshot],
    path: &Path,
) -> Result<(), ExportError> {
    let batch = snapshots_to_record_batch(snapshots)?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write trades to a Parquet file at `path`.
pub fn trades_to_parquet(trades: &[Trade], path: &Path) -> Result<(), ExportError> {
    let batch = trades_to_record_batch(trades)?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::{OrderId, Price, Quantity, Timestamp, Trade};
    use crate::orderbook::book::OrderBookSnapshot;

    fn make_snapshot(ts: u64, bids: &[(f64, u64)], asks: &[(f64, u64)]) -> OrderBookSnapshot {
        OrderBookSnapshot {
            timestamp: Timestamp(ts),
            bids: bids.iter().map(|(p, q)| (Price::from_f64(*p), Quantity(*q))).collect(),
            asks: asks.iter().map(|(p, q)| (Price::from_f64(*p), Quantity(*q))).collect(),
        }
    }

    fn make_trade(ts: u64, price: f64, qty: u64, maker: u64, taker: u64) -> Trade {
        Trade {
            timestamp: Timestamp(ts),
            price: Price::from_f64(price),
            quantity: Quantity(qty),
            maker_id: OrderId(maker),
            taker_id: OrderId(taker),
        }
    }

    // ── Schema checks ────────────────────────────────────────────────────────

    #[test]
    fn snapshot_schema_has_six_fields() {
        let s = snapshot_schema();
        assert_eq!(s.fields().len(), 6);
        assert_eq!(s.field(0).name(), "timestamp");
        assert_eq!(s.field(1).name(), "level");
        assert_eq!(s.field(2).name(), "bid_price");
    }

    #[test]
    fn trade_schema_has_five_fields() {
        let s = trade_schema();
        assert_eq!(s.fields().len(), 5);
        assert_eq!(s.field(0).name(), "timestamp");
        assert_eq!(s.field(4).name(), "taker_id");
    }

    // ── RecordBatch content ──────────────────────────────────────────────────

    #[test]
    fn snapshots_to_batch_row_count() {
        let snaps = vec![
            make_snapshot(1_000, &[(99.0, 10), (98.0, 20)], &[(101.0, 5), (102.0, 15)]),
            make_snapshot(2_000, &[(99.5, 8)], &[(100.5, 12)]),
        ];
        let batch = snapshots_to_record_batch(&snaps).unwrap();
        // First snapshot: min(2,2) = 2 rows; second: min(1,1) = 1 row → total 3
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn snapshots_to_batch_values_correct() {
        let snaps = vec![make_snapshot(
            1_234_000_000,
            &[(99.0, 10), (98.0, 20)],
            &[(101.0, 5), (102.0, 15)],
        )];
        let batch = snapshots_to_record_batch(&snaps).unwrap();

        let ts_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(ts_col.value(0), 1_234_000_000);
        assert_eq!(ts_col.value(1), 1_234_000_000);

        let level_col = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(level_col.value(0), 1);
        assert_eq!(level_col.value(1), 2);

        let bid_px = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(bid_px.value(0), Price::from_f64(99.0).0);
        assert_eq!(bid_px.value(1), Price::from_f64(98.0).0);

        let ask_px = batch.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ask_px.value(0), Price::from_f64(101.0).0);
    }

    #[test]
    fn snapshots_to_batch_truncates_to_shorter_side() {
        // bids has 3 levels, asks has 2 → only 2 rows produced
        let snaps = vec![make_snapshot(
            0,
            &[(99.0, 1), (98.0, 1), (97.0, 1)],
            &[(101.0, 1), (102.0, 1)],
        )];
        let batch = snapshots_to_record_batch(&snaps).unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn empty_snapshots_produces_empty_batch() {
        let batch = snapshots_to_record_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn trades_to_batch_values_correct() {
        let trades = vec![
            make_trade(1_000, 100.0, 50, 1, 2),
            make_trade(2_000, 100.5, 30, 3, 4),
        ];
        let batch = trades_to_record_batch(&trades).unwrap();
        assert_eq!(batch.num_rows(), 2);

        let qty_col = batch.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(qty_col.value(0), 50);
        assert_eq!(qty_col.value(1), 30);

        let maker_col = batch.column(3).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(maker_col.value(0), 1);
        assert_eq!(maker_col.value(1), 3);
    }

    #[test]
    fn empty_trades_produces_empty_batch() {
        let batch = trades_to_record_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 5);
    }

    // ── Parquet round-trip ───────────────────────────────────────────────────

    #[test]
    fn snapshots_parquet_write_produces_nonempty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("qf_test_snapshots.parquet");

        let snaps = vec![make_snapshot(
            0,
            &[(99.0, 100), (98.0, 200)],
            &[(101.0, 50), (102.0, 80)],
        )];
        snapshots_to_parquet(&snaps, &path).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 0, "parquet file must not be empty");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn trades_parquet_write_produces_nonempty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("qf_test_trades.parquet");

        let trades = vec![make_trade(1_000, 100.0, 10, 1, 2)];
        trades_to_parquet(&trades, &path).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 0);
        let _ = std::fs::remove_file(&path);
    }
}
