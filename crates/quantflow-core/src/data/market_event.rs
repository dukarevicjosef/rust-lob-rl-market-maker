use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum MarketEventError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Unknown event type dim: {0}")]
    UnknownEventType(usize),
}

// ── MarketEventType ───────────────────────────────────────────────────────────

/// Unified 12-dimensional event type — identical whether source is
/// Binance, LOBSTER, or the Hawkes simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MarketEventType {
    MarketBuy      = 0,   // Dim  0: aggressive buyer
    MarketSell     = 1,   // Dim  1: aggressive seller
    LimitBuyBest   = 2,   // Dim  2: new limit bid at best
    LimitSellBest  = 3,   // Dim  3: new limit ask at best
    LimitBuyDeep   = 4,   // Dim  4: new limit bid deeper in the book
    LimitSellDeep  = 5,   // Dim  5: new limit ask deeper in the book
    CancelBuyBest  = 6,   // Dim  6: cancel at best bid
    CancelSellBest = 7,   // Dim  7: cancel at best ask
    CancelBuyDeep  = 8,   // Dim  8: cancel bid deeper
    CancelSellDeep = 9,   // Dim  9: cancel ask deeper
    ModifyBuy      = 10,  // Dim 10
    ModifySell     = 11,  // Dim 11
}

impl MarketEventType {
    /// Hawkes process dimension index (0–11).
    #[inline]
    pub fn hawkes_dim(self) -> usize {
        self as usize
    }

    /// Construct from a Hawkes dimension index.
    pub fn from_hawkes_dim(dim: usize) -> Option<Self> {
        match dim {
            0  => Some(Self::MarketBuy),
            1  => Some(Self::MarketSell),
            2  => Some(Self::LimitBuyBest),
            3  => Some(Self::LimitSellBest),
            4  => Some(Self::LimitBuyDeep),
            5  => Some(Self::LimitSellDeep),
            6  => Some(Self::CancelBuyBest),
            7  => Some(Self::CancelSellBest),
            8  => Some(Self::CancelBuyDeep),
            9  => Some(Self::CancelSellDeep),
            10 => Some(Self::ModifyBuy),
            11 => Some(Self::ModifySell),
            _  => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::MarketBuy      => "MarketBuy",
            Self::MarketSell     => "MarketSell",
            Self::LimitBuyBest   => "LimitBuyBest",
            Self::LimitSellBest  => "LimitSellBest",
            Self::LimitBuyDeep   => "LimitBuyDeep",
            Self::LimitSellDeep  => "LimitSellDeep",
            Self::CancelBuyBest  => "CancelBuyBest",
            Self::CancelSellBest => "CancelSellBest",
            Self::CancelBuyDeep  => "CancelBuyDeep",
            Self::CancelSellDeep => "CancelSellDeep",
            Self::ModifyBuy      => "ModifyBuy",
            Self::ModifySell     => "ModifySell",
        }
    }
}

// ── MarketEvent ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MarketEvent {
    /// Seconds since session start (t=0 at first event).
    pub timestamp: f64,
    pub event_type: MarketEventType,
    pub price: f64,
    pub quantity: f64,
    /// Original exchange timestamp in milliseconds since epoch.
    pub raw_timestamp_ms: u64,
}

// ── Schema ────────────────────────────────────────────────────────────────────

/// Arrow schema for MarketEvent. event_type stored as Int32 for Parquet
/// compatibility (Parquet has no native UInt8 logical type).
pub fn market_event_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp",        DataType::Float64, false),
        Field::new("event_type",       DataType::Int32,   false),
        Field::new("price",            DataType::Float64, false),
        Field::new("quantity",         DataType::Float64, false),
        Field::new("raw_timestamp_ms", DataType::UInt64,  false),
    ]))
}

// ── Batch builder ─────────────────────────────────────────────────────────────

pub fn events_to_batch(events: &[MarketEvent]) -> Result<RecordBatch, MarketEventError> {
    let n = events.len();
    let timestamps:    Vec<f64> = events.iter().map(|e| e.timestamp).collect();
    let event_types:   Vec<i32> = events.iter().map(|e| e.event_type.hawkes_dim() as i32).collect();
    let prices:        Vec<f64> = events.iter().map(|e| e.price).collect();
    let quantities:    Vec<f64> = events.iter().map(|e| e.quantity).collect();
    let raw_ts:        Vec<u64> = events.iter().map(|e| e.raw_timestamp_ms).collect();

    let _ = n; // capacity hint used above
    RecordBatch::try_new(
        market_event_schema(),
        vec![
            Arc::new(Float64Array::from(timestamps)) as ArrayRef,
            Arc::new(Int32Array::from(event_types)),
            Arc::new(Float64Array::from(prices)),
            Arc::new(Float64Array::from(quantities)),
            Arc::new(UInt64Array::from(raw_ts)),
        ],
    )
    .map_err(MarketEventError::Arrow)
}

// ── Parquet I/O ───────────────────────────────────────────────────────────────

pub fn events_to_parquet(events: &[MarketEvent], path: &Path) -> Result<(), MarketEventError> {
    let batch = events_to_batch(events)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, market_event_schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub fn events_from_parquet(path: &Path) -> Result<Vec<MarketEvent>, MarketEventError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut events = Vec::new();

    for batch in reader {
        let batch = batch?;
        let timestamps  = batch.column(0).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| MarketEventError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected Float64 for timestamp")))?;
        let event_types = batch.column(1).as_any().downcast_ref::<Int32Array>()
            .ok_or_else(|| MarketEventError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected Int32 for event_type")))?;
        let prices      = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| MarketEventError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected Float64 for price")))?;
        let quantities  = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| MarketEventError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected Float64 for quantity")))?;
        let raw_ts      = batch.column(4).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| MarketEventError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected UInt64 for raw_timestamp_ms")))?;

        for i in 0..batch.num_rows() {
            let dim = event_types.value(i) as usize;
            if let Some(et) = MarketEventType::from_hawkes_dim(dim) {
                events.push(MarketEvent {
                    timestamp:        timestamps.value(i),
                    event_type:       et,
                    price:            prices.value(i),
                    quantity:         quantities.value(i),
                    raw_timestamp_ms: raw_ts.value(i),
                });
            }
        }
    }

    Ok(events)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hawkes_dim_round_trip() {
        for dim in 0..12usize {
            let et = MarketEventType::from_hawkes_dim(dim).unwrap();
            assert_eq!(et.hawkes_dim(), dim);
        }
    }

    #[test]
    fn from_hawkes_dim_out_of_range_returns_none() {
        assert!(MarketEventType::from_hawkes_dim(12).is_none());
        assert!(MarketEventType::from_hawkes_dim(100).is_none());
    }

    #[test]
    fn events_to_batch_schema_matches() {
        let events = vec![MarketEvent {
            timestamp: 1.5,
            event_type: MarketEventType::MarketBuy,
            price: 70_000.0,
            quantity: 0.01,
            raw_timestamp_ms: 1_000,
        }];
        let batch = events_to_batch(&events).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 5);
        let et_col = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(et_col.value(0), 0); // MarketBuy = dim 0
    }

    #[test]
    fn parquet_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("qf_market_events_test.parquet");

        let events: Vec<MarketEvent> = (0..50).map(|i| MarketEvent {
            timestamp:        i as f64 * 0.1,
            event_type:       MarketEventType::from_hawkes_dim(i % 12).unwrap(),
            price:            70_000.0 + i as f64,
            quantity:         0.001 * (i + 1) as f64,
            raw_timestamp_ms: 1_000 + i as u64 * 100,
        }).collect();

        events_to_parquet(&events, &path).unwrap();
        let loaded = events_from_parquet(&path).unwrap();

        assert_eq!(loaded.len(), 50);
        assert_eq!(loaded[0].event_type, MarketEventType::MarketBuy);
        assert!((loaded[10].timestamp - 1.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }
}
