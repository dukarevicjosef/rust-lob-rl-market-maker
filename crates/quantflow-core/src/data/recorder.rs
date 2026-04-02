use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Float64Builder, ListBuilder, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use super::binance::{AggTrade, BookTicker, DepthUpdate};
use super::binance_ws::BinanceMessage;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Schemas ───────────────────────────────────────────────────────────────────

fn depth_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("event_time", DataType::UInt64, false),
        Field::new("receive_time", DataType::UInt64, false),
        Field::new("first_update_id", DataType::UInt64, false),
        Field::new("last_update_id", DataType::UInt64, false),
        Field::new(
            "bid_prices",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "bid_quantities",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "ask_prices",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "ask_quantities",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ]))
}

fn trade_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("event_time", DataType::UInt64, false),
        Field::new("receive_time", DataType::UInt64, false),
        Field::new("agg_trade_id", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("is_buy", DataType::Boolean, false),
        Field::new("first_trade_id", DataType::UInt64, false),
        Field::new("last_trade_id", DataType::UInt64, false),
    ]))
}

fn bookticker_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("receive_time", DataType::UInt64, false),
        Field::new("update_id", DataType::UInt64, false),
        Field::new("best_bid_price", DataType::Float64, false),
        Field::new("best_bid_qty", DataType::Float64, false),
        Field::new("best_ask_price", DataType::Float64, false),
        Field::new("best_ask_qty", DataType::Float64, false),
    ]))
}

// ── BatchBuffer ───────────────────────────────────────────────────────────────

pub struct BatchBuffer<T> {
    items: Vec<T>,
    batch_size: usize,
}

impl<T> BatchBuffer<T> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            items: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Push an item. Returns a full batch for flushing when the threshold is reached.
    pub fn push(&mut self, item: T) -> Option<Vec<T>> {
        self.items.push(item);
        if self.items.len() >= self.batch_size {
            Some(std::mem::take(&mut self.items))
        } else {
            None
        }
    }

    /// Drain all buffered items regardless of threshold.
    pub fn flush(&mut self) -> Vec<T> {
        std::mem::take(&mut self.items)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── RecorderStats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RecorderStats {
    pub depth_count: u64,
    pub trade_count: u64,
    pub ticker_count: u64,
    pub start_time: Instant,
    pub bytes_written: u64,
}

impl RecorderStats {
    fn new() -> Self {
        Self {
            depth_count: 0,
            trade_count: 0,
            ticker_count: 0,
            start_time: Instant::now(),
            bytes_written: 0,
        }
    }
}

// ── ParquetRecorder ───────────────────────────────────────────────────────────

pub struct ParquetRecorder {
    depth_path: PathBuf,
    trade_path: PathBuf,
    ticker_path: PathBuf,
    depth_buffer: BatchBuffer<(u64, DepthUpdate)>,
    trade_buffer: BatchBuffer<(u64, AggTrade)>,
    ticker_buffer: BatchBuffer<(u64, BookTicker)>,
    depth_writer: Option<ArrowWriter<File>>,
    trade_writer: Option<ArrowWriter<File>>,
    ticker_writer: Option<ArrowWriter<File>>,
    pub stats: RecorderStats,
}

impl ParquetRecorder {
    pub fn new(output_dir: &Path, symbol: &str) -> Result<Self, RecorderError> {
        let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let dir = output_dir.join(symbol).join("raw");
        std::fs::create_dir_all(&dir)?;

        let depth_path = dir.join(format!("{}_depth.parquet", date));
        let trade_path = dir.join(format!("{}_trades.parquet", date));
        let ticker_path = dir.join(format!("{}_bookticker.parquet", date));

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let depth_writer = ArrowWriter::try_new(
            File::create(&depth_path)?,
            depth_schema(),
            Some(props.clone()),
        )?;
        let trade_writer = ArrowWriter::try_new(
            File::create(&trade_path)?,
            trade_schema(),
            Some(props.clone()),
        )?;
        let ticker_writer = ArrowWriter::try_new(
            File::create(&ticker_path)?,
            bookticker_schema(),
            Some(props),
        )?;

        Ok(Self {
            depth_path,
            trade_path,
            ticker_path,
            depth_buffer: BatchBuffer::new(1_000),
            trade_buffer: BatchBuffer::new(5_000),
            ticker_buffer: BatchBuffer::new(10_000),
            depth_writer: Some(depth_writer),
            trade_writer: Some(trade_writer),
            ticker_writer: Some(ticker_writer),
            stats: RecorderStats::new(),
        })
    }

    /// Process one `BinanceMessage`, buffering and flushing as needed.
    pub fn record(&mut self, msg: BinanceMessage) -> Result<(), RecorderError> {
        let recv_time = chrono::Utc::now().timestamp_millis() as u64;
        match msg {
            BinanceMessage::Depth(d) => {
                self.stats.depth_count += 1;
                if let Some(batch) = self.depth_buffer.push((recv_time, d)) {
                    self.flush_depth(batch)?;
                }
            }
            BinanceMessage::Trade(t) => {
                self.stats.trade_count += 1;
                if let Some(batch) = self.trade_buffer.push((recv_time, t)) {
                    self.flush_trades(batch)?;
                }
            }
            BinanceMessage::BookTicker(b) => {
                self.stats.ticker_count += 1;
                if let Some(batch) = self.ticker_buffer.push((recv_time, b)) {
                    self.flush_tickers(batch)?;
                }
            }
        }
        Ok(())
    }

    /// Flush all remaining buffers and close the Parquet files.
    pub fn finish(&mut self) -> Result<RecorderStats, RecorderError> {
        let depth_rem = self.depth_buffer.flush();
        if !depth_rem.is_empty() {
            self.flush_depth(depth_rem)?;
        }
        let trade_rem = self.trade_buffer.flush();
        if !trade_rem.is_empty() {
            self.flush_trades(trade_rem)?;
        }
        let ticker_rem = self.ticker_buffer.flush();
        if !ticker_rem.is_empty() {
            self.flush_tickers(ticker_rem)?;
        }

        if let Some(w) = self.depth_writer.take() {
            w.close()?;
        }
        if let Some(w) = self.trade_writer.take() {
            w.close()?;
        }
        if let Some(w) = self.ticker_writer.take() {
            w.close()?;
        }

        self.stats.bytes_written =
            std::fs::metadata(&self.depth_path).map(|m| m.len()).unwrap_or(0)
                + std::fs::metadata(&self.trade_path).map(|m| m.len()).unwrap_or(0)
                + std::fs::metadata(&self.ticker_path).map(|m| m.len()).unwrap_or(0);

        Ok(self.stats.clone())
    }

    // ── Private flush helpers ─────────────────────────────────────────────────

    fn flush_depth(&mut self, items: Vec<(u64, DepthUpdate)>) -> Result<(), RecorderError> {
        let batch = Self::depth_to_batch(&items)?;
        if let Some(w) = &mut self.depth_writer {
            w.write(&batch)?;
        }
        Ok(())
    }

    fn flush_trades(&mut self, items: Vec<(u64, AggTrade)>) -> Result<(), RecorderError> {
        let batch = Self::trade_to_batch(&items)?;
        if let Some(w) = &mut self.trade_writer {
            w.write(&batch)?;
        }
        Ok(())
    }

    fn flush_tickers(&mut self, items: Vec<(u64, BookTicker)>) -> Result<(), RecorderError> {
        let batch = Self::ticker_to_batch(&items)?;
        if let Some(w) = &mut self.ticker_writer {
            w.write(&batch)?;
        }
        Ok(())
    }

    // ── Private batch builders ────────────────────────────────────────────────

    pub fn depth_to_batch(items: &[(u64, DepthUpdate)]) -> Result<RecordBatch, RecorderError> {
        let n = items.len();
        let mut event_times = Vec::with_capacity(n);
        let mut receive_times = Vec::with_capacity(n);
        let mut first_ids = Vec::with_capacity(n);
        let mut last_ids = Vec::with_capacity(n);

        let mut bp_builder = ListBuilder::new(Float64Builder::new());
        let mut bq_builder = ListBuilder::new(Float64Builder::new());
        let mut ap_builder = ListBuilder::new(Float64Builder::new());
        let mut aq_builder = ListBuilder::new(Float64Builder::new());

        for (recv, d) in items {
            event_times.push(d.event_time);
            receive_times.push(*recv);
            first_ids.push(d.first_update_id);
            last_ids.push(d.last_update_id);

            let bids = d.parsed_bids();
            let asks = d.parsed_asks();

            for lvl in bids.iter().take(10) {
                bp_builder.values().append_value(lvl.price);
            }
            bp_builder.append(true);
            for lvl in bids.iter().take(10) {
                bq_builder.values().append_value(lvl.quantity);
            }
            bq_builder.append(true);
            for lvl in asks.iter().take(10) {
                ap_builder.values().append_value(lvl.price);
            }
            ap_builder.append(true);
            for lvl in asks.iter().take(10) {
                aq_builder.values().append_value(lvl.quantity);
            }
            aq_builder.append(true);
        }

        RecordBatch::try_new(
            depth_schema(),
            vec![
                Arc::new(UInt64Array::from(event_times)) as ArrayRef,
                Arc::new(UInt64Array::from(receive_times)),
                Arc::new(UInt64Array::from(first_ids)),
                Arc::new(UInt64Array::from(last_ids)),
                Arc::new(bp_builder.finish()),
                Arc::new(bq_builder.finish()),
                Arc::new(ap_builder.finish()),
                Arc::new(aq_builder.finish()),
            ],
        )
        .map_err(RecorderError::Arrow)
    }

    pub fn trade_to_batch(items: &[(u64, AggTrade)]) -> Result<RecordBatch, RecorderError> {
        let n = items.len();
        let mut event_times = Vec::with_capacity(n);
        let mut receive_times = Vec::with_capacity(n);
        let mut trade_ids = Vec::with_capacity(n);
        let mut prices = Vec::with_capacity(n);
        let mut quantities = Vec::with_capacity(n);
        let mut is_buy = Vec::with_capacity(n);
        let mut first_ids = Vec::with_capacity(n);
        let mut last_ids = Vec::with_capacity(n);

        for (recv, t) in items {
            event_times.push(t.event_time);
            receive_times.push(*recv);
            trade_ids.push(t.agg_trade_id);
            prices.push(t.parsed_price());
            quantities.push(t.parsed_quantity());
            is_buy.push(t.is_buy());
            first_ids.push(t.first_trade_id);
            last_ids.push(t.last_trade_id);
        }

        RecordBatch::try_new(
            trade_schema(),
            vec![
                Arc::new(UInt64Array::from(event_times)) as ArrayRef,
                Arc::new(UInt64Array::from(receive_times)),
                Arc::new(UInt64Array::from(trade_ids)),
                Arc::new(Float64Array::from(prices)),
                Arc::new(Float64Array::from(quantities)),
                Arc::new(BooleanArray::from(is_buy)),
                Arc::new(UInt64Array::from(first_ids)),
                Arc::new(UInt64Array::from(last_ids)),
            ],
        )
        .map_err(RecorderError::Arrow)
    }

    pub fn ticker_to_batch(items: &[(u64, BookTicker)]) -> Result<RecordBatch, RecorderError> {
        let n = items.len();
        let mut receive_times = Vec::with_capacity(n);
        let mut update_ids = Vec::with_capacity(n);
        let mut bid_prices = Vec::with_capacity(n);
        let mut bid_qtys = Vec::with_capacity(n);
        let mut ask_prices = Vec::with_capacity(n);
        let mut ask_qtys = Vec::with_capacity(n);

        for (recv, b) in items {
            receive_times.push(*recv);
            update_ids.push(b.update_id);
            bid_prices.push(b.best_bid_price.parse::<f64>().unwrap_or(0.0));
            bid_qtys.push(b.best_bid_qty.parse::<f64>().unwrap_or(0.0));
            ask_prices.push(b.best_ask_price.parse::<f64>().unwrap_or(0.0));
            ask_qtys.push(b.best_ask_qty.parse::<f64>().unwrap_or(0.0));
        }

        RecordBatch::try_new(
            bookticker_schema(),
            vec![
                Arc::new(UInt64Array::from(receive_times)) as ArrayRef,
                Arc::new(UInt64Array::from(update_ids)),
                Arc::new(Float64Array::from(bid_prices)),
                Arc::new(Float64Array::from(bid_qtys)),
                Arc::new(Float64Array::from(ask_prices)),
                Arc::new(Float64Array::from(ask_qtys)),
            ],
        )
        .map_err(RecorderError::Arrow)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binance::{AggTrade, BookTicker, DepthUpdate};

    fn make_depth(event_time: u64) -> DepthUpdate {
        serde_json::from_str(&format!(
            r#"{{"e":"depthUpdate","E":{event_time},"s":"BTCUSDT","U":1,"u":2,
               "b":[["70100.00","0.5"],["70099.00","1.2"],["70098.00","0.8"]],"a":[["70101.00","0.3"]]}}"#
        ))
        .unwrap()
    }

    fn make_trade(event_time: u64, is_buyer_maker: bool) -> AggTrade {
        serde_json::from_str(&format!(
            r#"{{"e":"aggTrade","E":{event_time},"s":"BTCUSDT","a":1,"p":"70100.00",
               "q":"0.01","f":1,"l":1,"T":{event_time},"m":{is_buyer_maker}}}"#
        ))
        .unwrap()
    }

    fn make_ticker(update_id: u64) -> BookTicker {
        serde_json::from_str(&format!(
            r#"{{"u":{update_id},"s":"BTCUSDT","b":"70100.00","B":"0.5","a":"70101.00","A":"0.3"}}"#
        ))
        .unwrap()
    }

    // ── BatchBuffer ───────────────────────────────────────────────────────────

    #[test]
    fn batch_buffer_no_flush_before_threshold() {
        let mut buf: BatchBuffer<u32> = BatchBuffer::new(3);
        assert!(buf.push(1).is_none());
        assert!(buf.push(2).is_none());
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn batch_buffer_flushes_at_threshold() {
        let mut buf: BatchBuffer<u32> = BatchBuffer::new(3);
        buf.push(1);
        buf.push(2);
        let batch = buf.push(3);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap(), vec![1, 2, 3]);
        assert!(buf.is_empty(), "buffer must be cleared after flush");
    }

    #[test]
    fn batch_buffer_manual_flush() {
        let mut buf: BatchBuffer<u32> = BatchBuffer::new(100);
        buf.push(10);
        buf.push(20);
        let items = buf.flush();
        assert_eq!(items, vec![10, 20]);
        assert!(buf.is_empty());
    }

    // ── depth_to_batch ────────────────────────────────────────────────────────

    #[test]
    fn depth_to_batch_schema_correct() {
        let items = vec![(0u64, make_depth(1_000))];
        let batch = ParquetRecorder::depth_to_batch(&items).unwrap();
        let schema = batch.schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "event_time");
        assert_eq!(schema.field(1).name(), "receive_time");
        assert_eq!(schema.field(4).name(), "bid_prices");
        assert_eq!(schema.field(6).name(), "ask_prices");
    }

    #[test]
    fn depth_to_batch_row_count() {
        let items: Vec<_> = (0..50).map(|i| (i as u64, make_depth(i as u64 * 100))).collect();
        let batch = ParquetRecorder::depth_to_batch(&items).unwrap();
        assert_eq!(batch.num_rows(), 50);
    }

    #[test]
    fn depth_to_batch_empty() {
        let batch = ParquetRecorder::depth_to_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 8);
    }

    // ── trade_to_batch ────────────────────────────────────────────────────────

    #[test]
    fn trade_to_batch_100_rows() {
        let items: Vec<_> = (0..100)
            .map(|i| (i as u64, make_trade(i as u64 * 100, i % 2 == 0)))
            .collect();
        let batch = ParquetRecorder::trade_to_batch(&items).unwrap();
        assert_eq!(batch.num_rows(), 100);
        assert_eq!(batch.num_columns(), 8);
    }

    #[test]
    fn trade_to_batch_schema_correct() {
        let items = vec![(0u64, make_trade(1_000, false))];
        let batch = ParquetRecorder::trade_to_batch(&items).unwrap();
        let schema = batch.schema();
        assert_eq!(schema.field(5).name(), "is_buy");
        assert_eq!(*schema.field(5).data_type(), DataType::Boolean);
    }

    #[test]
    fn trade_to_batch_is_buy_flag() {
        let buy = make_trade(1_000, false); // is_buyer_maker=false → is_buy=true
        let sell = make_trade(2_000, true); // is_buyer_maker=true → is_buy=false
        let batch = ParquetRecorder::trade_to_batch(&[(0, buy), (0, sell)]).unwrap();

        let is_buy_col = batch
            .column(5)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(is_buy_col.value(0));
        assert!(!is_buy_col.value(1));
    }

    // ── ticker_to_batch ───────────────────────────────────────────────────────

    #[test]
    fn ticker_to_batch_schema_and_values() {
        let items = vec![(12345u64, make_ticker(99))];
        let batch = ParquetRecorder::ticker_to_batch(&items).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);

        let bid = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((bid.value(0) - 70100.0).abs() < 1e-9);
    }
}
