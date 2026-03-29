//! LOBSTER (Limit Order Book System — The Efficient Reconstructor) format parser.
//!
//! Reference: https://lobsterdata.com/info/DataStructure.php
//!
//! Message file columns: Time, Type, OrderID, Size, Price, Direction
//! Order book file columns per level: AskPrice_i, AskSize_i, BidPrice_i, BidSize_i

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

use crate::orderbook::{Order, OrderBook, OrderId, OrderType, Price, Quantity, Side, Timestamp};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum LobsterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {detail}")]
    Parse { line: usize, detail: String },
    #[error("Unknown event type {code} at line {line}")]
    UnknownEvent { code: i32, line: usize },
}

// ── Event type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// New resting limit order added to the book.
    NewLimitOrder = 1,
    /// Partial cancellation: size field is the reduction, not the new size.
    PartialCancel = 2,
    /// Full cancellation of a resting order.
    FullCancel = 3,
    /// Execution of a visible resting order (size = executed shares).
    ExecutionVisible = 4,
    /// Execution of a hidden/iceberg resting order.
    ExecutionHidden = 5,
    /// Trading halt.
    TradingHalt = 7,
}

impl EventType {
    fn from_code(code: i32, line: usize) -> Result<Self, LobsterError> {
        match code {
            1 => Ok(EventType::NewLimitOrder),
            2 => Ok(EventType::PartialCancel),
            3 => Ok(EventType::FullCancel),
            4 => Ok(EventType::ExecutionVisible),
            5 => Ok(EventType::ExecutionHidden),
            7 => Ok(EventType::TradingHalt),
            _ => Err(LobsterError::UnknownEvent { code, line }),
        }
    }
}

// ── LobsterMessage ────────────────────────────────────────────────────────────

/// One row of a LOBSTER message file.
#[derive(Debug, Clone)]
pub struct LobsterMessage {
    /// Seconds since midnight.
    pub timestamp: f64,
    pub event_type: EventType,
    pub order_id: u64,
    /// For Type 1: order size. For Type 2: size reduction. For Type 4/5: executed size.
    pub size: u64,
    /// Raw price as stored in the source file (units depend on the data provider).
    pub price: i64,
    /// 1 = buy (bid), -1 = sell (ask).
    pub direction: i8,
}

impl LobsterMessage {
    /// Convert the raw price field to our internal `Price` using an explicit
    /// scale factor. LOBSTER data typically uses `scale = 10_000`; our
    /// internal `PRICE_SCALE = 100_000_000`. Pass `1` if prices are already
    /// in internal ticks.
    #[inline]
    pub fn to_price(&self, src_scale: i64) -> Price {
        use crate::orderbook::PRICE_SCALE;
        Price::from_ticks(self.price * (PRICE_SCALE / src_scale))
    }

    #[inline]
    pub fn to_side(&self) -> Side {
        if self.direction == 1 { Side::Bid } else { Side::Ask }
    }
}

// ── MessageIter ───────────────────────────────────────────────────────────────

/// Streaming iterator over a LOBSTER message CSV file.
pub struct MessageIter {
    reader: BufReader<File>,
    buf: String,
    line: usize,
}

impl MessageIter {
    fn open(path: &Path) -> Result<Self, LobsterError> {
        let file = File::open(path)?;
        Ok(MessageIter { reader: BufReader::new(file), buf: String::new(), line: 0 })
    }
}

impl Iterator for MessageIter {
    type Item = Result<LobsterMessage, LobsterError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();
        match self.reader.read_line(&mut self.buf) {
            Err(e) => return Some(Err(e.into())),
            Ok(0) => return None, // EOF
            Ok(_) => {}
        }
        self.line += 1;
        let ln = self.line;
        let row = self.buf.trim();
        if row.is_empty() {
            return self.next(); // skip blank lines
        }
        Some(parse_message_row(row, ln))
    }
}

fn parse_message_row(row: &str, line: usize) -> Result<LobsterMessage, LobsterError> {
    let err = |detail: &str| LobsterError::Parse { line, detail: detail.to_owned() };

    let mut cols = row.split(',');
    let timestamp: f64 = cols
        .next()
        .ok_or_else(|| err("missing timestamp"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid timestamp"))?;

    let type_code: i32 = cols
        .next()
        .ok_or_else(|| err("missing type"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid type"))?;

    let order_id: u64 = cols
        .next()
        .ok_or_else(|| err("missing order_id"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid order_id"))?;

    let size: u64 = cols
        .next()
        .ok_or_else(|| err("missing size"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid size"))?;

    let price: i64 = cols
        .next()
        .ok_or_else(|| err("missing price"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid price"))?;

    let direction: i8 = cols
        .next()
        .ok_or_else(|| err("missing direction"))?
        .trim()
        .parse()
        .map_err(|_| err("invalid direction"))?;

    Ok(LobsterMessage {
        timestamp,
        event_type: EventType::from_code(type_code, line)?,
        order_id,
        size,
        price,
        direction,
    })
}

// ── LobsterSnapshot ───────────────────────────────────────────────────────────

/// One price level entry within a LOBSTER order book snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LobsterLevel {
    pub ask_price: i64,
    pub ask_size: u64,
    pub bid_price: i64,
    pub bid_size: u64,
}

/// One row of a LOBSTER order book file — a depth snapshot at `levels` levels.
#[derive(Debug, Clone)]
pub struct LobsterSnapshot {
    pub levels: Vec<LobsterLevel>,
}

impl LobsterSnapshot {
    /// Number of price levels stored.
    #[inline]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
}

// ── SnapshotIter ──────────────────────────────────────────────────────────────

/// Streaming iterator over a LOBSTER order book CSV file.
pub struct SnapshotIter {
    reader: BufReader<File>,
    buf: String,
    line: usize,
}

impl SnapshotIter {
    fn open(path: &Path) -> Result<Self, LobsterError> {
        let file = File::open(path)?;
        Ok(SnapshotIter { reader: BufReader::new(file), buf: String::new(), line: 0 })
    }
}

impl Iterator for SnapshotIter {
    type Item = Result<LobsterSnapshot, LobsterError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();
        match self.reader.read_line(&mut self.buf) {
            Err(e) => return Some(Err(e.into())),
            Ok(0) => return None,
            Ok(_) => {}
        }
        self.line += 1;
        let ln = self.line;
        let row = self.buf.trim();
        if row.is_empty() {
            return self.next();
        }
        Some(parse_snapshot_row(row, ln))
    }
}

fn parse_snapshot_row(row: &str, line: usize) -> Result<LobsterSnapshot, LobsterError> {
    let err = |detail: &str| LobsterError::Parse { line, detail: detail.to_owned() };

    let cols: Vec<&str> = row.split(',').map(|c| c.trim()).collect();
    if cols.len() % 4 != 0 {
        return Err(err(&format!(
            "expected 4*L columns, got {}",
            cols.len()
        )));
    }

    let n_levels = cols.len() / 4;
    let mut levels = Vec::with_capacity(n_levels);

    for i in 0..n_levels {
        let base = i * 4;
        let ask_price: i64 = cols[base]
            .parse()
            .map_err(|_| err(&format!("invalid ask_price at level {}", i + 1)))?;
        let ask_size: u64 = cols[base + 1]
            .parse()
            .map_err(|_| err(&format!("invalid ask_size at level {}", i + 1)))?;
        let bid_price: i64 = cols[base + 2]
            .parse()
            .map_err(|_| err(&format!("invalid bid_price at level {}", i + 1)))?;
        let bid_size: u64 = cols[base + 3]
            .parse()
            .map_err(|_| err(&format!("invalid bid_size at level {}", i + 1)))?;
        levels.push(LobsterLevel { ask_price, ask_size, bid_price, bid_size });
    }

    Ok(LobsterSnapshot { levels })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Open `path` and return a streaming iterator over `LobsterMessage` records.
pub fn replay_messages(path: &Path) -> Result<MessageIter, LobsterError> {
    MessageIter::open(path)
}

/// Open `path` and return a streaming iterator over `LobsterSnapshot` records.
pub fn load_snapshots(path: &Path) -> Result<SnapshotIter, LobsterError> {
    SnapshotIter::open(path)
}

// ── LobsterReplayer ───────────────────────────────────────────────────────────

/// Applies LOBSTER messages to an `OrderBook`, maintaining the internal size
/// map needed to resolve partial executions and partial cancels.
///
/// # Price scale
/// Set `price_scale` to the scale factor of the raw price integers in the
/// source file relative to `PRICE_SCALE` (10^8). Use `1` if prices are
/// already in internal ticks; use `10_000` for standard LOBSTER 4-decimal
/// prices.
pub struct LobsterReplayer {
    pub book: OrderBook,
    /// Remaining shares for each resting order, keyed by order_id.
    sizes: HashMap<u64, u64>,
    price_scale: i64,
}

impl LobsterReplayer {
    pub fn new(price_scale: i64) -> Self {
        LobsterReplayer {
            book: OrderBook::new(),
            sizes: HashMap::new(),
            price_scale,
        }
    }

    /// Apply one LOBSTER message. Unknown order IDs in execution/cancel
    /// events are silently ignored (consistent with incomplete replays).
    pub fn apply(&mut self, msg: &LobsterMessage) {
        let ts = Timestamp((msg.timestamp * 1e9) as u64);
        let price = msg.to_price(self.price_scale);
        let side = msg.to_side();

        match msg.event_type {
            EventType::NewLimitOrder => {
                let order = Order::new(
                    OrderId(msg.order_id),
                    side,
                    price,
                    Quantity(msg.size),
                    ts,
                    OrderType::Limit,
                );
                // Insert directly as a resting order — matching is represented
                // by separate Type 4/5 events in LOBSTER format.
                self.book.insert_resting(order);
                self.sizes.insert(msg.order_id, msg.size);
            }

            EventType::PartialCancel => {
                // Size field = shares cancelled (reduction), not new size.
                if let Some(current) = self.sizes.get_mut(&msg.order_id) {
                    let new_qty = current.saturating_sub(msg.size);
                    if self.book.modify_order(OrderId(msg.order_id), Quantity(new_qty)) {
                        *current = new_qty;
                    }
                }
            }

            EventType::FullCancel => {
                self.book.cancel_order(OrderId(msg.order_id));
                self.sizes.remove(&msg.order_id);
            }

            EventType::ExecutionVisible | EventType::ExecutionHidden => {
                // Size = shares executed in this event.
                let maybe_current = self.sizes.get(&msg.order_id).copied();
                if let Some(current) = maybe_current {
                    let new_qty = current.saturating_sub(msg.size);
                    if new_qty == 0 {
                        self.book.cancel_order(OrderId(msg.order_id));
                        self.sizes.remove(&msg.order_id);
                    } else if self.book.modify_order(OrderId(msg.order_id), Quantity(new_qty)) {
                        self.sizes.insert(msg.order_id, new_qty);
                    }
                }
            }

            EventType::TradingHalt => { /* no book state change */ }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MSGS: &str = "34200.000,1,1,100,9990000000,1\n\
                        34200.001,1,2,50,10010000000,-1\n\
                        34200.002,4,2,50,10010000000,-1\n\
                        34200.003,3,1,100,9990000000,1\n";

    const SNAPS: &str = "10010000000,50,9990000000,100,10020000000,80,9980000000,200\n\
                         10005000000,100,9980000000,300,10020000000,80,9970000000,400\n";

    fn parse_msgs(src: &str) -> Vec<LobsterMessage> {
        src.lines()
            .enumerate()
            .map(|(i, line)| parse_message_row(line.trim(), i + 1).unwrap())
            .collect()
    }

    fn parse_snaps(src: &str) -> Vec<LobsterSnapshot> {
        src.lines()
            .enumerate()
            .map(|(i, line)| parse_snapshot_row(line.trim(), i + 1).unwrap())
            .collect()
    }

    #[test]
    fn parse_message_row_all_fields() {
        let msgs = parse_msgs(MSGS);
        assert_eq!(msgs.len(), 4);

        let m0 = &msgs[0];
        assert_eq!(m0.event_type, EventType::NewLimitOrder);
        assert_eq!(m0.order_id, 1);
        assert_eq!(m0.size, 100);
        assert_eq!(m0.price, 9990000000);
        assert_eq!(m0.direction, 1);
        assert!((m0.timestamp - 34200.0).abs() < 1e-6);

        let m1 = &msgs[1];
        assert_eq!(m1.event_type, EventType::NewLimitOrder);
        assert_eq!(m1.direction, -1);

        assert_eq!(msgs[2].event_type, EventType::ExecutionVisible);
        assert_eq!(msgs[3].event_type, EventType::FullCancel);
    }

    #[test]
    fn parse_snapshot_row_three_levels() {
        let snaps = parse_snaps(SNAPS);
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].depth(), 2);

        let l0 = &snaps[0].levels[0];
        assert_eq!(l0.ask_price, 10010000000);
        assert_eq!(l0.ask_size, 50);
        assert_eq!(l0.bid_price, 9990000000);
        assert_eq!(l0.bid_size, 100);

        let l1 = &snaps[0].levels[1];
        assert_eq!(l1.ask_price, 10020000000);
        assert_eq!(l1.bid_price, 9980000000);
    }

    #[test]
    fn snapshot_odd_column_count_is_error() {
        assert!(parse_snapshot_row("100,50,99", 1).is_err());
    }

    #[test]
    fn unknown_event_type_is_error() {
        assert!(parse_message_row("34200.0,9,1,100,10000000000,1", 1).is_err());
    }

    #[test]
    fn to_price_applies_scale_factor() {
        let msg = LobsterMessage {
            timestamp: 0.0,
            event_type: EventType::NewLimitOrder,
            order_id: 1,
            size: 1,
            price: 10_000,   // LOBSTER 10^-4 scale → $1.00
            direction: 1,
        };
        // price=10_000, src_scale=10_000 → 10_000 * (100_000_000/10_000) = 100_000_000 ticks = $1.00
        assert_eq!(msg.to_price(10_000), Price::from_f64(1.0));
    }

    #[test]
    fn replayer_new_order_and_cancel() {
        let msgs = parse_msgs(MSGS);
        let mut r = LobsterReplayer::new(1);
        for m in &msgs {
            r.apply(m);
        }
        // After: NewBid(1), NewAsk(2), ExecAsk(2→gone), CancelBid(1→gone)
        assert!(r.book.best_bid().is_none());
        assert!(r.book.best_ask().is_none());
    }

    #[test]
    fn replayer_partial_execution_reduces_quantity() {
        let src = "0.0,1,1,100,10000000000,-1\n\
                   0.1,4,1,40,10000000000,-1\n";
        let msgs = parse_msgs(src);
        let mut r = LobsterReplayer::new(1);
        for m in &msgs {
            r.apply(m);
        }
        // Ask id=1: started 100, executed 40 → 60 remaining
        let (_, qty) = r.book.best_ask().unwrap();
        assert_eq!(qty, Quantity(60));
    }

    #[test]
    fn replayer_partial_cancel_reduces_quantity() {
        let src = "0.0,1,1,200,9900000000,1\n\
                   0.1,2,1,80,9900000000,1\n";
        let msgs = parse_msgs(src);
        let mut r = LobsterReplayer::new(1);
        for m in &msgs {
            r.apply(m);
        }
        // Bid id=1: started 200, cancelled 80 → 120 remaining
        let (_, qty) = r.book.best_bid().unwrap();
        assert_eq!(qty, Quantity(120));
    }
}
