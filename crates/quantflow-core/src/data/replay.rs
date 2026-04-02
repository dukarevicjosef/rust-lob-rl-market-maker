use std::path::Path;

use crate::orderbook::{Order, OrderBook, OrderBookSnapshot};
use crate::orderbook::types::{OrderId, OrderType, Price, Quantity, Side, Timestamp};

use super::market_event::{MarketEvent, MarketEventError, MarketEventType, events_from_parquet};

// ── ReplayEngine ──────────────────────────────────────────────────────────────

/// Plays back normalised `MarketEvent`s from Parquet as a live LOB stream.
///
/// Can be used as a drop-in replacement for `HawkesLobSimulator` in the
/// agent evaluation and dashboard backends.
pub struct ReplayEngine {
    events: Vec<MarketEvent>,
    cursor: usize,
    book: OrderBook,
    next_order_id: u64,
    pub speed: f64,
}

impl ReplayEngine {
    /// Load events from a `{date}_events.parquet` file produced by
    /// `normalize_btcusdt`.
    pub fn from_parquet(path: &Path) -> Result<Self, MarketEventError> {
        let events = events_from_parquet(path)?;
        Ok(Self::from_events(events))
    }

    /// Construct directly from an already-loaded event vec (useful for tests
    /// and the normalize pipeline which has events in memory).
    pub fn from_events(events: Vec<MarketEvent>) -> Self {
        Self {
            events,
            cursor: 0,
            book: OrderBook::new(),
            next_order_id: 1,
            speed: 1.0,
        }
    }

    /// Advance by one event and apply it to the LOB.
    ///
    /// Returns `None` when the session is exhausted.
    pub fn next_event(&mut self) -> Option<&MarketEvent> {
        if self.cursor >= self.events.len() {
            return None;
        }
        // Clone to release the immutable borrow on self.events before apply_event.
        let event = self.events[self.cursor].clone();
        self.cursor += 1;
        self.apply_event(&event);
        Some(&self.events[self.cursor - 1])
    }

    /// Advance by up to `n` events and return a cloned batch.
    pub fn step_n(&mut self, n: usize) -> Vec<MarketEvent> {
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            match self.next_event() {
                Some(e) => batch.push(e.clone()),
                None => break,
            }
        }
        batch
    }

    /// Current LOB snapshot at `levels` depth per side.
    pub fn snapshot(&self, levels: usize) -> OrderBookSnapshot {
        self.book.snapshot(levels)
    }

    /// Mid-price as f64, or `None` if the book is empty.
    pub fn mid_price(&self) -> Option<f64> {
        self.book.mid_price().map(|p| p.to_f64())
    }

    /// Best bid as f64, or `None`.
    pub fn best_bid(&self) -> Option<f64> {
        self.book.best_bid().map(|(p, _)| p.to_f64())
    }

    /// Best ask as f64, or `None`.
    pub fn best_ask(&self) -> Option<f64> {
        self.book.best_ask().map(|(p, _)| p.to_f64())
    }

    /// Reset to the beginning of the session.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.book = OrderBook::new();
        self.next_order_id = 1;
    }

    /// Fraction of events consumed (0.0–1.0).
    pub fn progress(&self) -> f64 {
        self.cursor as f64 / self.events.len().max(1) as f64
    }

    /// Number of events not yet consumed.
    pub fn remaining(&self) -> usize {
        self.events.len().saturating_sub(self.cursor)
    }

    /// Total events in the session.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn next_id(&mut self) -> OrderId {
        let id = OrderId(self.next_order_id);
        self.next_order_id += 1;
        id
    }

    fn apply_event(&mut self, event: &MarketEvent) {
        // Scale BTC quantities to internal lots (×1000 → 3-decimal precision).
        // 0.001 BTC → 1, 1.0 BTC → 1000, 10 BTC → 10 000.
        let qty = Quantity((event.quantity * 1_000.0).round().max(1.0) as u64);
        let price = Price::from_f64(event.price);
        let ts = Timestamp(event.raw_timestamp_ms.saturating_mul(1_000_000)); // ms → ns

        match event.event_type {
            // ── Market orders — consume opposing side ─────────────────────────
            MarketEventType::MarketBuy => {
                let id = self.next_id();
                let order = Order::new(id, Side::Bid, Price(0), qty, ts, OrderType::Market);
                self.book.add_limit_order(order);
            }
            MarketEventType::MarketSell => {
                let id = self.next_id();
                let order = Order::new(id, Side::Ask, Price(0), qty, ts, OrderType::Market);
                self.book.add_limit_order(order);
            }

            // ── Limit orders — rest on the book without triggering matching ───
            MarketEventType::LimitBuyBest | MarketEventType::LimitBuyDeep => {
                let id = self.next_id();
                let order = Order::new(id, Side::Bid, price, qty, ts, OrderType::Limit);
                self.book.insert_resting(order);
            }
            MarketEventType::LimitSellBest | MarketEventType::LimitSellDeep => {
                let id = self.next_id();
                let order = Order::new(id, Side::Ask, price, qty, ts, OrderType::Limit);
                self.book.insert_resting(order);
            }

            // ── Cancel events — FIFO reduction at that price level ────────────
            MarketEventType::CancelBuyBest | MarketEventType::CancelBuyDeep => {
                self.book.reduce_quantity_at_price(Side::Bid, price, qty);
            }
            MarketEventType::CancelSellBest | MarketEventType::CancelSellDeep => {
                self.book.reduce_quantity_at_price(Side::Ask, price, qty);
            }

            // ── Modify — treat as cancel + re-add at same price ───────────────
            MarketEventType::ModifyBuy => {
                self.book.reduce_quantity_at_price(Side::Bid, price, qty);
                let id = self.next_id();
                let order = Order::new(id, Side::Bid, price, qty, ts, OrderType::Limit);
                self.book.insert_resting(order);
            }
            MarketEventType::ModifySell => {
                self.book.reduce_quantity_at_price(Side::Ask, price, qty);
                let id = self.next_id();
                let order = Order::new(id, Side::Ask, price, qty, ts, OrderType::Limit);
                self.book.insert_resting(order);
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::market_event::{events_to_parquet, MarketEvent, MarketEventType};

    fn make_events(n: usize) -> Vec<MarketEvent> {
        // Alternating LimitBuyDeep / LimitSellDeep at prices 99.x / 101.x
        (0..n).map(|i| {
            let (et, price) = if i % 2 == 0 {
                (MarketEventType::LimitBuyDeep, 99.0 - (i as f64) * 0.01)
            } else {
                (MarketEventType::LimitSellDeep, 101.0 + (i as f64) * 0.01)
            };
            MarketEvent {
                timestamp:        i as f64 * 0.1,
                event_type:       et,
                price,
                quantity:         0.01,
                raw_timestamp_ms: 1_000 + i as u64 * 100,
            }
        }).collect()
    }

    #[test]
    fn replay_engine_from_events_initial_state() {
        let engine = ReplayEngine::from_events(make_events(100));
        assert_eq!(engine.remaining(), 100);
        assert_eq!(engine.progress(), 0.0);
        assert!(engine.mid_price().is_none()); // book empty before first event
    }

    #[test]
    fn replay_engine_mid_price_after_100_events() {
        let mut engine = ReplayEngine::from_events(make_events(100));
        engine.step_n(100);
        assert!(engine.mid_price().is_some(), "book should have quotes after 100 events");
    }

    #[test]
    fn replay_engine_reset_clears_book_and_cursor() {
        let mut engine = ReplayEngine::from_events(make_events(50));
        engine.step_n(50);
        assert_eq!(engine.remaining(), 0);
        engine.reset();
        assert_eq!(engine.remaining(), 50);
        assert!(engine.mid_price().is_none()); // book empty after reset
    }

    #[test]
    fn replay_engine_progress_fraction() {
        let mut engine = ReplayEngine::from_events(make_events(10));
        engine.step_n(5);
        assert!((engine.progress() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn replay_engine_returns_none_at_end() {
        let mut engine = ReplayEngine::from_events(make_events(3));
        engine.next_event();
        engine.next_event();
        engine.next_event();
        assert!(engine.next_event().is_none());
    }

    #[test]
    fn replay_engine_from_parquet_roundtrip() {
        let events = make_events(30);
        let dir = std::env::temp_dir();
        let path = dir.join("qf_replay_test.parquet");
        events_to_parquet(&events, &path).unwrap();

        let mut engine = ReplayEngine::from_parquet(&path).unwrap();
        assert_eq!(engine.len(), 30);
        engine.step_n(30);
        assert_eq!(engine.remaining(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reduce_quantity_at_price_partial_cancel() {
        let mut book = OrderBook::new();
        use crate::orderbook::types::{OrderId, OrderType, Timestamp};
        // Add 10 lots at price 100
        let order = Order::new(
            OrderId(1), Side::Bid, Price::from_f64(100.0),
            Quantity(10), Timestamp(0), OrderType::Limit,
        );
        book.insert_resting(order);
        book.reduce_quantity_at_price(Side::Bid, Price::from_f64(100.0), Quantity(3));
        assert_eq!(book.best_bid().unwrap().1, Quantity(7));
    }

    #[test]
    fn reduce_quantity_at_price_full_cancel() {
        let mut book = OrderBook::new();
        use crate::orderbook::types::{OrderId, OrderType, Timestamp};
        let order = Order::new(
            OrderId(1), Side::Ask, Price::from_f64(101.0),
            Quantity(5), Timestamp(0), OrderType::Limit,
        );
        book.insert_resting(order);
        book.reduce_quantity_at_price(Side::Ask, Price::from_f64(101.0), Quantity(5));
        assert!(book.best_ask().is_none(), "level should be empty after full cancel");
    }
}
