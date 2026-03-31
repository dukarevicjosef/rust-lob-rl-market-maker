use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::order::{ExecutionReport, Order};
use super::types::{OrderId, Price, Quantity, Side, Timestamp};

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// Point-in-time view of the book at a given depth on each side.
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: Vec<(Price, Quantity)>,
    pub asks: Vec<(Price, Quantity)>,
    pub timestamp: Timestamp,
}

// ── OrderBook ─────────────────────────────────────────────────────────────────

/// Central limit order book backed by BTreeMaps for price-time priority.
///
/// Bids: `Reverse<Price>` key so the map iterates highest-to-lowest.
/// Asks: `Price` key so the map iterates lowest-to-highest.
/// Each price level holds a FIFO `VecDeque<Order>` — the front is the oldest
/// (highest priority) order at that price.
#[derive(Clone)]
pub struct OrderBook {
    pub(crate) bids: BTreeMap<Reverse<Price>, VecDeque<Order>>,
    pub(crate) asks: BTreeMap<Price, VecDeque<Order>>,
    /// O(1) lookup for cancel and modify: maps OrderId → (Side, Price).
    pub(crate) order_index: HashMap<OrderId, (Side, Price)>,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            order_index: HashMap::new(),
        }
    }

    /// Submit an order. Routes to the matching engine; limit orders that are
    /// not immediately crossed rest on the book.
    pub fn add_limit_order(&mut self, order: Order) -> ExecutionReport {
        super::matching::process_order(self, order)
    }

    /// Remove a resting order by ID. Returns the order if it was found.
    /// Also cleans the order_index and prunes empty price levels.
    pub fn cancel_order(&mut self, id: OrderId) -> Option<Order> {
        let (side, price) = self.order_index.remove(&id)?;
        let order = match side {
            Side::Bid => {
                let level = self.bids.get_mut(&Reverse(price))?;
                let pos = level.iter().position(|o| o.id == id)?;
                let order = level.remove(pos)?;
                if level.is_empty() {
                    self.bids.remove(&Reverse(price));
                }
                order
            }
            Side::Ask => {
                let level = self.asks.get_mut(&price)?;
                let pos = level.iter().position(|o| o.id == id)?;
                let order = level.remove(pos)?;
                if level.is_empty() {
                    self.asks.remove(&price);
                }
                order
            }
        };
        Some(order)
    }

    /// Reduce a resting order's quantity in-place. Only downward modification
    /// is supported — increasing quantity would require a re-queue and priority
    /// loss (Cont, Stoikov & Talreja, 2010, §3.3). Returns `false` if the
    /// order is not found or `new_qty >= current_qty`.
    pub fn modify_order(&mut self, id: OrderId, new_qty: Quantity) -> bool {
        let (side, price) = match self.order_index.get(&id).copied() {
            Some(v) => v,
            None => return false,
        };
        let level: &mut VecDeque<Order> = match side {
            Side::Bid => match self.bids.get_mut(&Reverse(price)) {
                Some(l) => l,
                None => return false,
            },
            Side::Ask => match self.asks.get_mut(&price) {
                Some(l) => l,
                None => return false,
            },
        };
        if let Some(order) = level.iter_mut().find(|o| o.id == id) {
            if new_qty < order.quantity {
                order.quantity = new_qty;
                return true;
            }
        }
        false
    }

    /// Best bid: (price, total quantity at that level).
    #[inline]
    pub fn best_bid(&self) -> Option<(Price, Quantity)> {
        self.bids.iter().next().map(|(rp, level)| {
            let total: u64 = level.iter().map(|o| o.quantity.0).sum();
            (rp.0, Quantity(total))
        })
    }

    /// Best ask: (price, total quantity at that level).
    #[inline]
    pub fn best_ask(&self) -> Option<(Price, Quantity)> {
        self.asks.iter().next().map(|(price, level)| {
            let total: u64 = level.iter().map(|o| o.quantity.0).sum();
            (*price, Quantity(total))
        })
    }

    /// Integer mid-price: (best_bid + best_ask) / 2. Truncates toward zero.
    #[inline]
    pub fn mid_price(&self) -> Option<Price> {
        let (bid, _) = self.best_bid()?;
        let (ask, _) = self.best_ask()?;
        Some(Price((bid.0 + ask.0) / 2))
    }

    /// Bid-ask spread: best_ask − best_bid.
    #[inline]
    pub fn spread(&self) -> Option<Price> {
        let (bid, _) = self.best_bid()?;
        let (ask, _) = self.best_ask()?;
        Some(ask - bid)
    }

    /// Aggregated depth for up to `levels` price levels on one side.
    /// Each entry is (price, total_quantity_at_level).
    pub fn depth(&self, side: Side, levels: usize) -> Vec<(Price, Quantity)> {
        match side {
            Side::Bid => self
                .bids
                .iter()
                .take(levels)
                .map(|(rp, level)| {
                    let qty: u64 = level.iter().map(|o| o.quantity.0).sum();
                    (rp.0, Quantity(qty))
                })
                .collect(),
            Side::Ask => self
                .asks
                .iter()
                .take(levels)
                .map(|(price, level)| {
                    let qty: u64 = level.iter().map(|o| o.quantity.0).sum();
                    (*price, Quantity(qty))
                })
                .collect(),
        }
    }

    /// Full book snapshot at `levels` depth per side.
    pub fn snapshot(&self, levels: usize) -> OrderBookSnapshot {
        let timestamp = self
            .bids
            .values()
            .flat_map(|l| l.iter())
            .chain(self.asks.values().flat_map(|l| l.iter()))
            .map(|o| o.timestamp)
            .max()
            .unwrap_or(Timestamp(0));

        OrderBookSnapshot {
            bids: self.depth(Side::Bid, levels),
            asks: self.depth(Side::Ask, levels),
            timestamp,
        }
    }

    // ── Internal helpers used exclusively by matching.rs ─────────────────────

    /// Append a resting order at the back of its price level (new arrival).
    pub(crate) fn insert_resting(&mut self, order: Order) {
        self.order_index.insert(order.id, (order.side, order.price));
        match order.side {
            Side::Bid => self
                .bids
                .entry(Reverse(order.price))
                .or_default()
                .push_back(order),
            Side::Ask => self
                .asks
                .entry(order.price)
                .or_default()
                .push_back(order),
        }
    }

    /// Re-insert a partially-filled maker at the FRONT of its level.
    /// The order retains its original time priority (Cont et al., 2010, §3.1).
    pub(crate) fn insert_resting_front(&mut self, order: Order) {
        self.order_index.insert(order.id, (order.side, order.price));
        match order.side {
            Side::Bid => self
                .bids
                .entry(Reverse(order.price))
                .or_default()
                .push_front(order),
            Side::Ask => self
                .asks
                .entry(order.price)
                .or_default()
                .push_front(order),
        }
    }

    /// Pop the oldest (front) order from an ask level. Prunes the level if empty.
    pub(crate) fn pop_front_ask(&mut self, price: Price) -> Option<Order> {
        let level = self.asks.get_mut(&price)?;
        let order = level.pop_front()?;
        self.order_index.remove(&order.id);
        if level.is_empty() {
            self.asks.remove(&price);
        }
        Some(order)
    }

    /// Pop the oldest (front) order from a bid level. Prunes the level if empty.
    pub(crate) fn pop_front_bid(&mut self, price: Price) -> Option<Order> {
        let level = self.bids.get_mut(&Reverse(price))?;
        let order = level.pop_front()?;
        self.order_index.remove(&order.id);
        if level.is_empty() {
            self.bids.remove(&Reverse(price));
        }
        Some(order)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::order::Order;
    use super::super::types::{OrderId, OrderType, Timestamp};

    fn make_order(id: u64, side: Side, price: f64, qty: u64, ts: u64, ot: OrderType) -> Order {
        Order::new(
            OrderId(id),
            side,
            Price::from_f64(price),
            Quantity(qty),
            Timestamp(ts),
            ot,
        )
    }

    #[test]
    fn new_book_is_empty() {
        let book = OrderBook::new();
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert!(book.mid_price().is_none());
        assert!(book.spread().is_none());
        assert!(book.depth(Side::Bid, 5).is_empty());
        assert!(book.depth(Side::Ask, 5).is_empty());
    }

    #[test]
    fn best_bid_aggregates_level_quantity() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 5, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Bid, 100.0, 3, 2, OrderType::Limit));
        book.insert_resting(make_order(3, Side::Bid, 99.0, 10, 3, OrderType::Limit));

        let (price, qty) = book.best_bid().unwrap();
        assert_eq!(price, Price::from_f64(100.0));
        assert_eq!(qty, Quantity(8));
    }

    #[test]
    fn best_ask_is_lowest_price() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Ask, 102.0, 5, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Ask, 101.0, 3, 2, OrderType::Limit));

        let (price, qty) = book.best_ask().unwrap();
        assert_eq!(price, Price::from_f64(101.0));
        assert_eq!(qty, Quantity(3));
    }

    #[test]
    fn mid_price_and_spread_correct() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 99.0, 1, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Ask, 101.0, 1, 2, OrderType::Limit));

        assert_eq!(book.mid_price().unwrap(), Price::from_f64(100.0));
        assert_eq!(book.spread().unwrap(), Price::from_f64(2.0));
    }

    #[test]
    fn cancel_removes_order_and_prunes_level() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));

        let order = book.cancel_order(OrderId(1)).unwrap();
        assert_eq!(order.quantity, Quantity(10));
        assert!(book.best_bid().is_none());
        assert!(!book.order_index.contains_key(&OrderId(1)));
    }

    #[test]
    fn cancel_unknown_id_returns_none() {
        let mut book = OrderBook::new();
        assert!(book.cancel_order(OrderId(999)).is_none());
    }

    #[test]
    fn cancel_mid_queue_preserves_others() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Ask, 100.0, 5, 2, OrderType::Limit));
        book.insert_resting(make_order(3, Side::Ask, 100.0, 5, 3, OrderType::Limit));

        book.cancel_order(OrderId(2));

        let (price, qty) = book.best_ask().unwrap();
        assert_eq!(price, Price::from_f64(100.0));
        assert_eq!(qty, Quantity(10)); // orders 1 and 3 remain
    }

    #[test]
    fn modify_down_succeeds() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));

        assert!(book.modify_order(OrderId(1), Quantity(4)));
        assert_eq!(book.best_bid().unwrap().1, Quantity(4));
    }

    #[test]
    fn modify_up_rejected() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));

        assert!(!book.modify_order(OrderId(1), Quantity(20)));
        assert_eq!(book.best_bid().unwrap().1, Quantity(10)); // unchanged
    }

    #[test]
    fn modify_same_qty_rejected() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));

        assert!(!book.modify_order(OrderId(1), Quantity(10)));
    }

    #[test]
    fn depth_returns_correct_levels() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 100.0, 5, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Bid, 100.0, 3, 2, OrderType::Limit));
        book.insert_resting(make_order(3, Side::Bid, 99.0, 10, 3, OrderType::Limit));
        book.insert_resting(make_order(4, Side::Bid, 98.0, 7, 4, OrderType::Limit));

        let depth = book.depth(Side::Bid, 2);
        assert_eq!(depth.len(), 2);
        assert_eq!(depth[0], (Price::from_f64(100.0), Quantity(8)));
        assert_eq!(depth[1], (Price::from_f64(99.0), Quantity(10)));
    }

    #[test]
    fn snapshot_covers_both_sides() {
        let mut book = OrderBook::new();
        book.insert_resting(make_order(1, Side::Bid, 99.0, 5, 1, OrderType::Limit));
        book.insert_resting(make_order(2, Side::Ask, 101.0, 3, 2, OrderType::Limit));

        let snap = book.snapshot(5);
        assert_eq!(snap.bids.len(), 1);
        assert_eq!(snap.asks.len(), 1);
    }
}
