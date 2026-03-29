use serde::{Deserialize, Serialize};

use super::types::{OrderId, OrderType, Price, Quantity, Side, Timestamp};

// ── Order ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
    pub order_type: OrderType,
}

impl Order {
    pub fn new(
        id: OrderId,
        side: Side,
        price: Price,
        quantity: Quantity,
        timestamp: Timestamp,
        order_type: OrderType,
    ) -> Self {
        Order { id, side, price, quantity, timestamp, order_type }
    }

    /// True for order types that do not rest on the book after partial fill.
    #[inline]
    pub fn is_immediate(self: &Order) -> bool {
        matches!(
            self.order_type,
            OrderType::Market | OrderType::ImmediateOrCancel | OrderType::FillOrKill
        )
    }
}

// ── Trade ─────────────────────────────────────────────────────────────────────

/// A single matched execution between a passive (maker) and aggressive (taker) order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trade {
    pub price: Price,
    pub quantity: Quantity,
    pub maker_id: OrderId,
    pub taker_id: OrderId,
    pub timestamp: Timestamp,
}

// ── ExecutionReport ───────────────────────────────────────────────────────────

/// Result of submitting an order to the matching engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// All fills generated during matching, in time order.
    pub trades: Vec<Trade>,
    /// Residual order resting on the book (None if fully filled or cancelled).
    pub remaining: Option<Order>,
}

impl ExecutionReport {
    pub fn filled(trades: Vec<Trade>) -> Self {
        ExecutionReport { trades, remaining: None }
    }

    pub fn partial(trades: Vec<Trade>, remaining: Order) -> Self {
        ExecutionReport { trades, remaining: Some(remaining) }
    }

    pub fn rejected() -> Self {
        ExecutionReport { trades: vec![], remaining: None }
    }

    #[inline]
    pub fn is_fully_filled(&self) -> bool {
        self.remaining.is_none() && !self.trades.is_empty()
    }

    pub fn filled_quantity(&self) -> Quantity {
        Quantity(self.trades.iter().map(|t| t.quantity.0).sum())
    }
}

// ── BookDelta ─────────────────────────────────────────────────────────────────

/// Incremental update event emitted by the LOB — sufficient to reconstruct
/// full book state from a stream of deltas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BookDelta {
    /// New resting order added to the book.
    Add(Order),
    /// Resting order fully removed (cancel or complete fill).
    Remove(OrderId),
    /// Resting order quantity reduced (partial fill or partial cancel).
    Modify { id: OrderId, new_quantity: Quantity },
    /// Execution between a maker and taker.
    Trade(Trade),
}
