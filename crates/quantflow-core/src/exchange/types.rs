use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Order side ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Buy  => "BUY",
            Self::Sell => "SELL",
        }
    }
}

// ── REST response types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct OrderResponse {
    #[serde(rename = "orderId")]
    pub order_id: u64,
    pub symbol:   String,
    /// "NEW" | "FILLED" | "CANCELED" | "PARTIALLY_FILLED" | "EXPIRED"
    pub status:   String,
    pub price:    String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    pub side:     String,
    #[serde(rename = "updateTime")]
    pub update_time: u64,
}

impl OrderResponse {
    pub fn price_f64(&self)        -> f64 { self.price.parse().unwrap_or(0.0) }
    pub fn orig_qty_f64(&self)     -> f64 { self.orig_qty.parse().unwrap_or(0.0) }
    pub fn executed_qty_f64(&self) -> f64 { self.executed_qty.parse().unwrap_or(0.0) }
    pub fn is_filled(&self)        -> bool { self.status == "FILLED" }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CancelResponse {
    #[serde(rename = "orderId")]
    pub order_id: u64,
    pub status:   String,
}

// ── Account / position types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AccountInfo {
    #[serde(rename = "totalWalletBalance")]
    pub total_balance: String,
    #[serde(rename = "totalUnrealizedProfit")]
    pub unrealized_pnl: String,
    #[serde(rename = "availableBalance")]
    pub available_balance: String,
    pub positions: Vec<PositionInfo>,
}

impl AccountInfo {
    pub fn total_balance_f64(&self)     -> f64 { self.total_balance.parse().unwrap_or(0.0) }
    pub fn unrealized_pnl_f64(&self)   -> f64 { self.unrealized_pnl.parse().unwrap_or(0.0) }
    pub fn available_balance_f64(&self) -> f64 { self.available_balance.parse().unwrap_or(0.0) }

    /// Returns only positions with non-zero size.
    pub fn open_positions(&self) -> Vec<&PositionInfo> {
        self.positions.iter()
            .filter(|p| p.position_amt_f64().abs() > 1e-9)
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PositionInfo {
    pub symbol: String,
    /// Positive = long, negative = short
    #[serde(rename = "positionAmt")]
    pub position_amt: String,
    #[serde(rename = "entryPrice")]
    pub entry_price: String,
    #[serde(rename = "unrealizedProfit")]
    pub unrealized_profit: String,
    pub leverage: String,
}

impl PositionInfo {
    pub fn position_amt_f64(&self)     -> f64 { self.position_amt.parse().unwrap_or(0.0) }
    pub fn entry_price_f64(&self)      -> f64 { self.entry_price.parse().unwrap_or(0.0) }
    pub fn unrealized_profit_f64(&self) -> f64 { self.unrealized_profit.parse().unwrap_or(0.0) }
    pub fn leverage_f64(&self)         -> f64 { self.leverage.parse().unwrap_or(1.0) }
}

// ── User data stream types ────────────────────────────────────────────────────

/// Top-level envelope for Binance Futures user-data-stream events.
/// Deserialized via the "e" field as the serde tag.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "e")]
pub enum UserDataEvent {
    #[serde(rename = "ORDER_TRADE_UPDATE")]
    OrderUpdate {
        /// Transaction time (ms since epoch)
        #[serde(rename = "T")]
        transaction_time: u64,
        #[serde(rename = "o")]
        order: OrderTradeUpdate,
    },
    #[serde(rename = "ACCOUNT_UPDATE")]
    AccountUpdate {
        #[serde(rename = "T")]
        transaction_time: u64,
        #[serde(rename = "a")]
        data: AccountUpdateData,
    },
    /// Keep-alive pong from the server — safe to ignore.
    #[serde(rename = "listenKeyExpired")]
    ListenKeyExpired,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrderTradeUpdate {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "S")]
    pub side: String,          // "BUY" | "SELL"
    #[serde(rename = "o")]
    pub order_type: String,    // "LIMIT" | "MARKET" | …
    #[serde(rename = "X")]
    pub order_status: String,  // "NEW" | "FILLED" | "CANCELED" | "PARTIALLY_FILLED"
    #[serde(rename = "i")]
    pub order_id: u64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub orig_qty: String,
    #[serde(rename = "z")]
    pub filled_qty: String,
    /// Last fill price (0.0 for NEW events)
    #[serde(rename = "L")]
    pub last_fill_price: String,
    /// Last fill quantity (0.0 for NEW events)
    #[serde(rename = "l")]
    pub last_fill_qty: String,
    #[serde(rename = "T")]
    pub trade_time: u64,
}

impl OrderTradeUpdate {
    pub fn price_f64(&self)          -> f64 { self.price.parse().unwrap_or(0.0) }
    pub fn orig_qty_f64(&self)       -> f64 { self.orig_qty.parse().unwrap_or(0.0) }
    pub fn filled_qty_f64(&self)     -> f64 { self.filled_qty.parse().unwrap_or(0.0) }
    pub fn last_fill_price_f64(&self) -> f64 { self.last_fill_price.parse().unwrap_or(0.0) }
    pub fn last_fill_qty_f64(&self)  -> f64 { self.last_fill_qty.parse().unwrap_or(0.0) }
    pub fn is_filled(&self) -> bool  { self.order_status == "FILLED" }
    pub fn is_partial(&self) -> bool { self.order_status == "PARTIALLY_FILLED" }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AccountUpdateData {
    /// Reason: "ORDER" | "FUNDING_FEE" | "WITHDRAW" | …
    #[serde(rename = "m")]
    pub event_reason: String,
    #[serde(rename = "B")]
    pub balances: Vec<BalanceUpdate>,
    #[serde(rename = "P")]
    pub positions: Vec<PositionUpdate>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BalanceUpdate {
    #[serde(rename = "a")]
    pub asset: String,
    #[serde(rename = "wb")]
    pub wallet_balance: String,
    #[serde(rename = "cw")]
    pub cross_wallet_balance: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PositionUpdate {
    #[serde(rename = "s")]
    pub symbol: String,
    /// Positive = long, negative = short
    #[serde(rename = "pa")]
    pub position_amt: String,
    #[serde(rename = "ep")]
    pub entry_price: String,
    #[serde(rename = "up")]
    pub unrealized_pnl: String,
    #[serde(rename = "mt")]
    pub margin_type: String,
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct BinanceApiError {
    pub code: i64,
    pub msg:  String,
}

#[derive(Debug, Error)]
pub enum ExchangeError {
    #[error("Binance API error {}: {}", .0.code, .0.msg)]
    ApiError(BinanceApiError),

    #[error("HTTP {0}: {1}")]
    HttpError(u16, String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Risk violation: {0}")]
    RiskViolation(String),
}

impl From<reqwest::Error> for ExchangeError {
    fn from(e: reqwest::Error) -> Self {
        ExchangeError::NetworkError(e.to_string())
    }
}
