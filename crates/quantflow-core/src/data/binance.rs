use serde::Deserialize;

/// Binance Depth Update (@depth@100ms)
#[derive(Debug, Clone, Deserialize)]
pub struct DepthUpdate {
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub last_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
}

/// Binance Aggregated Trade (@aggTrade)
#[derive(Debug, Clone, Deserialize)]
pub struct AggTrade {
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "a")]
    pub agg_trade_id: u64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "f")]
    pub first_trade_id: u64,
    #[serde(rename = "l")]
    pub last_trade_id: u64,
    #[serde(rename = "T")]
    pub trade_time: u64,
    /// true = sell aggressor, false = buy aggressor
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Binance Book Ticker (@bookTicker)
#[derive(Debug, Clone, Deserialize)]
pub struct BookTicker {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub best_bid_price: String,
    #[serde(rename = "B")]
    pub best_bid_qty: String,
    #[serde(rename = "a")]
    pub best_ask_price: String,
    #[serde(rename = "A")]
    pub best_ask_qty: String,
}

/// Parsed price/qty pair (String → f64 conversion)
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

impl DepthUpdate {
    pub fn parsed_bids(&self) -> Vec<PriceLevel> {
        self.bids
            .iter()
            .filter_map(|[p, q]| {
                Some(PriceLevel {
                    price: p.parse().ok()?,
                    quantity: q.parse().ok()?,
                })
            })
            .collect()
    }

    pub fn parsed_asks(&self) -> Vec<PriceLevel> {
        self.asks
            .iter()
            .filter_map(|[p, q]| {
                Some(PriceLevel {
                    price: p.parse().ok()?,
                    quantity: q.parse().ok()?,
                })
            })
            .collect()
    }
}

impl AggTrade {
    pub fn parsed_price(&self) -> f64 {
        self.price.parse().unwrap_or(0.0)
    }

    pub fn parsed_quantity(&self) -> f64 {
        self.quantity.parse().unwrap_or(0.0)
    }

    /// true = buyer initiated (buy aggressor), false = seller initiated
    pub fn is_buy(&self) -> bool {
        !self.is_buyer_maker
    }
}
