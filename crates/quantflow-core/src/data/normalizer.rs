use super::binance::{AggTrade, DepthUpdate, PriceLevel};
use super::market_event::{MarketEvent, MarketEventType};

// ── BinanceNormalizer ─────────────────────────────────────────────────────────

/// Converts Binance depth updates and aggregated trades into the unified
/// `MarketEvent` format. Derives limit/cancel events from depth diffs.
pub struct BinanceNormalizer {
    session_start_ms: u64,
    prev_bids: Vec<PriceLevel>,
    prev_asks: Vec<PriceLevel>,
    best_bid: f64,
    best_ask: f64,
    initialized: bool,
}

impl Default for BinanceNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BinanceNormalizer {
    pub fn new() -> Self {
        Self {
            session_start_ms: 0,
            prev_bids: Vec::new(),
            prev_asks: Vec::new(),
            best_bid: 0.0,
            best_ask: f64::INFINITY,
            initialized: false,
        }
    }

    fn to_ts(&self, event_time_ms: u64) -> f64 {
        if self.session_start_ms == 0 {
            0.0
        } else {
            (event_time_ms.saturating_sub(self.session_start_ms)) as f64 / 1000.0
        }
    }

    fn init(&mut self, event_time_ms: u64) {
        if !self.initialized {
            self.session_start_ms = event_time_ms;
            self.initialized = true;
        }
    }

    /// Normalize one `AggTrade` → exactly one `MarketEvent`.
    pub fn normalize_trade(&mut self, trade: &AggTrade) -> MarketEvent {
        self.init(trade.event_time);
        MarketEvent {
            timestamp:        self.to_ts(trade.event_time),
            event_type:       if trade.is_buy() { MarketEventType::MarketBuy } else { MarketEventType::MarketSell },
            price:            trade.parsed_price(),
            quantity:         trade.parsed_quantity(),
            raw_timestamp_ms: trade.event_time,
        }
    }

    /// Normalize one `DepthUpdate` → 0-N `MarketEvent`s by diffing against
    /// the previous LOB state.
    ///
    /// Classification logic (bid side shown; ask side is symmetric):
    /// - Quantity increased at a price → `LimitBuy{Best,Deep}`
    /// - Quantity decreased at a price → `CancelBuy{Best,Deep}`
    /// - Price level absent in prev → `LimitBuy{Best,Deep}` (new entry)
    /// - Price level absent in new (quantity = 0) → `CancelBuy{Best,Deep}`
    pub fn normalize_depth(&mut self, depth: &DepthUpdate) -> Vec<MarketEvent> {
        self.init(depth.event_time);
        let ts = self.to_ts(depth.event_time);
        let raw_ts = depth.event_time;

        let new_bids = depth.parsed_bids();
        let new_asks = depth.parsed_asks();

        let new_best_bid = new_bids.first().map(|l| l.price).unwrap_or(0.0);
        let new_best_ask = new_asks.first().map(|l| l.price).unwrap_or(f64::INFINITY);

        let mut events = Vec::new();

        // ── Bid side ──────────────────────────────────────────────────────────
        for lvl in &new_bids {
            let prev_qty = Self::find_prev_qty(&self.prev_bids, lvl.price);
            let delta = lvl.quantity - prev_qty;
            if delta > 1e-12 {
                let is_best = (lvl.price - new_best_bid).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::LimitBuyBest } else { MarketEventType::LimitBuyDeep },
                    price: lvl.price,
                    quantity: delta,
                    raw_timestamp_ms: raw_ts,
                });
            } else if delta < -1e-12 {
                let is_best = (lvl.price - new_best_bid).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::CancelBuyBest } else { MarketEventType::CancelBuyDeep },
                    price: lvl.price,
                    quantity: -delta,
                    raw_timestamp_ms: raw_ts,
                });
            }
        }
        // Levels that disappeared entirely (qty dropped to zero in prev but absent in new)
        for prev_lvl in &self.prev_bids {
            if Self::find_prev_qty(&new_bids, prev_lvl.price) < 1e-12
                && prev_lvl.quantity > 1e-12
            {
                let is_best = (prev_lvl.price - new_best_bid).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::CancelBuyBest } else { MarketEventType::CancelBuyDeep },
                    price: prev_lvl.price,
                    quantity: prev_lvl.quantity,
                    raw_timestamp_ms: raw_ts,
                });
            }
        }

        // ── Ask side ──────────────────────────────────────────────────────────
        for lvl in &new_asks {
            let prev_qty = Self::find_prev_qty(&self.prev_asks, lvl.price);
            let delta = lvl.quantity - prev_qty;
            if delta > 1e-12 {
                let is_best = (lvl.price - new_best_ask).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::LimitSellBest } else { MarketEventType::LimitSellDeep },
                    price: lvl.price,
                    quantity: delta,
                    raw_timestamp_ms: raw_ts,
                });
            } else if delta < -1e-12 {
                let is_best = (lvl.price - new_best_ask).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::CancelSellBest } else { MarketEventType::CancelSellDeep },
                    price: lvl.price,
                    quantity: -delta,
                    raw_timestamp_ms: raw_ts,
                });
            }
        }
        for prev_lvl in &self.prev_asks {
            if Self::find_prev_qty(&new_asks, prev_lvl.price) < 1e-12
                && prev_lvl.quantity > 1e-12
            {
                let is_best = (prev_lvl.price - new_best_ask).abs() < 1e-9;
                events.push(MarketEvent {
                    timestamp: ts,
                    event_type: if is_best { MarketEventType::CancelSellBest } else { MarketEventType::CancelSellDeep },
                    price: prev_lvl.price,
                    quantity: prev_lvl.quantity,
                    raw_timestamp_ms: raw_ts,
                });
            }
        }

        // Update state
        self.best_bid = new_best_bid;
        self.best_ask = new_best_ask;
        self.prev_bids = new_bids;
        self.prev_asks = new_asks;

        events
    }

    fn find_prev_qty(levels: &[PriceLevel], price: f64) -> f64 {
        levels
            .iter()
            .find(|l| (l.price - price).abs() < 1e-9)
            .map(|l| l.quantity)
            .unwrap_or(0.0)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(event_time: u64, price: &str, qty: &str, is_maker: bool) -> AggTrade {
        serde_json::from_str(&format!(
            r#"{{"e":"aggTrade","E":{event_time},"s":"BTCUSDT","a":1,"p":"{price}",
               "q":"{qty}","f":1,"l":1,"T":{event_time},"m":{is_maker}}}"#
        ))
        .unwrap()
    }

    fn make_depth(event_time: u64, bids: &[(&str, &str)], asks: &[(&str, &str)]) -> DepthUpdate {
        let b: Vec<String> = bids.iter().map(|(p, q)| format!(r#"["{p}","{q}"]"#)).collect();
        let a: Vec<String> = asks.iter().map(|(p, q)| format!(r#"["{p}","{q}"]"#)).collect();
        serde_json::from_str(&format!(
            r#"{{"e":"depthUpdate","E":{event_time},"s":"BTCUSDT","U":1,"u":2,
               "b":[{}],"a":[{}]}}"#,
            b.join(","),
            a.join(","),
        ))
        .unwrap()
    }

    #[test]
    fn normalize_trade_market_buy() {
        let mut n = BinanceNormalizer::new();
        let trade = make_trade(1000, "70100.00", "0.01", false); // is_buyer_maker=false → buy
        let event = n.normalize_trade(&trade);
        assert_eq!(event.event_type, MarketEventType::MarketBuy);
        assert!((event.price - 70_100.0).abs() < 1e-6);
        assert_eq!(event.timestamp, 0.0);
    }

    #[test]
    fn normalize_trade_market_sell() {
        let mut n = BinanceNormalizer::new();
        let trade = make_trade(1000, "70100.00", "0.01", true); // is_buyer_maker=true → sell
        let event = n.normalize_trade(&trade);
        assert_eq!(event.event_type, MarketEventType::MarketSell);
    }

    #[test]
    fn normalize_depth_qty_increase_is_limit_buy() {
        let mut n = BinanceNormalizer::new();
        // First depth update establishes baseline
        let d1 = make_depth(1000, &[("70100.00", "1.0"), ("70099.00", "2.0")], &[("70101.00", "1.5")]);
        n.normalize_depth(&d1);

        // Second update: qty at 70100 increased by 0.5 → LimitBuy
        let d2 = make_depth(2000, &[("70100.00", "1.5"), ("70099.00", "2.0")], &[("70101.00", "1.5")]);
        let events = n.normalize_depth(&d2);
        let limit_buys: Vec<_> = events.iter()
            .filter(|e| matches!(e.event_type, MarketEventType::LimitBuyBest | MarketEventType::LimitBuyDeep))
            .collect();
        assert!(!limit_buys.is_empty(), "qty increase should yield LimitBuy");
        assert!((limit_buys[0].quantity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn normalize_depth_qty_decrease_is_cancel_buy() {
        let mut n = BinanceNormalizer::new();
        let d1 = make_depth(1000, &[("70100.00", "2.0"), ("70099.00", "1.0")], &[("70101.00", "1.0")]);
        n.normalize_depth(&d1);

        // qty at 70099 dropped by 0.5 → CancelBuy (deep, since 70099 < best bid 70100)
        let d2 = make_depth(2000, &[("70100.00", "2.0"), ("70099.00", "0.5")], &[("70101.00", "1.0")]);
        let events = n.normalize_depth(&d2);
        let cancels: Vec<_> = events.iter()
            .filter(|e| matches!(e.event_type, MarketEventType::CancelBuyBest | MarketEventType::CancelBuyDeep))
            .collect();
        assert!(!cancels.is_empty(), "qty decrease should yield CancelBuy");
        assert_eq!(cancels[0].event_type, MarketEventType::CancelBuyDeep);
    }

    #[test]
    fn normalize_depth_new_price_level_is_limit_buy() {
        let mut n = BinanceNormalizer::new();
        let d1 = make_depth(1000, &[("70100.00", "1.0")], &[("70101.00", "1.0")]);
        n.normalize_depth(&d1);

        // New price level 70098 appears → LimitBuyDeep
        let d2 = make_depth(2000, &[("70100.00", "1.0"), ("70098.00", "0.5")], &[("70101.00", "1.0")]);
        let events = n.normalize_depth(&d2);
        let new_limit: Vec<_> = events.iter()
            .filter(|e| e.event_type == MarketEventType::LimitBuyDeep && (e.price - 70098.0).abs() < 1e-6)
            .collect();
        assert!(!new_limit.is_empty(), "new price level should yield LimitBuyDeep");
    }

    #[test]
    fn normalize_depth_disappeared_level_is_cancel() {
        let mut n = BinanceNormalizer::new();
        let d1 = make_depth(1000, &[("70100.00", "1.0"), ("70099.00", "0.5")], &[("70101.00", "1.0")]);
        n.normalize_depth(&d1);

        // 70099 disappeared → CancelBuyDeep
        let d2 = make_depth(2000, &[("70100.00", "1.0")], &[("70101.00", "1.0")]);
        let events = n.normalize_depth(&d2);
        let cancel: Vec<_> = events.iter()
            .filter(|e| e.event_type == MarketEventType::CancelBuyDeep && (e.price - 70099.0).abs() < 1e-6)
            .collect();
        assert!(!cancel.is_empty(), "disappeared level should yield CancelBuyDeep");
    }

    #[test]
    fn normalize_depth_timestamp_advances() {
        let mut n = BinanceNormalizer::new();
        let d1 = make_depth(1_000_000, &[("70100.00", "1.0")], &[("70101.00", "1.0")]);
        n.normalize_depth(&d1);
        let d2 = make_depth(2_000_000, &[("70100.00", "1.5")], &[("70101.00", "1.0")]);
        let events = n.normalize_depth(&d2);
        // Elapsed = (2_000_000 - 1_000_000) / 1000 = 1000.0 seconds
        assert!(!events.is_empty());
        assert!((events[0].timestamp - 1000.0).abs() < 1e-6);
    }
}
