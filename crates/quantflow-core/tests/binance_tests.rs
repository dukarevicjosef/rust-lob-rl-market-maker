use quantflow_core::data::binance::{AggTrade, BookTicker, DepthUpdate};
use quantflow_core::data::binance_ws::BinanceClient;

// ── DepthUpdate ───────────────────────────────────────────────────────────────

const DEPTH_JSON: &str = r#"{
    "e": "depthUpdate",
    "E": 1712000000000,
    "s": "BTCUSDT",
    "U": 100,
    "u": 200,
    "b": [["70100.00", "0.5"], ["70099.00", "1.2"]],
    "a": [["70101.00", "0.3"], ["70102.00", "0.8"]]
}"#;

#[test]
fn test_depth_update_deserialize() {
    let d: DepthUpdate = serde_json::from_str(DEPTH_JSON).expect("valid DepthUpdate");
    assert_eq!(d.event_time, 1_712_000_000_000);
    assert_eq!(d.symbol, "BTCUSDT");
    assert_eq!(d.first_update_id, 100);
    assert_eq!(d.last_update_id, 200);
    assert_eq!(d.bids.len(), 2);
    assert_eq!(d.asks.len(), 2);
}

#[test]
fn test_depth_parsed_levels() {
    let d: DepthUpdate = serde_json::from_str(DEPTH_JSON).unwrap();

    let bids = d.parsed_bids();
    assert_eq!(bids.len(), 2);
    assert!((bids[0].price - 70100.0).abs() < 1e-9);
    assert!((bids[0].quantity - 0.5).abs() < 1e-9);
    assert!((bids[1].price - 70099.0).abs() < 1e-9);
    assert!((bids[1].quantity - 1.2).abs() < 1e-9);

    let asks = d.parsed_asks();
    assert_eq!(asks.len(), 2);
    assert!((asks[0].price - 70101.0).abs() < 1e-9);
    assert!((asks[0].quantity - 0.3).abs() < 1e-9);
}

// ── AggTrade ──────────────────────────────────────────────────────────────────

// buy aggressor: is_buyer_maker = false → is_buy() = true
const AGG_TRADE_BUY_JSON: &str = r#"{
    "e": "aggTrade",
    "E": 1712000000000,
    "s": "BTCUSDT",
    "a": 123456,
    "p": "70123.40",
    "q": "0.001",
    "f": 100,
    "l": 100,
    "T": 1712000000000,
    "m": false
}"#;

// sell aggressor: is_buyer_maker = true → is_buy() = false
const AGG_TRADE_SELL_JSON: &str = r#"{
    "e": "aggTrade",
    "E": 1712000000001,
    "s": "BTCUSDT",
    "a": 123457,
    "p": "70120.00",
    "q": "0.005",
    "f": 101,
    "l": 101,
    "T": 1712000000001,
    "m": true
}"#;

#[test]
fn test_agg_trade_deserialize() {
    let t: AggTrade = serde_json::from_str(AGG_TRADE_BUY_JSON).expect("valid AggTrade");
    assert_eq!(t.agg_trade_id, 123456);
    assert_eq!(t.symbol, "BTCUSDT");
    assert!((t.parsed_price() - 70123.40).abs() < 1e-6);
    assert!((t.parsed_quantity() - 0.001).abs() < 1e-9);
}

#[test]
fn test_agg_trade_is_buy_correct() {
    let buy: AggTrade = serde_json::from_str(AGG_TRADE_BUY_JSON).unwrap();
    assert!(buy.is_buy(), "is_buyer_maker=false should be a buy aggressor");

    let sell: AggTrade = serde_json::from_str(AGG_TRADE_SELL_JSON).unwrap();
    assert!(!sell.is_buy(), "is_buyer_maker=true should be a sell aggressor");
}

// ── BookTicker ────────────────────────────────────────────────────────────────

const BOOK_TICKER_JSON: &str = r#"{
    "u": 100,
    "s": "BTCUSDT",
    "b": "70100.00",
    "B": "0.500",
    "a": "70101.00",
    "A": "0.300"
}"#;

#[test]
fn test_book_ticker_deserialize() {
    let bt: BookTicker = serde_json::from_str(BOOK_TICKER_JSON).expect("valid BookTicker");
    assert_eq!(bt.update_id, 100);
    assert_eq!(bt.symbol, "BTCUSDT");
    assert_eq!(bt.best_bid_price, "70100.00");
    assert_eq!(bt.best_ask_price, "70101.00");

    let bid: f64 = bt.best_bid_price.parse().unwrap();
    let ask: f64 = bt.best_ask_price.parse().unwrap();
    assert!((bid - 70100.0).abs() < 1e-9);
    assert!((ask - 70101.0).abs() < 1e-9);
}

// ── BinanceClient::ws_url ─────────────────────────────────────────────────────

#[test]
fn test_ws_url_spot() {
    let client = BinanceClient::new("BTCUSDT", false);
    let url = client.ws_url();
    assert!(url.starts_with("wss://stream.binance.com:9443/stream?streams="));
    assert!(url.contains("btcusdt@depth@100ms"));
    assert!(url.contains("btcusdt@aggTrade"));
    assert!(url.contains("btcusdt@bookTicker"));
}

#[test]
fn test_ws_url_futures() {
    let client = BinanceClient::new("btcusdt", true);
    let url = client.ws_url();
    assert!(url.starts_with("wss://fstream.binance.com/stream?streams="));
    assert!(url.contains("btcusdt@depth@100ms"));
    assert!(url.contains("btcusdt@aggTrade"));
    assert!(url.contains("btcusdt@bookTicker"));
}

#[test]
fn test_ws_url_symbol_lowercased() {
    let client = BinanceClient::new("ETHUSDT", true);
    let url = client.ws_url();
    assert!(url.contains("ethusdt@depth@100ms"));
    assert!(!url.contains("ETHUSDT"));
}
