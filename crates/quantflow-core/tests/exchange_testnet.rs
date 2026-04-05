//! Testnet integration tests — skipped automatically when credentials are absent.
//!
//! Run with:
//!   BINANCE_API_KEY=... BINANCE_API_SECRET=... BINANCE_TESTNET=1 \
//!   cargo test -p quantflow-core --test exchange_testnet -- --ignored --nocapture

use quantflow_core::exchange::{BinanceRestClient, ExchangeConfig, OrderSide};

fn client() -> Option<BinanceRestClient> {
    let cfg = ExchangeConfig::load_dotenv(".env").ok()?;
    if !cfg.testnet {
        eprintln!("SKIPPING: BINANCE_TESTNET is not set to true — refusing to run against mainnet");
        return None;
    }
    Some(BinanceRestClient::new(&cfg.api_key, &cfg.api_secret, true))
}

#[tokio::test]
#[ignore = "requires testnet credentials (BINANCE_API_KEY / BINANCE_API_SECRET)"]
async fn testnet_server_time() {
    let Some(rest) = client() else { return };
    let ts = rest.get_server_time().await.expect("server time");
    let local = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap()
        .as_millis() as u64;
    let skew_ms = (ts as i64 - local as i64).unsigned_abs();
    println!("Server time: {ts}  local: {local}  skew: {skew_ms} ms");
    assert!(skew_ms < 2_000, "clock skew {skew_ms} ms exceeds recv_window safety margin");
}

#[tokio::test]
#[ignore = "requires testnet credentials"]
async fn testnet_ticker_price() {
    let Some(rest) = client() else { return };
    let price = rest.get_ticker_price("BTCUSDT").await.expect("ticker price");
    println!("BTCUSDT mark price: {price}");
    assert!(price > 1_000.0, "suspiciously low mark price: {price}");
}

#[tokio::test]
#[ignore = "requires testnet credentials"]
async fn testnet_account_info() {
    let Some(rest) = client() else { return };
    let acct = rest.get_account().await.expect("account info");
    let balance = acct.total_balance_f64();
    let upnl    = acct.unrealized_pnl_f64();
    let avail   = acct.available_balance_f64();
    println!("Balance: {balance:.4}  uPnL: {upnl:.4}  Available: {avail:.4}");
    println!("Open positions: {:?}", acct.open_positions());
}

#[tokio::test]
#[ignore = "requires testnet credentials"]
async fn testnet_open_orders() {
    let Some(rest) = client() else { return };
    let orders = rest.get_open_orders("BTCUSDT").await.expect("open orders");
    println!("Open orders ({}): {:?}", orders.len(), orders);
}

#[tokio::test]
#[ignore = "requires testnet credentials — places and immediately cancels a limit order"]
async fn testnet_place_and_cancel_limit_order() {
    let Some(rest) = client() else { return };

    // Place a far-away limit buy that will never fill
    let price = rest.get_ticker_price("BTCUSDT").await.expect("ticker");
    // Round to nearest $100 tick, place 50% below market (will never fill)
    let limit_price = ((price * 0.50) / 100.0).floor() * 100.0;
    // Minimum Futures notional = $100; round qty up to 3 decimal places
    let qty = ((105.0 / limit_price) * 1000.0).ceil() / 1000.0;

    println!("Placing limit BUY @ {limit_price:.2}  qty={qty:.3}  notional=${:.2}  (50% below market @ {price:.2})",
             limit_price * qty);

    let order = rest
        .place_limit_order("BTCUSDT", OrderSide::Buy, limit_price, qty, "GTC")
        .await
        .expect("place order");

    println!("Order placed: id={} status={}", order.order_id, order.status);
    assert_eq!(order.status, "NEW");

    // Cancel immediately
    let cancel = rest
        .cancel_order("BTCUSDT", order.order_id)
        .await
        .expect("cancel order");

    println!("Cancelled: id={} status={}", cancel.order_id, cancel.status);
    assert_eq!(cancel.status, "CANCELED");
}

#[tokio::test]
#[ignore = "requires testnet credentials"]
async fn testnet_cancel_all_orders_is_idempotent() {
    let Some(rest) = client() else { return };
    // Should succeed even when no orders are open (code -2011 handled gracefully)
    rest.cancel_all_orders("BTCUSDT").await.expect("cancel_all");
    println!("cancel_all_orders: OK (idempotent)");
}

#[tokio::test]
#[ignore = "requires testnet credentials"]
async fn testnet_listen_key_lifecycle() {
    let Some(rest) = client() else { return };

    let key = rest.start_user_data_stream().await.expect("listen key");
    println!("listenKey: {}", &key[..20.min(key.len())]);
    assert!(!key.is_empty());

    rest.keepalive_user_data_stream().await.expect("keepalive");
    println!("keepalive: OK");

    rest.close_user_data_stream().await.expect("close stream");
    println!("stream closed: OK");
}
