//! Bidirectional Binance Futures exchange connector.
//!
//! # Modules
//! - [`config`]       — load API credentials from environment variables
//! - [`types`]        — shared data types (orders, positions, events, errors)
//! - [`binance_rest`] — authenticated REST client (order management, account)
//! - [`binance_ws`]   — user data stream WebSocket (fill / account updates)
//!
//! # Quick start (testnet)
//! ```no_run
//! # tokio_test::block_on(async {
//! use quantflow_core::exchange::{
//!     config::ExchangeConfig, BinanceRestClient, types::OrderSide,
//! };
//!
//! let cfg = ExchangeConfig::load_dotenv(".env").unwrap();
//! let rest = BinanceRestClient::new(&cfg.api_key, &cfg.api_secret, cfg.testnet);
//!
//! let price = rest.get_ticker_price(&cfg.symbol).await.unwrap();
//! println!("BTCUSDT mark price: {price}");
//!
//! let order = rest.place_limit_order(
//!     &cfg.symbol, OrderSide::Buy, price * 0.99, 0.001, "GTC"
//! ).await.unwrap();
//! println!("Order placed: {:?}", order);
//! # });
//! ```

pub mod binance_rest;
pub mod binance_ws;
pub mod config;
pub mod risk_manager;
pub mod types;

// Re-export the most-used items at the module level
pub use binance_rest::BinanceRestClient;
pub use binance_ws::UserDataStream;
pub use config::ExchangeConfig;
pub use risk_manager::{RiskConfig, RiskManager, RiskStatus, RiskViolation};
pub use types::{
    AccountInfo, CancelResponse, ExchangeError, OrderResponse, OrderSide, UserDataEvent,
};
