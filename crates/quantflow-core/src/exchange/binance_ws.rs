//! User data stream WebSocket client for Binance Futures.
//!
//! Connects to the private user-data stream via a `listenKey` obtained from
//! the REST API and forwards fill / account-update events over a channel.
//!
//! Keep-alive loop renews the `listenKey` every 25 minutes in the background
//! so the stream never silently expires.
//!
//! # Example
//! ```no_run
//! # tokio_test::block_on(async {
//! use quantflow_core::exchange::{
//!     BinanceRestClient, UserDataStream, types::UserDataEvent,
//! };
//! use tokio::sync::mpsc;
//!
//! let rest = BinanceRestClient::new("key", "secret", true);
//! let listen_key = rest.start_user_data_stream().await.unwrap();
//!
//! let (tx, mut rx) = mpsc::channel(256);
//! let stream = UserDataStream::new(&listen_key, true);
//! tokio::spawn(stream.run(tx, rest));
//!
//! while let Some(event) = rx.recv().await {
//!     println!("{event:?}");
//! }
//! # });
//! ```

use futures_util::StreamExt;
use tokio::time::{interval, Duration};
use tokio_tungstenite::connect_async;

use super::binance_rest::BinanceRestClient;
use super::types::{ExchangeError, UserDataEvent};

/// WebSocket base URLs for the user data stream.
const USER_STREAM_TESTNET: &str = "wss://stream.binancefuture.com/ws";
const USER_STREAM_MAINNET: &str = "wss://fstream.binance.com/ws";

// ── Stream handle ─────────────────────────────────────────────────────────────

pub struct UserDataStream {
    ws_url: String,
}

impl UserDataStream {
    pub fn new(listen_key: &str, testnet: bool) -> Self {
        let base = if testnet { USER_STREAM_TESTNET } else { USER_STREAM_MAINNET };
        Self {
            ws_url: format!("{}/{}", base, listen_key),
        }
    }

    /// Connect and stream events into `tx`.
    ///
    /// Spawns a background task that calls `keepalive_user_data_stream` every
    /// 25 minutes so the `listenKey` does not expire after 60 minutes.
    ///
    /// Runs until the channel receiver is dropped or a fatal WebSocket error
    /// occurs.  Reconnects automatically on transient failures (up to 5 times
    /// with exponential back-off).
    pub async fn run(
        self,
        tx:   tokio::sync::mpsc::Sender<UserDataEvent>,
        rest: BinanceRestClient,
    ) -> Result<(), ExchangeError> {
        // ── Keep-alive task ───────────────────────────────────────────────────
        let ka_rest = rest;
        tokio::spawn(async move {
            // Renew every 25 min — listenKey expires after 60 min
            let mut timer = interval(Duration::from_secs(25 * 60));
            timer.tick().await; // skip first immediate tick
            loop {
                timer.tick().await;
                match ka_rest.keepalive_user_data_stream().await {
                    Ok(_) => tracing::debug!("listenKey renewed"),
                    Err(e) => tracing::warn!("listenKey keepalive failed: {e}"),
                }
            }
        });

        // ── WebSocket loop with reconnect ─────────────────────────────────────
        let mut retries: u32 = 0;
        const MAX_RETRIES: u32 = 5;

        loop {
            tracing::info!("Connecting to user data stream: {}", self.ws_url);

            match connect_async(self.ws_url.as_str()).await {
                Ok((ws_stream, _)) => {
                    retries = 0; // reset on successful connect
                    let (_, mut read) = ws_stream.split();

                    while let Some(msg) = read.next().await {
                        let msg = match msg {
                            Ok(m)  => m,
                            Err(e) => {
                                tracing::warn!("WS read error: {e}");
                                break;
                            }
                        };

                        if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                            match serde_json::from_str::<UserDataEvent>(&text) {
                                Ok(event) => {
                                    if tx.send(event).await.is_err() {
                                        // Receiver dropped — exit cleanly
                                        return Ok(());
                                    }
                                }
                                Err(e) => {
                                    tracing::debug!("Skipping unparseable event: {e} — {text:.120}");
                                }
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("Connection failed: {e}"),
            }

            retries += 1;
            if retries > MAX_RETRIES {
                return Err(ExchangeError::NetworkError(format!(
                    "user data stream: exceeded {MAX_RETRIES} reconnect attempts"
                )));
            }

            let backoff = Duration::from_millis(500 * (1 << retries.min(6)));
            tracing::info!("Reconnecting in {}ms (attempt {retries}/{MAX_RETRIES}) …",
                           backoff.as_millis());
            tokio::time::sleep(backoff).await;
        }
    }
}
