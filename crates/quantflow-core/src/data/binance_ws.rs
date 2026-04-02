use futures_util::StreamExt;
use tokio_tungstenite::connect_async;
use url::Url;

use super::binance::{AggTrade, BookTicker, DepthUpdate};

/// Unified message enum for all three Binance streams.
#[derive(Debug)]
pub enum BinanceMessage {
    Depth(DepthUpdate),
    Trade(AggTrade),
    BookTicker(BookTicker),
}

pub struct BinanceClient {
    symbol: String,
    use_futures: bool,
}

impl BinanceClient {
    pub fn new(symbol: &str, use_futures: bool) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
            use_futures,
        }
    }

    pub fn ws_url(&self) -> String {
        let base = if self.use_futures {
            "wss://fstream.binance.com/stream"
        } else {
            "wss://stream.binance.com:9443/stream"
        };

        let streams = format!(
            "{}@depth@100ms/{}@aggTrade/{}@bookTicker",
            self.symbol, self.symbol, self.symbol
        );

        format!("{}?streams={}", base, streams)
    }

    /// Connects and streams messages over a channel.
    /// Runs until the sender is dropped or an error occurs.
    pub async fn stream(
        &self,
        tx: tokio::sync::mpsc::Sender<BinanceMessage>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let raw_url = self.ws_url();
        // Validate the URL before connecting.
        let _validated = Url::parse(&raw_url)?;
        tracing::info!("Connecting to {}", raw_url);

        let (ws_stream, _) = connect_async(raw_url.as_str()).await?;
        tracing::info!("Connected to Binance WebSocket");

        let (_, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                // Binance combined stream format: {"stream":"...", "data":{...}}
                if let Ok(wrapper) = serde_json::from_str::<serde_json::Value>(&text) {
                    let stream = wrapper["stream"].as_str().unwrap_or("");
                    let data = &wrapper["data"];

                    let parsed = if stream.contains("@depth") {
                        serde_json::from_value(data.clone())
                            .ok()
                            .map(BinanceMessage::Depth)
                    } else if stream.contains("@aggTrade") {
                        serde_json::from_value(data.clone())
                            .ok()
                            .map(BinanceMessage::Trade)
                    } else if stream.contains("@bookTicker") {
                        serde_json::from_value(data.clone())
                            .ok()
                            .map(BinanceMessage::BookTicker)
                    } else {
                        None
                    };

                    if let Some(parsed_msg) = parsed {
                        if tx.send(parsed_msg).await.is_err() {
                            break; // receiver dropped
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
