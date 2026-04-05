use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;

use super::types::{
    AccountInfo, BinanceApiError, CancelResponse, ExchangeError, OrderResponse, OrderSide,
};

/// Binance Futures REST endpoints.
pub const FUTURES_TESTNET: &str = "https://testnet.binancefuture.com";
pub const FUTURES_MAINNET: &str = "https://fapi.binance.com";

type HmacSha256 = Hmac<Sha256>;

// ── Client ────────────────────────────────────────────────────────────────────

pub struct BinanceRestClient {
    client:       Client,
    base_url:     String,
    api_key:      String,
    api_secret:   String,
    /// Maximum clock skew tolerated by the exchange (ms).
    recv_window:  u64,
}

impl BinanceRestClient {
    pub fn new(api_key: &str, api_secret: &str, testnet: bool) -> Self {
        Self {
            client:      Client::new(),
            base_url:    if testnet { FUTURES_TESTNET } else { FUTURES_MAINNET }.to_string(),
            api_key:     api_key.to_string(),
            api_secret:  api_secret.to_string(),
            recv_window: 5_000,
        }
    }

    // ── Signing ───────────────────────────────────────────────────────────────

    /// HMAC-SHA256 over `query_string`. Returns lowercase hex digest.
    fn sign(&self, query: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(query.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn timestamp_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_millis() as u64
    }

    // ── Order management ──────────────────────────────────────────────────────

    /// Place a LIMIT order (GTC / IOC / FOK).
    pub async fn place_limit_order(
        &self,
        symbol:        &str,
        side:          OrderSide,
        price:         f64,
        quantity:      f64,
        time_in_force: &str,
    ) -> Result<OrderResponse, ExchangeError> {
        let params = format!(
            "symbol={}&side={}&type=LIMIT&timeInForce={}&quantity={:.4}&price={:.2}\
             &recvWindow={}&timestamp={}",
            symbol.to_uppercase(),
            side.as_str(),
            time_in_force,
            quantity,
            price,
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!("{}/fapi/v1/order?{}&signature={}", self.base_url, params, sig);
        let resp = self.client.post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        self.parse_response(resp).await
    }

    /// Place a MARKET order.
    pub async fn place_market_order(
        &self,
        symbol:   &str,
        side:     OrderSide,
        quantity: f64,
    ) -> Result<OrderResponse, ExchangeError> {
        let params = format!(
            "symbol={}&side={}&type=MARKET&quantity={:.4}&recvWindow={}&timestamp={}",
            symbol.to_uppercase(),
            side.as_str(),
            quantity,
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!("{}/fapi/v1/order?{}&signature={}", self.base_url, params, sig);
        let resp = self.client.post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        self.parse_response(resp).await
    }

    /// Cancel a single order by `order_id`.
    pub async fn cancel_order(
        &self,
        symbol:   &str,
        order_id: u64,
    ) -> Result<CancelResponse, ExchangeError> {
        let params = format!(
            "symbol={}&orderId={}&recvWindow={}&timestamp={}",
            symbol.to_uppercase(),
            order_id,
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!("{}/fapi/v1/order?{}&signature={}", self.base_url, params, sig);
        let resp = self.client.delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        self.parse_response(resp).await
    }

    /// Cancel ALL open orders for `symbol` in a single request.
    ///
    /// Used as a kill-switch: call this before shutdown to guarantee no
    /// resting quotes remain on the exchange.
    pub async fn cancel_all_orders(&self, symbol: &str) -> Result<(), ExchangeError> {
        let params = format!(
            "symbol={}&recvWindow={}&timestamp={}",
            symbol.to_uppercase(),
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!(
            "{}/fapi/v1/allOpenOrders?{}&signature={}",
            self.base_url, params, sig
        );
        let resp = self.client.delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            if let Ok(err) = serde_json::from_str::<BinanceApiError>(&text) {
                // Code -2011 = no open orders — not an error in this context
                if err.code != -2011 {
                    return Err(ExchangeError::ApiError(err));
                }
            }
        }
        Ok(())
    }

    // ── Account info ──────────────────────────────────────────────────────────

    /// Full account snapshot: balances + all positions.
    pub async fn get_account(&self) -> Result<AccountInfo, ExchangeError> {
        let params = format!(
            "recvWindow={}&timestamp={}",
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!("{}/fapi/v2/account?{}&signature={}", self.base_url, params, sig);
        let resp = self.client.get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        self.parse_response(resp).await
    }

    /// All currently open orders for `symbol`.
    pub async fn get_open_orders(
        &self,
        symbol: &str,
    ) -> Result<Vec<OrderResponse>, ExchangeError> {
        let params = format!(
            "symbol={}&recvWindow={}&timestamp={}",
            symbol.to_uppercase(),
            self.recv_window,
            self.timestamp_ms(),
        );
        let sig = self.sign(&params);
        let url = format!(
            "{}/fapi/v1/openOrders?{}&signature={}",
            self.base_url, params, sig
        );
        let resp = self.client.get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        self.parse_response(resp).await
    }

    // ── Market data (public, unauthenticated) ─────────────────────────────────

    /// Best-price mark for `symbol`.
    pub async fn get_ticker_price(&self, symbol: &str) -> Result<f64, ExchangeError> {
        let url = format!(
            "{}/fapi/v1/ticker/price?symbol={}",
            self.base_url,
            symbol.to_uppercase()
        );
        let resp = self.client.get(&url).send().await?;
        let data: serde_json::Value = resp.json().await?;
        data["price"].as_str()
            .and_then(|p| p.parse::<f64>().ok())
            .ok_or_else(|| ExchangeError::ParseError("price field missing or non-numeric".into()))
    }

    /// Binance server time in ms — useful for clock-skew diagnostics.
    pub async fn get_server_time(&self) -> Result<u64, ExchangeError> {
        let url = format!("{}/fapi/v1/time", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let data: serde_json::Value = resp.json().await?;
        data["serverTime"].as_u64()
            .ok_or_else(|| ExchangeError::ParseError("serverTime field missing".into()))
    }

    // ── User data stream ──────────────────────────────────────────────────────

    /// Obtain a `listenKey` to subscribe to the user data WebSocket stream.
    /// The key expires after 60 minutes without a keep-alive call.
    pub async fn start_user_data_stream(&self) -> Result<String, ExchangeError> {
        let url = format!("{}/fapi/v1/listenKey", self.base_url);
        let resp = self.client.post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        let data: serde_json::Value = resp.json().await?;
        data["listenKey"].as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| ExchangeError::ParseError("listenKey field missing".into()))
    }

    /// Extend the `listenKey` lifetime by another 60 minutes.
    /// Must be called at least every 30 minutes to keep the stream alive.
    pub async fn keepalive_user_data_stream(&self) -> Result<(), ExchangeError> {
        let url = format!("{}/fapi/v1/listenKey", self.base_url);
        self.client.put(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        Ok(())
    }

    /// Explicitly close the user data stream.
    pub async fn close_user_data_stream(&self) -> Result<(), ExchangeError> {
        let url = format!("{}/fapi/v1/listenKey", self.base_url);
        self.client.delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send().await?;
        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn parse_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: reqwest::Response,
    ) -> Result<T, ExchangeError> {
        let status = resp.status();
        let text   = resp.text().await?;

        if !status.is_success() {
            // Binance error envelope: {"code": -XXXX, "msg": "..."}
            if let Ok(api_err) = serde_json::from_str::<BinanceApiError>(&text) {
                return Err(ExchangeError::ApiError(api_err));
            }
            return Err(ExchangeError::HttpError(status.as_u16(), text));
        }

        let truncated = &text[..200.min(text.len())];
        serde_json::from_str(&text)
            .map_err(|e| ExchangeError::ParseError(format!("{e}: {truncated}")))
    }
}
