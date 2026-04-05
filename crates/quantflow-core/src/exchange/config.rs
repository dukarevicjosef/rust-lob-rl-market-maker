/// Exchange configuration loaded from environment variables.
///
/// API keys must never appear in source code or version control.
/// Set them via a `.env` file (excluded from git) or through the
/// process environment before launching the binary.
///
/// Required env vars:
///   BINANCE_API_KEY    — Binance (Testnet) API key
///   BINANCE_API_SECRET — Binance (Testnet) API secret
///
/// Optional:
///   BINANCE_TESTNET    — "1" or "true" enables testnet (default: true)
///   BINANCE_SYMBOL     — trading pair (default: "BTCUSDT")

#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub api_key:    String,
    pub api_secret: String,
    /// When true, all requests go to the Binance Futures Testnet.
    /// Default is true so production credentials can never be used
    /// accidentally by omission.
    pub testnet:    bool,
    pub symbol:     String,
}

impl ExchangeConfig {
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            api_key: std::env::var("BINANCE_API_KEY")
                .map_err(|_| "BINANCE_API_KEY not set".to_string())?,
            api_secret: std::env::var("BINANCE_API_SECRET")
                .map_err(|_| "BINANCE_API_SECRET not set".to_string())?,
            testnet: std::env::var("BINANCE_TESTNET")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),   // safe default: testnet
            symbol: std::env::var("BINANCE_SYMBOL")
                .unwrap_or_else(|_| "BTCUSDT".to_string()),
        })
    }

    /// Load a `.env` file from the given path before reading env vars.
    /// Silently ignores missing files — variables already in the environment
    /// take precedence.
    pub fn load_dotenv(path: &str) -> Result<Self, String> {
        if let Ok(contents) = std::fs::read_to_string(path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    // Only set if not already present in the environment
                    if std::env::var(k).is_err() {
                        std::env::set_var(k, v.trim());
                    }
                }
            }
        }
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_testnet_is_true() {
        // Without BINANCE_TESTNET set, config must default to testnet=true.
        // This test unsets the var to guarantee a clean state.
        std::env::remove_var("BINANCE_TESTNET");
        std::env::set_var("BINANCE_API_KEY",    "test_key");
        std::env::set_var("BINANCE_API_SECRET", "test_secret");
        let cfg = ExchangeConfig::from_env().unwrap();
        assert!(cfg.testnet, "default must be testnet=true");
        // Clean up
        std::env::remove_var("BINANCE_API_KEY");
        std::env::remove_var("BINANCE_API_SECRET");
    }
}
