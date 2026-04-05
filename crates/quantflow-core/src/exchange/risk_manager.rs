use std::time::Instant;

use super::types::OrderSide;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Max absolute position size in base-asset units (e.g. 0.1 BTC).
    pub max_position: f64,
    /// Max allowed intraday loss in quote-asset units (e.g. 100 USDT).
    pub max_daily_loss: f64,
    /// Max drawdown from the intraday PnL peak (e.g. 200 USDT).
    pub max_drawdown: f64,
    /// Orders per second — sliding-window rate limit.
    pub max_orders_per_second: u32,
    /// Max size of a single order in base-asset units.
    pub max_order_size: f64,
    /// Max number of simultaneously open (resting) orders.
    pub max_open_orders: u32,
    /// Max notional value of a single order in quote-asset units.
    pub max_notional: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self::testnet()
    }
}

impl RiskConfig {
    /// Testnet / paper-trading profile — deliberately loose limits so
    /// strategy exploration is not impeded by risk rules.
    pub fn testnet() -> Self {
        Self {
            max_position:          0.1,
            max_daily_loss:        100.0,
            max_drawdown:          200.0,
            max_orders_per_second: 5,
            max_order_size:        0.05,
            max_open_orders:       10,
            max_notional:          10_000.0,
        }
    }

    /// Mainnet / live-trading profile — tight limits to cap real-money risk.
    pub fn mainnet() -> Self {
        Self {
            max_position:          0.01,
            max_daily_loss:         50.0,
            max_drawdown:          100.0,
            max_orders_per_second:   3,
            max_order_size:         0.005,
            max_open_orders:          5,
            max_notional:          1_000.0,
        }
    }

    /// Load from `RISK_*` environment variables.  Falls back to testnet defaults
    /// for any variable that is absent or cannot be parsed.
    pub fn from_env() -> Self {
        let f = |key: &str, def: f64| -> f64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
        };
        let u = |key: &str, def: u32| -> u32 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
        };
        let defaults = Self::testnet();
        Self {
            max_position:          f("RISK_MAX_POSITION",       defaults.max_position),
            max_daily_loss:        f("RISK_MAX_DAILY_LOSS",     defaults.max_daily_loss),
            max_drawdown:          f("RISK_MAX_DRAWDOWN",       defaults.max_drawdown),
            max_orders_per_second: u("RISK_MAX_ORDERS_PER_SEC", defaults.max_orders_per_second),
            max_order_size:        f("RISK_MAX_ORDER_SIZE",     defaults.max_order_size),
            max_open_orders:       u("RISK_MAX_OPEN_ORDERS",    defaults.max_open_orders),
            max_notional:          f("RISK_MAX_NOTIONAL",       defaults.max_notional),
        }
    }
}

// ── Violation type ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum RiskViolation {
    PositionLimitExceeded { current: f64, order_qty: f64, limit: f64 },
    DailyLossExceeded     { current_loss: f64, limit: f64 },
    DrawdownExceeded      { drawdown: f64, limit: f64 },
    OrderRateExceeded     { rate: u32, limit: u32 },
    OrderSizeExceeded     { size: f64, limit: f64 },
    OpenOrdersExceeded    { count: u32, limit: u32 },
    NotionalExceeded      { notional: f64, limit: f64 },
    KillSwitchActive,
}

impl std::fmt::Display for RiskViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PositionLimitExceeded { current, order_qty, limit } =>
                write!(f, "Position limit: current={current:.4}, order={order_qty:.4}, limit={limit:.4}"),
            Self::DailyLossExceeded { current_loss, limit } =>
                write!(f, "Daily loss: {current_loss:.2} > limit {limit:.2}"),
            Self::DrawdownExceeded { drawdown, limit } =>
                write!(f, "Drawdown: {drawdown:.2} > limit {limit:.2}"),
            Self::OrderRateExceeded { rate, limit } =>
                write!(f, "Order rate: {rate}/s > limit {limit}/s"),
            Self::OrderSizeExceeded { size, limit } =>
                write!(f, "Order size: {size:.4} > limit {limit:.4}"),
            Self::OpenOrdersExceeded { count, limit } =>
                write!(f, "Open orders: {count} > limit {limit}"),
            Self::NotionalExceeded { notional, limit } =>
                write!(f, "Notional: {notional:.2} > limit {limit:.2}"),
            Self::KillSwitchActive =>
                write!(f, "KILL SWITCH ACTIVE — all trading halted"),
        }
    }
}

// ── Status snapshot ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct RiskStatus {
    pub is_killed:            bool,
    pub position:             f64,
    pub daily_pnl:            f64,
    pub peak_pnl:             f64,
    pub drawdown:             f64,
    pub open_orders:          u32,
    pub violations_today:     usize,
    /// |position| / max_position  → [0, ∞)
    pub position_utilization: f64,
    /// |daily_pnl| / max_daily_loss → [0, ∞)
    pub loss_utilization:     f64,
}

// ── Risk manager ──────────────────────────────────────────────────────────────

pub struct RiskManager {
    config: RiskConfig,

    // ── State ────────────────────────────────────────────────────────────────
    is_killed:   bool,
    position:    f64,    // signed (+ = long, − = short)
    daily_pnl:   f64,
    peak_pnl:    f64,    // intraday PnL high-water mark
    open_orders: u32,

    // ── Rate limiting (sliding 1-second window) ───────────────────────────────
    order_timestamps: Vec<Instant>,

    // ── Diagnostics ───────────────────────────────────────────────────────────
    violations_today: usize,
}

impl RiskManager {
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            is_killed:        false,
            position:         0.0,
            daily_pnl:        0.0,
            peak_pnl:         0.0,
            open_orders:      0,
            order_timestamps: Vec::new(),
            violations_today: 0,
        }
    }

    // ── Pre-trade check ───────────────────────────────────────────────────────

    /// Gate every outbound order through this method.
    ///
    /// Returns `Ok(())` if the order is within all limits.  Returns
    /// `Err(RiskViolation)` and logs a warning for any breach.
    ///
    /// On detecting a daily-loss or drawdown breach the kill switch is
    /// activated as a side-effect; the caller must then cancel all open orders
    /// and flatten the position.
    pub fn check_order(
        &mut self,
        side:     OrderSide,
        quantity: f64,
        price:    f64,
    ) -> Result<(), RiskViolation> {
        // 0. Kill switch — checked first, no logging (already active)
        if self.is_killed {
            return Err(RiskViolation::KillSwitchActive);
        }

        // 1. Single-order size
        if quantity > self.config.max_order_size {
            return Err(self.violation(RiskViolation::OrderSizeExceeded {
                size:  quantity,
                limit: self.config.max_order_size,
            }));
        }

        // 2. Post-fill projected position
        let projected = match side {
            OrderSide::Buy  => self.position + quantity,
            OrderSide::Sell => self.position - quantity,
        };
        if projected.abs() > self.config.max_position {
            return Err(self.violation(RiskViolation::PositionLimitExceeded {
                current:   self.position,
                order_qty: quantity,
                limit:     self.config.max_position,
            }));
        }

        // 3. Daily loss (auto-kill)
        if self.daily_pnl < -self.config.max_daily_loss {
            self.trigger_kill_switch("daily loss limit exceeded");
            return Err(self.violation(RiskViolation::DailyLossExceeded {
                current_loss: self.daily_pnl.abs(),
                limit:        self.config.max_daily_loss,
            }));
        }

        // 4. Drawdown (auto-kill)
        let drawdown = self.peak_pnl - self.daily_pnl;
        if drawdown > self.config.max_drawdown {
            self.trigger_kill_switch("max drawdown exceeded");
            return Err(self.violation(RiskViolation::DrawdownExceeded {
                drawdown,
                limit: self.config.max_drawdown,
            }));
        }

        // 5. Open orders
        if self.open_orders >= self.config.max_open_orders {
            return Err(self.violation(RiskViolation::OpenOrdersExceeded {
                count: self.open_orders,
                limit: self.config.max_open_orders,
            }));
        }

        // 6. Single-order notional
        let notional = quantity * price;
        if notional > self.config.max_notional {
            return Err(self.violation(RiskViolation::NotionalExceeded {
                notional,
                limit: self.config.max_notional,
            }));
        }

        // 7. Rate limit — sliding 1-second window
        let now = Instant::now();
        self.order_timestamps.retain(|t| now.duration_since(*t).as_secs_f64() < 1.0);
        if self.order_timestamps.len() as u32 >= self.config.max_orders_per_second {
            return Err(self.violation(RiskViolation::OrderRateExceeded {
                rate:  self.order_timestamps.len() as u32,
                limit: self.config.max_orders_per_second,
            }));
        }

        self.order_timestamps.push(now);
        Ok(())
    }

    // ── Post-trade callbacks ──────────────────────────────────────────────────

    /// Call after an order is successfully submitted to the exchange.
    pub fn on_order_placed(&mut self) {
        self.open_orders += 1;
    }

    /// Call when an order is fully or partially filled.
    /// Partial fills: call with the filled quantity, not the full order qty.
    pub fn on_fill(&mut self, side: OrderSide, quantity: f64, _price: f64) {
        match side {
            OrderSide::Buy  => self.position += quantity,
            OrderSide::Sell => self.position -= quantity,
        }
        self.open_orders = self.open_orders.saturating_sub(1);
    }

    /// Call when an order is cancelled (not filled).
    pub fn on_order_cancelled(&mut self) {
        self.open_orders = self.open_orders.saturating_sub(1);
    }

    // ── PnL update ────────────────────────────────────────────────────────────

    /// Update realized + unrealized PnL.  Auto-triggers the kill switch if
    /// either daily-loss or drawdown limits are breached.
    pub fn update_pnl(&mut self, realized_pnl: f64, unrealized_pnl: f64) {
        self.daily_pnl = realized_pnl + unrealized_pnl;
        if self.daily_pnl > self.peak_pnl {
            self.peak_pnl = self.daily_pnl;
        }
        if self.daily_pnl < -self.config.max_daily_loss {
            self.trigger_kill_switch("daily loss limit on PnL update");
        }
        let drawdown = self.peak_pnl - self.daily_pnl;
        if drawdown > self.config.max_drawdown {
            self.trigger_kill_switch("drawdown limit on PnL update");
        }
    }

    // ── Kill switch ───────────────────────────────────────────────────────────

    /// Activate the kill switch.  After this all `check_order` calls return
    /// `Err(KillSwitchActive)` until `reset_kill_switch` is called.
    ///
    /// The caller is responsible for cancelling all open orders and flattening
    /// the position via a market order.
    pub fn trigger_kill_switch(&mut self, reason: &str) {
        if !self.is_killed {
            self.is_killed = true;
            tracing::error!("KILL SWITCH ACTIVATED: {reason}");
            tracing::error!(
                "  position={:.4}  daily_pnl={:.2}  drawdown={:.2}",
                self.position,
                self.daily_pnl,
                self.peak_pnl - self.daily_pnl,
            );
        }
    }

    /// Operator-initiated kill (e.g. dashboard button).
    pub fn manual_kill(&mut self) {
        self.trigger_kill_switch("manual kill from operator");
    }

    /// Re-enable trading after the kill switch was tripped.
    /// Should only be called after manual review.
    pub fn reset_kill_switch(&mut self) {
        tracing::warn!("kill switch RESET — trading will resume");
        self.is_killed = false;
    }

    // ── Daily reset ───────────────────────────────────────────────────────────

    /// Reset intraday counters at the start of a new trading day.
    pub fn reset_daily(&mut self) {
        self.daily_pnl        = 0.0;
        self.peak_pnl         = 0.0;
        self.violations_today = 0;
        tracing::info!("daily risk counters reset");
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn is_killed(&self)    -> bool { self.is_killed }
    pub fn position(&self)     -> f64  { self.position }
    pub fn daily_pnl(&self)    -> f64  { self.daily_pnl }
    pub fn drawdown(&self)     -> f64  { self.peak_pnl - self.daily_pnl }
    pub fn open_orders(&self)  -> u32  { self.open_orders }
    pub fn config(&self)       -> &RiskConfig { &self.config }

    pub fn status(&self) -> RiskStatus {
        RiskStatus {
            is_killed:            self.is_killed,
            position:             self.position,
            daily_pnl:            self.daily_pnl,
            peak_pnl:             self.peak_pnl,
            drawdown:             self.peak_pnl - self.daily_pnl,
            open_orders:          self.open_orders,
            violations_today:     self.violations_today,
            position_utilization: self.position.abs() / (self.config.max_position + f64::EPSILON),
            loss_utilization:     self.daily_pnl.abs() / (self.config.max_daily_loss + f64::EPSILON),
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Log the violation, bump the counter, and return it so the caller
    /// can use it directly in `Err(self.violation(...))`.
    fn violation(&mut self, v: RiskViolation) -> RiskViolation {
        self.violations_today += 1;
        tracing::warn!("risk violation: {v}");
        v
    }
}
