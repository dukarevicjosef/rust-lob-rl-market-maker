use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use quantflow_core::exchange::{
    risk_manager::{RiskConfig, RiskManager},
    OrderSide,
};

// ── PyRiskManager ─────────────────────────────────────────────────────────────

/// Pre-trade risk gate sitting between the agent and the exchange.
///
/// Every outbound order must pass through ``check_order``.  The manager
/// tracks position, intraday PnL, drawdown, open-order count, and order
/// rate.  Breaching any limit raises ``ValueError``; breaching the daily-loss
/// or drawdown limits also activates the kill switch which blocks all further
/// orders until manually reset.
///
/// Config keys (all optional, testnet defaults if omitted)
/// -------------------------------------------------------
/// max_position          float  — max |position| in base asset   (default 0.1)
/// max_daily_loss        float  — max intraday loss in quote asset (default 100)
/// max_drawdown          float  — max peak-to-trough drawdown      (default 200)
/// max_orders_per_second int    — rate limit                       (default 5)
/// max_order_size        float  — max single order in base asset   (default 0.05)
/// max_open_orders       int    — max resting orders               (default 10)
/// max_notional          float  — max single-order notional        (default 10000)
///
/// Example
/// -------
/// >>> rm = RiskManager({"max_position": 0.05, "max_daily_loss": 50.0})
/// >>> rm.check_order("buy", 0.01, 67000.0)   # True
/// >>> rm.on_fill("buy", 0.01, 67000.0)
/// >>> rm.update_pnl(-55.0, 0.0)              # raises ValueError (kill switch)
#[pyclass(name = "RiskManager")]
pub struct PyRiskManager {
    inner: RiskManager,
}

#[pymethods]
impl PyRiskManager {
    /// Construct a RiskManager from an optional config dict.
    ///
    /// Pass ``testnet=True`` (default) to use testnet defaults as the baseline;
    /// pass ``testnet=False`` to use the stricter mainnet baseline.
    /// Any keys present in the dict override the baseline value.
    #[new]
    #[pyo3(signature = (config=None, testnet=true))]
    pub fn new(config: Option<&Bound<'_, PyDict>>, testnet: bool) -> PyResult<Self> {
        let base = if testnet { RiskConfig::testnet() } else { RiskConfig::mainnet() };

        let cfg = if let Some(d) = config {
            let get_f = |key: &str, default: f64| -> PyResult<f64> {
                match d.get_item(key)? {
                    Some(v) => v.extract::<f64>(),
                    None    => Ok(default),
                }
            };
            let get_u = |key: &str, default: u32| -> PyResult<u32> {
                match d.get_item(key)? {
                    Some(v) => v.extract::<u32>(),
                    None    => Ok(default),
                }
            };
            RiskConfig {
                max_position:          get_f("max_position",          base.max_position)?,
                max_daily_loss:        get_f("max_daily_loss",        base.max_daily_loss)?,
                max_drawdown:          get_f("max_drawdown",          base.max_drawdown)?,
                max_orders_per_second: get_u("max_orders_per_second", base.max_orders_per_second)?,
                max_order_size:        get_f("max_order_size",        base.max_order_size)?,
                max_open_orders:       get_u("max_open_orders",       base.max_open_orders)?,
                max_notional:          get_f("max_notional",          base.max_notional)?,
            }
        } else {
            base
        };

        Ok(Self { inner: RiskManager::new(cfg) })
    }

    // ── Pre-trade gate ────────────────────────────────────────────────────────

    /// Check whether an order is permitted.
    ///
    /// Parameters
    /// ----------
    /// side : str
    ///     ``"buy"`` or ``"sell"`` (case-insensitive).
    /// quantity : float
    ///     Order size in base-asset units.
    /// price : float
    ///     Limit price in quote-asset units (used for notional check).
    ///
    /// Returns
    /// -------
    /// True if the order passes all checks.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     With a human-readable message describing the violated limit.
    pub fn check_order(&mut self, side: &str, quantity: f64, price: f64) -> PyResult<bool> {
        let s = parse_side(side)?;
        self.inner
            .check_order(s, quantity, price)
            .map(|()| true)
            .map_err(|v| PyValueError::new_err(v.to_string()))
    }

    // ── Post-trade callbacks ──────────────────────────────────────────────────

    /// Notify the manager that an order was fully or partially filled.
    ///
    /// Updates position and decrements the open-order counter.
    pub fn on_fill(&mut self, side: &str, quantity: f64, price: f64) -> PyResult<()> {
        let s = parse_side(side)?;
        self.inner.on_fill(s, quantity, price);
        Ok(())
    }

    /// Notify the manager that an order was accepted by the exchange.
    ///
    /// Increments the open-order counter.
    pub fn on_order_placed(&mut self) {
        self.inner.on_order_placed();
    }

    /// Notify the manager that an order was cancelled (not filled).
    ///
    /// Decrements the open-order counter.
    pub fn on_order_cancelled(&mut self) {
        self.inner.on_order_cancelled();
    }

    // ── PnL tracking ─────────────────────────────────────────────────────────

    /// Update the intraday PnL.
    ///
    /// Triggers the kill switch automatically if either the daily-loss or
    /// drawdown limit is breached.
    pub fn update_pnl(&mut self, realized: f64, unrealized: f64) {
        self.inner.update_pnl(realized, unrealized);
    }

    // ── Kill switch ───────────────────────────────────────────────────────────

    /// Activate the kill switch immediately (operator-triggered).
    ///
    /// After this all ``check_order`` calls raise ``ValueError`` until
    /// ``reset_kill_switch()`` is called.
    pub fn manual_kill(&mut self) {
        self.inner.manual_kill();
    }

    /// Re-enable trading after the kill switch was tripped.
    ///
    /// Should only be called after a manual review of the position and risk state.
    pub fn reset_kill_switch(&mut self) {
        self.inner.reset_kill_switch();
    }

    /// Reset intraday PnL, peak PnL, and violation counter.
    ///
    /// Typically called at the start of a new trading day.
    pub fn reset_daily(&mut self) {
        self.inner.reset_daily();
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[getter]
    pub fn is_killed(&self) -> bool { self.inner.is_killed() }

    #[getter]
    pub fn position(&self) -> f64 { self.inner.position() }

    #[getter]
    pub fn daily_pnl(&self) -> f64 { self.inner.daily_pnl() }

    #[getter]
    pub fn drawdown(&self) -> f64 { self.inner.drawdown() }

    /// Full status snapshot as a dict.
    ///
    /// Keys: is_killed, position, daily_pnl, peak_pnl, drawdown,
    ///       open_orders, violations_today, position_utilization,
    ///       loss_utilization
    pub fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.status();
        let d = PyDict::new(py);
        d.set_item("is_killed",            s.is_killed)?;
        d.set_item("position",             s.position)?;
        d.set_item("daily_pnl",            s.daily_pnl)?;
        d.set_item("peak_pnl",             s.peak_pnl)?;
        d.set_item("drawdown",             s.drawdown)?;
        d.set_item("open_orders",          s.open_orders)?;
        d.set_item("violations_today",     s.violations_today)?;
        d.set_item("position_utilization", s.position_utilization)?;
        d.set_item("loss_utilization",     s.loss_utilization)?;
        Ok(d)
    }

    pub fn __repr__(&self) -> String {
        let s = self.inner.status();
        format!(
            "RiskManager(killed={}, pos={:.4}, pnl={:.2}, drawdown={:.2}, open_orders={})",
            s.is_killed, s.position, s.daily_pnl, s.drawdown, s.open_orders,
        )
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_side(side: &str) -> PyResult<OrderSide> {
    match side.to_ascii_lowercase().as_str() {
        "buy"  => Ok(OrderSide::Buy),
        "sell" => Ok(OrderSide::Sell),
        other  => Err(PyValueError::new_err(format!(
            "invalid side '{other}': expected 'buy' or 'sell'"
        ))),
    }
}
