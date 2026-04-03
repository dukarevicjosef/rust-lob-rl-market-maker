use std::sync::Arc;

use arrow::array::{Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_arrow::PyRecordBatch;

use quantflow_core::simulator::HawkesLobSimulator;

use crate::orderbook::PyOrderBook;

// ── PyHawkesSimulator ─────────────────────────────────────────────────────────

/// Python-facing wrapper around the 12-dimensional Hawkes-driven LOB simulator.
///
/// Example
/// -------
/// ```python
/// from quantflow import HawkesSimulator
/// sim = HawkesSimulator.new()
/// sim.reset(42)
/// while (event := sim.step()) is not None:
///     if event["trades"]:
///         print(event["trades"])
/// ```
#[pyclass(name = "HawkesSimulator")]
pub struct PyHawkesSimulator {
    inner: HawkesLobSimulator,
}

#[pymethods]
impl PyHawkesSimulator {
    /// Create a simulator with the default 12-dimensional parameter set.
    ///
    /// Optional ``config`` dict may contain:
    /// - ``t_max`` (float): trading day length in seconds (default 23400)
    /// - ``snapshot_interval`` (int): events between book snapshots (default 100)
    /// - ``initial_mid`` (float): initial mid-price (default 100.0)
    /// - ``tick_size_f`` (float): tick size in price units (default 1.0)
    ///
    /// Calibrated Hawkes parameters (all three must be provided together):
    /// - ``hawkes_mu``    (list[float]): baseline rates μ_i, length 12
    /// - ``hawkes_alpha`` (list[list[float]]): branching-ratio matrix n*_ij = α_py/β_py, 12×12
    /// - ``hawkes_beta``  (list[list[float]]): decay-rate matrix β_ij, 12×12
    ///
    /// When ``hawkes_mu/alpha/beta`` are all present the calibrated constructor
    /// is used instead of ``default_12d()``.
    #[staticmethod]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<&Bound<PyDict>>) -> PyResult<Self> {
        use quantflow_core::orderbook::types::{Price, PRICE_SCALE};
        use quantflow_core::simulator::SimulatorConfig;

        // Start with a base SimulatorConfig (not the full simulator yet).
        let mut base_cfg = SimulatorConfig::default();

        // Pre-parse scalar overrides so they apply regardless of Hawkes path.
        let mut hawkes_mu:    Option<Vec<f64>>         = None;
        let mut hawkes_alpha: Option<Vec<Vec<f64>>>    = None;
        let mut hawkes_beta:  Option<Vec<Vec<f64>>>    = None;

        if let Some(cfg) = config {
            if let Ok(Some(v)) = cfg.get_item("t_max") {
                base_cfg.t_max = v.extract::<f64>()?;
            }
            if let Ok(Some(v)) = cfg.get_item("snapshot_interval") {
                base_cfg.snapshot_interval = v.extract::<usize>()?;
            }
            if let Ok(Some(v)) = cfg.get_item("initial_mid") {
                base_cfg.initial_mid = Price::from_f64(v.extract::<f64>()?);
            }
            if let Ok(Some(v)) = cfg.get_item("tick_size_f") {
                let tick_f = v.extract::<f64>()?;
                base_cfg.tick_size = (tick_f * PRICE_SCALE as f64).round() as i64;
            }
            if let Ok(Some(v)) = cfg.get_item("hawkes_mu") {
                hawkes_mu = Some(v.extract::<Vec<f64>>()?);
            }
            if let Ok(Some(v)) = cfg.get_item("hawkes_alpha") {
                hawkes_alpha = Some(v.extract::<Vec<Vec<f64>>>()?);
            }
            if let Ok(Some(v)) = cfg.get_item("hawkes_beta") {
                hawkes_beta = Some(v.extract::<Vec<Vec<f64>>>()?);
            }
        }

        // Choose constructor: calibrated if all three Hawkes params are present.
        let sim = match (hawkes_mu, hawkes_alpha, hawkes_beta) {
            (Some(mu), Some(alpha), Some(beta)) => {
                HawkesLobSimulator::from_calibrated(mu, alpha, beta, base_cfg)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            _ => {
                let mut s = HawkesLobSimulator::default_12d()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                s.config = base_cfg;
                s
            }
        };

        Ok(PyHawkesSimulator { inner: sim })
    }

    /// Re-initialise the simulator with a new random seed.
    pub fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// Advance by one event.
    ///
    /// Returns a dict with keys:
    ///   ``sim_time`` (float), ``event_type`` (int 0-11),
    ///   ``trades`` (list[dict]), ``has_snapshot`` (bool).
    ///
    /// Returns ``None`` when the simulated day is exhausted.
    pub fn step(&mut self, py: Python) -> PyResult<PyObject> {
        match self.inner.step() {
            None => Ok(py.None()),
            Some(event) => {
                let trades_list = PyList::empty(py);
                for t in &event.trades {
                    let td = PyDict::new(py);
                    td.set_item("price",    t.price.to_f64())?;
                    td.set_item("qty",      t.quantity.0)?;
                    td.set_item("maker_id", t.maker_id.0)?;
                    td.set_item("taker_id", t.taker_id.0)?;
                    trades_list.append(td)?;
                }
                let d = PyDict::new(py);
                d.set_item("sim_time",    event.sim_time)?;
                d.set_item("event_type",  event.event_type)?;
                d.set_item("trades",      trades_list)?;
                d.set_item("has_snapshot", event.snapshot.is_some())?;
                Ok(d.into_any().unbind())
            }
        }
    }

    /// Run a complete simulated trading day and return statistics.
    ///
    /// Returns a dict with keys:
    ///   ``n_events`` (int), ``sim_time`` (float),
    ///   ``trades`` (Arrow RecordBatch), ``n_trades`` (int).
    pub fn simulate_day(&mut self, py: Python, seed: u64) -> PyResult<PyObject> {
        let result = self.inner.simulate_trading_day(seed);

        let n_trades = result.trades.len();
        let trades_batch = trades_to_record_batch(&result.trades)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let trades_rb = PyRecordBatch::new(trades_batch);
        let trades_py = trades_rb.to_pyarrow(py)?;

        let d = PyDict::new(py);
        d.set_item("n_events",  result.n_events)?;
        d.set_item("sim_time",  result.sim_time)?;
        d.set_item("n_trades",  n_trades)?;
        d.set_item("trades",    trades_py)?;
        Ok(d.into_any().unbind())
    }

    /// Return a snapshot of the current order book as a ``PyOrderBook``.
    pub fn get_book(&self) -> PyOrderBook {
        PyOrderBook::from_book(self.inner.book().clone())
    }

    /// Current mid-price, or ``None`` if the book is empty.
    pub fn mid_price(&self) -> Option<f64> {
        self.inner.mid_price()
    }

    /// Place a limit order on the simulator's live book.
    ///
    /// Returns the order ID (u64) that can be passed to ``cancel_agent_order``.
    pub fn place_limit_order(&mut self, side: &str, price: f64, qty: u64) -> PyResult<u64> {
        use quantflow_core::orderbook::types::{Price, Quantity, Side};
        let side = match side.to_lowercase().as_str() {
            "bid" | "buy"  => Side::Bid,
            "ask" | "sell" => Side::Ask,
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown side '{other}'; use 'bid' or 'ask'"),
            )),
        };
        let oid = self.inner.place_limit_order(side, Price::from_f64(price), Quantity(qty));
        Ok(oid.0)
    }

    /// Cancel a previously placed agent order by ID.
    pub fn cancel_agent_order(&mut self, order_id: u64) {
        use quantflow_core::orderbook::types::OrderId;
        self.inner.cancel_agent_order(OrderId(order_id));
    }

    /// Total simulation horizon in seconds (from config).
    pub fn t_max(&self) -> f64 {
        self.inner.config.t_max
    }

    /// Set the initial mid-price used when the simulator is next reset.
    pub fn set_initial_mid(&mut self, mid: f64) {
        use quantflow_core::orderbook::types::Price;
        self.inner.config.initial_mid = Price::from_f64(mid);
    }

    /// Set the log-normal order-size σ for the next reset onward.
    ///
    /// Higher σ → wider order-size distribution → greater average price impact.
    pub fn set_lognormal_sigma(&mut self, sigma: f64) {
        self.inner.config.lognormal_sigma = sigma.max(0.1);
    }

    /// Set the log-normal order-size μ for the next reset onward.
    pub fn set_lognormal_mu(&mut self, mu: f64) {
        self.inner.config.lognormal_mu = mu.max(0.0);
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn trades_to_record_batch(
    trades: &[quantflow_core::orderbook::order::Trade],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let prices:    Float64Array = trades.iter().map(|t| t.price.to_f64()).collect();
    let qtys:      UInt64Array  = trades.iter().map(|t| t.quantity.0).collect();
    let maker_ids: UInt64Array  = trades.iter().map(|t| t.maker_id.0).collect();
    let taker_ids: UInt64Array  = trades.iter().map(|t| t.taker_id.0).collect();
    let ts:        UInt64Array  = trades.iter().map(|t| t.timestamp.0).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("price",     DataType::Float64, false),
        Field::new("qty",       DataType::UInt64,  false),
        Field::new("maker_id",  DataType::UInt64,  false),
        Field::new("taker_id",  DataType::UInt64,  false),
        Field::new("timestamp", DataType::UInt64,  false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(prices),
            Arc::new(qtys),
            Arc::new(maker_ids),
            Arc::new(taker_ids),
            Arc::new(ts),
        ],
    )
}
