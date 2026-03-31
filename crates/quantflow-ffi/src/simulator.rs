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
    #[staticmethod]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<&Bound<PyDict>>) -> PyResult<Self> {
        let mut sim = HawkesLobSimulator::default_12d()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if let Some(cfg) = config {
            if let Ok(Some(v)) = cfg.get_item("t_max") {
                sim.config.t_max = v.extract::<f64>()?;
            }
            if let Ok(Some(v)) = cfg.get_item("snapshot_interval") {
                sim.config.snapshot_interval = v.extract::<usize>()?;
            }
            if let Ok(Some(v)) = cfg.get_item("initial_mid") {
                use quantflow_core::orderbook::types::Price;
                sim.config.initial_mid = Price::from_f64(v.extract::<f64>()?);
            }
        }

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
