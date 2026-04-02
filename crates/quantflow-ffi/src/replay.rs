use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use quantflow_core::data::replay::ReplayEngine;

/// Python-facing wrapper around `ReplayEngine`.
///
/// Example
/// -------
/// ```python
/// from quantflow import ReplayEngine
/// engine = ReplayEngine("data/btcusdt/processed/2026-04-02_events.parquet")
/// engine.reset()
/// while (event := engine.next_event()) is not None:
///     print(event["timestamp"], event["price"])
/// ```
#[pyclass(name = "ReplayEngine")]
pub struct PyReplayEngine {
    inner: ReplayEngine,
}

#[pymethods]
impl PyReplayEngine {
    /// Load events from a ``{date}_events.parquet`` file produced by
    /// ``normalize_btcusdt``.
    #[new]
    fn new(parquet_path: &str) -> PyResult<Self> {
        let engine = ReplayEngine::from_parquet(std::path::Path::new(parquet_path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: engine })
    }

    /// Advance by one event and return it as a dict, or ``None`` at session end.
    ///
    /// Keys: ``timestamp`` (float), ``event_type`` (int 0–11),
    ///       ``price`` (float), ``quantity`` (float), ``raw_timestamp_ms`` (int).
    pub fn next_event(&mut self, py: Python) -> PyResult<PyObject> {
        match self.inner.next_event() {
            None => Ok(py.None()),
            Some(e) => {
                let d = PyDict::new(py);
                d.set_item("timestamp",        e.timestamp)?;
                d.set_item("event_type",       e.event_type.hawkes_dim())?;
                d.set_item("price",            e.price)?;
                d.set_item("quantity",         e.quantity)?;
                d.set_item("raw_timestamp_ms", e.raw_timestamp_ms)?;
                Ok(d.into_any().unbind())
            }
        }
    }

    /// Advance by up to ``n`` events and return them as a list of dicts.
    pub fn step_n(&mut self, py: Python, n: usize) -> PyResult<PyObject> {
        let batch = self.inner.step_n(n);
        let list = PyList::empty(py);
        for e in &batch {
            let d = PyDict::new(py);
            d.set_item("timestamp",        e.timestamp)?;
            d.set_item("event_type",       e.event_type.hawkes_dim())?;
            d.set_item("price",            e.price)?;
            d.set_item("quantity",         e.quantity)?;
            d.set_item("raw_timestamp_ms", e.raw_timestamp_ms)?;
            list.append(d)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Current mid-price, or ``None`` if the book has no quotes yet.
    pub fn mid_price(&self) -> Option<f64> {
        self.inner.mid_price()
    }

    /// Best bid price, or ``None``.
    pub fn best_bid(&self) -> Option<f64> {
        self.inner.best_bid()
    }

    /// Best ask price, or ``None``.
    pub fn best_ask(&self) -> Option<f64> {
        self.inner.best_ask()
    }

    /// Return a snapshot of the current LOB as a dict with keys
    /// ``bids`` and ``asks``, each a list of ``[price, quantity]`` pairs.
    pub fn snapshot(&self, py: Python, levels: usize) -> PyResult<PyObject> {
        let snap = self.inner.snapshot(levels);
        let bids = PyList::empty(py);
        for (p, q) in &snap.bids {
            bids.append(vec![p.to_f64(), q.0 as f64 / 1_000.0])?;
        }
        let asks = PyList::empty(py);
        for (p, q) in &snap.asks {
            asks.append(vec![p.to_f64(), q.0 as f64 / 1_000.0])?;
        }
        let d = PyDict::new(py);
        d.set_item("bids", bids)?;
        d.set_item("asks", asks)?;
        Ok(d.into_any().unbind())
    }

    /// Reset to the beginning of the session (cursor = 0, empty book).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Fraction of events consumed (0.0–1.0).
    pub fn progress(&self) -> f64 {
        self.inner.progress()
    }

    /// Number of events not yet consumed.
    pub fn remaining(&self) -> usize {
        self.inner.remaining()
    }

    /// Total events in the session.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
