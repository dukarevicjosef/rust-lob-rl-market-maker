use std::sync::Arc;

use arrow::array::{Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_arrow::PyRecordBatch;

use quantflow_core::orderbook::book::OrderBook;
use quantflow_core::orderbook::order::Order;
use quantflow_core::orderbook::types::{OrderId, OrderType, Price, Quantity, Side, Timestamp};

// ── PyOrderBook ───────────────────────────────────────────────────────────────

/// Python-facing wrapper around `OrderBook`.
///
/// Example
/// -------
/// ```python
/// from quantflow import OrderBook
/// book = OrderBook()
/// report = book.add_limit_order("bid", 99.5, 100)
/// print(book.best_bid())   # (99.5, 100)
/// rb = book.snapshot(5)    # pyarrow-compatible RecordBatch
/// ```
#[pyclass(name = "OrderBook")]
pub struct PyOrderBook {
    pub(crate) inner: OrderBook,
    next_id: u64,
    next_ts: u64,
}

impl PyOrderBook {
    pub(crate) fn from_book(book: OrderBook) -> Self {
        PyOrderBook { inner: book, next_id: 1_000_000, next_ts: 0 }
    }
}

fn parse_side(s: &str) -> PyResult<Side> {
    match s.to_lowercase().as_str() {
        "bid" | "buy" | "b" => Ok(Side::Bid),
        "ask" | "sell" | "a" | "s" => Ok(Side::Ask),
        other => Err(PyRuntimeError::new_err(format!("unknown side '{other}'; use 'bid' or 'ask'"))),
    }
}

#[pymethods]
impl PyOrderBook {
    /// Create an empty order book.
    #[new]
    pub fn new() -> Self {
        PyOrderBook { inner: OrderBook::new(), next_id: 1, next_ts: 0 }
    }

    /// Place a limit order.
    ///
    /// Parameters
    /// ----------
    /// side : str
    ///     ``"bid"`` / ``"buy"`` or ``"ask"`` / ``"sell"``.
    /// price : float
    ///     Limit price in native units (e.g. 99.50).
    /// qty : int
    ///     Order quantity in lots.
    ///
    /// Returns
    /// -------
    /// dict with keys ``order_id``, ``filled_qty``, ``remaining_qty``,
    /// ``is_fully_filled``, ``trades`` (list of dicts).
    pub fn add_limit_order(
        &mut self,
        py: Python,
        side: &str,
        price: f64,
        qty: u64,
    ) -> PyResult<PyObject> {
        let side = parse_side(side)?;
        let order = Order::new(
            OrderId(self.next_id),
            side,
            Price::from_f64(price),
            Quantity(qty),
            Timestamp(self.next_ts),
            OrderType::Limit,
        );
        let oid = self.next_id;
        self.next_id += 1;
        self.next_ts += 1;

        let report = self.inner.add_limit_order(order);

        let trades_list = PyList::empty(py);
        for t in &report.trades {
            let td = PyDict::new(py);
            td.set_item("price", t.price.to_f64())?;
            td.set_item("qty", t.quantity.0)?;
            td.set_item("maker_id", t.maker_id.0)?;
            td.set_item("taker_id", t.taker_id.0)?;
            trades_list.append(td)?;
        }

        let d = PyDict::new(py);
        d.set_item("order_id", oid)?;
        d.set_item("filled_qty", report.filled_quantity().0)?;
        d.set_item("remaining_qty", report.remaining.as_ref().map(|o| o.quantity.0).unwrap_or(0))?;
        d.set_item("is_fully_filled", report.is_fully_filled())?;
        d.set_item("trades", trades_list)?;
        Ok(d.into_any().unbind())
    }

    /// Cancel a resting order by its ID.
    ///
    /// Returns a dict describing the cancelled order, or ``None`` if not found.
    pub fn cancel_order(&mut self, py: Python, order_id: u64) -> PyResult<PyObject> {
        match self.inner.cancel_order(OrderId(order_id)) {
            None => Ok(py.None()),
            Some(o) => {
                let d = PyDict::new(py);
                d.set_item("order_id", o.id.0)?;
                d.set_item("side", format!("{:?}", o.side).to_lowercase())?;
                d.set_item("price", o.price.to_f64())?;
                d.set_item("remaining_qty", o.quantity.0)?;
                Ok(d.into_any().unbind())
            }
        }
    }

    /// Best bid as ``(price: float, qty: int)`` or ``None``.
    pub fn best_bid(&self, py: Python) -> PyObject {
        match self.inner.best_bid() {
            Some((p, q)) => (p.to_f64(), q.0).into_pyobject(py).unwrap().into_any().unbind(),
            None => py.None(),
        }
    }

    /// Best ask as ``(price: float, qty: int)`` or ``None``.
    pub fn best_ask(&self, py: Python) -> PyObject {
        match self.inner.best_ask() {
            Some((p, q)) => (p.to_f64(), q.0).into_pyobject(py).unwrap().into_any().unbind(),
            None => py.None(),
        }
    }

    /// Mid-price as ``float`` or ``None`` (empty book).
    pub fn mid_price(&self) -> Option<f64> {
        self.inner.mid_price().map(|p| p.to_f64())
    }

    /// Bid-ask spread as ``float`` or ``None``.
    pub fn spread(&self) -> Option<f64> {
        self.inner.spread().map(|p| p.to_f64().abs())
    }

    /// Return the top *levels* of the book as a zero-copy Arrow RecordBatch.
    ///
    /// Columns: ``bid_price`` (f64), ``bid_qty`` (u64),
    ///          ``ask_price`` (f64), ``ask_qty`` (u64).
    ///
    /// Rows are ordered best-to-worst.  Missing levels are padded with 0.0 / 0.
    #[pyo3(signature = (levels=10))]
    pub fn snapshot(&self, py: Python, levels: usize) -> PyResult<PyObject> {
        let bids = self.inner.depth(Side::Bid, levels);
        let asks = self.inner.depth(Side::Ask, levels);

        let mut bid_prices = vec![f64::NAN; levels];
        let mut bid_qtys   = vec![0u64;   levels];
        let mut ask_prices = vec![f64::NAN; levels];
        let mut ask_qtys   = vec![0u64;   levels];

        for (i, (p, q)) in bids.iter().enumerate() {
            bid_prices[i] = p.to_f64();
            bid_qtys[i]   = q.0;
        }
        for (i, (p, q)) in asks.iter().enumerate() {
            ask_prices[i] = p.to_f64();
            ask_qtys[i]   = q.0;
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("bid_price", DataType::Float64, true),
            Field::new("bid_qty",   DataType::UInt64,  false),
            Field::new("ask_price", DataType::Float64, true),
            Field::new("ask_qty",   DataType::UInt64,  false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(bid_prices)),
                Arc::new(UInt64Array::from(bid_qtys)),
                Arc::new(Float64Array::from(ask_prices)),
                Arc::new(UInt64Array::from(ask_qtys)),
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        PyRecordBatch::new(batch).to_pyarrow(py)
    }
}
