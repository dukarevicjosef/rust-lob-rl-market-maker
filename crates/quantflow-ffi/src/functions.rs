use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int8Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_arrow::PyRecordBatch;

use quantflow_core::data::lobster::{replay_messages, load_snapshots};
use quantflow_core::hawkes::calibration::calibrate_exponential;
use quantflow_core::hawkes::process::HawkesEvent;

// ── calibrate_hawkes ──────────────────────────────────────────────────────────

/// Calibrate an exponential Hawkes process to a sequence of events.
///
/// Parameters
/// ----------
/// events : list[dict]
///     Each dict must have ``"time"`` (float) and ``"event_type"`` (int).
/// d : int, optional
///     Number of event types (dimensions).  If omitted, inferred as
///     ``max(event_type) + 1``.
///
/// Returns
/// -------
/// dict with keys:
///   ``mu`` (list[float]), ``alpha`` (list[list[float]]),
///   ``beta`` (list[list[float]]), ``nll`` (float),
///   ``converged`` (bool), ``n_iter`` (int).
#[pyfunction]
#[pyo3(signature = (events, d=None))]
pub fn calibrate_hawkes(
    py: Python,
    events: &Bound<PyList>,
    d: Option<usize>,
) -> PyResult<PyObject> {
    // Parse event list.
    let mut raw: Vec<HawkesEvent> = Vec::with_capacity(events.len());
    for item in events.iter() {
        let item = item.downcast::<pyo3::types::PyDict>()?;
        let time: f64    = item.get_item("time")?.ok_or_else(|| {
            PyRuntimeError::new_err("each event dict must have key 'time'")
        })?.extract()?;
        let etype: usize = item.get_item("event_type")?.ok_or_else(|| {
            PyRuntimeError::new_err("each event dict must have key 'event_type'")
        })?.extract()?;
        raw.push(HawkesEvent { time, event_type: etype });
    }

    let dim = d.unwrap_or_else(|| {
        raw.iter().map(|e| e.event_type).max().map(|m| m + 1).unwrap_or(1)
    });

    let t_max = raw.iter().map(|e| e.time).fold(0.0_f64, f64::max);
    let result = calibrate_exponential(&raw, dim, t_max)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let p = &result.params;
    let out = PyDict::new(py);
    out.set_item("mu",        p.mu.clone())?;
    out.set_item("alpha",     p.alpha.clone())?;
    out.set_item("beta",      p.beta.clone())?;
    out.set_item("nll",       result.nll)?;
    out.set_item("converged", result.converged)?;
    out.set_item("n_iter",    result.n_iter)?;
    Ok(out.into_any().unbind())
}

// ── load_lobster ──────────────────────────────────────────────────────────────

/// Parse LOBSTER message and book snapshot CSV files.
///
/// Parameters
/// ----------
/// message_path : str
///     Path to the LOBSTER message file (``*_message_*.csv``).
/// book_path : str
///     Path to the LOBSTER orderbook snapshot file (``*_orderbook_*.csv``).
///     Pass an empty string ``""`` to skip snapshot loading.
///
/// Returns
/// -------
/// dict with keys:
///   ``messages`` (Arrow RecordBatch): columns timestamp(f64),
///      event_type(str), order_id(u64), size(u64), price(i64), direction(i8).
///   ``snapshots`` (Arrow RecordBatch or None): first 10 bid/ask levels
///      as f64 columns.  None if book_path is empty.
#[pyfunction]
pub fn load_lobster(
    py: Python,
    message_path: &str,
    book_path: &str,
) -> PyResult<PyObject> {
    // ── messages ──────────────────────────────────────────────────────────────
    let msg_iter = replay_messages(Path::new(message_path))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut timestamps:   Vec<f64>   = Vec::new();
    let mut event_types:  Vec<String> = Vec::new();
    let mut order_ids:    Vec<u64>   = Vec::new();
    let mut sizes:        Vec<u64>   = Vec::new();
    let mut prices:       Vec<i64>   = Vec::new();
    let mut directions:   Vec<i8>    = Vec::new();

    for msg in msg_iter {
        let msg = msg.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        timestamps.push(msg.timestamp);
        event_types.push(format!("{:?}", msg.event_type));
        order_ids.push(msg.order_id);
        sizes.push(msg.size);
        prices.push(msg.price);
        directions.push(msg.direction);
    }

    let msg_schema = Arc::new(Schema::new(vec![
        Field::new("timestamp",  DataType::Float64, false),
        Field::new("event_type", DataType::Utf8,    false),
        Field::new("order_id",   DataType::UInt64,  false),
        Field::new("size",       DataType::UInt64,  false),
        Field::new("price",      DataType::Int64,   false),
        Field::new("direction",  DataType::Int8,    false),
    ]));

    let msg_batch = RecordBatch::try_new(
        msg_schema,
        vec![
            Arc::new(Float64Array::from(timestamps)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(UInt64Array::from(order_ids)),
            Arc::new(UInt64Array::from(sizes)),
            Arc::new(Int64Array::from(prices)),
            Arc::new(Int8Array::from(directions)),
        ],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let msg_rb = PyRecordBatch::new(msg_batch);
    let msg_py = msg_rb.to_pyarrow(py)?;

    // ── snapshots (optional) ──────────────────────────────────────────────────
    let snap_py = if book_path.is_empty() {
        py.None()
    } else {
        let snap_iter = load_snapshots(Path::new(book_path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        const LEVELS: usize = 10;
        let mut ts_vec: Vec<f64> = Vec::new();
        let mut bid_p: Vec<Vec<f64>> = vec![Vec::new(); LEVELS];
        let mut bid_q: Vec<Vec<u64>> = vec![Vec::new(); LEVELS];
        let mut ask_p: Vec<Vec<f64>> = vec![Vec::new(); LEVELS];
        let mut ask_q: Vec<Vec<u64>> = vec![Vec::new(); LEVELS];

        for snap in snap_iter {
            let snap = snap.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let d = snap.depth().min(LEVELS);
            ts_vec.push(0.0); // LOBSTER snapshots have no independent timestamp
            for lvl in 0..LEVELS {
                if lvl < d {
                    bid_p[lvl].push(snap.levels[lvl].bid_price as f64 / 10_000.0);
                    bid_q[lvl].push(snap.levels[lvl].bid_size);
                    ask_p[lvl].push(snap.levels[lvl].ask_price as f64 / 10_000.0);
                    ask_q[lvl].push(snap.levels[lvl].ask_size);
                } else {
                    bid_p[lvl].push(f64::NAN);
                    bid_q[lvl].push(0);
                    ask_p[lvl].push(f64::NAN);
                    ask_q[lvl].push(0);
                }
            }
        }

        let n = ts_vec.len();
        if n == 0 {
            py.None()
        } else {
            let mut fields = vec![Field::new("row", DataType::UInt64, false)];
            let mut arrays: Vec<Arc<dyn arrow::array::Array>> =
                vec![Arc::new(UInt64Array::from_iter(0..n as u64))];

            for lvl in 0..LEVELS {
                fields.push(Field::new(format!("bid_price_{lvl}"), DataType::Float64, true));
                fields.push(Field::new(format!("bid_qty_{lvl}"),   DataType::UInt64,  false));
                fields.push(Field::new(format!("ask_price_{lvl}"), DataType::Float64, true));
                fields.push(Field::new(format!("ask_qty_{lvl}"),   DataType::UInt64,  false));
                arrays.push(Arc::new(Float64Array::from(bid_p[lvl].clone())));
                arrays.push(Arc::new(UInt64Array::from(bid_q[lvl].clone())));
                arrays.push(Arc::new(Float64Array::from(ask_p[lvl].clone())));
                arrays.push(Arc::new(UInt64Array::from(ask_q[lvl].clone())));
            }

            let snap_batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            PyRecordBatch::new(snap_batch).to_pyarrow(py)?
        }
    };

    let out = PyDict::new(py);
    out.set_item("messages",  msg_py)?;
    out.set_item("snapshots", snap_py)?;
    Ok(out.into_any().unbind())
}
