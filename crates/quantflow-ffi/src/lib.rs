mod functions;
mod orderbook;
mod replay;
mod simulator;
mod strategy;

use pyo3::prelude::*;

use functions::{calibrate_hawkes, load_lobster};
use orderbook::PyOrderBook;
use replay::PyReplayEngine;
use simulator::PyHawkesSimulator;
use strategy::PyAvellanedaStoikov;

/// High-performance LOB engine and Hawkes-process market simulator.
///
/// Classes
/// -------
/// OrderBook            -- BTreeMap-based LOB with price-time priority matching
/// HawkesSimulator      -- 12-dimensional Hawkes-driven LOB event generator
/// AvellanedaStoikov    -- Optimal market-making quotes (AS 2008)
/// ReplayEngine         -- Historical Parquet playback through the LOB engine
///
/// Functions
/// ---------
/// calibrate_hawkes     -- MLE fit of exponential Hawkes process
/// load_lobster         -- Parse LOBSTER CSV files → Arrow RecordBatches
#[pymodule]
fn quantflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrderBook>()?;
    m.add_class::<PyHawkesSimulator>()?;
    m.add_class::<PyAvellanedaStoikov>()?;
    m.add_class::<PyReplayEngine>()?;
    m.add_function(wrap_pyfunction!(calibrate_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(load_lobster, m)?)?;
    Ok(())
}
