use pyo3::prelude::*;

/// Python-facing quantflow module — thin wrappers over quantflow-core types.
#[pymodule]
fn quantflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
