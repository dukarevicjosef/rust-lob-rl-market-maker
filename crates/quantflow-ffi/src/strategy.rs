use pyo3::prelude::*;

use quantflow_core::strategy::AvellanedaStoikov;

// ── PyAvellanedaStoikov ───────────────────────────────────────────────────────

/// Python-facing wrapper around the Avellaneda-Stoikov market-making strategy.
///
/// Example
/// -------
/// ```python
/// from quantflow import AvellanedaStoikov
///
/// # Fixed σ
/// strat = AvellanedaStoikov(gamma=0.1, kappa=1.5, t_end=3600.0, sigma=0.02)
///
/// # Auto-calibrated σ (recommended)
/// strat = AvellanedaStoikov(gamma=0.05, kappa=0.5, t_end=3600.0)
///
/// bid, ask = strat.compute_quotes(mid=100.0, inventory=5, t=1800.0)
/// ```
#[pyclass(name = "AvellanedaStoikov")]
pub struct PyAvellanedaStoikov {
    inner: AvellanedaStoikov,
}

#[pymethods]
impl PyAvellanedaStoikov {
    /// Construct the strategy.
    ///
    /// Parameters
    /// ----------
    /// gamma : float
    ///     Risk-aversion coefficient γ > 0.
    /// kappa : float
    ///     Fill-arrival intensity κ near the best quotes.
    /// t_end : float
    ///     Trading horizon T (seconds from simulation origin).
    /// inventory_limit : int, optional
    ///     Hard inventory cap |q| ≤ inventory_limit.  Default 50.
    /// sigma : float, optional
    ///     Volatility σ (price units per second).  If omitted, σ is
    ///     auto-calibrated from the first warm-up events.
    /// spread_floor : float, optional
    ///     Minimum half-spread.  Default 0.0.
    #[new]
    #[pyo3(signature = (gamma, kappa, t_end, inventory_limit=50, sigma=None, spread_floor=0.0))]
    pub fn new(
        gamma: f64,
        kappa: f64,
        t_end: f64,
        inventory_limit: i64,
        sigma: Option<f64>,
        spread_floor: f64,
    ) -> Self {
        let inner = match sigma {
            Some(s) => {
                let mut strat = AvellanedaStoikov::new(gamma, s, kappa, t_end, inventory_limit);
                strat.spread_floor = spread_floor;
                strat
            }
            None => AvellanedaStoikov::with_auto_sigma(gamma, kappa, t_end, inventory_limit, spread_floor),
        };
        PyAvellanedaStoikov { inner }
    }

    /// Compute the optimal bid and ask quotes.
    ///
    /// Parameters
    /// ----------
    /// mid : float
    ///     Current mid-price.
    /// inventory : int
    ///     Current signed inventory (positive = long).
    /// t : float
    ///     Current simulation time (seconds).
    ///
    /// Returns
    /// -------
    /// (bid: float, ask: float)
    pub fn compute_quotes(&self, mid: f64, inventory: i64, t: f64) -> (f64, f64) {
        self.inner.compute_quotes(mid, inventory, t)
    }

    /// Current γ parameter.
    #[getter]
    pub fn gamma(&self) -> f64 { self.inner.gamma }

    /// Current κ parameter.
    #[getter]
    pub fn kappa(&self) -> f64 { self.inner.kappa }

    /// σ value used for quoting (fixed or auto-calibrated).
    #[getter]
    pub fn sigma(&self) -> f64 { self.inner.sigma }

    /// Whether σ is auto-calibrated.
    #[getter]
    pub fn sigma_auto(&self) -> bool { self.inner.sigma_auto }

    /// Trading horizon T.
    #[getter]
    pub fn t_end(&self) -> f64 { self.inner.t_end }

    /// Compute quotes with active inventory skewing.
    ///
    /// Returns ``((bid, ask), mode)`` where mode is one of:
    /// ``"normal"``, ``"skew"``, ``"suppress"``, ``"dump"``.
    ///
    /// When mode is ``"dump"`` the caller should place an aggressive order
    /// through mid instead of resting at the returned quotes.
    pub fn compute_quotes_skewed(
        &self,
        mid: f64,
        inventory: i64,
        t: f64,
    ) -> ((f64, f64), &'static str) {
        let (quotes, mode) = self.inner.compute_quotes_skewed(mid, inventory, t);
        (quotes, mode.as_str())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "AvellanedaStoikov(gamma={:.4}, kappa={:.4}, t_end={:.1}, sigma_auto={}, spread_floor={:.4})",
            self.inner.gamma, self.inner.kappa, self.inner.t_end,
            self.inner.sigma_auto, self.inner.spread_floor,
        )
    }
}
