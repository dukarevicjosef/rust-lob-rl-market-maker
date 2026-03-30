//! MLE calibration for multivariate Hawkes processes with exponential kernels.
//!
//! ## Algorithm
//!
//! Minimise the negative log-likelihood using L-BFGS with an unconstrained
//! change of variables (Ogata 1978; Bowsher 2007):
//!
//!   θ_μ  = log μ         (μ  = exp θ_μ > 0)
//!   θ_α  = logit α       (α  = sigmoid θ_α ∈ (0,1))
//!   θ_β  = log β         (β  = exp θ_β > 0)
//!
//! The gradient is computed analytically in O(n·d²) using the recursive
//! excitation state R_ij and C_ij (Daley & Vere-Jones 2003, §7.2):
//!
//!   R_ij(t_k) = exp(−β_ij·δ) · R_ij(t_{k−1})          (decayed sum)
//!   C_ij(t_k) = exp(−β_ij·δ) · C_ij(t_{k−1}) + δ·R_ij(t_k)
//!
//! where δ = t_k − t_{k−1} and R_ij is already advanced before C.
//!
//! ## Goodness of fit
//!
//! The time-rescaling theorem (Brown et al. 2002) states that for a correct
//! model the integrated intensities between events:
//!
//!   Λ_k = ∫_{t_{k−1}}^{t_k} Σ_i λ_i(t) dt
//!
//! are i.i.d. Exp(1).  A KS-test against Exp(1) gives a p-value; sorted Λ_k
//! against theoretical Exp(1) quantiles gives Q-Q data for Python plots.

use thiserror::Error;

use super::process::HawkesEvent;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, PartialEq)]
pub enum CalibrationError {
    #[error("events slice is empty")]
    NoEvents,
    #[error("t_max {t_max} must exceed the last event time {last}")]
    InvalidTMax { t_max: f64, last: f64 },
    #[error("event_type {0} out of range for dimension {1}")]
    EventTypeOutOfRange(usize, usize),
    #[error("dimension must be ≥ 1")]
    ZeroDimension,
}

// ── Parameter types ───────────────────────────────────────────────────────────

/// Calibrated parameters for a multivariate Hawkes process with exponential
/// kernels.  Indexing: `alpha[i][j]` = φ_ij excitation from type j → type i.
#[derive(Debug, Clone)]
pub struct ExpParams {
    pub mu: Vec<f64>,
    pub alpha: Vec<Vec<f64>>,
    pub beta: Vec<Vec<f64>>,
}

impl ExpParams {
    /// Build a uniform initialisation: all μ_i = initial_rate, α_ij = 0.1, β_ij = 1.
    pub fn initial(d: usize, initial_rate: f64) -> Self {
        ExpParams {
            mu: vec![initial_rate; d],
            alpha: vec![vec![0.1; d]; d],
            beta: vec![vec![1.0; d]; d],
        }
    }

    /// Row-wise branching ratio: n*_i = Σ_j α_ij.  Process is stationary iff
    /// all row sums < 1 (sufficient condition via infinity-norm bound).
    pub fn branching_row_sums(&self) -> Vec<f64> {
        self.alpha
            .iter()
            .map(|row| row.iter().sum())
            .collect()
    }

    /// Returns `true` if every row sum of the α matrix is < 1.
    pub fn is_stationary(&self) -> bool {
        self.branching_row_sums().iter().all(|&s| s < 1.0)
    }
}

/// Result returned by `calibrate_exponential`.
#[derive(Debug)]
pub struct CalibratedParams {
    pub params: ExpParams,
    /// Final negative log-likelihood (lower = better fit).
    pub nll: f64,
    /// Number of L-BFGS iterations taken.
    pub n_iter: usize,
    /// `true` if the gradient norm fell below the convergence tolerance.
    pub converged: bool,
}

// ── Goodness of fit ───────────────────────────────────────────────────────────

/// Diagnostics based on the time-rescaling theorem (Brown et al. 2002).
#[derive(Debug)]
pub struct GoodnessOfFit {
    /// Kolmogorov–Smirnov statistic D against Exp(1).
    pub ks_statistic: f64,
    /// Approximate p-value (Kolmogorov asymptotic distribution).
    pub ks_p_value: f64,
    /// Sorted integrated inter-arrival intensities {Λ_k} (empirical axis).
    pub qq_empirical: Vec<f64>,
    /// Corresponding theoretical Exp(1) quantiles (theoretical axis).
    pub qq_theoretical: Vec<f64>,
}

// ── Parameter packing ─────────────────────────────────────────────────────────

/// Layout of the unconstrained parameter vector x (length d + 2·d²):
///   x[i]               = log μ_i            for i ∈ [0, d)
///   x[d + i·d + j]     = logit α_ij         for (i,j) ∈ [0,d)²
///   x[d + d² + i·d + j] = log β_ij          for (i,j) ∈ [0,d)²
fn pack(p: &ExpParams) -> Vec<f64> {
    let d = p.mu.len();
    let mut x = Vec::with_capacity(d + 2 * d * d);
    for &m in &p.mu {
        x.push(m.ln());
    }
    for row in &p.alpha {
        for &a in row {
            let a = a.clamp(1e-9, 1.0 - 1e-9);
            x.push((a / (1.0 - a)).ln());
        }
    }
    for row in &p.beta {
        for &b in row {
            x.push(b.ln());
        }
    }
    x
}

#[inline]
fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn unpack(x: &[f64], d: usize) -> ExpParams {
    let mu: Vec<f64> = x[..d].iter().map(|&v| v.exp()).collect();
    let alpha: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| sigmoid(x[d + i * d + j])).collect())
        .collect();
    let beta: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| x[d + d * d + i * d + j].exp()).collect())
        .collect();
    ExpParams { mu, alpha, beta }
}

// ── NLL and analytical gradient ───────────────────────────────────────────────

/// Compute (NLL, gradient-in-transformed-space) in a single O(n·d²) forward pass.
///
/// Recursive state maintained for each (i,j) pair:
///   R_ij = Σ_{past type-j events m} exp(−β_ij·(t_current − t_m))
///   C_ij = Σ_{past type-j events m} (t_current − t_m)·exp(−β_ij·(t_current − t_m))
fn nll_and_grad(x: &[f64], events: &[HawkesEvent], t_max: f64, d: usize) -> (f64, Vec<f64>) {
    let p = unpack(x, d);
    let mu = &p.mu;
    let alpha = &p.alpha;
    let beta = &p.beta;

    // Recursive excitation state — at prev_t BEFORE adding the event at prev_t.
    let mut r = vec![vec![0.0_f64; d]; d]; // R_ij
    let mut c = vec![vec![0.0_f64; d]; d]; // C_ij

    // Gradient accumulators in original-parameter space.
    let mut dmu = vec![0.0_f64; d];
    let mut dalpha = vec![vec![0.0_f64; d]; d];
    let mut dbeta = vec![vec![0.0_f64; d]; d];

    let mut nll = 0.0_f64;
    let mut prev_t = 0.0_f64;

    for event in events {
        let t_k = event.time;
        let jstar = event.event_type;
        let delta = t_k - prev_t;

        // ── Advance R and C to t_k (before adding current event) ─────────────
        // C_ij(t_k) = decay · C_ij(prev) + δ · decay · R_ij(prev)
        //           = decay · (C_ij(prev) + δ · R_ij(prev))
        // R_ij(t_k) = decay · R_ij(prev)   [must update C before R]
        for i in 0..d {
            for j in 0..d {
                let decay = (-beta[i][j] * delta).exp();
                c[i][j] = decay * (c[i][j] + delta * r[i][j]);
                r[i][j] *= decay;
            }
        }

        // ── Compute λ_jstar(t_k) and accumulate log term ──────────────────────
        let lambda_i: f64 = mu[jstar]
            + (0..d)
                .map(|j| alpha[jstar][j] * beta[jstar][j] * r[jstar][j])
                .sum::<f64>();
        let lambda_safe = lambda_i.max(f64::MIN_POSITIVE);
        nll -= lambda_safe.ln();

        let inv_lambda = 1.0 / lambda_safe;
        dmu[jstar] += inv_lambda;
        for j in 0..d {
            let r_ij = r[jstar][j];
            let c_ij = c[jstar][j];
            dalpha[jstar][j] += beta[jstar][j] * r_ij * inv_lambda;
            dbeta[jstar][j] += alpha[jstar][j] * (r_ij - beta[jstar][j] * c_ij) * inv_lambda;
        }

        // ── Add current event to R (C unchanged: elapsed = 0) ────────────────
        for i in 0..d {
            r[i][jstar] += 1.0;
        }

        prev_t = t_k;
    }

    // ── Advance to t_max for compensator terms ────────────────────────────────
    let delta_final = t_max - prev_t;
    for i in 0..d {
        for j in 0..d {
            let decay = (-beta[i][j] * delta_final).exp();
            c[i][j] = decay * (c[i][j] + delta_final * r[i][j]);
            r[i][j] *= decay;
        }
    }
    // Now r[i][j] = R_ij(T) = Σ_m exp(−β_ij·(T−t_m)) over all type-j events.
    // And c[i][j] = C_ij(T) = Σ_m (T−t_m)·exp(−β_ij·(T−t_m)).

    // Count type-j events once, reuse below.
    let mut n_j = vec![0usize; d];
    for e in events {
        n_j[e.event_type] += 1;
    }

    // ── Compensator and its gradient ─────────────────────────────────────────
    // Compensator_i = μ_i·T + Σ_j α_ij·(n_j − R_ij(T))
    // ∂Comp_i/∂μ_i   = T
    // ∂Comp_i/∂α_ij  = n_j − R_ij(T)
    // ∂Comp_i/∂β_ij  = α_ij · C_ij(T)
    for i in 0..d {
        nll += mu[i] * t_max;
        dmu[i] -= t_max;
        for j in 0..d {
            let s_ij = n_j[j] as f64 - r[i][j]; // = Σ_m (1 − exp(−β_ij·(T−t_m)))
            nll += alpha[i][j] * s_ij;
            dalpha[i][j] -= s_ij;
            dbeta[i][j] -= alpha[i][j] * c[i][j];
        }
    }

    // ── Chain rule: original-space → transformed-space gradient ──────────────
    // d/d(log μ)   = d/dμ   · μ
    // d/d(logit α) = d/dα   · α·(1−α)
    // d/d(log β)   = d/dβ   · β
    // Gradient of NLL = −gradient of ℓ
    let mut grad = vec![0.0_f64; d + 2 * d * d];
    for i in 0..d {
        grad[i] = -dmu[i] * mu[i];
        for j in 0..d {
            let a = alpha[i][j];
            grad[d + i * d + j] = -dalpha[i][j] * a * (1.0 - a);
            grad[d + d * d + i * d + j] = -dbeta[i][j] * beta[i][j];
        }
    }

    (nll, grad)
}

// ── L-BFGS optimizer ──────────────────────────────────────────────────────────

const LBFGS_M: usize = 10; // number of history pairs stored
const C1: f64 = 1e-4; // Armijo sufficient-decrease constant

/// Unconstrained L-BFGS minimiser.
///
/// Returns `(x_opt, f_opt, n_iter, converged)`.
fn lbfgs(
    f_and_g: impl Fn(&[f64]) -> (f64, Vec<f64>),
    x0: Vec<f64>,
    max_iter: usize,
    grad_tol: f64,
) -> (Vec<f64>, f64, usize, bool) {
    let mut x = x0;
    let (mut f, mut g) = f_and_g(&x);

    // Circular buffers for the L-BFGS pairs.
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_M);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_M);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(LBFGS_M);

    for iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < grad_tol {
            return (x, f, iter, true);
        }

        // ── Two-loop L-BFGS recursion ─────────────────────────────────────────
        let m = s_hist.len();
        let mut q = g.clone();
        let mut a = vec![0.0_f64; m];

        for k in (0..m).rev() {
            a[k] = rho_hist[k] * dot(&s_hist[k], &q);
            axpy(-a[k], &y_hist[k], &mut q);
        }

        // Initial Hessian scaling γ = (s_{k-1}·y_{k-1}) / (y_{k-1}·y_{k-1}).
        let gamma = if m > 0 {
            dot(&s_hist[m - 1], &y_hist[m - 1])
                / dot(&y_hist[m - 1], &y_hist[m - 1]).max(1e-30)
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();

        for k in 0..m {
            let beta = rho_hist[k] * dot(&y_hist[k], &r);
            axpy(a[k] - beta, &s_hist[k], &mut r);
        }

        // Descent direction d = -H·g = -r.
        let d: Vec<f64> = r.iter().map(|&v| -v).collect();

        // Verify descent; reset if not (e.g. after numerical issues).
        let slope = dot(&g, &d);
        let d = if slope >= 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            g.iter().map(|&v| -v).collect()
        } else {
            d
        };

        // ── Backtracking Armijo line search ───────────────────────────────────
        let slope = dot(&g, &d);
        let mut alpha = 1.0_f64;
        let x_new;
        let f_new;
        let g_new;

        loop {
            let x_try: Vec<f64> =
                x.iter().zip(d.iter()).map(|(&xi, &di)| xi + alpha * di).collect();
            let (ft, gt) = f_and_g(&x_try);
            if ft <= f + C1 * alpha * slope || alpha < 1e-14 {
                x_new = x_try;
                f_new = ft;
                g_new = gt;
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-14 {
                // Line search failed; take a tiny step and reset history.
                x_new = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + 1e-14 * di).collect();
                let (ft, gt) = f_and_g(&x_new);
                f_new = ft;
                g_new = gt;
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                break;
            }
        }

        // ── Update L-BFGS history ─────────────────────────────────────────────
        let s_k: Vec<f64> = x_new.iter().zip(x.iter()).map(|(&xn, &xo)| xn - xo).collect();
        let y_k: Vec<f64> = g_new.iter().zip(g.iter()).map(|(&gn, &go)| gn - go).collect();
        let sy = dot(&s_k, &y_k);

        x = x_new;
        f = f_new;
        g = g_new;

        if sy > 1e-15 {
            if s_hist.len() == LBFGS_M {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_k);
            y_hist.push(y_k);
            rho_hist.push(1.0 / sy);
        }
    }

    let converged =
        g.iter().map(|v| v * v).sum::<f64>().sqrt() < grad_tol;
    (x, f, max_iter, converged)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[inline]
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Calibrate a d-dimensional Hawkes process with exponential kernels via MLE.
///
/// Uses L-BFGS with logit/log change of variables; gradient computed
/// analytically in O(n·d²).
pub fn calibrate_exponential(
    events: &[HawkesEvent],
    dim: usize,
    t_max: f64,
) -> Result<CalibratedParams, CalibrationError> {
    if dim == 0 {
        return Err(CalibrationError::ZeroDimension);
    }
    if events.is_empty() {
        return Err(CalibrationError::NoEvents);
    }
    if let Some(last) = events.last() {
        if last.time >= t_max {
            return Err(CalibrationError::InvalidTMax { t_max, last: last.time });
        }
    }
    for e in events {
        if e.event_type >= dim {
            return Err(CalibrationError::EventTypeOutOfRange(e.event_type, dim));
        }
    }

    // ── Initial parameters ────────────────────────────────────────────────────
    // μ_i = total_rate / d; α = 0.1; β = 1.
    let total_rate = events.len() as f64 / t_max;
    let init = ExpParams::initial(dim, (total_rate / dim as f64).max(1e-6));
    let x0 = pack(&init);

    // ── Optimise ──────────────────────────────────────────────────────────────
    let (x_opt, nll_opt, n_iter, converged) = lbfgs(
        |x| nll_and_grad(x, events, t_max, dim),
        x0,
        500,
        1e-5,
    );

    let params = unpack(&x_opt, dim);
    Ok(CalibratedParams { params, nll: nll_opt, n_iter, converged })
}

// ── Goodness of fit ───────────────────────────────────────────────────────────

/// Compute goodness-of-fit diagnostics via the time-rescaling theorem.
///
/// Integrated inter-arrival intensities:
///   Λ_k = ∫_{t_{k−1}}^{t_k} Σ_i λ_i(t) dt
///         = Σ_i μ_i · δ + Σ_i Σ_j α_ij · R_ij(t_{k−1}) · (1 − exp(−β_ij·δ))
///
/// If the model is correctly specified, {Λ_k} ~ i.i.d. Exp(1).
pub fn goodness_of_fit(
    events: &[HawkesEvent],
    params: &ExpParams,
    _t_max: f64,
) -> GoodnessOfFit {
    let d = params.mu.len();
    let n = events.len();

    // ── Compute Λ_k values ────────────────────────────────────────────────────
    let mut lambdas: Vec<f64> = Vec::with_capacity(n);

    // r[i][j] = running R_ij INCLUDING the most recent event.
    let mut r = vec![vec![0.0_f64; d]; d];
    let mut prev_t = 0.0_f64;
    let mu_sum: f64 = params.mu.iter().sum();

    for event in events {
        let t_k = event.time;
        let jstar = event.event_type;
        let delta = t_k - prev_t;

        // Λ_k using r BEFORE advancing (= state at t_{k-1}).
        let mut lambda_k = mu_sum * delta;
        for i in 0..d {
            for j in 0..d {
                lambda_k +=
                    params.alpha[i][j] * r[i][j] * (1.0 - (-params.beta[i][j] * delta).exp());
            }
        }
        lambdas.push(lambda_k.max(0.0));

        // Advance r to t_k (decay + add new event).
        for i in 0..d {
            for j in 0..d {
                r[i][j] *= (-params.beta[i][j] * delta).exp();
            }
        }
        for i in 0..d {
            r[i][jstar] += 1.0;
        }

        prev_t = t_k;
    }

    // ── KS statistic against Exp(1) ───────────────────────────────────────────
    lambdas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ks_stat = 0.0_f64;
    for (k, &lk) in lambdas.iter().enumerate() {
        let ecdf = (k + 1) as f64 / n as f64;
        let theoretical = 1.0 - (-lk).exp(); // Exp(1) CDF
        ks_stat = ks_stat.max((ecdf - theoretical).abs());
        if k > 0 {
            let ecdf_prev = k as f64 / n as f64;
            ks_stat = ks_stat.max((ecdf_prev - theoretical).abs());
        }
    }

    // ── Q-Q data ─────────────────────────────────────────────────────────────
    // Theoretical quantiles of Exp(1) at plotting positions (k−0.5)/n.
    let qq_theoretical: Vec<f64> = (1..=n)
        .map(|k| -((1.0 - (k as f64 - 0.5) / n as f64).max(1e-15)).ln())
        .collect();
    let qq_empirical = lambdas.clone();

    // ── KS p-value (Kolmogorov asymptotic) ───────────────────────────────────
    let z = (n as f64).sqrt() * ks_stat;
    let ks_p_value = ks_p(z);

    GoodnessOfFit { ks_statistic: ks_stat, ks_p_value, qq_empirical, qq_theoretical }
}

/// Approximate KS p-value via the asymptotic Kolmogorov distribution.
///
/// P(√n · D_n > z) ≈ 2 Σ_{k=1}^{K} (−1)^{k+1} exp(−2k²z²)
///
/// For z < 0.27 the series requires thousands of terms; in that regime the
/// p-value is always ≥ 0.99, so we return 1.0 directly.  For z ≥ 0.27 the
/// alternating series converges with ≤ 50 terms.
fn ks_p(z: f64) -> f64 {
    if z < 0.27 {
        return 1.0;
    }
    let mut p = 0.0_f64;
    for k in 1_i64..=50 {
        let term = (-(2.0 * (k * k) as f64 * z * z)).exp();
        if k % 2 == 0 { p -= term; } else { p += term; }
        if term < 1e-15 { break; }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkes::kernel::ExponentialKernel;
    use crate::hawkes::kernel::ExcitationKernel;
    use crate::hawkes::process::MultivariateHawkes;

    fn make_1d_hawkes(mu: f64, alpha: f64, beta: f64) -> MultivariateHawkes {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
            vec![vec![Box::new(ExponentialKernel::new(alpha, beta).unwrap())]];
        MultivariateHawkes::new(vec![mu], kernels).unwrap()
    }

    // ── Constructor / validation ──────────────────────────────────────────────

    #[test]
    fn calibrate_rejects_empty_events() {
        let result = calibrate_exponential(&[], 1, 100.0);
        assert!(matches!(result, Err(CalibrationError::NoEvents)));
    }

    #[test]
    fn calibrate_rejects_t_max_before_last_event() {
        let events = vec![HawkesEvent { time: 10.0, event_type: 0 }];
        let result = calibrate_exponential(&events, 1, 5.0);
        assert!(matches!(result, Err(CalibrationError::InvalidTMax { .. })));
    }

    #[test]
    fn calibrate_rejects_out_of_range_event_type() {
        let events = vec![HawkesEvent { time: 1.0, event_type: 2 }];
        let result = calibrate_exponential(&events, 2, 10.0);
        assert!(matches!(result, Err(CalibrationError::EventTypeOutOfRange(2, 2))));
    }

    #[test]
    fn calibrate_rejects_zero_dimension() {
        let events = vec![HawkesEvent { time: 1.0, event_type: 0 }];
        let result = calibrate_exponential(&events, 0, 10.0);
        assert!(matches!(result, Err(CalibrationError::ZeroDimension)));
    }

    // ── NLL and gradient ─────────────────────────────────────────────────────

    #[test]
    fn nll_no_events_equals_mu_times_t() {
        // ℓ = 0 − μ·T  →  NLL = μ·T
        let p = ExpParams { mu: vec![2.0], alpha: vec![vec![0.3]], beta: vec![vec![1.0]] };
        let x = pack(&p);
        let (nll, _) = nll_and_grad(&x, &[], 5.0, 1);
        assert!((nll - 10.0).abs() < 1e-10, "nll={nll}");
    }

    #[test]
    fn nll_single_event_known_value() {
        // Mirror of process::log_likelihood_single_event_known_value.
        // μ=1, α=0.5, β=1, t=1 on [0,2]: ℓ = −(2 + 0.5·(1−e⁻¹))
        let p = ExpParams { mu: vec![1.0], alpha: vec![vec![0.5]], beta: vec![vec![1.0]] };
        let x = pack(&p);
        let events = vec![HawkesEvent { time: 1.0, event_type: 0 }];
        let (nll, _) = nll_and_grad(&x, &events, 2.0, 1);
        let expected = 2.0 + 0.5 * (1.0 - (-1.0_f64).exp());
        assert!((nll - expected).abs() < 1e-10, "nll={nll:.12}, expected={expected:.12}");
    }

    #[test]
    fn gradient_finite_difference_check_1d() {
        let p = ExpParams { mu: vec![1.5], alpha: vec![vec![0.4]], beta: vec![vec![2.0]] };
        let events = vec![
            HawkesEvent { time: 0.5, event_type: 0 },
            HawkesEvent { time: 1.2, event_type: 0 },
            HawkesEvent { time: 2.8, event_type: 0 },
        ];
        let x = pack(&p);
        let (f0, g) = nll_and_grad(&x, &events, 4.0, 1);
        let eps = 1e-5;
        for k in 0..x.len() {
            let mut xp = x.clone();
            xp[k] += eps;
            let (fp, _) = nll_and_grad(&xp, &events, 4.0, 1);
            let fd = (fp - f0) / eps;
            let rel_err = (fd - g[k]).abs() / (fd.abs().max(1e-8));
            assert!(
                rel_err < 1e-4,
                "param {k}: fd={fd:.8}, analytical={:.8}, rel_err={rel_err:.2e}",
                g[k]
            );
        }
    }

    #[test]
    fn gradient_finite_difference_check_2d() {
        let p = ExpParams {
            mu: vec![1.0, 0.8],
            alpha: vec![vec![0.3, 0.1], vec![0.2, 0.25]],
            beta: vec![vec![2.0, 1.5], vec![1.0, 3.0]],
        };
        let events = vec![
            HawkesEvent { time: 0.3, event_type: 0 },
            HawkesEvent { time: 0.7, event_type: 1 },
            HawkesEvent { time: 1.1, event_type: 0 },
            HawkesEvent { time: 1.9, event_type: 1 },
            HawkesEvent { time: 2.5, event_type: 0 },
        ];
        let x = pack(&p);
        let (f0, g) = nll_and_grad(&x, &events, 3.0, 2);
        let eps = 1e-5;
        for k in 0..x.len() {
            let mut xp = x.clone();
            xp[k] += eps;
            let (fp, _) = nll_and_grad(&xp, &events, 3.0, 2);
            let fd = (fp - f0) / eps;
            let rel_err = (fd - g[k]).abs() / (fd.abs().max(1e-8));
            assert!(
                rel_err < 1e-3,
                "param {k}: fd={fd:.8}, analytical={:.8}, rel_err={rel_err:.2e}",
                g[k]
            );
        }
    }

    // ── calibrate_exponential: parameter recovery ─────────────────────────────

    #[test]
    fn calibrate_1d_recovers_true_params_within_10pct() {
        // Generate 10 000 events from known parameters; calibrate; compare.
        let true_mu = 1.0_f64;
        let true_alpha = 0.5_f64;
        let true_beta = 2.0_f64;
        let t_max = 5_000.0_f64;

        let hawkes = make_1d_hawkes(true_mu, true_alpha, true_beta);
        let events = hawkes.simulate(t_max, 42);

        let result = calibrate_exponential(&events, 1, t_max).unwrap();
        assert!(result.converged, "optimiser did not converge");

        let est_mu = result.params.mu[0];
        let est_alpha = result.params.alpha[0][0];
        let est_beta = result.params.beta[0][0];

        let tol = 0.10; // 10 %
        assert!(
            (est_mu - true_mu).abs() / true_mu < tol,
            "μ: est={est_mu:.3}, true={true_mu:.3}"
        );
        assert!(
            (est_alpha - true_alpha).abs() / true_alpha < tol,
            "α: est={est_alpha:.3}, true={true_alpha:.3}"
        );
        assert!(
            (est_beta - true_beta).abs() / true_beta < tol,
            "β: est={est_beta:.3}, true={true_beta:.3}"
        );
    }

    #[test]
    fn calibrate_1d_estimated_params_are_stationary() {
        let hawkes = make_1d_hawkes(0.8, 0.4, 1.5);
        let events = hawkes.simulate(2_000.0, 7);
        let result = calibrate_exponential(&events, 1, 2_000.0).unwrap();
        assert!(
            result.params.is_stationary(),
            "calibrated params are not stationary: row sums = {:?}",
            result.params.branching_row_sums()
        );
    }

    #[test]
    fn calibrate_nll_lower_than_initial() {
        let hawkes = make_1d_hawkes(1.0, 0.5, 2.0);
        let events = hawkes.simulate(1_000.0, 99);
        let t_max = 1_000.0;

        let init = ExpParams::initial(1, events.len() as f64 / t_max);
        let x_init = pack(&init);
        let (nll_init, _) = nll_and_grad(&x_init, &events, t_max, 1);

        let result = calibrate_exponential(&events, 1, t_max).unwrap();
        assert!(
            result.nll < nll_init,
            "calibrated NLL {:.2} ≥ initial NLL {nll_init:.2}",
            result.nll
        );
    }

    // ── goodness_of_fit ───────────────────────────────────────────────────────

    #[test]
    fn gof_ks_p_value_high_for_correct_params() {
        // Well-specified model → KS p-value should exceed 0.05.
        let hawkes = make_1d_hawkes(1.0, 0.5, 2.0);
        let t_max = 5_000.0;
        let events = hawkes.simulate(t_max, 42);

        let result = calibrate_exponential(&events, 1, t_max).unwrap();
        let gof = goodness_of_fit(&events, &result.params, t_max);

        assert!(
            gof.ks_p_value > 0.05,
            "KS p={:.4}, D={:.4}",
            gof.ks_p_value,
            gof.ks_statistic
        );
    }

    #[test]
    fn gof_ks_p_value_low_for_misspecified_params() {
        // Wrong β (10× too large) → residuals far from Exp(1) → low p-value.
        let hawkes = make_1d_hawkes(1.0, 0.5, 2.0);
        let t_max = 2_000.0;
        let events = hawkes.simulate(t_max, 1);

        let wrong = ExpParams { mu: vec![1.0], alpha: vec![vec![0.5]], beta: vec![vec![20.0]] };
        let gof = goodness_of_fit(&events, &wrong, t_max);

        assert!(
            gof.ks_p_value < 0.05,
            "expected low p-value for misspecified params, got {:.4}",
            gof.ks_p_value
        );
    }

    #[test]
    fn gof_qq_data_lengths_match() {
        let hawkes = make_1d_hawkes(2.0, 0.3, 1.0);
        let events = hawkes.simulate(500.0, 5);
        let n = events.len();
        let p = ExpParams { mu: vec![2.0], alpha: vec![vec![0.3]], beta: vec![vec![1.0]] };
        let gof = goodness_of_fit(&events, &p, 500.0);
        assert_eq!(gof.qq_empirical.len(), n);
        assert_eq!(gof.qq_theoretical.len(), n);
    }

    #[test]
    fn gof_qq_empirical_sorted_ascending() {
        let hawkes = make_1d_hawkes(2.0, 0.3, 1.0);
        let events = hawkes.simulate(200.0, 9);
        let p = ExpParams { mu: vec![2.0], alpha: vec![vec![0.3]], beta: vec![vec![1.0]] };
        let gof = goodness_of_fit(&events, &p, 200.0);
        for w in gof.qq_empirical.windows(2) {
            assert!(w[0] <= w[1], "qq_empirical not sorted");
        }
    }

    // ── KS p-value helper ─────────────────────────────────────────────────────

    #[test]
    fn ks_p_extreme_values() {
        // Very large z → p ≈ 0.
        assert!(ks_p(3.0) < 1e-4);
        // Very small z → p ≈ 1.
        assert!(ks_p(0.01) > 0.99);
        // At z=0: returns 1.
        assert!((ks_p(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ks_p_canonical_value() {
        // D=0.1, n=100 → z = 1.0.  Known p ≈ 0.270 (Kolmogorov table).
        let p = ks_p(1.0);
        assert!((p - 0.270).abs() < 0.005, "ks_p(1.0)={p:.4}");
    }
}
