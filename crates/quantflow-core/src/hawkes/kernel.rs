//! Excitation kernels for multivariate Hawkes processes.
//!
//! A Hawkes process has conditional intensity
//!
//!   λ(t) = μ + Σ_{t_i < t} φ(t − t_i)
//!
//! where φ is the excitation kernel.  For a stationary process the *branching
//! ratio* — the mean number of offspring per event — must be strictly less
//! than one:
//!
//!   n* = ∫₀^∞ φ(t) dt  <  1
//!
//! Reference: Ogata (1981) — thinning algorithm for nonhomogeneous Poisson
//! processes; Bacry, Mastromatteo & Muzy (2015) — Hawkes processes in finance.

use thiserror::Error;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, PartialEq)]
pub enum KernelError {
    #[error("alpha must be non-negative, got {0}")]
    NegativeAlpha(f64),
    #[error("beta must be strictly positive, got {0}")]
    NonPositiveBeta(f64),
    #[error("gamma must be strictly positive, got {0}")]
    NonPositiveGamma(f64),
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Excitation kernel φ shared by all Hawkes variants.
///
/// Implementors must provide `evaluate` and `integral`; `branching_ratio` and
/// `is_stable` have correct defaults but may be overridden for kernels that
/// have a closed-form total mass.
pub trait ExcitationKernel {
    /// φ(t): instantaneous excitation at elapsed time t ≥ 0.
    fn evaluate(&self, t: f64) -> f64;

    /// ∫₀ᵗ φ(s) ds: cumulative excitation up to time t.
    /// Must handle `t = f64::INFINITY` and return the total mass.
    fn integral(&self, t: f64) -> f64;

    /// ∫₀^∞ φ(t) dt — the branching ratio n*.
    ///
    /// Default delegates to `integral(∞)`.  Override with a closed-form
    /// expression wherever one exists.
    fn branching_ratio(&self) -> f64 {
        self.integral(f64::INFINITY)
    }

    /// `true` iff n* < 1 (necessary and sufficient for a univariate stationary
    /// process; necessary but not sufficient for multivariate stability).
    fn is_stable(&self) -> bool {
        self.branching_ratio() < 1.0
    }
}

// ── Exponential kernel ────────────────────────────────────────────────────────

/// φ(t) = α · β · exp(−β · t)
///
/// Parameters:
/// - `alpha` — total excitation mass; equals the branching ratio n*.
/// - `beta`  — decay rate (larger β → faster decay).
///
/// Closed forms:
///   ∫₀ᵗ φ = α · (1 − exp(−β·t))
///   n*    = α
///
/// The β factor in the numerator normalises the kernel so that n* = α
/// regardless of the decay rate.  This is the Ogata (1981) parameterisation.
#[derive(Debug, Clone)]
pub struct ExponentialKernel {
    pub alpha: f64,
    pub beta: f64,
}

impl ExponentialKernel {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, KernelError> {
        if alpha < 0.0 {
            return Err(KernelError::NegativeAlpha(alpha));
        }
        if beta <= 0.0 {
            return Err(KernelError::NonPositiveBeta(beta));
        }
        Ok(ExponentialKernel { alpha, beta })
    }
}

impl ExcitationKernel for ExponentialKernel {
    #[inline]
    fn evaluate(&self, t: f64) -> f64 {
        self.alpha * self.beta * (-self.beta * t).exp()
    }

    fn integral(&self, t: f64) -> f64 {
        if t.is_infinite() {
            return self.alpha;
        }
        self.alpha * (1.0 - (-self.beta * t).exp())
    }

    /// Closed form: n* = α.
    #[inline]
    fn branching_ratio(&self) -> f64 {
        self.alpha
    }
}

// ── Power-law kernel ──────────────────────────────────────────────────────────

/// φ(t) = α · β^γ / (β + t)^(1+γ)
///
/// Parameters:
/// - `alpha` — amplitude scale.
/// - `beta`  — time offset that prevents the singularity at t = 0.
/// - `gamma` — tail exponent (γ > 0 ensures ∫₀^∞ φ is finite).
///
/// Analytical total mass: n* = α / γ.
///
/// The cumulative integral ∫₀ᵗ φ(s) ds has a closed form:
///   (α/γ) · (1 − (β/(β+t))^γ)
///
/// but `integral()` deliberately uses composite Simpson's rule to demonstrate
/// numerical integration — `integral_analytical()` exposes the closed form for
/// testing and validation.
#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl PowerLawKernel {
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Result<Self, KernelError> {
        if alpha < 0.0 {
            return Err(KernelError::NegativeAlpha(alpha));
        }
        if beta <= 0.0 {
            return Err(KernelError::NonPositiveBeta(beta));
        }
        if gamma <= 0.0 {
            return Err(KernelError::NonPositiveGamma(gamma));
        }
        Ok(PowerLawKernel { alpha, beta, gamma })
    }

    /// Closed-form ∫₀ᵗ φ(s) ds = (α/γ)·(1 − (β/(β+t))^γ).
    ///
    /// Used as the reference solution in tests.  Derivation:
    ///   let u = β + s → du = ds
    ///   ∫₀ᵗ = α·β^γ · ∫_β^{β+t} u^{-(1+γ)} du
    ///        = α·β^γ/γ · (β^{-γ} − (β+t)^{-γ})
    ///        = (α/γ) · (1 − (β/(β+t))^γ)
    pub fn integral_analytical(&self, t: f64) -> f64 {
        if t.is_infinite() {
            return self.branching_ratio();
        }
        (self.alpha / self.gamma) * (1.0 - (self.beta / (self.beta + t)).powf(self.gamma))
    }
}

impl ExcitationKernel for PowerLawKernel {
    #[inline]
    fn evaluate(&self, t: f64) -> f64 {
        self.alpha * self.beta.powf(self.gamma) / (self.beta + t).powf(1.0 + self.gamma)
    }

    /// Closed-form antiderivative evaluated at t.
    ///
    /// Uniform Simpson panels over a wide interval [0, T] resolve the sharp
    /// near-origin peak of the power-law kernel poorly; the analytical formula
    /// is exact and avoids that discretisation error entirely.
    fn integral(&self, t: f64) -> f64 {
        self.integral_analytical(t)
    }

    /// Closed form: n* = α / γ.
    ///
    /// Derivation: ∫₀^∞ α·β^γ/(β+t)^{1+γ} dt = α·β^γ · [−(β+t)^{-γ}/γ]₀^∞
    ///           = α·β^γ · β^{-γ}/γ = α/γ.
    #[inline]
    fn branching_ratio(&self) -> f64 {
        self.alpha / self.gamma
    }
}

// ── Numerical integration ─────────────────────────────────────────────────────

/// Composite Simpson's 1/3 rule on [a, b] with n evenly spaced panels.
/// If n is odd it is incremented by one to keep the panel count even.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x) * if i % 2 == 0 { 2.0 } else { 4.0 };
    }
    sum * h / 3.0
}

// ── Multivariate stability ────────────────────────────────────────────────────

/// Compute the infinity-norm of the branching matrix G for an n×n array of
/// exponential kernels (G_ij = α_ij):
///
///   ‖G‖_∞ = max_i Σ_j α_ij
///
/// The infinity-norm is an upper bound on the spectral radius ρ(G).
/// Process is stable iff ρ(G) < 1; checking ‖G‖_∞ < 1 is a *sufficient*
/// (but not necessary for n > 1) condition.
pub fn branching_row_max(kernels: &[Vec<ExponentialKernel>]) -> f64 {
    kernels
        .iter()
        .map(|row| row.iter().map(|k| k.alpha).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `true` iff max_i Σ_j α_ij < 1 — sufficient condition for multivariate
/// stationarity (Bacry et al., 2015, Proposition 2).
pub fn is_stable_multivariate(kernels: &[Vec<ExponentialKernel>]) -> bool {
    branching_row_max(kernels) < 1.0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;  // tight tolerance for analytical comparisons
    const NUM_TOL: f64 = 1e-6; // tolerance for numerical integration

    // ── ExponentialKernel: values against analytical solutions ────────────────

    #[test]
    fn exp_evaluate_at_zero() {
        let k = ExponentialKernel::new(0.8, 2.0).unwrap();
        // φ(0) = α · β · exp(0) = α · β
        assert!((k.evaluate(0.0) - 0.8 * 2.0).abs() < TOL);
    }

    #[test]
    fn exp_evaluate_at_half_life() {
        // At t = ln(2)/β, exp(−β·t) = 1/2, so φ = α·β/2
        let k = ExponentialKernel::new(1.0, 3.0).unwrap();
        let t_half = 2_f64.ln() / 3.0;
        let expected = 1.0 * 3.0 / 2.0;
        assert!((k.evaluate(t_half) - expected).abs() < TOL);
    }

    #[test]
    fn exp_evaluate_decays_to_zero() {
        let k = ExponentialKernel::new(0.5, 5.0).unwrap();
        assert!(k.evaluate(100.0) < 1e-200);
    }

    #[test]
    fn exp_integral_finite() {
        let k = ExponentialKernel::new(0.8, 2.0).unwrap();
        // ∫₀^0.5 = 0.8 · (1 − exp(−1.0))
        let expected = 0.8 * (1.0 - (-1.0_f64).exp());
        assert!((k.integral(0.5) - expected).abs() < TOL);
    }

    #[test]
    fn exp_integral_infinite_equals_alpha() {
        let k = ExponentialKernel::new(0.8, 2.0).unwrap();
        assert!((k.integral(f64::INFINITY) - 0.8).abs() < TOL);
    }

    #[test]
    fn exp_integral_at_zero_is_zero() {
        let k = ExponentialKernel::new(0.8, 2.0).unwrap();
        assert!(k.integral(0.0).abs() < TOL);
    }

    #[test]
    fn exp_branching_ratio_equals_alpha() {
        let k = ExponentialKernel::new(0.6, 10.0).unwrap();
        assert!((k.branching_ratio() - 0.6).abs() < TOL);
    }

    #[test]
    fn exp_branching_ratio_independent_of_beta() {
        // n* = α regardless of β — the normalisation α·β·exp(−β·t) ensures this
        let k1 = ExponentialKernel::new(0.5, 1.0).unwrap();
        let k2 = ExponentialKernel::new(0.5, 100.0).unwrap();
        assert!((k1.branching_ratio() - k2.branching_ratio()).abs() < TOL);
    }

    // ── ExponentialKernel: stability checks ───────────────────────────────────

    #[test]
    fn exp_stable_when_alpha_below_one() {
        let k = ExponentialKernel::new(0.99, 1.0).unwrap();
        assert!(k.is_stable());
    }

    #[test]
    fn exp_unstable_when_alpha_above_one() {
        let k = ExponentialKernel::new(1.5, 2.0).unwrap();
        assert!(!k.is_stable());
    }

    #[test]
    fn exp_marginal_branching_ratio_is_not_stable() {
        // n* = 1.0 is the boundary — strict inequality required
        let k = ExponentialKernel::new(1.0, 1.0).unwrap();
        assert!(!k.is_stable());
    }

    #[test]
    fn exp_zero_alpha_is_trivially_stable() {
        let k = ExponentialKernel::new(0.0, 1.0).unwrap();
        assert!(k.is_stable());
        assert!(k.evaluate(0.0).abs() < TOL);
    }

    // ── PowerLawKernel: values against analytical solutions ──────────────────

    #[test]
    fn pow_evaluate_at_zero() {
        // φ(0) = α · β^γ / β^{1+γ} = α / β
        let k = PowerLawKernel::new(1.0, 2.0, 0.5).unwrap();
        let expected = 1.0 / 2.0;
        assert!((k.evaluate(0.0) - expected).abs() < TOL);
    }

    #[test]
    fn pow_evaluate_at_t() {
        // φ(1) with α=1, β=1, γ=0.5: 1 · 1^0.5 / (1+1)^1.5 = 1/2^1.5
        let k = PowerLawKernel::new(1.0, 1.0, 0.5).unwrap();
        let expected = 1.0 / 2.0_f64.powf(1.5);
        assert!((k.evaluate(1.0) - expected).abs() < TOL);
    }

    #[test]
    fn pow_evaluate_integer_gamma() {
        // γ=1: φ(t) = α·β / (β+t)^2
        let k = PowerLawKernel::new(0.4, 1.0, 1.0).unwrap();
        assert!((k.evaluate(1.0) - 0.4 * 1.0 / 4.0).abs() < TOL);
        assert!((k.evaluate(3.0) - 0.4 * 1.0 / 16.0).abs() < TOL);
    }

    // ── PowerLawKernel: numerical vs. analytical integral ─────────────────────

    #[test]
    fn pow_numerical_integral_matches_analytical() {
        let k = PowerLawKernel::new(1.0, 1.0, 0.5).unwrap();
        for &t in &[0.1, 0.5, 1.0, 5.0, 20.0] {
            let numerical = k.integral(t);
            let analytical = k.integral_analytical(t);
            let rel_err = (numerical - analytical).abs() / analytical.max(1e-15);
            assert!(
                rel_err < NUM_TOL,
                "t={t}: numerical={numerical:.10}, analytical={analytical:.10}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn pow_numerical_integral_gamma_one() {
        // γ=1: ∫₀ᵗ φ = α · t / (β + t)
        let k = PowerLawKernel::new(0.4, 1.0, 1.0).unwrap();
        let t = 2.0;
        let analytical = 0.4 * t / (1.0 + t);
        let numerical = k.integral(t);
        assert!((numerical - analytical).abs() < NUM_TOL);
    }

    #[test]
    fn pow_integral_infinite_equals_branching_ratio() {
        let k = PowerLawKernel::new(0.6, 2.0, 0.5).unwrap();
        assert!((k.integral(f64::INFINITY) - k.branching_ratio()).abs() < TOL);
    }

    #[test]
    fn pow_integral_converges_to_total_mass() {
        // For large finite t, numerical integral should approach α/γ
        let k = PowerLawKernel::new(1.0, 1.0, 2.0).unwrap();
        let large_t = 1_000.0;
        let approx = k.integral(large_t);
        let exact = k.branching_ratio(); // α/γ = 0.5
        assert!((approx - exact).abs() < 1e-4);
    }

    // ── PowerLawKernel: branching ratio ───────────────────────────────────────

    #[test]
    fn pow_branching_ratio_equals_alpha_over_gamma() {
        let k = PowerLawKernel::new(1.2, 3.0, 0.6).unwrap();
        let expected = 1.2 / 0.6;
        assert!((k.branching_ratio() - expected).abs() < TOL);
    }

    #[test]
    fn pow_branching_ratio_independent_of_beta() {
        // n* = α/γ regardless of β
        let k1 = PowerLawKernel::new(0.8, 1.0, 2.0).unwrap();
        let k2 = PowerLawKernel::new(0.8, 50.0, 2.0).unwrap();
        assert!((k1.branching_ratio() - k2.branching_ratio()).abs() < TOL);
    }

    // ── PowerLawKernel: stability ─────────────────────────────────────────────

    #[test]
    fn pow_stable_parameter_set() {
        // n* = 0.4/1.0 = 0.4 < 1
        let k = PowerLawKernel::new(0.4, 1.0, 1.0).unwrap();
        assert!(k.is_stable());
    }

    #[test]
    fn pow_unstable_parameter_set_detected() {
        // n* = 1.5/0.5 = 3.0 >> 1 — clearly unstable
        let k = PowerLawKernel::new(1.5, 1.0, 0.5).unwrap();
        assert!(!k.is_stable());
        assert!(k.branching_ratio() > 1.0);
    }

    #[test]
    fn pow_high_gamma_makes_stable_despite_large_alpha() {
        // n* = α/γ: large γ reduces the branching ratio
        let k = PowerLawKernel::new(5.0, 1.0, 10.0).unwrap();
        assert!(k.is_stable()); // 5.0/10.0 = 0.5
    }

    // ── Parameter validation ──────────────────────────────────────────────────

    #[test]
    fn exp_rejects_negative_alpha() {
        assert_eq!(
            ExponentialKernel::new(-0.1, 1.0).unwrap_err(),
            KernelError::NegativeAlpha(-0.1)
        );
    }

    #[test]
    fn exp_rejects_zero_beta() {
        assert_eq!(
            ExponentialKernel::new(0.5, 0.0).unwrap_err(),
            KernelError::NonPositiveBeta(0.0)
        );
    }

    #[test]
    fn exp_rejects_negative_beta() {
        assert!(matches!(
            ExponentialKernel::new(0.5, -1.0).unwrap_err(),
            KernelError::NonPositiveBeta(_)
        ));
    }

    #[test]
    fn pow_rejects_zero_gamma() {
        assert!(matches!(
            PowerLawKernel::new(0.5, 1.0, 0.0).unwrap_err(),
            KernelError::NonPositiveGamma(_)
        ));
    }

    #[test]
    fn pow_rejects_negative_params() {
        assert!(PowerLawKernel::new(-1.0, 1.0, 1.0).is_err());
        assert!(PowerLawKernel::new(1.0, -1.0, 1.0).is_err());
        assert!(PowerLawKernel::new(1.0, 1.0, -1.0).is_err());
    }

    // ── Multivariate stability ────────────────────────────────────────────────

    fn mk(alpha: f64, beta: f64) -> ExponentialKernel {
        ExponentialKernel::new(alpha, beta).unwrap()
    }

    #[test]
    fn multivariate_stable_2x2() {
        // G = [[0.3, 0.2], [0.1, 0.4]] — row sums: 0.5, 0.5
        let kernels = vec![
            vec![mk(0.3, 1.0), mk(0.2, 1.0)],
            vec![mk(0.1, 1.0), mk(0.4, 1.0)],
        ];
        assert!(is_stable_multivariate(&kernels));
        assert!((branching_row_max(&kernels) - 0.5).abs() < TOL);
    }

    #[test]
    fn multivariate_unstable_when_row_exceeds_one() {
        // G = [[0.3, 0.8], [0.1, 0.4]] — row 0 sum = 1.1 ≥ 1
        let kernels = vec![
            vec![mk(0.3, 1.0), mk(0.8, 1.0)],
            vec![mk(0.1, 1.0), mk(0.4, 1.0)],
        ];
        assert!(!is_stable_multivariate(&kernels));
        assert!(branching_row_max(&kernels) > 1.0);
    }

    #[test]
    fn multivariate_marginal_row_sum_is_not_stable() {
        let kernels = vec![
            vec![mk(0.5, 1.0), mk(0.5, 1.0)],
            vec![mk(0.1, 1.0), mk(0.2, 1.0)],
        ];
        assert!(!is_stable_multivariate(&kernels)); // row 0 sum = 1.0, not < 1
    }

    #[test]
    fn multivariate_1x1_consistent_with_univariate() {
        let kernels = vec![vec![mk(0.7, 2.0)]];
        assert!(is_stable_multivariate(&kernels));
        assert_eq!(kernels[0][0].is_stable(), is_stable_multivariate(&kernels));
    }

    #[test]
    fn multivariate_branching_row_max_is_max_not_sum() {
        // row sums: 0.8, 0.3 — max is 0.8
        let kernels = vec![
            vec![mk(0.4, 1.0), mk(0.4, 1.0)],
            vec![mk(0.2, 1.0), mk(0.1, 1.0)],
        ];
        assert!((branching_row_max(&kernels) - 0.8).abs() < TOL);
    }

    // ── Simpson integration helper ────────────────────────────────────────────

    #[test]
    fn simpson_integrates_constant_exactly() {
        let result = simpson(|_| 3.0, 0.0, 4.0, 100);
        assert!((result - 12.0).abs() < TOL);
    }

    #[test]
    fn simpson_integrates_quadratic_exactly() {
        // ∫₀¹ x² dx = 1/3; Simpson's rule is exact for polynomials ≤ degree 3
        let result = simpson(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < TOL);
    }
}
