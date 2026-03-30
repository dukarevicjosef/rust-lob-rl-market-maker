//! Multivariate Hawkes process simulation via Ogata thinning (Ogata 1981).
//!
//! The conditional intensity of component i at time t is
//!
//!   λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} φ_ij(t − t_k^j)
//!
//! where μ_i is the background rate and φ_ij is the excitation kernel
//! describing how a type-j event raises the intensity of type i.
//!
//! ## Thinning algorithm (Ogata 1981, §2)
//!
//! 1. Set λ* = Σ_i λ_i(t) — the current total intensity is an upper bound
//!    because all kernels are monotone-decreasing for t > 0.
//! 2. Draw inter-arrival dt ~ Exp(λ*).  t_cand = t + dt.
//! 3. Evaluate λ(t_cand) by scanning the active-event set.
//! 4. Accept with probability λ(t_cand) / λ*; sample type ∝ λ_i(t_cand).
//! 5. Whether accepted or rejected, advance t = t_cand and tighten
//!    λ* = λ(t_cand) (resp. λ(t_cand⁺) after acceptance).
//!
//! ## Incremental intensity updates
//!
//! The running excitation vector r[i] = Σ_{active k} φ_{i,type_k}(t − t_k)
//! is recomputed from the pruned active-event set at each candidate time.
//! Events are pruned from the active set once every contribution drops below
//! PRUNE_THRESHOLD, keeping the active set O(1/β) deep for exponential
//! kernels.  For exponential kernels a true O(1) per-step recursive update
//! r[i] → r[i]·exp(−β·Δt) is possible but requires exposing the decay
//! structure through the trait; the general dyn approach here is O(|active|·d).

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand_distr::Exp;
use rayon::prelude::*;
use thiserror::Error;

use super::kernel::ExcitationKernel;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, PartialEq)]
pub enum ProcessError {
    #[error("dimension mismatch: baselines length {baselines} != kernel rows {kernels}")]
    DimensionMismatch { baselines: usize, kernels: usize },
    #[error("kernel matrix row {row} has {got} columns, expected {expected}")]
    KernelRowLength { row: usize, got: usize, expected: usize },
    #[error("baseline μ_{index} must be non-negative, got {value}")]
    NegativeBaseline { index: usize, value: f64 },
}

// ── Event ─────────────────────────────────────────────────────────────────────

/// A single realised event in a multivariate Hawkes process.
#[derive(Debug, Clone, PartialEq)]
pub struct HawkesEvent {
    /// Arrival time (seconds from simulation origin).
    pub time: f64,
    /// Component index ∈ [0, d).
    pub event_type: usize,
}

// ── Process ───────────────────────────────────────────────────────────────────

/// Multivariate Hawkes process with d event types.
///
/// `kernels[i][j]` = φ_ij: the kernel describing how type-j events
/// excite dimension i.  For a LOB with d = 12 the index layout is
/// (market/limit/cancel) × (buy/sell) × (best/deep).
pub struct MultivariateHawkes {
    pub dimension: usize,
    pub baselines: Vec<f64>,
    /// φ_ij stored row-major: `kernels[i][j]`.
    pub kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>>,
}

impl MultivariateHawkes {
    /// Construct and validate a multivariate Hawkes process.
    pub fn new(
        baselines: Vec<f64>,
        kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>>,
    ) -> Result<Self, ProcessError> {
        let d = baselines.len();
        if kernels.len() != d {
            return Err(ProcessError::DimensionMismatch {
                baselines: d,
                kernels: kernels.len(),
            });
        }
        for (row, row_vec) in kernels.iter().enumerate() {
            if row_vec.len() != d {
                return Err(ProcessError::KernelRowLength {
                    row,
                    got: row_vec.len(),
                    expected: d,
                });
            }
        }
        for (index, &value) in baselines.iter().enumerate() {
            if value < 0.0 {
                return Err(ProcessError::NegativeBaseline { index, value });
            }
        }
        Ok(MultivariateHawkes { dimension: d, baselines, kernels })
    }

    /// Conditional intensity vector λ(t) given a sorted event history.
    ///
    /// λ_i(t) = μ_i + Σ_{k: t_k < t} φ_{i, type_k}(t − t_k)
    ///
    /// History must be sorted by ascending `time`; iteration stops at the
    /// first event with time ≥ t.
    pub fn intensity(&self, t: f64, history: &[HawkesEvent]) -> Vec<f64> {
        let d = self.dimension;
        let mut lambda = self.baselines.clone();
        for event in history {
            if event.time >= t {
                break;
            }
            let elapsed = t - event.time;
            let j = event.event_type;
            for i in 0..d {
                lambda[i] += self.kernels[i][j].evaluate(elapsed);
            }
        }
        lambda
    }

    /// Log-likelihood of an event sequence on [0, t_max].
    ///
    /// ℓ = Σ_k log λ_{type_k}(t_k⁻)  −  Σ_i ∫₀^{t_max} λ_i(t) dt
    ///
    /// The compensator uses the closed form (Daley & Vere-Jones 2003, Ch. 7):
    ///
    ///   ∫₀^{t_max} λ_i(t) dt = μ_i · t_max
    ///                         + Σ_{k: t_k < t_max} φ_{i,type_k}.integral(t_max − t_k)
    pub fn log_likelihood(&self, events: &[HawkesEvent], t_max: f64) -> f64 {
        let d = self.dimension;

        // Sum of log-intensities at each event time (left limit λ(t_k⁻)).
        let mut ll = 0.0_f64;
        for (k, event) in events.iter().enumerate() {
            if event.time > t_max {
                break;
            }
            let lambda = self.intensity(event.time, &events[..k]);
            // Guard against log(0) for degenerate parameter sets.
            ll += lambda[event.event_type].max(f64::MIN_POSITIVE).ln();
        }

        // Compensator: Σ_i ∫₀^{t_max} λ_i(t) dt.
        let mut compensator = 0.0_f64;
        for i in 0..d {
            compensator += self.baselines[i] * t_max;
        }
        for event in events {
            if event.time >= t_max {
                break;
            }
            let remaining = t_max - event.time;
            let j = event.event_type;
            for i in 0..d {
                compensator += self.kernels[i][j].integral(remaining);
            }
        }

        ll - compensator
    }

    /// Simulate a single realisation on [0, t_max] via Ogata thinning.
    pub fn simulate(&self, t_max: f64, seed: u64) -> Vec<HawkesEvent> {
        thinning(self, t_max, seed).0
    }

    /// Simulate `n` independent realisations in parallel using rayon.
    ///
    /// Each run uses seed `base_seed + k` for k ∈ [0, n).
    pub fn simulate_n(&self, t_max: f64, base_seed: u64, n: usize) -> Vec<Vec<HawkesEvent>> {
        (0..n as u64)
            .into_par_iter()
            .map(|k| thinning(self, t_max, base_seed + k).0)
            .collect()
    }

    /// Simulate and return `(events, n_proposed)` for testing the acceptance rate.
    #[cfg(test)]
    pub(crate) fn simulate_traced(
        &self,
        t_max: f64,
        seed: u64,
    ) -> (Vec<HawkesEvent>, usize) {
        thinning(self, t_max, seed)
    }
}

// ── Ogata thinning ────────────────────────────────────────────────────────────

/// Events contributing less than this to every dimension are pruned from the
/// active set.  Keeps the set O(1/β) deep for exponential kernels.
const PRUNE_THRESHOLD: f64 = 1e-12;

/// Prune only when the active set exceeds this size to amortise the cost.
const PRUNE_MIN_ACTIVE: usize = 128;

/// Core Ogata thinning algorithm.  Returns `(events, n_proposed_candidates)`.
fn thinning(
    process: &MultivariateHawkes,
    t_max: f64,
    seed: u64,
) -> (Vec<HawkesEvent>, usize) {
    let d = process.dimension;
    let mut rng = SmallRng::seed_from_u64(seed);

    // Full realisation returned to the caller.
    let mut history: Vec<HawkesEvent> = Vec::new();

    // Active subset: (arrival_time, event_type) for events still contributing
    // non-negligibly.  Separated from `history` to keep the hot-path scan tight.
    let mut active: Vec<(f64, usize)> = Vec::new();

    let mut t = 0.0_f64;

    // r[i] = Σ_{active k} φ_{i, type_k}(t_current − t_k)
    let mut r: Vec<f64> = vec![0.0_f64; d];

    // λ* — upper bound on total intensity.  Initialised to Σ μ_i (no events yet).
    let mut lambda_star: f64 = process.baselines.iter().sum();

    let mut n_proposed: usize = 0;

    loop {
        if lambda_star <= 0.0 {
            break;
        }

        // Draw next candidate inter-arrival dt ~ Exp(λ*).
        let dt: f64 = rng.sample(Exp::new(lambda_star).unwrap());
        let t_cand = t + dt;
        if t_cand > t_max {
            break;
        }
        n_proposed += 1;

        // Recompute r at t_cand from the active set (incremental over pruned events).
        let mut r_cand = vec![0.0_f64; d];
        for &(t_k, j) in &active {
            let elapsed = t_cand - t_k;
            for i in 0..d {
                r_cand[i] += process.kernels[i][j].evaluate(elapsed);
            }
        }
        let lambda_cand_total: f64 = process
            .baselines
            .iter()
            .zip(r_cand.iter())
            .map(|(&mu, &ri)| mu + ri)
            .sum();

        // Thinning acceptance test (Ogata 1981, §2).
        let u: f64 = rng.gen();
        t = t_cand; // advance time regardless of acceptance

        if u * lambda_star <= lambda_cand_total {
            // ── Accept ──────────────────────────────────────────────────────

            // Sample event type ∝ λ_i(t_cand).
            let threshold = rng.gen::<f64>() * lambda_cand_total;
            let mut cumsum = 0.0_f64;
            let mut event_type = d - 1;
            for i in 0..d {
                cumsum += process.baselines[i] + r_cand[i];
                if threshold < cumsum {
                    event_type = i;
                    break;
                }
            }

            // Include the new event's φ(0) contribution in r_cand so that the
            // new λ* = λ(t_cand⁺) is immediately available without another scan.
            for i in 0..d {
                r_cand[i] += process.kernels[i][event_type].evaluate(0.0);
            }

            history.push(HawkesEvent { time: t_cand, event_type });
            active.push((t_cand, event_type));
            r = r_cand;

            // New upper bound: intensity right after the accepted event.
            lambda_star = process
                .baselines
                .iter()
                .zip(r.iter())
                .map(|(&mu, &ri)| mu + ri)
                .sum();

            // Prune negligible events to bound the active-set scan cost.
            if active.len() >= PRUNE_MIN_ACTIVE {
                active.retain(|&(t_k, j)| {
                    let elapsed = t_cand - t_k;
                    (0..d).any(|i| process.kernels[i][j].evaluate(elapsed) > PRUNE_THRESHOLD)
                });
                // Recompute r from the freshly pruned set.
                r = vec![0.0_f64; d];
                for &(t_k, j) in &active {
                    let elapsed = t_cand - t_k;
                    for i in 0..d {
                        r[i] += process.kernels[i][j].evaluate(elapsed);
                    }
                }
                lambda_star = process
                    .baselines
                    .iter()
                    .zip(r.iter())
                    .map(|(&mu, &ri)| mu + ri)
                    .sum();
            }
        } else {
            // ── Reject ──────────────────────────────────────────────────────
            // Tighten the upper bound to the actual intensity at t_cand.
            // r is not updated: the next iteration recomputes r_cand from active.
            lambda_star = lambda_cand_total;
        }
    }

    (history, n_proposed)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkes::kernel::{ExcitationKernel, ExponentialKernel};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a 1D Hawkes(μ, α, β) with one ExponentialKernel.
    fn make_1d(
        mu: f64,
        alpha: f64,
        beta: f64,
    ) -> MultivariateHawkes {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
            vec![vec![Box::new(ExponentialKernel::new(alpha, beta).unwrap())]];
        MultivariateHawkes::new(vec![mu], kernels).unwrap()
    }

    /// Build a symmetric 2D Hawkes with self-excitation α_self and
    /// cross-excitation α_cross.
    fn make_2d_symmetric(
        mu: f64,
        alpha_self: f64,
        alpha_cross: f64,
        beta: f64,
    ) -> MultivariateHawkes {
        let mk = |a: f64| -> Box<dyn ExcitationKernel + Send + Sync> {
            Box::new(ExponentialKernel::new(a, beta).unwrap())
        };
        let kernels = vec![
            vec![mk(alpha_self), mk(alpha_cross)],
            vec![mk(alpha_cross), mk(alpha_self)],
        ];
        MultivariateHawkes::new(vec![mu, mu], kernels).unwrap()
    }

    // ── Constructor validation ─────────────────────────────────────────────────

    #[test]
    fn constructor_rejects_baseline_kernel_dimension_mismatch() {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
            vec![vec![Box::new(ExponentialKernel::new(0.3, 1.0).unwrap())]];
        let result = MultivariateHawkes::new(vec![1.0, 2.0], kernels);
        assert!(
            matches!(result, Err(ProcessError::DimensionMismatch { baselines: 2, kernels: 1 }))
        );
    }

    #[test]
    fn constructor_rejects_kernel_row_wrong_length() {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> = vec![
            vec![
                Box::new(ExponentialKernel::new(0.2, 1.0).unwrap()),
                Box::new(ExponentialKernel::new(0.1, 1.0).unwrap()),
            ],
            vec![Box::new(ExponentialKernel::new(0.1, 1.0).unwrap())], // short row
        ];
        let result = MultivariateHawkes::new(vec![1.0, 1.0], kernels);
        assert!(matches!(
            result,
            Err(ProcessError::KernelRowLength { row: 1, got: 1, expected: 2 })
        ));
    }

    #[test]
    fn constructor_rejects_negative_baseline() {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
            vec![vec![Box::new(ExponentialKernel::new(0.3, 1.0).unwrap())]];
        let result = MultivariateHawkes::new(vec![-1.0], kernels);
        assert!(matches!(result, Err(ProcessError::NegativeBaseline { index: 0, .. })));
    }

    // ── intensity() ──────────────────────────────────────────────────────────

    #[test]
    fn intensity_at_origin_equals_baselines() {
        let h = make_1d(3.0, 0.5, 2.0);
        let lambda = h.intensity(0.0, &[]);
        assert_eq!(lambda, vec![3.0]);
    }

    #[test]
    fn intensity_increases_immediately_after_event() {
        let h = make_1d(1.0, 0.6, 1.0);
        let events = vec![HawkesEvent { time: 1.0, event_type: 0 }];
        // λ(1.5) includes φ(0.5) from the event at t=1
        let lambda_after = h.intensity(1.5, &events);
        // Must exceed baseline
        assert!(lambda_after[0] > h.baselines[0]);
    }

    #[test]
    fn intensity_decays_back_toward_baseline() {
        let h = make_1d(1.0, 0.5, 5.0); // fast decay β=5
        let events = vec![HawkesEvent { time: 0.0, event_type: 0 }];
        let lambda_near = h.intensity(0.1, &events)[0];
        let lambda_far = h.intensity(10.0, &events)[0];
        assert!(lambda_far < lambda_near, "intensity should decay");
        // After 10s with β=5: contribution ≈ α·β·exp(-50) ≈ 0; should be near μ=1
        assert!((lambda_far - 1.0).abs() < 1e-6);
    }

    // ── log_likelihood() ─────────────────────────────────────────────────────

    #[test]
    fn log_likelihood_empty_history_equals_neg_mu_t() {
        // ℓ = 0 (no events) − μ · T
        let h = make_1d(2.0, 0.3, 1.0);
        let ll = h.log_likelihood(&[], 5.0);
        assert!((ll - (-10.0)).abs() < 1e-12, "ll={ll}");
    }

    #[test]
    fn log_likelihood_single_event_known_value() {
        // 1D Hawkes μ=1, α=0.5, β=1 on [0, 2] with one event at t=1.
        //   λ(1⁻) = μ = 1.0
        //   compensator = μ·2 + ∫₁^2 φ(s−1) ds = 2 + kernel.integral(1)
        //                = 2 + 0.5·(1 − e⁻¹)
        //   ℓ = ln(1) − (2 + 0.5·(1−e⁻¹)) = −2 − 0.5·(1−e⁻¹)
        let h = make_1d(1.0, 0.5, 1.0);
        let events = vec![HawkesEvent { time: 1.0, event_type: 0 }];
        let ll = h.log_likelihood(&events, 2.0);
        let expected = -(2.0 + 0.5 * (1.0 - (-1.0_f64).exp()));
        assert!((ll - expected).abs() < 1e-12, "ll={ll:.12}, expected={expected:.12}");
    }

    #[test]
    fn log_likelihood_two_events_known_value() {
        // 1D Hawkes μ=2, α=0.3, β=1 on [0, 3] with events at t=1 and t=2.
        //   λ(1⁻) = μ = 2.0
        //   λ(2⁻) = μ + φ(1) = 2 + 0.3·1·e⁻¹ = 2 + 0.3·e⁻¹
        //   compensator:
        //     μ·3 = 6
        //     event at t=1: kernel.integral(2) = 0.3·(1−e⁻²)
        //     event at t=2: kernel.integral(1) = 0.3·(1−e⁻¹)
        let h = make_1d(2.0, 0.3, 1.0);
        let events = vec![
            HawkesEvent { time: 1.0, event_type: 0 },
            HawkesEvent { time: 2.0, event_type: 0 },
        ];
        let ll = h.log_likelihood(&events, 3.0);
        let phi_at_1 = 0.3 * 1.0 * (-1.0_f64).exp(); // φ(1) = α·β·e^{-β}
        let expected = (2.0_f64).ln()
            + (2.0 + phi_at_1).ln()
            - (6.0 + 0.3 * (1.0 - (-2.0_f64).exp()) + 0.3 * (1.0 - (-1.0_f64).exp()));
        assert!((ll - expected).abs() < 1e-11, "ll={ll:.12}, expected={expected:.12}");
    }

    #[test]
    fn log_likelihood_higher_for_true_params_than_misspecified() {
        // Generate data from true parameters; true ll should exceed misspecified ll.
        let true_h = make_1d(1.0, 0.5, 2.0);
        let events = true_h.simulate(2_000.0, 777);
        let ll_true = true_h.log_likelihood(&events, 2_000.0);

        let wrong_h = make_1d(1.0, 0.05, 2.0); // α too low
        let ll_wrong = wrong_h.log_likelihood(&events, 2_000.0);

        assert!(ll_true > ll_wrong, "ll_true={ll_true:.2}, ll_wrong={ll_wrong:.2}");
    }

    // ── simulate() — structural correctness ──────────────────────────────────

    #[test]
    fn simulate_zero_baseline_produces_no_events() {
        let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
            vec![vec![Box::new(ExponentialKernel::new(0.0, 1.0).unwrap())]];
        let h = MultivariateHawkes::new(vec![0.0], kernels).unwrap();
        let events = h.simulate(1_000.0, 0);
        assert!(events.is_empty(), "expected no events when μ=0 and α=0");
    }

    #[test]
    fn simulate_events_sorted_by_time() {
        let h = make_1d(5.0, 0.3, 2.0);
        let events = h.simulate(200.0, 1);
        for w in events.windows(2) {
            assert!(w[0].time < w[1].time, "{} >= {}", w[0].time, w[1].time);
        }
    }

    #[test]
    fn simulate_all_event_types_in_range() {
        let h = make_2d_symmetric(1.0, 0.2, 0.1, 1.0);
        let events = h.simulate(500.0, 42);
        assert!(events.iter().all(|e| e.event_type < 2));
    }

    #[test]
    fn simulate_all_times_within_t_max() {
        let h = make_1d(2.0, 0.4, 1.0);
        let t_max = 100.0;
        let events = h.simulate(t_max, 55);
        assert!(events.iter().all(|e| e.time <= t_max));
    }

    // ── simulate() — statistical correctness ─────────────────────────────────

    #[test]
    fn empirical_rate_matches_stationary_theory_1d() {
        // For a stationary 1D Hawkes: E[N/T] = μ / (1 − α).
        // μ=1, α=0.5 → theoretical rate = 2.0.
        let h = make_1d(1.0, 0.5, 2.0);
        let t_max = 5_000.0;
        let n = h.simulate(t_max, 42).len() as f64;
        let empirical = n / t_max;
        let theoretical = 1.0 / (1.0 - 0.5);
        let rel_err = (empirical - theoretical).abs() / theoretical;
        assert!(rel_err < 0.05, "empirical={empirical:.3}, theoretical={theoretical:.3}");
    }

    #[test]
    fn stable_process_event_count_stays_bounded() {
        // μ=0.5, α=0.3 → E[N] = 0.5/0.7 · 1000 ≈ 714.  Allow 3× slack.
        let h = make_1d(0.5, 0.3, 1.0);
        let t_max = 1_000.0;
        let n = h.simulate(t_max, 7).len() as f64;
        let theoretical_mean = 0.5 / (1.0 - 0.3) * t_max;
        assert!(
            n < 3.0 * theoretical_mean,
            "n={n:.0}, 3×E[N]={:.0}",
            3.0 * theoretical_mean
        );
    }

    #[test]
    fn thinning_acceptance_rate_in_valid_range() {
        // Acceptance rate ∈ (0, 1) for any non-trivial stable process.
        let h = make_1d(2.0, 0.4, 3.0);
        let (events, n_proposed) = h.simulate_traced(1_000.0, 0);
        assert!(n_proposed > 0, "no candidates proposed");
        let acceptance_rate = events.len() as f64 / n_proposed as f64;
        assert!(
            acceptance_rate > 0.05 && acceptance_rate < 1.0,
            "acceptance_rate={acceptance_rate:.3}"
        );
        // Sanity: we never accept more events than we proposed.
        assert!(events.len() <= n_proposed);
    }

    #[test]
    fn thinning_acceptance_rate_worsens_with_higher_excitation() {
        // Higher α → more intense bursts → larger λ* vs. λ_avg → lower acceptance.
        let h_low = make_1d(2.0, 0.1, 3.0);
        let h_high = make_1d(2.0, 0.8, 3.0);
        let (events_low, prop_low) = h_low.simulate_traced(500.0, 1);
        let (events_high, prop_high) = h_high.simulate_traced(500.0, 1);
        let rate_low = events_low.len() as f64 / prop_low as f64;
        let rate_high = events_high.len() as f64 / prop_high as f64;
        assert!(
            rate_high < rate_low,
            "expected higher excitation → worse acceptance: rate_high={rate_high:.3}, rate_low={rate_low:.3}"
        );
    }

    // ── simulate_n() ──────────────────────────────────────────────────────────

    #[test]
    fn simulate_n_returns_correct_count() {
        let h = make_1d(1.0, 0.4, 2.0);
        let runs = h.simulate_n(100.0, 0, 8);
        assert_eq!(runs.len(), 8);
    }

    #[test]
    fn simulate_n_realisations_are_independent() {
        let h = make_1d(2.0, 0.3, 1.0);
        let runs = h.simulate_n(500.0, 100, 4);
        let counts: Vec<usize> = runs.iter().map(|r| r.len()).collect();
        let all_equal = counts.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_equal, "all realisations identical: {counts:?}");
    }

    // ── bivariate mutual excitation ───────────────────────────────────────────

    #[test]
    fn bivariate_both_dimensions_fire() {
        let h = make_2d_symmetric(0.5, 0.2, 0.1, 1.0);
        let events = h.simulate(1_000.0, 42);
        let n0 = events.iter().filter(|e| e.event_type == 0).count();
        let n1 = events.iter().filter(|e| e.event_type == 1).count();
        assert!(n0 > 0 && n1 > 0, "n0={n0}, n1={n1}");
    }

    #[test]
    fn bivariate_symmetric_rates_approximately_equal() {
        // Symmetric parameters → both dimensions fire at the same rate.
        let h = make_2d_symmetric(1.0, 0.2, 0.1, 2.0);
        let events = h.simulate(5_000.0, 9);
        let n0 = events.iter().filter(|e| e.event_type == 0).count() as f64;
        let n1 = events.iter().filter(|e| e.event_type == 1).count() as f64;
        let ratio = n0 / n1.max(1.0);
        assert!(
            (ratio - 1.0).abs() < 0.10,
            "n0={n0:.0}, n1={n1:.0}, ratio={ratio:.3}"
        );
    }
}
