// Simulate a 1D Hawkes process, calibrate it, compute goodness-of-fit,
// then print Q-Q data as CSV to stdout for the Python plotter.
//
// Usage:
//   cargo run -p quantflow-core --example hawkes_qq | python3 scripts/plot_qq.py

use quantflow_core::hawkes::{
    calibrate_exponential, goodness_of_fit, ExcitationKernel, ExponentialKernel,
    MultivariateHawkes,
};

fn main() {
    let true_mu = 1.0_f64;
    let true_alpha = 0.5_f64;
    let true_beta = 2.0_f64;
    let t_max = 5_000.0_f64;

    // Simulate
    let kernels: Vec<Vec<Box<dyn ExcitationKernel + Send + Sync>>> =
        vec![vec![Box::new(ExponentialKernel::new(true_alpha, true_beta).unwrap())]];
    let hawkes = MultivariateHawkes::new(vec![true_mu], kernels).unwrap();
    let events = hawkes.simulate(t_max, 42);

    eprintln!("Simulated {} events", events.len());

    // Calibrate
    let result = calibrate_exponential(&events, 1, t_max).unwrap();
    eprintln!(
        "Calibrated: μ={:.3}  α={:.3}  β={:.3}  (converged={}, iter={})",
        result.params.mu[0],
        result.params.alpha[0][0],
        result.params.beta[0][0],
        result.converged,
        result.n_iter,
    );

    // Goodness of fit
    let gof = goodness_of_fit(&events, &result.params, t_max);
    eprintln!(
        "KS: D={:.4}  p={:.4}",
        gof.ks_statistic, gof.ks_p_value
    );

    // Write CSV to stdout: empirical,theoretical
    println!("empirical,theoretical");
    for (e, t) in gof.qq_empirical.iter().zip(gof.qq_theoretical.iter()) {
        println!("{e},{t}");
    }
}
