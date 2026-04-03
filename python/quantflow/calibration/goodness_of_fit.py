"""
Goodness-of-fit diagnostics for calibrated Hawkes processes.

Uses the Time-Rescaling Theorem (Brown et al., 2002):

    If the model is correctly specified, the rescaled inter-event times
        τ_k = Λ(t_k) - Λ(t_{k-1})
    are i.i.d. Exp(1).  Equivalently, u_k = 1 - exp(-τ_k) ~ Uniform(0,1).

A KS test against Exp(1) at α=0.05 provides a formal model validity check.

Reference: Brown et al. (2002), "The Time-Rescaling Theorem and Its
Application to Neural Spike Train Data Analysis".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest, expon

from .hawkes_mle import HawkesParams


class HawkesGoodnessOfFit:
    """
    Diagnostics for a single fitted Hawkes dimension.

    Parameters
    ----------
    params : HawkesParams
        Fitted parameters (output of HawkesMLE).
    all_times : list[np.ndarray]
        Per-dimension event time arrays (same input used for calibration).
    T : float
        Observation window length (seconds).
    """

    def __init__(
        self,
        params:    HawkesParams,
        all_times: list[np.ndarray],
        T:         float,
    ) -> None:
        self.params    = params
        self.all_times = all_times
        self.T         = T
        self._rescaled: np.ndarray | None = None

    # ── rescaled times ────────────────────────────────────────────────────────

    def compute_rescaled_times(self) -> np.ndarray:
        """
        Compute the compensator Λ_d(t_k) for each event of dimension d.

        Returns τ_k = Λ(t_k) - Λ(t_{k-1}), shape (n_target - 1,).
        Under the null (correct model), τ_k ~ i.i.d. Exp(1).

        Efficient O(N_target × D) implementation: loops over target events
        only, using binary search + vectorised numpy for source contributions.

        Compensator integral over [t_{k-1}, t_k]:
            ΔΛ = μ·dt + Σ_j (α_j/β_j)·R_j·(1 - exp(-β_j·dt))
                      + Σ_j Σ_{t_{k-1} ≤ s_l^j < t_k} (α_j/β_j)·(1 - exp(-β_j·(t_k-s_l^j)))

        where R_j = Σ_{s < t_{k-1}, dim=j} exp(-β_j·(t_{k-1}-s)).
        """
        if self._rescaled is not None:
            return self._rescaled

        p         = self.params
        D         = p.n_dims
        mu        = p.mu
        alpha     = p.alpha
        beta      = p.beta
        all_times = self.all_times
        times     = all_times[p.dim]   # target events, sorted

        n_target = len(times)
        if n_target < 2:
            self._rescaled = np.array([], dtype=np.float64)
            return self._rescaled

        # R[j] = exponentially-weighted sum of past source events from dim j,
        #        evaluated just BEFORE the current target event.
        R          = np.zeros(D)
        ptrs       = np.zeros(D, dtype=np.int64)
        t_prev     = 0.0
        cumulative = 0.0
        cum_at     = np.empty(n_target)

        for i in range(n_target):
            t  = times[i]
            dt = t - t_prev

            # ΔΛ from the existing R (events before t_prev), decayed to [t_prev, t]
            exp_decay    = np.exp(-beta * dt)
            delta_lambda = mu * dt + float(np.sum((alpha / beta) * R * (1.0 - exp_decay)))
            R           *= exp_decay

            # ΔΛ from new source events in [t_prev, t) and update R
            for j in range(D):
                src = all_times[j]
                hi  = int(np.searchsorted(src, t, side="left"))
                lo  = int(ptrs[j])
                if hi > lo:
                    gaps = t - src[lo:hi]                    # (t - s) for each new source event
                    exp_gaps = np.exp(-beta[j] * gaps)
                    # Contribution to Λ: (α_j/β_j)·(1 - exp(-β_j·(t-s))) for each s
                    delta_lambda += float((alpha[j] / beta[j]) * np.sum(1.0 - exp_gaps))
                    # Update R[j] with these new events
                    R[j] += float(np.sum(exp_gaps))
                ptrs[j] = hi

            cumulative    += delta_lambda
            cum_at[i]      = cumulative
            t_prev         = t

        # Rescaled inter-event times: τ_k = Λ(t_k) - Λ(t_{k-1})
        self._rescaled = np.diff(cum_at)
        return self._rescaled

    # ── KS test ───────────────────────────────────────────────────────────────

    def ks_test(self) -> dict:
        """
        Kolmogorov-Smirnov test against Exp(1).

        Returns
        -------
        dict with keys:
            statistic   : KS statistic
            p_value     : two-sided p-value
            passed      : bool (p_value > 0.05)
            n_intervals : number of rescaled intervals tested
        """
        tau = self.compute_rescaled_times()
        if len(tau) < 2:
            return {
                "statistic":   float("nan"),
                "p_value":     float("nan"),
                "passed":      False,
                "n_intervals": len(tau),
            }
        stat, pval = kstest(tau, "expon", args=(0, 1))
        return {
            "statistic":   float(stat),
            "p_value":     float(pval),
            "passed":      bool(pval > 0.05),
            "n_intervals": len(tau),
        }

    # ── Q-Q data ──────────────────────────────────────────────────────────────

    def qq_data(self) -> dict:
        """
        Data for a Q-Q plot: empirical vs theoretical Exp(1) quantiles.

        Returns
        -------
        dict with keys:
            theoretical : np.ndarray  (sorted Exp(1) quantiles)
            empirical   : np.ndarray  (sorted rescaled inter-event times)
            n           : int
        """
        tau = self.compute_rescaled_times()
        if len(tau) == 0:
            return {"theoretical": np.array([]), "empirical": np.array([]), "n": 0}

        empirical   = np.sort(tau)
        n           = len(empirical)
        # Theoretical quantiles at the same probability points
        probs       = (np.arange(1, n + 1) - 0.5) / n
        theoretical = expon.ppf(probs)

        return {
            "theoretical": theoretical,
            "empirical":   empirical,
            "n":           n,
        }

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Combined diagnostics dict."""
        tau = self.compute_rescaled_times()
        ks  = self.ks_test()
        return {
            "dim":             self.params.dim,
            "dim_name":        self.params.dim_name,
            "n_events":        self.params.n_events,
            "n_intervals":     ks["n_intervals"],
            "ks_statistic":    ks["statistic"],
            "ks_p_value":      ks["p_value"],
            "ks_passed":       ks["passed"],
            "tau_mean":        float(np.mean(tau)) if len(tau) > 0 else float("nan"),
            "tau_std":         float(np.std(tau))  if len(tau) > 0 else float("nan"),
            "branching_ratio": self.params.branching_ratio,
            "mu":              self.params.mu,
        }
