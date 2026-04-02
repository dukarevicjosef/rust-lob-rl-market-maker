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

from .hawkes_mle import HawkesParams, _merge_events


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
        """
        if self._rescaled is not None:
            return self._rescaled

        p          = self.params
        D          = p.n_dims
        mu         = p.mu
        alpha      = p.alpha
        beta       = p.beta
        T          = self.T
        all_times  = self.all_times
        times      = all_times[p.dim]

        # Merge all events for recursion
        merged_times, merged_dims = _merge_events(all_times, D)

        if len(times) < 2 or len(merged_times) == 0:
            self._rescaled = np.array([], dtype=np.float64)
            return self._rescaled

        # Compute the integrated intensity at each target event via recursion.
        # Λ(t) = μ·t + Σ_j (α_j/β_j)·Σ_{t_j^l < t} (1 - exp(-β_j·(t - t_j^l)))
        #
        # Between consecutive events the integral accumulates:
        #   ΔΛ(t_{k-1}, t_k) = μ·dt + Σ_j α_j · R_j · (1 - exp(-β_j·dt)) / β_j × β_j
        # which simplifies to:
        #   ΔΛ = μ·dt + Σ_j (α_j/β_j) · R_j · (1 - exp(-β_j·dt))  ... (Papangelou)
        #
        # We track R_j at the moment just before t_k, then compute the integral
        # up to t_k accounting for the new event that may have arrived.

        R          = np.zeros(D)
        t_prev     = 0.0
        cumulative = 0.0    # Λ accumulated up to (not including) the previous event

        # We need cumulative Λ at each target event time
        # Build sorted target event positions in merged stream
        target_times_sorted = times  # already sorted (EventClassifier guarantees this)
        n_target = len(target_times_sorted)
        cum_at_target = np.empty(n_target)

        target_idx = 0   # pointer into target_times_sorted

        n_merged = len(merged_times)
        k = 0
        while k < n_merged and target_idx < n_target:
            t_k   = merged_times[k]
            dim_k = merged_dims[k]
            dt    = t_k - t_prev

            # Integrate Λ over [t_prev, t_k]
            # ΔΛ = μ·dt + Σ_j (α_j/β_j)·R_j·(1 - exp(-β_j·dt))
            exp_decay = np.exp(-beta * dt)
            delta_lambda = mu * dt + np.sum((alpha / beta) * R * (1.0 - exp_decay))
            cumulative  += delta_lambda

            # Update R
            R *= exp_decay
            R[dim_k] += 1.0

            # Record cumulative Λ for all target events that occur exactly at t_k
            while target_idx < n_target and target_times_sorted[target_idx] == t_k:
                cum_at_target[target_idx] = cumulative
                target_idx += 1

            t_prev = t_k
            k += 1

        # Handle any remaining target events (times after the last merged event)
        while target_idx < n_target:
            t_k = target_times_sorted[target_idx]
            dt  = t_k - t_prev
            exp_decay = np.exp(-beta * dt)
            delta_lambda = mu * dt + np.sum((alpha / beta) * R * (1.0 - exp_decay))
            cumulative  += delta_lambda
            R *= exp_decay
            cum_at_target[target_idx] = cumulative
            t_prev = t_k
            target_idx += 1

        # Rescaled inter-event times: τ_k = Λ(t_k) - Λ(t_{k-1})
        self._rescaled = np.diff(cum_at_target)
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
