"""
Hawkes process MLE calibration with exponential kernels.

Each dimension d has intensity:
    λ_d(t) = μ_d + Σ_j α_{dj} · Σ_{t_j^k < t} exp(-β_{dj} · (t - t_j^k))

Log-likelihood is computed via the recursive form (Ozaki, 1979):
    R_j(k) = exp(-β_{dj} · dt_k) · (R_j(k-1) + 1_{event from dim j at t_{k-1}})

Compensator (integrated intensity):
    Λ_d = μ_d · T + Σ_j (α_{dj}/β_{dj}) · Σ_k (1 - exp(-β_{dj} · (T - t_j^k)))

Stationarity: branching ratio ρ = Σ_j (α_{dj}/β_{dj}) < 1.

Reference: Ozaki (1979), Hawkes (1971), Bacry et al. (2015).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesParams:
    """Fitted parameters for one dimension of a multivariate Hawkes process."""

    dim:             int
    dim_name:        str
    n_dims:          int
    mu:              float                    # baseline intensity
    alpha:           np.ndarray              # excitation amplitudes  (n_dims,)
    beta:            np.ndarray              # decay rates            (n_dims,)
    log_likelihood:  float
    branching_ratio: float                   # Σ_j (α_j / β_j), must be < 1
    n_events:        int

    def excitation_matrix_row(self) -> dict[int, float]:
        """Map from source dim → α/β (mean number of triggered events)."""
        return {j: float(self.alpha[j] / self.beta[j]) for j in range(self.n_dims)}

    def to_dict(self) -> dict:
        return {
            "dim":             self.dim,
            "dim_name":        self.dim_name,
            "n_dims":          self.n_dims,
            "mu":              self.mu,
            "alpha":           self.alpha.tolist(),
            "beta":            self.beta.tolist(),
            "log_likelihood":  self.log_likelihood,
            "branching_ratio": self.branching_ratio,
            "n_events":        self.n_events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HawkesParams:
        return cls(
            dim=d["dim"],
            dim_name=d["dim_name"],
            n_dims=d["n_dims"],
            mu=d["mu"],
            alpha=np.array(d["alpha"]),
            beta=np.array(d["beta"]),
            log_likelihood=d["log_likelihood"],
            branching_ratio=d["branching_ratio"],
            n_events=d["n_events"],
        )


@dataclass
class CalibrationResult:
    """Full calibration result for a multivariate Hawkes process."""

    dim_params:      list[HawkesParams]
    dim_names:       list[str]
    t_span:          float
    total_events:    int
    source_path:     str
    calibrated_dims: list[int]                # dims that were actually fitted
    skipped_dims:    list[int]               # dims with insufficient data
    meta:            dict = field(default_factory=dict)

    @property
    def n_dims(self) -> int:
        return len(self.dim_names)

    def params_for(self, dim: int) -> HawkesParams | None:
        for p in self.dim_params:
            if p.dim == dim:
                return p
        return None

    def excitation_matrix(self) -> np.ndarray:
        """
        (D × D) matrix where entry [i, j] = α_{ij}/β_{ij}.
        Row i = how strongly other dims j excite dim i.
        """
        D = self.n_dims
        mat = np.zeros((D, D))
        for p in self.dim_params:
            for j in range(D):
                mat[p.dim, j] = p.alpha[j] / p.beta[j]
        return mat

    def top_cross_excitations(self, n: int = 10) -> list[tuple[int, int, float]]:
        """
        Return top-n off-diagonal excitation pairs (target_dim, source_dim, strength).
        """
        mat = self.excitation_matrix()
        D = self.n_dims
        pairs: list[tuple[int, int, float]] = []
        for i in range(D):
            for j in range(D):
                if i != j:
                    pairs.append((i, j, mat[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    def to_dict(self) -> dict:
        return {
            "dim_params":      [p.to_dict() for p in self.dim_params],
            "dim_names":       self.dim_names,
            "t_span":          self.t_span,
            "total_events":    self.total_events,
            "source_path":     self.source_path,
            "calibrated_dims": self.calibrated_dims,
            "skipped_dims":    self.skipped_dims,
            "meta":            self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationResult:
        return cls(
            dim_params=[HawkesParams.from_dict(p) for p in d["dim_params"]],
            dim_names=d["dim_names"],
            t_span=d["t_span"],
            total_events=d["total_events"],
            source_path=d["source_path"],
            calibrated_dims=d["calibrated_dims"],
            skipped_dims=d["skipped_dims"],
            meta=d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationResult:
        return cls.from_dict(json.loads(Path(path).read_text()))


class HawkesMLE:
    """
    MLE calibration of a multivariate Hawkes process with exponential kernels.

    Each dimension is fitted independently via L-BFGS-B, using the
    recursive log-likelihood formula (Ozaki, 1979).

    Parameters
    ----------
    n_dims : int
        Number of event dimensions.
    dim_names : list[str] | None
        Human-readable names for each dimension.
    max_iter : int
        L-BFGS-B maximum iterations.
    n_restarts : int
        Number of random restarts; best result is kept.
    beta_init : float
        Initial decay rate for all kernels.
    min_events : int
        Minimum events per dimension to attempt calibration.
    """

    def __init__(
        self,
        n_dims:    int,
        dim_names: list[str] | None = None,
        max_iter:  int = 500,
        n_restarts: int = 3,
        beta_init:  float = 10.0,
        min_events: int = 50,
    ) -> None:
        self.n_dims     = n_dims
        self.dim_names  = dim_names or [f"dim_{i}" for i in range(n_dims)]
        self.max_iter   = max_iter
        self.n_restarts = n_restarts
        self.beta_init  = beta_init
        self.min_events = min_events

    # ── public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        event_data,                           # HawkesEventData from EventClassifier
        dims: list[int] | None = None,
    ) -> CalibrationResult:
        """
        Calibrate all (or a subset of) dimensions.

        Parameters
        ----------
        event_data : HawkesEventData
            Output of EventClassifier.load_and_classify().
        dims : list[int] | None
            Specific dimensions to calibrate; defaults to all active dims.
        """
        target_dims   = dims if dims is not None else event_data.active_dims
        calibrated    = []
        skipped       = []
        dim_params    = []

        # Pre-fetch all event time arrays once
        all_times: list[np.ndarray] = [
            event_data.times_for_dim(d) for d in range(self.n_dims)
        ]
        T = event_data.t_span

        for d in range(self.n_dims):
            if d not in target_dims:
                skipped.append(d)
                continue
            times_d = all_times[d]
            if len(times_d) < self.min_events:
                skipped.append(d)
                continue

            params = self._calibrate_single_dim(d, all_times, T)
            dim_params.append(params)
            calibrated.append(d)

        return CalibrationResult(
            dim_params=dim_params,
            dim_names=self.dim_names,
            t_span=T,
            total_events=event_data.total_events,
            source_path=event_data.source_path,
            calibrated_dims=calibrated,
            skipped_dims=skipped,
        )

    # ── single-dim calibration ────────────────────────────────────────────────

    def _calibrate_single_dim(
        self,
        target_dim: int,
        all_times: list[np.ndarray],   # one array per dimension, absolute times
        T: float,
    ) -> HawkesParams:
        """
        Fit μ, α_0…α_{D-1}, β_0…β_{D-1} for one target dimension.

        Parameter vector layout (length 1 + 2·D):
            [mu, alpha_0, …, alpha_{D-1}, beta_0, …, beta_{D-1}]
        """
        D     = self.n_dims
        times = all_times[target_dim]

        # Identify source dimensions with no observed events.
        # Their α contributes nothing to either the log-intensity or the
        # compensator (empty sums), so the LL gradient w.r.t. those α is
        # identically zero. L-BFGS-B would leave them at whatever initial
        # value they were given — producing phantom excitation entries.
        # Fix: fix those α to 0 by setting lb = ub = 0.
        empty_src = [j for j in range(D) if len(all_times[j]) == 0]

        best_ll   = -np.inf
        best_res  = None

        rng = np.random.default_rng(seed=target_dim)

        for restart in range(self.n_restarts):
            if restart == 0:
                # Deterministic initialisation from empirical rates
                rate_target = len(times) / T if T > 0 else 1.0
                mu0    = rate_target * 0.5
                alpha0 = np.array([
                    0.0 if j in empty_src else rate_target * 0.1
                    for j in range(D)
                ])
                beta0  = np.full(D, self.beta_init)
            else:
                mu0    = rng.uniform(1e-4, 2.0)
                alpha0 = np.array([
                    0.0 if j in empty_src else rng.uniform(0.0, 1.0)
                    for j in range(D)
                ])
                beta0  = rng.uniform(1.0, 50.0, size=D)

            x0 = np.concatenate([[mu0], alpha0, beta0])

            bounds = (
                [(1e-8, None)]                                # mu
                + [
                    (0.0, 0.0) if j in empty_src else (0.0, None)
                    for j in range(D)
                ]                                             # alpha — frozen at 0 for empty dims
                + [(1e-4, None)] * D                         # beta
            )

            res = minimize(
                fun=self._neg_ll_and_grad,
                x0=x0,
                args=(target_dim, all_times, T),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )

            ll = -res.fun
            if ll > best_ll:
                best_ll  = ll
                best_res = res

        assert best_res is not None
        x     = best_res.x
        mu    = float(x[0])
        alpha = x[1 : 1 + D].copy()
        beta  = x[1 + D : 1 + 2 * D].copy()

        # Hard-zero any empty-source alphas (guard against numerical noise)
        for j in empty_src:
            alpha[j] = 0.0

        branching = float(np.sum(alpha / beta))

        # Stationarity constraint: branching ratio must be < 1.
        # Post-hoc rescaling: if ρ > 0.95, scale all α down proportionally
        # so that ρ_constrained = 0.95. β is unchanged (kernel shape preserved).
        BR_MAX = 0.95
        if branching > BR_MAX:
            alpha    = alpha * (BR_MAX / branching)
            branching = float(np.sum(alpha / beta))

        return HawkesParams(
            dim=target_dim,
            dim_name=self.dim_names[target_dim],
            n_dims=D,
            mu=mu,
            alpha=alpha,
            beta=beta,
            log_likelihood=best_ll,
            branching_ratio=branching,
            n_events=len(times),
        )

    # ── log-likelihood ────────────────────────────────────────────────────────

    def _neg_log_likelihood(
        self,
        x:          np.ndarray,
        target_dim: int,
        all_times:  list[np.ndarray],
        T:          float,
    ) -> float:
        """
        Negative log-likelihood for dimension target_dim.

        Efficient O(N_target × D) implementation: loops over target events only
        (not the full merged stream of all dims), using binary search +
        vectorised numpy for source contributions in each inter-event interval.

        R_j(t) = Σ_{s < t, dim=j} exp(-β_j · (t - s))

        Updating from t_prev to t_k:
            R_j *= exp(-β_j · dt)
            R_j += Σ_{t_prev ≤ s < t_k, dim=j} exp(-β_j · (t_k - s))

        Log-likelihood:
            ℓ = Σ_k log(μ + Σ_j α_j · R_j(t_k)) - Λ

        Compensator (vectorised):
            Λ = μ·T + Σ_j (α_j/β_j) · Σ_l (1 - exp(-β_j · (T - t_j^l)))
        """
        D      = self.n_dims
        mu     = x[0]
        alpha  = x[1 : 1 + D]
        beta   = x[1 + D : 1 + 2 * D]

        target_times = all_times[target_dim]
        n_target     = len(target_times)
        if n_target == 0:
            return np.inf

        # R[j] running sum; ptrs[j] = next unprocessed index in all_times[j]
        R    = np.zeros(D)
        ptrs = np.zeros(D, dtype=np.int64)
        t_prev  = 0.0
        ll_sum  = 0.0

        for i in range(n_target):
            t  = target_times[i]
            dt = t - t_prev

            # Decay all running sums over the interval
            R *= np.exp(-beta * dt)

            # Add contributions from each source dim's events in [t_prev, t)
            for j in range(D):
                src = all_times[j]
                hi  = int(np.searchsorted(src, t, side="left"))
                lo  = int(ptrs[j])
                if hi > lo:
                    R[j] += float(np.sum(np.exp(-beta[j] * (t - src[lo:hi]))))
                ptrs[j] = hi

            # Accumulate log-intensity at this target event
            intensity = mu + float(np.dot(alpha, R))
            if intensity <= 0.0:
                return np.inf
            ll_sum += np.log(intensity)

            t_prev = t

        # Compensator Λ = μ·T + Σ_j (α_j/β_j) · Σ_l (1 - exp(-β_j·(T-t_j^l)))
        compensator = mu * T
        for j in range(D):
            t_j = all_times[j]
            if len(t_j) == 0:
                continue
            compensator += (alpha[j] / beta[j]) * float(
                np.sum(1.0 - np.exp(-beta[j] * (T - t_j)))
            )

        return -(ll_sum - compensator)

    def _neg_ll_and_grad(
        self,
        x:          np.ndarray,
        target_dim: int,
        all_times:  list[np.ndarray],
        T:          float,
    ) -> tuple[float, np.ndarray]:
        """
        Negative log-likelihood and its analytical gradient, computed jointly
        to avoid redundant passes over the data.

        Auxiliary variable Q_j(t) = Σ_{s<t, dim=j} (t-s)·exp(-β_j·(t-s))
        satisfies: ∂R_j/∂β_j = -Q_j   →   ∂λ/∂β_j = -α_j·Q_j

        Q_j recursion (before decaying R):
            Q_j ← exp(-β_j·dt)·(Q_j + dt·R_j)
            Q_j += Σ_{new s} (t-s)·exp(-β_j·(t-s))

        Gradient of ℓ = Σ_k log λ(t_k) - Λ:
            ∂ℓ/∂μ   = Σ_k 1/λ_k  - T
            ∂ℓ/∂α_j = Σ_k R_j/λ_k - (1/β_j)·Σ_l(1-exp(-β_j·(T-s_l)))
            ∂ℓ/∂β_j = -α_j·Σ_k Q_j/λ_k
                       + (α_j/β_j²)·Σ_l(1-exp(-β_j·(T-s_l)))
                       - (α_j/β_j)·Σ_l(T-s_l)·exp(-β_j·(T-s_l))
        """
        D      = self.n_dims
        mu     = x[0]
        alpha  = x[1 : 1 + D]
        beta   = x[1 + D : 1 + 2 * D]

        target_times = all_times[target_dim]
        n_target     = len(target_times)
        if n_target == 0:
            return np.inf, np.zeros(1 + 2 * D)

        R    = np.zeros(D)
        Q    = np.zeros(D)   # Q_j = Σ_{s<t, dim=j} (t-s)·exp(-β_j·(t-s))
        ptrs = np.zeros(D, dtype=np.int64)
        t_prev = 0.0

        ll_sum  = 0.0
        g_mu    = 0.0
        g_alpha = np.zeros(D)
        g_beta  = np.zeros(D)

        for i in range(n_target):
            t  = target_times[i]
            dt = t - t_prev
            exp_b_dt = np.exp(-beta * dt)

            # Update Q before R (Q depends on the pre-decay R)
            Q  = exp_b_dt * (Q + dt * R)
            R *= exp_b_dt

            # Add source events in (t_prev, t) to both R and Q
            for j in range(D):
                src = all_times[j]
                hi  = int(np.searchsorted(src, t, side="left"))
                lo  = int(ptrs[j])
                if hi > lo:
                    gaps     = t - src[lo:hi]
                    exp_gaps = np.exp(-beta[j] * gaps)
                    R[j]    += float(np.sum(exp_gaps))
                    Q[j]    += float(np.sum(gaps * exp_gaps))
                ptrs[j] = hi

            # Intensity and per-event gradient contribution
            intensity = mu + float(np.dot(alpha, R))
            if intensity <= 0.0:
                return np.inf, np.zeros(1 + 2 * D)
            inv_lam  = 1.0 / intensity
            ll_sum  += np.log(intensity)
            g_mu    += inv_lam
            g_alpha += R * inv_lam
            g_beta  -= alpha * Q * inv_lam   # ∂log_λ/∂β_j = -α_j·Q_j / λ

            t_prev = t

        # Compensator Λ and its gradient
        compensator = mu * T
        for j in range(D):
            t_j = all_times[j]
            if len(t_j) == 0:
                continue
            gaps_end = T - t_j
            exp_end  = np.exp(-beta[j] * gaps_end)
            sum1     = float(np.sum(1.0 - exp_end))           # Σ(1-exp(-β_j·(T-s)))
            sum2     = float(np.sum(gaps_end * exp_end))      # Σ(T-s)·exp(-β_j·(T-s))

            compensator   += (alpha[j] / beta[j]) * sum1
            g_alpha[j]    -= sum1 / beta[j]                   # ∂(-Λ)/∂α_j
            g_beta[j]     += (alpha[j] / beta[j] ** 2) * sum1 - (alpha[j] / beta[j]) * sum2

        g_mu -= T   # ∂(-Λ)/∂μ

        neg_ll   = -(ll_sum - compensator)
        neg_grad = np.concatenate([[-g_mu], -g_alpha, -g_beta])
        return neg_ll, neg_grad


# ── helpers ───────────────────────────────────────────────────────────────────


def _merge_events(
    all_times: list[np.ndarray],
    n_dims:    int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-dimension time arrays into a single sorted stream.
    Returns (times, dims) both of shape (N,).
    Used by HawkesGoodnessOfFit for the compensator integral.
    """
    parts_t: list[np.ndarray] = []
    parts_d: list[np.ndarray] = []
    for j in range(n_dims):
        t = all_times[j]
        if len(t) > 0:
            parts_t.append(t)
            parts_d.append(np.full(len(t), j, dtype=np.int32))

    if not parts_t:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    times = np.concatenate(parts_t)
    dims  = np.concatenate(parts_d)
    idx   = np.argsort(times, kind="stable")
    return times[idx], dims[idx]
