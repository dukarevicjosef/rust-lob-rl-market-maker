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

        best_ll   = -np.inf
        best_res  = None

        rng = np.random.default_rng(seed=target_dim)

        for restart in range(self.n_restarts):
            if restart == 0:
                # Deterministic initialisation from empirical rates
                rate_target = len(times) / T if T > 0 else 1.0
                mu0    = rate_target * 0.5
                alpha0 = np.full(D, rate_target * 0.1)
                beta0  = np.full(D, self.beta_init)
            else:
                mu0    = rng.uniform(1e-4, 2.0)
                alpha0 = rng.uniform(0.0, 1.0, size=D)
                beta0  = rng.uniform(1.0, 50.0, size=D)

            x0 = np.concatenate([[mu0], alpha0, beta0])

            bounds = (
                [(1e-8, None)]          # mu
                + [(0.0, None)] * D     # alpha (non-negative)
                + [(1e-4, None)] * D    # beta  (strictly positive)
            )

            res = minimize(
                fun=self._neg_log_likelihood,
                x0=x0,
                args=(target_dim, all_times, T),
                method="L-BFGS-B",
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

        Uses the recursive formula (Ozaki, 1979):

            R_j(k) = exp(-β_j · (t_k - t_{k-1})) · (R_j(k-1) + N_j(t_{k-1}^+))

        where N_j(t_{k-1}^+) = 1 if the previous event belonged to dim j.

        Log-likelihood:
            ℓ = Σ_k log(μ + Σ_j α_j · R_j(k)) - Λ

        Compensator:
            Λ = μ·T + Σ_j (α_j/β_j) · Σ_l (1 - exp(-β_j · (T - t_j^l)))
        """
        D  = self.n_dims
        mu     = x[0]
        alpha  = x[1 : 1 + D]
        beta   = x[1 + D : 1 + 2 * D]

        # Merge all events into a single sorted stream
        times_target = all_times[target_dim]

        # Build merged event list: (time, source_dim)
        # We need all events to advance the R_j recursion
        merged_times, merged_dims = _merge_events(all_times, D)

        n_all = len(merged_times)
        if n_all == 0:
            return np.inf

        # R[j] = current recursive auxiliary variable for each source dim j
        R = np.zeros(D)

        ll_sum = 0.0
        t_prev = 0.0

        target_set = set(np.searchsorted(merged_times, times_target, side="left"))

        # Iterate over merged stream; only accumulate LL at target events
        target_event_idx = 0      # which target event we're at
        n_target = len(times_target)

        # Build a sorted index of target event positions in merged stream
        target_positions: set[int] = set()
        ti = 0
        for k in range(n_all):
            if ti < n_target and merged_times[k] == times_target[ti]:
                target_positions.add(k)
                ti += 1

        for k in range(n_all):
            t_k   = merged_times[k]
            dim_k = merged_dims[k]
            dt    = t_k - t_prev

            # Decay R
            decay = np.exp(-beta * dt)
            R *= decay

            # At target dim events: accumulate log-intensity
            if k in target_positions:
                intensity = mu + np.dot(alpha, R)
                if intensity <= 0.0:
                    return np.inf
                ll_sum += np.log(intensity)

            # Increment R for source dim of this event
            R[dim_k] += 1.0
            t_prev = t_k

        # Compensator Λ = μ·T + Σ_j (α_j/β_j) · Σ_l (1 - exp(-β_j·(T-t_j^l)))
        compensator = mu * T
        for j in range(D):
            t_j = all_times[j]
            if len(t_j) == 0:
                continue
            dt_to_end = T - t_j
            compensator += (alpha[j] / beta[j]) * np.sum(1.0 - np.exp(-beta[j] * dt_to_end))

        ll = ll_sum - compensator
        return -ll


# ── helpers ───────────────────────────────────────────────────────────────────


def _merge_events(
    all_times: list[np.ndarray],
    n_dims:    int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-dimension time arrays into a single sorted stream.
    Returns (times, dims) both of shape (N,).
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
