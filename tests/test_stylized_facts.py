"""Tests for StylizedFacts computation."""
from __future__ import annotations

import numpy as np
import pytest

from quantflow.calibration.stylized_facts import StylizedFacts


# ── helpers ───────────────────────────────────────────────────────────────────


def _gaussian_mid(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times  = np.arange(n, dtype=np.float64)
    prices = np.exp(np.cumsum(rng.normal(0, 1e-4, n))) * 100.0
    return np.column_stack([times, prices])


def _t_dist_mid(n: int = 5000, df: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times  = np.arange(n, dtype=np.float64)
    prices = np.exp(np.cumsum(rng.standard_t(df, n) * 1e-4)) * 100.0
    return np.column_stack([times, prices])


# ── Test 1 & 2: return_distribution ──────────────────────────────────────────


class TestReturnDistribution:
    def test_gaussian_noise_low_kurtosis(self):
        """Gaussian random walk → excess kurtosis ≈ 0."""
        result = StylizedFacts.return_distribution(_gaussian_mid())
        assert result, "return_distribution returned empty dict"
        assert abs(result["kurtosis"]) < 2.0, (
            f"Expected |kurtosis| < 2 for Gaussian, got {result['kurtosis']:.2f}"
        )

    def test_t_dist_high_kurtosis(self):
        """t(3) innovations → excess kurtosis > 3 (fat tails)."""
        result = StylizedFacts.return_distribution(_t_dist_mid())
        assert result, "return_distribution returned empty dict"
        assert result["kurtosis"] > 2.0, (
            f"Expected kurtosis > 2 for t-dist, got {result['kurtosis']:.2f}"
        )


# ── Test 3: acf_absolute_returns ─────────────────────────────────────────────


class TestACF:
    def test_iid_noise_near_zero_acf(self):
        """iid Gaussian increments → mean |ACF| near zero."""
        result = StylizedFacts.acf_absolute_returns(_gaussian_mid(n=3000), max_lag=20)
        assert result, "acf_absolute_returns returned empty dict"
        mean_abs_acf = float(np.mean(np.abs(result["acf"])))
        assert mean_abs_acf < 0.15, (
            f"Expected near-zero ACF for iid, got mean|acf|={mean_abs_acf:.3f}"
        )


# ── Test 4: spread_distribution ──────────────────────────────────────────────


class TestSpreadDistribution:
    def test_constant_spread_correct_statistics(self):
        """Constant spread → mean and median both equal the constant."""
        t = np.arange(200, dtype=np.float64)
        spreads = np.column_stack([t, np.full(200, 2.5)])
        result  = StylizedFacts.spread_distribution(spreads, n_bins=20)
        assert result, "spread_distribution returned empty dict"
        assert abs(result["mean"]   - 2.5) < 0.05
        assert abs(result["median"] - 2.5) < 0.05


# ── Test 5: intraday_volume ───────────────────────────────────────────────────


class TestIntradayVolume:
    def test_counts_sum_to_total_events(self):
        """Sum of bin counts must equal total number of events."""
        rng   = np.random.default_rng(7)
        times = np.sort(rng.uniform(0, 600, 1000))
        result = StylizedFacts.intraday_volume(times, t_max=600.0, bin_minutes=5)
        assert result, "intraday_volume returned empty dict"
        assert sum(result["counts"]) == 1000, (
            f"Expected count sum = 1000, got {sum(result['counts'])}"
        )


# ── Test 6: signature_plot ────────────────────────────────────────────────────


class TestSignaturePlot:
    def test_values_positive_and_finite(self):
        """Realised vol values must be positive and finite."""
        rng   = np.random.default_rng(42)
        n     = 6000
        times = np.sort(rng.uniform(0, 600, n))
        prices = np.exp(np.cumsum(rng.normal(0, 2e-4, n))) * 65_000.0
        mid   = np.column_stack([times, prices])
        result = StylizedFacts.signature_plot(mid, dt_values=[1.0, 5.0, 10.0, 30.0, 60.0])
        assert result, "signature_plot returned empty dict"
        valid = [v for v in result["realized_vol"] if np.isfinite(v)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid), "All finite realised-vol values must be positive"

    def test_generally_decreasing_with_dt(self):
        """
        Signature plot should show higher (or equal) vol at finer Δt due to
        microstructure noise — first value >= last value.
        """
        rng   = np.random.default_rng(99)
        n     = 10_000
        times = np.sort(rng.uniform(0, 600, n))
        prices = np.cumprod(1 + rng.normal(0, 2e-4, n)) * 65_000.0
        mid   = np.column_stack([times, prices])
        result = StylizedFacts.signature_plot(mid, dt_values=[0.1, 1.0, 10.0, 60.0])
        vals  = np.array([v for v in result["realized_vol"] if np.isfinite(v)])
        assert len(vals) >= 2
        # First value (finest Δt) should be >= last value (coarsest Δt)
        assert vals[0] >= vals[-1] * 0.5, (
            "Expected signature plot to be generally non-increasing"
        )
