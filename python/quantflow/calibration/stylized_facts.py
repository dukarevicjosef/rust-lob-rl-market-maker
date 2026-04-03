"""
Stylized Facts computation.

Identical static API for empirical (parquet/npz) and simulated (session dict) data.
"""
from __future__ import annotations

import numpy as np
import scipy.stats


class StylizedFacts:
    """
    Container for computed stylized facts.

    Attributes
    ----------
    iet       : inter-event time distribution dict
    returns   : log-return distribution dict
    acf       : ACF of absolute returns dict
    spread    : bid-ask spread distribution dict
    intraday  : intraday event-rate profile dict
    signature : realized-vol vs. sampling-frequency dict
    qq        : Q-Q data (IET × rate vs. Exp(1)) dict
    label     : short label for plots ("Empirical" / "Simulated")
    """

    def __init__(self, label: str = "") -> None:
        self.label    = label
        self.iet:      dict = {}
        self.returns:  dict = {}
        self.acf:      dict = {}
        self.spread:   dict = {}
        self.intraday: dict = {}
        self.signature: dict = {}
        self.qq:       dict = {}

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_session(cls, session: dict, label: str = "Simulated") -> StylizedFacts:
        """Compute facts from a simulated session dict (or aggregated sessions)."""
        sf = cls(label=label)
        mid    = session["mid_prices"]
        spreads = session["spreads"]
        t_max  = session["t_max"]
        all_t  = np.array([t for t, _ in session["events"]], dtype=np.float64)

        sf.iet       = cls.inter_event_distribution(all_t)
        sf.returns   = cls.return_distribution(mid)
        sf.acf       = cls.acf_absolute_returns(mid)
        sf.spread    = cls.spread_distribution(spreads)
        sf.intraday  = cls.intraday_volume(all_t, t_max)
        sf.signature = cls.signature_plot(mid)
        sf.qq        = cls.qq_from_iet(all_t)
        return sf

    @classmethod
    def from_parquet(
        cls,
        path: str,
        t_max: float = 600.0,
        label: str = "Empirical",
    ) -> StylizedFacts:
        """Compute facts from a processed events Parquet file."""
        import polars as pl

        df = pl.read_parquet(path).sort("timestamp")
        t0 = float(df["timestamp"].min())
        df = df.with_columns((pl.col("timestamp") - t0).alias("timestamp"))
        df = df.filter(pl.col("timestamp") <= t_max)

        all_t  = df["timestamp"].to_numpy().astype(np.float64)
        mid, spreads = cls._reconstruct_price_series(
            times=all_t,
            event_types=df["event_type"].to_numpy().astype(np.int32),
            prices=df["price"].to_numpy().astype(np.float64),
        )

        sf = cls(label=label)
        sf.iet       = cls.inter_event_distribution(all_t)
        sf.returns   = cls.return_distribution(mid)
        sf.acf       = cls.acf_absolute_returns(mid)
        sf.spread    = cls.spread_distribution(spreads)
        sf.intraday  = cls.intraday_volume(all_t, t_max)
        sf.signature = cls.signature_plot(mid)
        sf.qq        = cls.qq_from_iet(all_t)
        return sf

    @classmethod
    def from_npz(
        cls,
        path: str,
        t_max: float = 600.0,
        label: str = "Empirical",
    ) -> StylizedFacts:
        """
        Compute facts from a classified_events.npz file.
        Price-based facts (returns, spread, signature) are skipped as npz
        contains only event times, not prices.
        """
        d     = np.load(path, allow_pickle=True)
        all_t = d["all_times"].astype(np.float64)
        t0    = all_t[0]
        all_t = all_t - t0
        all_t = all_t[all_t <= t_max]

        sf = cls(label=label)
        sf.iet      = cls.inter_event_distribution(all_t)
        sf.intraday = cls.intraday_volume(all_t, t_max)
        sf.qq       = cls.qq_from_iet(all_t)
        # returns, acf, spread, signature require price data → left empty
        return sf

    # ── Static computation methods ────────────────────────────────────────────

    @staticmethod
    def inter_event_distribution(
        event_times: np.ndarray,
        max_lag:     float = 5.0,
        n_bins:      int   = 100,
    ) -> dict:
        """
        Distribution of inter-event times (all dims combined).
        Returns bin_centers, density, exponential fit, and summary stats.
        """
        iet = np.diff(np.sort(event_times))
        iet = iet[(iet > 0) & (iet < max_lag)]
        if len(iet) < 2:
            return {}
        hist, edges = np.histogram(iet, bins=n_bins, density=True)
        centers     = (edges[:-1] + edges[1:]) / 2
        rate        = 1.0 / float(np.mean(iet))
        exp_fit     = rate * np.exp(-rate * centers)
        return {
            "bin_centers": centers.tolist(),
            "density":     hist.tolist(),
            "exp_fit":     exp_fit.tolist(),
            "mean":        float(np.mean(iet)),
            "std":         float(np.std(iet)),
            "kurtosis":    float(scipy.stats.kurtosis(iet, fisher=True)),
        }

    @staticmethod
    def return_distribution(
        mid_prices: np.ndarray,
        dt:         float = 1.0,
        n_bins:     int   = 100,
    ) -> dict:
        """
        Log-return distribution at sampling interval dt.
        Normalised returns are compared to a Gaussian baseline.
        """
        if mid_prices is None or len(mid_prices) < 4:
            return {}
        times, prices = mid_prices[:, 0], mid_prices[:, 1]
        if times[-1] - times[0] < dt * 3:
            return {}
        grid      = np.arange(times[0], times[-1], dt)
        resampled = np.interp(grid, times, prices)
        log_ret   = np.diff(np.log(resampled))
        log_ret   = log_ret[np.isfinite(log_ret)]
        if len(log_ret) < 10:
            return {}
        std = float(np.std(log_ret))
        if std == 0.0:
            return {}
        norm_ret    = (log_ret - float(np.mean(log_ret))) / std
        hist, edges = np.histogram(norm_ret, bins=n_bins, density=True)
        centers     = (edges[:-1] + edges[1:]) / 2
        return {
            "bin_centers":  centers.tolist(),
            "log_density":  np.log10(np.maximum(hist, 1e-10)).tolist(),
            "log_gaussian": np.log10(np.maximum(scipy.stats.norm.pdf(centers), 1e-10)).tolist(),
            "kurtosis":     float(scipy.stats.kurtosis(norm_ret, fisher=True)),
            "skewness":     float(scipy.stats.skew(norm_ret)),
        }

    @staticmethod
    def acf_absolute_returns(
        mid_prices: np.ndarray,
        dt:         float = 1.0,
        max_lag:    int   = 200,
    ) -> dict:
        """
        Autocorrelation of absolute log-returns.
        Slow decay indicates volatility clustering (stylized fact of real markets).
        """
        if mid_prices is None or len(mid_prices) < 4:
            return {}
        times, prices = mid_prices[:, 0], mid_prices[:, 1]
        if times[-1] - times[0] < dt * 10:
            return {}
        grid      = np.arange(times[0], times[-1], dt)
        resampled = np.interp(grid, times, prices)
        log_ret   = np.diff(np.log(resampled))
        abs_ret   = np.abs(log_ret[np.isfinite(log_ret)])
        n = len(abs_ret)
        if n < 4:
            return {}
        mean = float(np.mean(abs_ret))
        var  = float(np.var(abs_ret))
        if var == 0.0:
            return {}
        n_lag = min(max_lag, n // 4)
        acf_vals = [
            float(np.mean((abs_ret[lag:] - mean) * (abs_ret[:-lag] - mean)) / var)
            for lag in range(1, n_lag + 1)
        ]
        return {"lags": list(range(1, len(acf_vals) + 1)), "acf": acf_vals}

    @staticmethod
    def spread_distribution(
        spreads: np.ndarray,
        n_bins:  int = 50,
    ) -> dict:
        """Distribution of bid-ask spreads."""
        if spreads is None or len(spreads) < 2:
            return {}
        s = spreads[:, 1]
        s = s[s > 0]
        if len(s) < 2:
            return {}
        hist, edges = np.histogram(s, bins=n_bins, density=True)
        centers     = (edges[:-1] + edges[1:]) / 2
        return {
            "bin_centers": centers.tolist(),
            "density":     hist.tolist(),
            "mean":        float(np.mean(s)),
            "median":      float(np.median(s)),
        }

    @staticmethod
    def intraday_volume(
        event_times: np.ndarray,
        t_max:       float,
        bin_minutes: int = 5,
    ) -> dict:
        """Event-rate profile over time (events per second in each bin)."""
        bin_size = bin_minutes * 60.0
        n_bins   = max(1, int(np.ceil(t_max / bin_size)))
        edges    = np.arange(0, (n_bins + 1) * bin_size, bin_size)
        counts   = np.histogram(event_times, bins=edges)[0]
        centers  = (edges[:-1] + edges[1:]) / 2 / 60.0  # convert to minutes
        return {
            "bin_centers_min": centers.tolist(),
            "counts":          counts.tolist(),
            "rates":           (counts / bin_size).tolist(),
        }

    @staticmethod
    def signature_plot(
        mid_prices: np.ndarray,
        dt_values:  list[float] | None = None,
    ) -> dict:
        """
        Realized volatility as a function of sampling frequency.
        Microstructure noise inflates vol at high frequencies (small Δt).
        """
        if mid_prices is None or len(mid_prices) < 3:
            return {}
        if dt_values is None:
            dt_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
        times, prices = mid_prices[:, 0], mid_prices[:, 1]
        rvols = []
        for dt in dt_values:
            grid = np.arange(times[0], times[-1], dt)
            if len(grid) < 3:
                rvols.append(float("nan"))
                continue
            resampled = np.interp(grid, times, prices)
            log_ret   = np.diff(np.log(resampled))
            log_ret   = log_ret[np.isfinite(log_ret)]
            if len(log_ret) == 0:
                rvols.append(float("nan"))
                continue
            # Annualised realised vol (per-second scaling)
            rv = float(np.sqrt(np.sum(log_ret ** 2) / (len(log_ret) * dt)))
            rvols.append(rv)
        return {"dt": dt_values, "realized_vol": rvols}

    @staticmethod
    def qq_from_iet(
        event_times: np.ndarray,
        rate:        float | None = None,
    ) -> dict:
        """
        Q-Q data: empirical IETs (scaled by rate) vs. Exp(1) theoretical quantiles.

        For a Poisson process, this is a straight diagonal.
        Hawkes clustering → concave deviation (more small IETs than Exp(1)).
        """
        iet = np.diff(np.sort(event_times))
        iet = iet[iet > 0]
        if len(iet) < 2:
            return {}
        rate_   = rate if rate is not None else 1.0 / float(np.mean(iet))
        tau     = np.sort(iet * rate_)        # scaled to Exp(1) under Poisson null
        n       = len(tau)
        probs   = (np.arange(1, n + 1) - 0.5) / n
        theor   = scipy.stats.expon.ppf(probs)
        # Subsample for plotting efficiency
        if n > 3000:
            idx   = np.round(np.linspace(0, n - 1, 3000)).astype(int)
            tau   = tau[idx]
            theor = theor[idx]
        return {
            "theoretical": theor.tolist(),
            "empirical":   tau.tolist(),
        }

    # ── Price reconstruction ──────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_price_series(
        times:       np.ndarray,
        event_types: np.ndarray,
        prices:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mid price and spread from LOB event stream.

        Dim 2 (LimitBuyBest)  → new best bid
        Dim 3 (LimitSellBest) → new best ask
        Dim 0 (MarketBuy)     → traded at ask (initialises best_ask if unknown)
        Dim 1 (MarketSell)    → traded at bid (initialises best_bid if unknown)
        """
        best_bid = np.nan
        best_ask = np.nan
        mid_recs:    list[tuple[float, float]] = []
        spread_recs: list[tuple[float, float]] = []

        for t, e, p in zip(times, event_types, prices):
            if e == 2:    # Limit Buy Best → new best bid
                best_bid = p
            elif e == 3:  # Limit Sell Best → new best ask
                best_ask = p
            elif e == 0 and np.isnan(best_ask):
                best_ask = p
            elif e == 1 and np.isnan(best_bid):
                best_bid = p

            if (not np.isnan(best_bid) and not np.isnan(best_ask)
                    and best_ask > best_bid):
                mid_recs.append((t, (best_bid + best_ask) / 2.0))
                spread_recs.append((t, best_ask - best_bid))

        if not mid_recs:
            # Fallback: use all event prices as a rough mid proxy
            valid = prices[prices > 0]
            t_valid = times[prices > 0]
            mid_recs    = list(zip(t_valid.tolist(), valid.tolist()))
            spread_recs = [(t, 1.0) for t, _ in mid_recs]

        return (
            np.array(mid_recs,    dtype=np.float64),
            np.array(spread_recs, dtype=np.float64),
        )
