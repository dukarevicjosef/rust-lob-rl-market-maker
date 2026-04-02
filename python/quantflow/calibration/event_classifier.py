from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


class EventClassifier:
    """
    Loads normalised MarketEvents from Parquet and prepares them
    for Hawkes MLE calibration.

    Input:  data/btcusdt/processed/{date}_events.parquet
    Output: HawkesEventData — per-dimension event time arrays
    """

    N_DIMS = 12

    DIM_NAMES = [
        "Market Buy",
        "Market Sell",
        "Limit Buy Best",
        "Limit Sell Best",
        "Limit Buy Deep",
        "Limit Sell Deep",
        "Cancel Buy Best",
        "Cancel Sell Best",
        "Cancel Buy Deep",
        "Cancel Sell Deep",
        "Modify Buy",
        "Modify Sell",
    ]

    def __init__(self, min_events_per_dim: int = 100) -> None:
        self.min_events_per_dim = min_events_per_dim

    # ── single-file loading ───────────────────────────────────────────────────

    def load_and_classify(self, parquet_path: str | Path) -> HawkesEventData:
        """Load events from Parquet, classify per dimension, return HawkesEventData."""
        parquet_path = Path(parquet_path)
        df = pl.read_parquet(parquet_path)

        for col in ("timestamp", "event_type", "price", "quantity"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        df = df.sort("timestamp")

        if len(df) == 0:
            return self._empty_data(str(parquet_path))

        t_start = float(df["timestamp"].min())
        t_end   = float(df["timestamp"].max())
        t_span  = t_end - t_start

        dim_events: list[dict] = []
        dim_stats:  list[dict] = []

        for dim in range(self.N_DIMS):
            sub = df.filter(pl.col("event_type") == dim)

            times      = sub["timestamp"].to_numpy().astype(np.float64)
            prices     = sub["price"].to_numpy().astype(np.float64)
            quantities = sub["quantity"].to_numpy().astype(np.float64)

            inter_times = np.diff(times) if len(times) > 1 else np.array([], dtype=np.float64)

            stats = {
                "dim":              dim,
                "name":             self.DIM_NAMES[dim],
                "count":            len(times),
                "rate_per_sec":     len(times) / t_span if t_span > 0 else 0.0,
                "pct_of_total":     len(times) / len(df) * 100 if len(df) > 0 else 0.0,
                "mean_inter_time":  float(np.mean(inter_times))   if len(inter_times) > 0 else float("inf"),
                "median_inter_time":float(np.median(inter_times)) if len(inter_times) > 0 else float("inf"),
                "std_inter_time":   float(np.std(inter_times))    if len(inter_times) > 0 else 0.0,
                "mean_quantity":    float(np.mean(quantities))    if len(quantities)  > 0 else 0.0,
                "active":           len(times) >= self.min_events_per_dim,
            }

            dim_events.append({
                "times":       times,
                "prices":      prices,
                "quantities":  quantities,
                "inter_times": inter_times,
            })
            dim_stats.append(stats)

        return HawkesEventData(
            dim_events=dim_events,
            dim_stats=dim_stats,
            t_start=t_start,
            t_end=t_end,
            t_span=t_span,
            total_events=len(df),
            source_path=str(parquet_path),
        )

    # ── multi-day loading ─────────────────────────────────────────────────────

    def load_multiple_days(
        self,
        parquet_dir: str | Path,
        dates: list[str] | None = None,
    ) -> HawkesEventData:
        """
        Load and concatenate multiple event files.
        Timestamps are shifted so that day N+1 follows day N sequentially
        (1-second gap between sessions).
        """
        parquet_dir = Path(parquet_dir)

        if dates is None:
            files = sorted(parquet_dir.glob("*_events.parquet"))
        else:
            files = [parquet_dir / f"{d}_events.parquet" for d in dates]

        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"No event files found in {parquet_dir}")

        dfs: list[pl.DataFrame] = []
        time_offset = 0.0

        for f in files:
            df = pl.read_parquet(f).sort("timestamp")
            if len(df) == 0:
                continue
            t_min = float(df["timestamp"].min())
            df = df.with_columns(
                (pl.col("timestamp") - t_min + time_offset).alias("timestamp")
            )
            time_offset = float(df["timestamp"].max()) + 1.0
            dfs.append(df)

        if not dfs:
            raise ValueError("All event files are empty.")

        combined = pl.concat(dfs)
        tmp_path = parquet_dir / "_combined_events_tmp.parquet"
        combined.write_parquet(tmp_path)
        try:
            result = self.load_and_classify(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return result

    # ── intraday profile ──────────────────────────────────────────────────────

    def compute_intraday_profile(
        self,
        data: HawkesEventData,
        bin_minutes: int = 30,
    ) -> dict:
        """
        Event counts and rates per time bin.
        Produces the U-shaped intraday activity profile typical of equity/crypto markets.

        Returns keys: bin_edges, bin_minutes, counts, rates, per_dim.
        """
        bin_size = bin_minutes * 60.0
        n_bins   = max(1, int(np.ceil(data.t_span / bin_size)))
        bin_edges = np.arange(0, (n_bins + 1) * bin_size, bin_size)

        all_times, _ = data.all_times_flat()
        total_counts = np.histogram(all_times, bins=bin_edges)[0]
        total_rates  = total_counts / bin_size

        per_dim: dict[int, list[int]] = {}
        for dim in range(data.n_dims):
            dt = data.times_for_dim(dim)
            per_dim[dim] = np.histogram(dt, bins=bin_edges)[0].tolist()

        return {
            "bin_edges":   bin_edges.tolist(),
            "bin_minutes": bin_minutes,
            "counts":      total_counts.tolist(),
            "rates":       total_rates.tolist(),
            "per_dim":     per_dim,
        }

    # ── outlier filtering ─────────────────────────────────────────────────────

    def filter_outliers(
        self,
        data: HawkesEventData,
        max_inter_time: float = 60.0,
        min_price: float = 0.01,
    ) -> HawkesEventData:
        """
        Remove events that would bias calibration:
        - Events preceded by a gap > max_inter_time (session breaks, maintenance windows)
        - Events with price < min_price or quantity <= 0

        Returns a filtered copy.  The first event of each dimension is always kept
        (it has no predecessor).
        """
        dim_events: list[dict] = []
        dim_stats:  list[dict] = []
        filtered_total = 0

        for dim in range(self.N_DIMS):
            src = data.dim_events[dim]
            times      = src["times"]
            prices     = src["prices"]
            quantities = src["quantities"]

            if len(times) == 0:
                dim_events.append({"times": times, "prices": prices,
                                   "quantities": quantities,
                                   "inter_times": np.array([], dtype=np.float64)})
                dim_stats.append({**data.dim_stats[dim], "count": 0,
                                  "rate_per_sec": 0.0, "active": False})
                continue

            # Price and quantity validity mask
            valid_mask = (prices >= min_price) & (quantities > 0)

            # Gap mask: first event always kept; subsequent events kept if gap ≤ max_inter_time
            inter_times_full = np.diff(times)
            gap_mask = np.concatenate([[True], inter_times_full <= max_inter_time])

            keep = valid_mask & gap_mask

            times_f      = times[keep]
            prices_f     = prices[keep]
            quantities_f = quantities[keep]
            inter_times_f = np.diff(times_f) if len(times_f) > 1 else np.array([], dtype=np.float64)

            filtered_total += len(times_f)

            old_stats = data.dim_stats[dim]
            t_span = data.t_span if data.t_span > 0 else 1.0
            stats = {
                **old_stats,
                "count":            len(times_f),
                "rate_per_sec":     len(times_f) / t_span,
                "pct_of_total":     0.0,        # recalculated below
                "mean_inter_time":  float(np.mean(inter_times_f))    if len(inter_times_f) > 0 else float("inf"),
                "median_inter_time":float(np.median(inter_times_f))  if len(inter_times_f) > 0 else float("inf"),
                "std_inter_time":   float(np.std(inter_times_f))     if len(inter_times_f) > 0 else 0.0,
                "mean_quantity":    float(np.mean(quantities_f))     if len(quantities_f)  > 0 else 0.0,
                "active":           len(times_f) >= self.min_events_per_dim,
            }

            dim_events.append({"times": times_f, "prices": prices_f,
                                "quantities": quantities_f,
                                "inter_times": inter_times_f})
            dim_stats.append(stats)

        # Fix pct_of_total now that we know filtered_total
        for s in dim_stats:
            s["pct_of_total"] = s["count"] / filtered_total * 100 if filtered_total > 0 else 0.0

        return HawkesEventData(
            dim_events=dim_events,
            dim_stats=dim_stats,
            t_start=data.t_start,
            t_end=data.t_end,
            t_span=data.t_span,
            total_events=filtered_total,
            source_path=data.source_path,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _empty_data(self, source_path: str) -> HawkesEventData:
        dim_events = [
            {"times": np.array([], dtype=np.float64),
             "prices": np.array([], dtype=np.float64),
             "quantities": np.array([], dtype=np.float64),
             "inter_times": np.array([], dtype=np.float64)}
            for _ in range(self.N_DIMS)
        ]
        dim_stats = [
            {"dim": d, "name": self.DIM_NAMES[d], "count": 0,
             "rate_per_sec": 0.0, "pct_of_total": 0.0,
             "mean_inter_time": float("inf"), "median_inter_time": float("inf"),
             "std_inter_time": 0.0, "mean_quantity": 0.0, "active": False}
            for d in range(self.N_DIMS)
        ]
        return HawkesEventData(
            dim_events=dim_events, dim_stats=dim_stats,
            t_start=0.0, t_end=0.0, t_span=0.0,
            total_events=0, source_path=source_path,
        )


class HawkesEventData:
    """Container for classified events, ready for MLE calibration."""

    def __init__(
        self,
        dim_events: list[dict],
        dim_stats: list[dict],
        t_start: float,
        t_end: float,
        t_span: float,
        total_events: int,
        source_path: str,
    ) -> None:
        self.dim_events   = dim_events
        self.dim_stats    = dim_stats
        self.t_start      = t_start
        self.t_end        = t_end
        self.t_span       = t_span
        self.total_events = total_events
        self.source_path  = source_path

    @property
    def n_dims(self) -> int:
        return len(self.dim_events)

    @property
    def active_dims(self) -> list[int]:
        """Dimensions with enough events for calibration."""
        return [s["dim"] for s in self.dim_stats if s["active"]]

    def times_for_dim(self, dim: int) -> np.ndarray:
        return self.dim_events[dim]["times"]

    def all_times_flat(self) -> tuple[np.ndarray, np.ndarray]:
        """All events as (times, dims) arrays sorted by time."""
        all_times: list[np.ndarray] = []
        all_dims:  list[np.ndarray] = []

        for dim in range(self.n_dims):
            t = self.dim_events[dim]["times"]
            all_times.append(t)
            all_dims.append(np.full(len(t), dim, dtype=np.int32))

        times = np.concatenate(all_times)
        dims  = np.concatenate(all_dims)
        sort_idx = np.argsort(times, kind="stable")
        return times[sort_idx], dims[sort_idx]

    def inter_times_for_dim(self, dim: int) -> np.ndarray:
        return self.dim_events[dim]["inter_times"]

    def print_summary(self) -> None:
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║  HAWKES EVENT CLASSIFICATION                                     ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"  Source:        {self.source_path}")
        print(f"  Time span:     {self.t_span:.1f}s ({self.t_span / 3600:.2f}h)")
        print(f"  Total events:  {self.total_events:,}")
        rate = self.total_events / self.t_span if self.t_span > 0 else 0.0
        print(f"  Overall rate:  {rate:.1f} events/s")
        print()
        print(f"  {'Dim':>3} │ {'Type':<20} │ {'Count':>8} │ {'Rate/s':>7} │ {'%':>6} │ {'Mean Δt':>8} │ Active")
        print(f"  ────┼──────────────────────┼──────────┼─────────┼────────┼──────────┼──────")

        for s in self.dim_stats:
            marker = "  ✓" if s["active"] else "  ✗"
            if s["count"] > 0:
                print(
                    f"  {s['dim']:>3} │ {s['name']:<20} │ {s['count']:>8,} │ "
                    f"{s['rate_per_sec']:>7.2f} │ {s['pct_of_total']:>5.1f}% │ "
                    f"{s['mean_inter_time']:>7.3f}s │ {marker}"
                )
            else:
                print(
                    f"  {s['dim']:>3} │ {s['name']:<20} │ {'0':>8} │ "
                    f"{'—':>7} │ {'—':>6} │ {'—':>8} │ {marker}"
                )

        print()
        active = self.active_dims
        print(f"  Active dims:  {len(active)}/{self.n_dims} → {active}")
        inactive = [s["dim"] for s in self.dim_stats if not s["active"]]
        if inactive:
            print(f"  Inactive (below min_events threshold): {inactive}")
