"""
Streamlit dashboard for the quantflow market-making research stack.

Run with:
    uv run streamlit run python/quantflow/dashboard/app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(
    page_title="quantflow",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ──────────────────────────────────────────────────────────────

C = {
    "SAC":             "#2196F3",
    "Optimized AS":    "#4CAF50",
    "Static AS":       "#FF9800",
    "Naive Symmetric": "#9E9E9E",
    "bid":             "#26A69A",
    "ask":             "#EF5350",
    "drawdown":        "rgba(239,83,80,0.15)",
    "purple":          "#9C27B0",
}

AGENT_ORDER = ["Naive Symmetric", "Static AS", "Optimized AS", "SAC"]

# ── Sidebar navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📈 quantflow")
    st.divider()
    _PAGES = {
        "🏛️  Live LOB":         "lob",
        "📊  Agent Performance": "performance",
        "📉  Stylized Facts":    "stylized",
        "⚡  Benchmarks":        "benchmarks",
    }
    page = st.radio(
        "page",
        list(_PAGES.keys()),
        key="nav",
        label_visibility="collapsed",
    )
    _page_key = _PAGES[page]
    st.divider()
    st.caption("Hawkes-driven LOB + SAC market-making agent")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _import_quantflow():
    try:
        import quantflow
        return quantflow
    except ImportError:
        st.error(
            "Rust extension not found. Build it first:\n\n"
            "```bash\nmaturin develop --release\n```"
        )
        st.stop()


def _acf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """ACF without scipy."""
    x   = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    n = len(x)
    return np.array([np.mean(x[:n-lag] * x[lag:]) / var for lag in range(nlags + 1)])


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — Live LOB
# ═══════════════════════════════════════════════════════════════════════════════

def _init_sim(seed: int) -> None:
    qf = _import_quantflow()
    sim = qf.HawkesSimulator.new({"t_max": 3600.0, "snapshot_interval": 1})
    sim.reset(seed)
    st.session_state.sim        = sim
    st.session_state.price_hist = []   # (sim_time, mid)
    st.session_state.trade_hist = []   # {price, qty, sim_time}
    st.session_state.running    = False


def _depth_chart(sim) -> go.Figure:
    rb  = sim.get_book().snapshot(10)
    bps = rb["bid_price"].to_pylist()
    bqs = rb["bid_qty"].to_pylist()
    aps = rb["ask_price"].to_pylist()
    aqs = rb["ask_qty"].to_pylist()

    bids = [(p, q) for p, q in zip(bps, bqs) if p and p == p and q > 0]
    asks = [(p, q) for p, q in zip(aps, aqs) if p and p == p and q > 0]

    fig = go.Figure()
    if bids:
        bp, bq = zip(*sorted(bids, reverse=True))
        fig.add_trace(go.Bar(x=list(bq), y=[f"{p:.4f}" for p in bp],
                             orientation="h", name="Bid",
                             marker_color=C["bid"], opacity=0.8))
    if asks:
        ap, aq = zip(*sorted(asks))
        fig.add_trace(go.Bar(x=list(aq), y=[f"{p:.4f}" for p in ap],
                             orientation="h", name="Ask",
                             marker_color=C["ask"], opacity=0.8))

    fig.update_layout(
        height=320, title="Order Book Depth",
        xaxis_title="Quantity", barmode="overlay",
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _price_chart(price_hist: list, trade_hist: list) -> go.Figure:
    fig = go.Figure()
    if price_hist:
        ts, ps = zip(*price_hist)
        fig.add_trace(go.Scatter(x=list(ts), y=list(ps), mode="lines",
                                 name="Mid", line=dict(color="#444", width=1.5)))
    if trade_hist:
        fig.add_trace(go.Scatter(
            x=[t["sim_time"] for t in trade_hist],
            y=[t["price"]    for t in trade_hist],
            mode="markers", name="Trades",
            marker=dict(color=C["ask"], size=5, opacity=0.65),
        ))
    fig.update_layout(
        height=320, title="Price & Trades",
        xaxis_title="Sim Time (s)", yaxis_title="Price",
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def page_lob() -> None:
    st.title("🏛️  Live LOB Visualization")

    if "sim" not in st.session_state:
        _init_sim(42)

    # ── Controls ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 3, 3])
    if c1.button("▶ Start",  use_container_width=True):
        st.session_state.running = True
    if c2.button("⏸ Pause",  use_container_width=True):
        st.session_state.running = False

    speed = c3.slider("Events per frame", 5, 300, 30, step=5)
    seed  = c4.number_input("Seed", value=42, min_value=0, max_value=9999, step=1)

    if st.button("🔄 New Trading Day", use_container_width=False):
        _init_sim(int(seed))
        st.rerun()

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    ph_depth = col_left.empty()
    ph_price = col_right.empty()

    m1, m2, m3, m4 = st.columns(4)
    ph_mid    = m1.empty()
    ph_spread = m2.empty()
    ph_trades = m3.empty()
    ph_time   = m4.empty()

    # ── Advance ────────────────────────────────────────────────────────────────
    if st.session_state.running:
        sim = st.session_state.sim
        for _ in range(speed):
            event = sim.step()
            if event is None:
                st.session_state.running = False
                st.info("Simulated trading day complete. Press 🔄 for a new day.")
                break
            t   = event["sim_time"]
            mid = sim.mid_price()
            if mid and mid > 0:
                st.session_state.price_hist.append((t, mid))
            for trade in event["trades"]:
                st.session_state.trade_hist.append({**trade, "sim_time": t})

        st.session_state.price_hist = st.session_state.price_hist[-600:]
        st.session_state.trade_hist = st.session_state.trade_hist[-300:]

    # ── Render ─────────────────────────────────────────────────────────────────
    sim = st.session_state.sim
    ph_depth.plotly_chart(_depth_chart(sim),
                          use_container_width=True, key="depth")
    ph_price.plotly_chart(_price_chart(st.session_state.price_hist,
                                       st.session_state.trade_hist),
                          use_container_width=True, key="price")

    mid    = sim.mid_price()
    spread = sim.get_book().spread()
    sim_t  = st.session_state.price_hist[-1][0] if st.session_state.price_hist else 0.0

    ph_mid.metric("Mid Price",    f"{mid:.4f}"    if mid    else "—")
    ph_spread.metric("Spread",    f"{spread:.4f}" if spread else "—")
    ph_trades.metric("Trades",    len(st.session_state.trade_hist))
    ph_time.metric("Sim Time",    f"{sim_t:.1f}s")

    if st.session_state.running:
        time.sleep(0.04)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — Agent Performance
# ═══════════════════════════════════════════════════════════════════════════════

def page_performance() -> None:
    st.title("📊  Agent Performance")

    res_path  = Path("results/evaluation/results.parquet")
    traj_path = Path("results/evaluation/trajectories.parquet")

    if not res_path.exists():
        st.warning(
            "No evaluation data found. Run the evaluation pipeline first:\n\n"
            "```bash\nbash scripts/run_evaluation.sh runs/sac_test/best_model.zip 50\n```"
        )
        return

    summary = pd.read_parquet(res_path)
    traj    = pd.read_parquet(traj_path) if traj_path.exists() else None
    agents  = [a for a in AGENT_ORDER if a in summary["agent"].unique()]

    # ── Controls ───────────────────────────────────────────────────────────────
    c1, c2, _ = st.columns([2, 2, 4])
    agent = c1.selectbox("Agent", agents, index=len(agents) - 1)
    ep_id = c2.selectbox(
        "Episode",
        sorted(summary[summary["agent"] == agent]["episode_id"].unique())[:20],
        index=0,
    )

    ag = summary[summary["agent"] == agent]

    # ── Metric row ─────────────────────────────────────────────────────────────
    cols = st.columns(5)
    cols[0].metric("Mean PnL",    f"{ag['final_pnl'].mean():+.2f}")
    cols[1].metric("Mean Sharpe", f"{ag['sharpe'].mean():+.3f}")
    cols[2].metric("Max DD",      f"{ag['max_drawdown'].mean():.2f}")
    cols[3].metric("Fill Rate",   f"{ag['fill_rate'].mean():.4f}")
    cols[4].metric("Inv Std",     f"{ag['inventory_std'].mean():.2f}")

    st.divider()

    # ── PnL / inventory / reward chart ─────────────────────────────────────────
    if traj is not None:
        ep = traj[(traj["agent"] == agent) & (traj["episode_id"] == ep_id)]
        if not ep.empty:
            steps = ep["step"].to_numpy()
            pnl   = ep["cumulative_pnl"].to_numpy()
            inv   = ep["inventory"].to_numpy()
            rews  = ep["reward"].to_numpy()
            peak  = np.maximum.accumulate(pnl)

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=["Cumulative PnL + Drawdown", "Inventory", "Step Reward"],
                vertical_spacing=0.08,
            )

            # PnL line
            color = C.get(agent, "#2196F3")
            fig.add_trace(go.Scatter(x=steps, y=pnl, mode="lines", name="PnL",
                                     line=dict(color=color, width=2)), row=1, col=1)
            # Drawdown shading
            fig.add_trace(go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([peak,  pnl[::-1]]),
                fill="toself", fillcolor=C["drawdown"],
                line=dict(width=0), name="Drawdown",
            ), row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          line_width=1, row=1, col=1)

            # Inventory
            fig.add_trace(go.Scatter(
                x=steps, y=inv, mode="lines", name="Inventory",
                line=dict(color=C["purple"], width=1.5),
                fill="tozeroy", fillcolor="rgba(156,39,176,0.12)",
            ), row=2, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          line_width=1, row=2, col=1)

            # Reward bars
            fig.add_trace(go.Bar(
                x=steps, y=rews, name="Step Reward",
                marker_color=[C["bid"] if r >= 0 else C["ask"] for r in rews],
                opacity=0.7,
            ), row=3, col=1)

            fig.update_layout(height=580, showlegend=True,
                               margin=dict(l=0, r=0, t=40, b=0),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, key="perf_main")

    st.divider()

    # ── Strategy comparison tabs ────────────────────────────────────────────────
    st.subheader("Strategy Comparison")
    tab_sharpe, tab_pnl, tab_fill, tab_decomp = st.tabs(
        ["Sharpe", "PnL", "Fill Rate", "PnL Decomposition"]
    )

    def _bar(col: str, ylab: str, tab) -> None:
        with tab:
            agg = (summary.groupby("agent")[col]
                   .agg(["mean", "std"])
                   .loc[[a for a in AGENT_ORDER if a in summary["agent"].unique()]]
                   .reset_index())
            fig = go.Figure(go.Bar(
                x=agg["agent"], y=agg["mean"],
                error_y=dict(type="data", array=agg["std"].tolist(), visible=True),
                marker_color=[C.get(a, "#555") for a in agg["agent"]],
                opacity=0.85,
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
            fig.update_layout(height=320, yaxis_title=ylab,
                              margin=dict(l=0, r=0, t=10, b=0),
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, key=f"cmp_{col}")

    _bar("sharpe",    "Sharpe Ratio", tab_sharpe)
    _bar("final_pnl", "Mean PnL",     tab_pnl)
    _bar("fill_rate", "Fill Rate",    tab_fill)

    with tab_decomp:
        agg = (summary.groupby("agent")[["spread_pnl", "inventory_pnl"]]
               .mean()
               .loc[[a for a in AGENT_ORDER if a in summary["agent"].unique()]]
               .reset_index())
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["agent"], y=agg["spread_pnl"],
                             name="Spread PnL",    marker_color="#4CAF50", opacity=0.85))
        fig.add_trace(go.Bar(x=agg["agent"], y=agg["inventory_pnl"],
                             name="Inventory PnL", marker_color="#EF5350", opacity=0.75))
        totals = agg["spread_pnl"] + agg["inventory_pnl"]
        fig.add_trace(go.Scatter(x=agg["agent"], y=totals, mode="markers",
                                 name="Total", marker=dict(color="black", size=8)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        fig.update_layout(height=320, barmode="relative", yaxis_title="Mean PnL",
                          margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, key="cmp_decomp")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — Stylized Facts
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_facts(seed: int, n_events: int) -> dict:
    qf  = _import_quantflow()
    sim = qf.HawkesSimulator.new({"t_max": 86400.0, "snapshot_interval": 1})
    sim.reset(seed)

    mids: list[float]    = []
    times: list[float]   = []
    spreads: list[float] = []

    for _ in range(n_events):
        event = sim.step()
        if event is None:
            break
        mid    = sim.mid_price()
        spread = sim.get_book().spread()
        if mid and mid > 0:
            mids.append(mid)
            times.append(event["sim_time"])
        if spread and spread > 0:
            spreads.append(spread)

    return {
        "mids":    np.array(mids),
        "times":   np.array(times),
        "spreads": np.array(spreads),
    }


def _facts_from_uploads(uploaded_trades) -> dict | None:
    """Compute facts from an uploaded CSV with columns: price, sim_time."""
    try:
        df    = pd.read_csv(uploaded_trades)
        mids  = df["price"].to_numpy(dtype=float)
        times = df["sim_time"].to_numpy(dtype=float) if "sim_time" in df.columns else np.arange(len(mids))
        return {"mids": mids, "times": times, "spreads": np.array([])}
    except Exception as e:
        st.error(f"Could not parse uploaded file: {e}")
        return None


def _render_facts(data: dict) -> None:
    mids    = data["mids"]
    times   = data["times"]
    spreads = data["spreads"]

    if len(mids) < 10:
        st.warning("Too few observations — increase event count or check the data.")
        return

    log_ret = np.log(mids[1:] / mids[:-1])
    abs_ret = np.abs(log_ret)
    mu, sig = log_ret.mean(), log_ret.std()
    kurtosis = (float(np.mean((log_ret - mu) ** 4)) / (sig ** 4 + 1e-12)) - 3.0

    st.caption(
        f"Observations: {len(mids):,} mid-prices · "
        f"{len(spreads):,} spread snapshots · "
        f"Excess kurtosis: **{kurtosis:.2f}**"
    )

    col1, col2 = st.columns(2)

    # Plot 1 — Return distribution
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log_ret, nbinsx=60, histnorm="probability density",
            name="Simulated", marker_color=C["SAC"], opacity=0.7,
        ))
        if sig > 1e-10:
            xs = np.linspace(log_ret.min(), log_ret.max(), 200)
            ys = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Normal",
                                     line=dict(color=C["ask"], width=2)))
        fig.update_layout(
            height=300, title=f"Return Distribution  (kurtosis excess = {kurtosis:.2f})",
            xaxis_title="Log-return", margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, key="sf_ret")

    # Plot 2 — ACF of |returns|
    with col2:
        nlags    = min(50, len(abs_ret) // 5)
        acf_vals = _acf(abs_ret, nlags=nlags)
        conf     = 1.96 / np.sqrt(len(abs_ret))
        lags     = np.arange(1, nlags + 1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags, y=acf_vals[1:], name="ACF(|r|)",
                             marker_color=C["Optimized AS"], opacity=0.8))
        fig.add_hline(y= conf, line_dash="dot", line_color="red",
                      annotation_text="95% CI", line_width=1)
        fig.add_hline(y=-conf, line_dash="dot", line_color="red", line_width=1)
        fig.update_layout(
            height=300, title="ACF of |Returns|  (volatility clustering)",
            xaxis_title="Lag", margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, key="sf_acf")

    col3, col4 = st.columns(2)

    # Plot 3 — Spread distribution
    with col3:
        if len(spreads) > 5:
            fig = go.Figure(go.Histogram(
                x=spreads, nbinsx=50, histnorm="probability density",
                marker_color=C["Static AS"], opacity=0.8,
            ))
            fig.update_layout(
                height=300, title="Spread Distribution  (right-skewed)",
                xaxis_title="Bid-Ask Spread",
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True, key="sf_spread")
        else:
            st.info("Not enough spread observations.")

    # Plot 4 — Intraday pattern
    with col4:
        if len(times) > 20:
            T    = float(times[-1]) if times[-1] > 0 else 1.0
            bins = np.linspace(0, T, 13)
            counts, edges = np.histogram(times, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2 / 3600

            fig = go.Figure(go.Bar(
                x=centers, y=counts,
                marker_color=C["purple"], opacity=0.8,
                width=[(edges[1] - edges[0]) / 3600 * 0.8] * len(counts),
            ))
            fig.update_layout(
                height=300, title="Intraday Event Distribution  (U-shape)",
                xaxis_title="Simulation Time (h)", yaxis_title="Event Count",
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True, key="sf_intraday")


def page_stylized_facts() -> None:
    st.title("📉  Stylized Facts")

    # ── Toggle: simulated vs empirical ────────────────────────────────────────
    mode = st.radio("Data source", ["Simulated", "Empirical (upload)"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Empirical (upload)":
        st.caption("Upload a CSV with columns `price` (required) and `sim_time` (optional).")
        uploaded = st.file_uploader("Trade CSV", type="csv")
        if uploaded:
            data = _facts_from_uploads(uploaded)
            if data:
                _render_facts(data)
        else:
            st.info("Upload a CSV file to see empirical stylized facts.")
        return

    # ── Simulated mode ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 3, 2])
    seed     = c1.slider("Seed",    0, 999, 42)
    n_events = c2.slider("Events",  500, 10_000, 4_000, step=500)
    run_btn  = c3.button("▶ Simulate", type="primary", use_container_width=True)

    cache_key = f"sf_{seed}_{n_events}"
    if run_btn or (cache_key not in st.session_state):
        with st.spinner("Running simulator…"):
            st.session_state[cache_key] = _simulate_facts(seed, n_events)

    data = st.session_state.get(cache_key)
    if data:
        _render_facts(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def _load_criterion() -> dict[str, dict]:
    criterion_dir = Path("target/criterion")
    results: dict[str, dict] = {}
    if not criterion_dir.exists():
        return results
    for bench_dir in sorted(criterion_dir.iterdir()):
        est_file = bench_dir / "new" / "estimates.json"
        if est_file.exists():
            try:
                d = json.loads(est_file.read_text())
                results[bench_dir.name] = {
                    "mean_ns": d["mean"]["point_estimate"],
                    "std_ns":  d["std_dev"]["point_estimate"],
                }
            except (KeyError, json.JSONDecodeError):
                pass
    return results


def page_benchmarks() -> None:
    st.title("⚡  Benchmarks")

    # ── Latency ────────────────────────────────────────────────────────────────
    st.subheader("LOB Engine Latency  (Criterion)")

    bench = _load_criterion()
    if bench:
        rows = []
        for name, v in bench.items():
            ns = v["mean_ns"]
            if ns < 1_000:
                val, unit = ns, "ns"
            elif ns < 1_000_000:
                val, unit = ns / 1_000, "µs"
            else:
                val, unit = ns / 1_000_000, "ms"
            rows.append({
                "Benchmark":      name,
                f"Mean ({unit})":  f"{val:.2f}",
                "Std (ns)":        f"{v['std_ns']:.1f}",
                "Throughput":      f"{1e9/ns:,.0f}/s" if ns > 0 else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        names = list(bench.keys())
        means = [v["mean_ns"] for v in bench.values()]
        stds  = [v["std_ns"]  for v in bench.values()]
        fig = go.Figure(go.Bar(
            x=names, y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=C["SAC"], opacity=0.8,
        ))
        fig.update_layout(height=280, yaxis_title="Latency (ns)",
                          margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, key="bench_latency")
    else:
        st.info(
            "No benchmark data found. Generate it with:\n\n"
            "```bash\ncargo bench --bench orderbook_bench -p quantflow-core\n```"
        )

    st.divider()

    # ── Agent comparison table ─────────────────────────────────────────────────
    st.subheader("Agent Comparison")

    res_path = Path("results/evaluation/results.parquet")
    if not res_path.exists():
        st.warning(
            "No evaluation data. Run:\n\n"
            "```bash\nbash scripts/run_evaluation.sh runs/sac_test/best_model.zip 50\n```"
        )
        return

    df     = pd.read_parquet(res_path)
    agents = [a for a in AGENT_ORDER if a in df["agent"].unique()]

    agg = (df.groupby("agent")
           .agg(
               PnL_mean      = ("final_pnl",      "mean"),
               PnL_std       = ("final_pnl",      "std"),
               Sharpe        = ("sharpe",          "mean"),
               MaxDD         = ("max_drawdown",    "mean"),
               FillRate      = ("fill_rate",       "mean"),
               InvStd        = ("inventory_std",   "mean"),
               Spread_PnL    = ("spread_pnl",      "mean"),
               Inventory_PnL = ("inventory_pnl",   "mean"),
           )
           .loc[agents]
           .reset_index())
    agg.columns = ["Agent", "PnL Mean", "PnL Std", "Sharpe",
                   "Max DD", "Fill Rate", "Inv Std", "Spread PnL", "Inv PnL"]

    st.dataframe(
        agg.style
           .format({c: "{:+.2f}" for c in ["PnL Mean", "PnL Std", "Sharpe", "Spread PnL", "Inv PnL"]})
           .format({"Max DD": "{:.2f}", "Fill Rate": "{:.4f}", "Inv Std": "{:.2f}"})
           .background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Robustness heatmap ─────────────────────────────────────────────────────
    st.subheader("Sharpe Heatmap  (Agent × Episode)")
    baseline_agents = [a for a in agents if a != "SAC"]
    if baseline_agents:
        sub    = df[df["agent"].isin(agents)][["agent", "episode_id", "sharpe"]]
        pivot  = sub.pivot_table(index="agent", columns="episode_id",
                                 values="sharpe").iloc[:, :40]
        pivot  = pivot.loc[[a for a in AGENT_ORDER if a in pivot.index]]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            colorbar_title="Sharpe",
        ))
        fig.update_layout(
            height=220,
            xaxis_title="Episode",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "lob":         page_lob,
    "performance": page_performance,
    "stylized":    page_stylized_facts,
    "benchmarks":  page_benchmarks,
}
_DISPATCH[_page_key]()
