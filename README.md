# rust-lob-rl-market-maker

A high-performance limit order book (LOB) engine written in Rust, with a full quantitative market-microstructure stack and a reinforcement learning training pipeline. The Rust core handles matching and market data simulation at low latency; a PyO3 extension module exposes the engine to Python; a Gymnasium-compatible environment drives SAC training via stable-baselines3; and an evaluation suite compares the learned policy against Avellaneda-Stoikov baselines.

## Repository layout

```
rust-lob-rl-market-maker/
├── crates/
│   ├── quantflow-core/              # Pure-Rust library (no Python dependency)
│   │   └── src/
│   │       ├── orderbook/
│   │       │   ├── types.rs         # Price, Quantity, OrderId, Side, OrderType, Timestamp
│   │       │   ├── order.rs         # Order, Trade, ExecutionReport, BookDelta
│   │       │   ├── book.rs          # OrderBook — BTreeMap-based, price-time priority
│   │       │   └── matching.rs      # Limit, Market, IOC, FOK matching engine
│   │       ├── data/
│   │       │   ├── lobster.rs       # LOBSTER message & snapshot CSV parser
│   │       │   ├── arrow_export.rs  # Arrow RecordBatch + Parquet export (snapshots)
│   │       │   ├── binance.rs       # Binance WebSocket message types (DepthUpdate, AggTrade, BookTicker)
│   │       │   ├── binance_ws.rs    # Async WebSocket client — combined stream (depth/trade/ticker)
│   │       │   ├── recorder.rs      # ParquetRecorder — BatchBuffer<T> → Arrow → Parquet
│   │       │   ├── market_event.rs  # Unified MarketEvent (12-type) + Parquet round-trip I/O
│   │       │   ├── normalizer.rs    # BinanceNormalizer — depth diffs + trades → MarketEvent
│   │       │   └── replay.rs        # ReplayEngine — Parquet playback through OrderBook
│   │       ├── hawkes/
│   │       │   ├── kernel.rs        # ExponentialKernel, PowerLawKernel, ExcitationKernel trait
│   │       │   ├── process.rs       # MultivariateHawkes — Ogata thinning simulation
│   │       │   └── calibration.rs   # MLE via L-BFGS, goodness-of-fit, KS test, Q-Q data
│   │       ├── simulator.rs         # HawkesLobSimulator — 12D Hawkes → LOB event dispatch
│   │       ├── metrics/
│   │       │   └── stylized_facts.rs # Return kurtosis, ACF, spread dist, signature plot, intraday
│   │       └── strategy/
│   │           └── avellaneda.rs    # Avellaneda-Stoikov (2008) + BacktestResult
│   └── quantflow-ffi/               # PyO3 extension module (maturin build)
│       └── src/
│           ├── lib.rs               # #[pymodule] entry point
│           ├── orderbook.rs         # PyOrderBook
│           ├── simulator.rs         # PyHawkesSimulator
│           ├── strategy.rs          # PyAvellanedaStoikov
│           ├── replay.rs            # PyReplayEngine — historical Parquet playback from Python
│           └── functions.rs         # calibrate_hawkes(), load_lobster()
├── python/
│   └── quantflow/
│       ├── envs/
│       │   ├── market_making.py     # MarketMakingEnv (gymnasium.Env) + RewardNormalizer
│       │   ├── obs_features.py      # Obs v2: OFI, trade rate, vol, spread pctile, fill imbalance
│       │   ├── domain_randomizer.py # DomainRandomizer wrapper — Hawkes param perturbation
│       │   └── curriculum.py        # CurriculumWrapper + CurriculumCallback (3-stage)
│       ├── training/
│       │   ├── train.py             # SAC training loop + callbacks
│       │   ├── feature_extractor.py # LobFeatureExtractor — 1D-CNN + scalar MLP for SB3
│       │   ├── ab_curriculum.py     # A/B: no-curriculum vs CurriculumWrapper
│       │   └── evaluate.py          # Quick policy evaluation helper
│       ├── evaluation/
│       │   ├── compare.py           # 4-strategy comparison → Parquet
│       │   ├── report.py            # Formatted report + PnL decomposition
│       │   └── plots.py             # 6 matplotlib plots
│       └── features.py              # LOB microstructure feature engineering
├── tests/
│   ├── test_ffi.py                  # 27 PyO3 binding tests
│   ├── test_env.py                  # 18 Gymnasium env tests
│   ├── test_features.py             # 49 feature + RunningNormalizer tests
│   ├── test_domain_randomizer.py    # DomainRandomizer tests
│   └── test_curriculum.py           # 13 CurriculumWrapper + CurriculumCallback tests
├── scripts/
│   ├── plot_qq.py                   # Q-Q plot visualisation (matplotlib)
│   └── run_evaluation.sh            # End-to-end evaluation pipeline
├── backend/                         # FastAPI application server
│   ├── routers/
│   │   ├── simulation.py            # WebSocket /ws/live + REST control endpoints
│   │   ├── evaluation.py            # Strategy comparison endpoints
│   │   └── metrics.py               # Stylized-facts endpoints
│   └── services/
│       └── websocket.py             # SimulationRunner → HawkesSimulator + AvellanedaStoikov
├── dashboard/                       # Next.js 16 / Tailwind v4 Bloomberg-terminal UI
│   └── src/
│       ├── app/
│       │   ├── live/page.tsx        # F1 — real-time LOB + agent monitor
│       │   ├── arena/page.tsx       # F2 — strategy comparison (PnL, Sharpe, decomposition)
│       │   ├── metrics/page.tsx     # F3 — stylized facts
│       │   └── simulator/page.tsx   # F4 — Hawkes backtest explorer
│       ├── components/
│       │   ├── live/                # LobDepthChart, TradeFlowChart, StatsPanel, TradeFeed, ControlBar
│       │   ├── charts/              # PnlChart, SharpeBar, PnlDecomposition, DistributionBox
│       │   └── terminal/            # Bloomberg-style Panel, DataCell, TopBar, FunctionKeyBar
│       └── hooks/
│           └── useSimulation.ts     # WebSocket hook with auto-reconnect, rolling state
└── pyproject.toml                   # uv project; maturin build backend
```

## Building

### Rust

```bash
# Full Rust test suite (214 tests)
cargo test -p quantflow-core

# Release build
cargo build -p quantflow-core --release

# Q-Q plot: simulate, calibrate, visualise goodness-of-fit
cargo run -p quantflow-core --example hawkes_qq | python3 scripts/plot_qq.py

# AS backtest — full diagnostic report, σ auto-calibrated
cargo run -p quantflow-core --example as_backtest --release -- --auto-sigma

# AS backtest — manual parameters
cargo run -p quantflow-core --example as_backtest --release -- \
  --gamma 0.05 --kappa 0.5 --inv-limit 30 --auto-sigma

# AS parameter grid search (γ × κ × inv_limit, rayon-parallelised)
cargo run -p quantflow-core --example as_grid_search --release

# Criterion benchmarks
cargo bench --bench orderbook_bench -p quantflow-core
```

### Binance data pipeline

```bash
# Step 1 — record live Binance Futures streams to Parquet (depth@100ms, aggTrade, bookTicker)
cargo run -p quantflow-core --example record_btcusdt --release -- \
  --duration 3600 --output data/btcusdt/raw --symbol btcusdt --futures

# Step 2 — normalise raw Parquet to unified MarketEvent format + LOB snapshots
cargo run -p quantflow-core --example normalize_btcusdt --release -- \
  --input data/btcusdt/raw --output data/btcusdt/processed --date 2026-04-02

# Step 3 — replay events through the LOB engine and inspect statistics
cargo run -p quantflow-core --example replay_btcusdt --release -- \
  --input data/btcusdt/processed/2026-04-02_events.parquet
```

### Python (PyO3 extension + training)

```bash
# Create virtualenv and install dependencies
uv venv --python 3.12 && source .venv/bin/activate
uv sync

# Build the Rust extension in-place (required before any Python import)
maturin develop --release

# Run Python test suite (94 tests: env, features, FFI bindings)
uv run pytest tests/ -v

# Quick training run (50k steps, no W&B)
uv run python -m quantflow.training.train --timesteps 50000 --run-dir runs/sac_test

# Full training run with Weights & Biases
uv run python -m quantflow.training.train --final --wandb --wandb-project quantflow-mm

# End-to-end evaluation (compare → report → 6 plots)
bash scripts/run_evaluation.sh runs/sac_test/best_model.zip 50
```

### Dashboard

```bash
# Terminal 1 — FastAPI backend (port 8000)
uv run uvicorn backend.main:app --reload

# Terminal 2 — Next.js frontend (port 3000)
cd dashboard && npm run dev
```

Open `http://localhost:3000`.  The UI runs four screens navigable via F1–F4 or mouse:

| Key | Screen | Data source |
|-----|---------|-------------|
| F1  | Live trading monitor | `HawkesSimulator` streaming via WebSocket; AS agent places real resting orders into the Rust LOB |
| F2  | Strategy arena | 50-episode backtest comparison: SAC, Optimized AS, Static AS, Naive symmetric |
| F3  | Stylized facts | Return distribution, ACF, spread, intraday volume profile |
| F4  | Simulator | Interactive Hawkes backtest explorer |

**F1 architecture:** the FastAPI server runs `SimulationRunner`, which drives `HawkesSimulator` (Rust) via PyO3 at up to 10× real-time.  `AvellanedaStoikov.compute_quotes_skewed()` (Rust) computes optimal quotes with active inventory skewing; quotes are placed as real limit orders into the Rust `OrderBook`; fills are detected by exact `maker_id` matching.  The LOB depth chart, price chart, and trade feed are updated at 10 Hz over a WebSocket.

---

## 1. Hawkes Excitation Kernels (`hawkes/kernel.rs`)

### Financial motivation

In an efficient market, price moves should be unpredictable. But order *arrivals* are not — they cluster. A burst of market buys is followed by more market buys, followed by limit-sell cancellations, followed by further buys. This self-reinforcing dynamic is the defining feature of high-frequency order flow and is responsible for phenomena like short-term momentum, price impact, and intraday volatility clustering.

The **Hawkes process** (Hawkes, 1971) captures this by making the arrival *intensity* of each event type a function of past events:

```
λ_i(t) = μ_i  +  Σ_{k: t_k < t} φ_{i, type_k}(t − t_k)
```

- `μ_i` — background rate: how often event type *i* would arrive in the absence of any past activity (e.g., 3 limit buys per second on a quiet day).
- `φ_{ij}(τ)` — the **excitation kernel**: how much a past event of type *j* raises the current intensity of type *i* after elapsed time τ.

The kernel must eventually decay to zero — without decay, a single spike in activity would permanently elevate intensity and the market would eventually "explode" (infinite event count in finite time). The **branching ratio**

```
n*_ij = ∫₀^∞ φ_ij(τ) dτ
```

is the mean number of type-*i* events directly caused by one type-*j* event. For the process to be stationary (non-explosive), the spectral radius of the matrix `[n*_ij]` must be strictly less than 1. In the univariate case this reduces to `n* < 1`.

Empirically, equity markets operate with branching ratios of 0.5–0.9 (Bacry, Mastromatteo & Muzy, 2015) — close to criticality, which explains the apparent long-memory in order flow without requiring an explosive process.

### `ExponentialKernel`

```
φ(τ) = α · β · exp(−β · τ)
```

Parameters:
- `alpha` — total excitation mass; equals the branching ratio n\*.
- `beta` — decay rate. Large β means excitement dissipates in milliseconds; small β means it lingers for seconds.

The β factor in the numerator normalises the kernel so that `n* = α` regardless of decay speed. This is the Ogata (1981) parameterisation and makes `alpha` directly interpretable as the branching ratio.

Closed forms:
```
φ(τ)       = α · β · e^{−βτ}
∫₀ᵗ φ dτ   = α · (1 − e^{−βt})
∫₀^∞ φ dτ  = α  (= n*)
```

**Financial interpretation:** With β = 3 (mean excitement duration ≈ 0.33 s), an exponential kernel models the "mechanical" self-excitation of HFT algorithms reacting to recent activity. With β = 0.1 (≈ 10 s), it captures the slower rebalancing of institutional execution algorithms.

### `PowerLawKernel`

```
φ(τ) = α · β^γ · (β + τ)^{−(1+γ)}
```

Parameters:
- `alpha` — branching ratio n\*.
- `beta` — location scale (prevents singularity at τ = 0).
- `gamma` — tail decay exponent.

The power-law kernel decays much more slowly than the exponential. Empirical studies of equity and FX order flow (Bacry et al., 2015; Hawkes, 2018) consistently find power-law memory with γ ≈ 0.5–0.8. The long tail means that a burst of activity from 10 minutes ago still influences current intensity — consistent with intraday volatility persistence.

Closed form integral:
```
∫₀ᵗ φ dτ = (α/γ) · [1 − (β/(β+t))^γ]
∫₀^∞ φ dτ = α/γ
```

Note: for the power-law kernel, the branching ratio is `α/γ`, not `α`. This means stability requires `α/γ < 1`, so large α can be offset by a large decay exponent γ.

---

## 2. Multivariate Hawkes Process (`hawkes/process.rs`)

### From univariate to multivariate

Real markets have many event types happening simultaneously and influencing each other. A market buy arrival raises the probability of:
- More market buys (momentum, order splitting),
- Limit sell cancellations (adverse selection avoidance),
- New limit sell placements (liquidity replenishment).

The **multivariate Hawkes process** models d event types jointly. The kernel matrix `φ_ij` has d² entries: `φ_ij(τ)` is the influence that a past type-*j* event exerts on the current intensity of type *i*.

### The 12-dimensional LOB event space

The implementation uses d = 12 event types mapping to the six fundamental LOB interactions:

| Dim | Event | Financial interpretation |
|-----|-------|--------------------------|
| 0 | Market buy | Aggressive buyer crossing the spread; signals short-term bullishness |
| 1 | Market sell | Aggressive seller; signals short-term bearishness |
| 2 | Limit best buy | Passive buyer improving the bid; reduces spread |
| 3 | Limit best sell | Passive seller improving the ask; reduces spread |
| 4 | Limit deep buy | Order placed away from best; adds depth without narrowing spread |
| 5 | Limit deep sell | Same on the ask side |
| 6 | Cancel best buy | Withdrawal of the best bid; widens spread, reduces visible liquidity |
| 7 | Cancel best sell | Withdrawal of the best ask |
| 8 | Cancel deep buy | Stealth cancellation; minimal immediate price impact |
| 9 | Cancel deep sell | Same on the ask side |
| 10 | Modify buy | Downward quantity revision; reduces exposed size without repricing |
| 11 | Modify sell | Same on the ask side |

**Cross-excitation structure:** The default excitation matrix uses three tiers:
- **Self-excitation** (α = 0.20): each event type excites itself — bursts of market buys beget more market buys.
- **Symmetric-side excitation** (α = 0.10): market buys excite market sells (and vice versa) — the two sides respond to each other.
- **Cross-class excitation** (α = 0.03): weak coupling across event classes — a cancellation weakly predicts future market orders.

Row sums ≈ 0.20 + 0.10 + 10 × 0.03 = 0.60 < 1, ensuring stability.

### Ogata thinning algorithm

Simulating a Hawkes process requires generating a non-stationary Poisson process whose rate λ(t) depends on all previous events. The standard approach (Ogata, 1981) is **thinning** (acceptance-rejection):

1. Maintain λ\*, an upper bound on the current total intensity.
2. Draw the next candidate arrival dt ~ Exp(λ\*).
3. At t_cand = t + dt, evaluate the true intensity λ(t_cand) from the active event history.
4. Accept the candidate with probability λ(t_cand)/λ\*; assign type proportional to λ_i(t_cand).
5. Tighten λ\* = λ(t_cand⁺) and repeat.

This is valid because all excitation kernels are monotone-decreasing — λ(t) can only decrease (decay) or jump up (new event), so the current intensity is always a valid upper bound for the next interval. The acceptance rate reflects how "bursty" the process is: near-critical processes (high branching ratio) have very variable intensity and therefore low acceptance rates during quiet periods.

**Computational efficiency:** Past events whose kernel contribution falls below 10⁻¹² are pruned from the active set. For exponential kernels with β = 3, a contribution decays to 10⁻¹² after about 9 seconds — the active set stays bounded at O(9 × μ_total) entries regardless of simulation length.

---

## 3. MLE Calibration (`hawkes/calibration.rs`)

### Why calibrate?

A simulator with wrong parameters is useless for strategy development. If the model fires too many market orders, the strategy will learn to earn from phantom liquidity; if cancellation rates are too low, the strategy will never experience adverse selection. Calibration ties the simulator to observed market reality.

### The log-likelihood of a point process

Given a sequence of events {(t_k, type_k)} on [0, T], the log-likelihood under the Hawkes model is:

```
ℓ(θ) = Σ_k log λ_{type_k}(t_k⁻)  −  Σ_i ∫₀ᵀ λ_i(t) dt
```

The first term rewards the model for assigning high intensity to the times and types where events actually occurred. The second term (the **compensator**) penalises the model for predicting high intensity everywhere — it equals the expected total event count and prevents the trivial solution of infinite intensity.

For exponential kernels, both terms have closed-form O(n·d²) recursions, making calibration tractable even for large event sequences. The key recursive quantities are:

```
R_ij(t_k) = Σ_{t_m < t_k, type_m = j} exp(−β_ij · (t_k − t_m))
```

R_ij tracks the "running excitation" from past type-j events on dimension i. It updates multiplicatively:
```
R_ij(t_k) = exp(−β_ij · Δt) · R_ij(t_{k-1}) + 1_{type_{k-1} = j}
```
This reduces the naive O(n²) per-step cost to O(n·d²) total.

### L-BFGS optimisation

The negative log-likelihood is minimised via **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton algorithm that approximates the inverse Hessian using only the last m gradient vectors. This requires an exact gradient of ℓ(θ) — provided by the same O(n·d²) forward pass — and converges superlinearly without computing or storing a full d²×d² Hessian matrix.

**Reparameterisation:** The parameters (μ > 0, α ∈ [0,1), β > 0) are transformed to unconstrained space for the optimiser:
```
μ → log μ         (ensures μ > 0)
α → logit α       (ensures 0 < α < 1)
β → log β         (ensures β > 0)
```
Chain-rule corrects the gradient accordingly. This avoids constrained optimisation and box-projection steps.

### Goodness-of-fit: the time-rescaling theorem

A correctly specified Hawkes model satisfies the **time-rescaling theorem** (Daley & Vere-Jones, 2003): the transformed inter-arrival times

```
Λ_k = ∫_{t_{k-1}}^{t_k} λ(t) dt
```

should be i.i.d. Exp(1). This gives a natural goodness-of-fit test:

- **KS test:** Compare the empirical CDF of {Λ_k} to the Exp(1) CDF. A small KS statistic D and large p-value indicate a good fit.
- **Q-Q plot:** Plot empirical quantiles of {Λ_k} against the theoretical Exp(1) quantiles. Points on the 45° diagonal = model fits; systematic deviations reveal misspecification.

```bash
# Run the full pipeline: simulate → calibrate → Q-Q plot
cargo run -p quantflow-core --example hawkes_qq | python3 scripts/plot_qq.py
```

The Q-Q plot shows two panels: the scatter against the diagonal, and residuals (empirical − theoretical). KS D < 0.03 with p > 0.50 indicates acceptable fit.

---

## 4. Hawkes-Driven LOB Simulator (`simulator.rs`, `metrics/stylized_facts.rs`)

### Why not i.i.d. Poisson?

The simplest LOB simulator draws each event type from an independent Poisson process. This is computationally convenient but fails to reproduce any of the empirical regularities of real markets:

| Property | i.i.d. Poisson | Hawkes | Empirical |
|---|---|---|---|
| Order clustering | ✗ | ✓ | ✓ |
| Short-term momentum | ✗ | ✓ | ✓ |
| Volatility clustering | ✗ | ✓ | ✓ |
| Intraday U-shape | ✗ | ✓ (with thinning) | ✓ |
| Heavy-tailed returns | ✗ | ✓ | ✓ |

A strategy trained on i.i.d. Poisson data will systematically overfit to the absence of clustering — it will, for example, never learn to widen quotes after a burst of market orders because the i.i.d. model says such bursts have no predictive content.

### Event-to-order mapping

The simulator consumes a pre-simulated Hawkes event stream and maps each event to an order book action:

- **Market buy/sell (dims 0/1):** Submit a market order with an extreme price (crossing all resting liquidity). Quantity drawn from `LogNormal(μ=3, σ=1)`, giving a realistic right-skewed size distribution (Gopikrishnan et al., 2000).

- **Limit best buy/sell (dims 2/3):** Place a limit order one tick inside the current spread. Suppressed if it would cross the book (prevents accidental immediate execution).

- **Limit deep buy/sell (dims 4/5):** Place 2–10 ticks behind the best, adding depth without affecting the touch.

- **Cancel best buy/sell (dims 6/7):** Remove the time-priority order at the best bid or ask — the most impactful cancellation type as it widens the spread.

- **Cancel deep buy/sell (dims 8/9):** Remove a randomly selected order from levels 2+.

- **Modify buy/sell (dims 10/11):** Halve the quantity of a randomly selected resting order — models the common pattern of risk managers scaling back exposed size without full cancellation.

### Intraday U-shape

Trading activity is empirically highest near market open (9:30) and close (16:00), and lowest around midday — the U-shape (Admati & Pfleiderer, 1988). The mechanism: informed traders concentrate at open/close when price discovery is most valuable; market makers accommodate.

The simulator replicates this by thinning the Hawkes event stream with a time-of-day acceptance function:

```
p_accept(t) = f(t) / peak_factor
f(t) = 1 + (peak_factor − 1) · (2t/T − 1)²
```

At open and close, `f(t) = peak_factor` and nearly all events are accepted. At midday, `f(t) = 1` and only `1/peak_factor` of events pass through. The U-shape emerges without altering the self-exciting Hawkes dynamics.

### Stylized facts validation (`metrics/stylized_facts.rs`)

Five functions verify that the simulator reproduces empirically documented market properties (Cont, 2001):

| Function | Measures | What to look for |
|---|---|---|
| `compute_return_kurtosis` | Excess kurtosis of log-returns | Should be ≫ 0 (fat tails; empirical equity kurtosis ≈ 5–20) |
| `compute_acf_absolute_returns` | ACF of \|r_t\| at lags 1–N | Should decay slowly and stay positive (volatility clustering / GARCH effect) |
| `compute_spread_distribution` | Sorted bid-ask spreads | Should be right-skewed; mode at 1–2 ticks |
| `compute_signature_plot` | Realized variance vs. sampling scale | Should increase with scale (microstructure noise at fine scales) |
| `compute_intraday_pattern` | Volume binned by time-of-day | Should be U-shaped: high at open/close, low at midday |

---

## 5. Avellaneda-Stoikov Market Making (`strategy/avellaneda.rs`)

### The market-making problem

A market maker continuously quotes a bid and an ask, earning the bid-ask spread on each round-trip (buy at bid, sell at ask). The core tension is **inventory risk**: if the market moves against a position built up through unbalanced fills, the spread income is more than offset by mark-to-market losses.

Avellaneda & Stoikov (2008) solve this formally using a Hamilton-Jacobi-Bellman (HJB) equation. The agent maximises expected exponential utility of terminal wealth:

```
max E[−exp(−γ · W_T)]
```

where γ > 0 is the **risk-aversion parameter** and W_T is wealth at time T. Under the model assumptions (geometric Brownian motion for mid-price, Poisson arrivals for fills), the optimal policy has a closed-form solution.

### The reservation price

The key insight is that a long inventory position is a liability: if the mid-price falls, the agent loses on every unit held. The **reservation price** (also called the *indifference price*) adjusts the mid for this inventory risk:

```
r(t) = s(t) − q · γ · σ² · (T − t)
```

- `s(t)` — current mid-price.
- `q` — current inventory (positive = long, negative = short).
- `γ · σ² · (T − t)` — inventory adjustment: risk aversion × variance × time remaining.

**Interpretation:**
- A long agent (`q > 0`) quotes *below* mid — lower bid and ask to attract sellers and reduce exposure.
- A short agent (`q < 0`) quotes *above* mid — higher bid and ask to attract buyers and reduce short exposure.
- The adjustment grows with time remaining `τ = T − t`: early in the day, inventory risk is high (long time for adverse moves); near close, it shrinks toward zero.

### The optimal spread

The full bid and ask are:

```
bid(t) = r(t) − δ*(t) / 2
ask(t) = r(t) + δ*(t) / 2
```

where the **optimal spread** is:

```
δ*(t) = γ · σ² · (T − t)  +  (2/γ) · ln(1 + γ/κ)
```

Two terms:
1. `γ · σ² · (T − t)` — **inventory risk term**: wider spread when volatility is high or time horizon is long. This compensates for the risk of being filled on one side and then seeing the price move against the position.

2. `(2/γ) · ln(1 + γ/κ)` — **fill-intensity term**: depends on κ, the order arrival intensity near the quotes. When the book is liquid (high κ), the agent can quote more aggressively (narrower spread) and still earn from rapid round-trips. When the book is illiquid (low κ), wider spreads compensate for infrequent fills.

**Key properties:**
- The spread narrows toward zero as τ → 0 (inventory risk vanishes at day-end).
- The spread narrows as κ → ∞ (extremely liquid market).
- For a flat mid (σ = 0), the spread collapses to the pure liquidity term.

### σ auto-calibration

In practice, the correct value of σ for a given instrument is not known in advance and changes intraday. Using a misconfigured σ breaks the strategy: a σ that is too small produces quotes that are too tight, creating severe adverse selection; a σ that is too large produces quotes far from mid that never fill.

The `with_auto_sigma` constructor enables a warm-up phase: the strategy observes the first `warm_up_events` (default 500) mid-price movements without placing any quotes, then estimates σ from the realised mid-price series using quadratic variation:

```
σ̂² = Σ_k (log s_{t_k} − log s_{t_{k−1}})² / T_total
```

This is the continuous-time realised variance estimator. Dividing the sum of squared log-returns by total elapsed time T (not by the number of observations) gives σ in price units per second, consistent with the AS model's Brownian motion assumption.

```rust
// σ estimated from first 500 mid-price observations; no quotes during warm-up
let strat = AvellanedaStoikov::with_auto_sigma(
    0.05,   // γ
    0.5,    // κ
    3600.0, // T
    30,     // inventory limit
    0.0,    // spread floor
);
```

### Spread floor

The AS formula produces arbitrarily tight spreads as `τ → 0` (end of day) or when σ is small. In practice, a minimum spread is necessary because:
- Exchange fees impose a minimum cost per round-trip.
- Adverse selection from informed flow is never zero.
- Wide markets in the underlying can cause fills at prices away from the quoted mid.

The `spread_floor` parameter sets a minimum half-spread. Any quote where the AS formula yields a half-spread below `spread_floor` is widened to `spread_floor`:

```
half_spread = max(γσ²τ/2 + (1/γ)·ln(1 + γ/κ), spread_floor)
```

### Backtest mechanics

The backtest runs against the Hawkes LOB simulator, refreshing quotes every second of simulation time:

1. **Fill detection:** After each market event, check whether any of the agent's resting orders appear as `maker_id` in the returned trades. If so, update cash and inventory.

2. **Quote refresh:** Cancel the current bid/ask, compute new quotes via the AS formula, and place fresh limit orders — provided the pre-fill inventory check `|inventory ± quote_qty| ≤ inventory_limit` is satisfied.

3. **Mark-to-market PnL:** At each book snapshot, compute `cash + inventory × mid_price`.

### PnL decomposition

`BacktestResult` decomposes total PnL into two components that diagnose strategy health:

```
Net PnL = Spread PnL + Inventory PnL
```

- **Spread PnL** (`spread_pnl`): income from capturing the bid-ask spread — `Σ |fill_price − mid_at_fill| × qty`. Always ≥ 0. This is the pure market-making revenue.

- **Inventory PnL** (`inventory_pnl`): mark-to-market gains/losses from holding a net position while prices drift — `Net PnL − Spread PnL`. Typically negative when the strategy is adversely selected.

The ratio `|Inventory PnL| / Spread PnL` is the key diagnostic: values below 50% indicate a well-hedged strategy where spread income comfortably exceeds directional loss; values above 100% mean the strategy is losing money overall despite earning the spread.

### Adverse selection analysis

For each fill, `BacktestResult` records the mid-price move at 1 s, 5 s, and 10 s after execution:

- **Ask fill:** adverse if mid moves up after we sold (we were hit by an informed buyer who knew the price would rise).
- **Bid fill:** adverse if mid moves down after we bought (we were hit by an informed seller who knew the price would fall).

```
adverse_move(δt) = E[sign(fill_direction) · (mid_{t+δt} − mid_t)]
```

Positive values indicate systematic adverse selection — counterparties have information advantage. If `adverse_move(1s) > half_spread`, the strategy cannot break even in expectation and must either widen quotes or reduce exposure.

### Performance metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Sharpe ratio** | `mean(ΔPnL) / std(ΔPnL) · √n` | Risk-adjusted return; > 1.0 indicates a viable strategy |
| **Max drawdown** | `max(peak − current PnL)` | Worst peak-to-trough loss; measures tail risk |
| **Calmar ratio** | `final_pnl / max_drawdown` | Return per unit of drawdown risk |
| **Fill rate** | `fills / quotes_placed` | Fraction of placed orders executed; low = quotes too far from mid |
| **Time-in-market** | `events_with_open_quote / total_events` | Quote exposure fraction; low = frequent warm-up / inventory blocks |

```rust
use quantflow_core::strategy::AvellanedaStoikov;
use quantflow_core::simulator::HawkesLobSimulator;

let mut sim = HawkesLobSimulator::default_12d().unwrap();
let strat = AvellanedaStoikov::with_auto_sigma(0.05, 0.5, 3600.0, 30, 0.0);
let result = strat.run_backtest(&mut sim, /*seed=*/ 42);

println!("Spread PnL:   {:+.2}", result.spread_pnl);
println!("Inventory PnL:{:+.2}", result.inventory_pnl);
println!("Sharpe:        {:.3}", result.sharpe);
println!("Calmar:        {:.3}", result.calmar_ratio());
println!("Adverse (1s):  {:.5}", result.post_fill_move_1s);
```

---

## 6. Parameter Grid Search (`examples/as_grid_search.rs`)

### Motivation

The AS formula is a three-parameter family (γ, κ, inv_limit). No single set of parameters is universally optimal: the correct γ depends on realised volatility; the correct κ depends on book depth; the correct inv_limit depends on capital and risk appetite. Rather than hand-tuning, the grid search exhaustively evaluates the cross-product of candidate values and surfaces which combinations produce robust positive Sharpe across multiple market regimes (seeds).

### Grid axes

| Parameter | Values | Rationale |
|---|---|---|
| γ (risk aversion) | 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0 | Spans from near-neutral (0.05) to strongly risk-averse (1.0) |
| κ (fill intensity) | 0.5, 1.0, 1.5, 2.0, 3.0 | Liquid (high κ, tight spread) to illiquid (low κ, wide spread) |
| inv_limit | 10, 20, 30, 50 | Conservative to permissive inventory constraint |
| σ | auto | Quadratic variation warm-up; no manual σ in search |

Total: 7 × 5 × 4 = 140 combinations × 5 seeds = 700 independent backtest runs.

### Parallelisation

Each combination is independent — no shared state between (γ, κ, inv_limit) runs. The search uses **rayon**'s `par_iter` to distribute work across all available cores. On an 8-core machine a 1-hour grid search completes in under 3 minutes in release mode.

### Output

```
  Rank │ γ     │ σ    │ κ   │ inv_lim │ Sharpe │ PnL (mean±std)    │ MaxDD   │ FillRate
  ─────┼───────┼──────┼─────┼─────────┼────────┼───────────────────┼─────────┼─────────
  1    │ 0.05  │ auto │ 0.5 │ 30      │  +2.18 │ +1475 ± 537       │     348 │    30.4%
  2    │ 0.05  │ auto │ 0.5 │ 10      │  +1.67 │ +480 ± 226        │     196 │    34.2%
  ...
```

Followed by **marginal Sharpe bar charts** — average Sharpe holding each parameter fixed while marginalising over the others. This reveals the dominant sensitivity: if the γ bar chart shows a monotone decline from γ=0.05 to γ=1.0, risk aversion is the primary lever; if the inv_limit chart is flat, the strategy is insensitive to position limits at these scales.

The best configuration is reproduced as a ready-to-run command:
```bash
cargo run -p quantflow-core --example as_backtest --release -- \
  --gamma 0.05 --kappa 0.5 --inv-limit 30 --auto-sigma
```

---

## 7. Limit Order Book Engine (`orderbook/`)

### Price representation

All prices are stored as `i64` integer ticks with a fixed scale of `PRICE_SCALE = 100_000_000` (10⁸ ticks per dollar). This means $1.00 is represented as `100_000_000`, $99.90 as `9_990_000_000`.

**Why integer ticks?**
- Floating-point keys in a `BTreeMap` are unsound (`f64` does not implement `Ord`).
- Tick arithmetic is exact: `a + b - b == a` always holds, unlike float arithmetic.
- The fixed scale gives 8 decimal places of precision — sufficient for equities and most derivatives.

### Order book (`orderbook/book.rs`)

`OrderBook` is backed by two `BTreeMap`s:

```
bids: BTreeMap<Reverse<Price>, VecDeque<Order>>   — highest price first
asks: BTreeMap<Price, VecDeque<Order>>            — lowest price first
order_index: HashMap<OrderId, (Side, Price)>      — O(1) cancel/modify lookup
```

Each price level holds a `VecDeque<Order>`. The front is the oldest (highest-priority) order — FIFO price-time priority, consistent with all major exchange protocols.

| Method | Complexity | Notes |
|---|---|---|
| `add_limit_order` | O(log L + Q) | L levels, Q makers consumed |
| `cancel_order` | O(log L + Q) | `order_index` eliminates side scan |
| `modify_order` | O(log L + Q) | Downward only; in-place, no priority loss |
| `best_bid / best_ask` | O(1) | First entry of each BTreeMap |
| `mid_price / spread` | O(1) | Derived from best bid/ask |

### Matching engine (`orderbook/matching.rs`)

Four order types: **Limit** (rests on book if not crossed), **Market** (sweeps all available liquidity), **IOC** (cancels residual), **FOK** (atomic: rejects entirely if not fully fillable).

Execution price is always the **resting (maker) order's price** — consistent with continuous double auction mechanics. Partially consumed maker orders are re-inserted at the *front* of their level, preserving time priority (Cont, Stoikov & Talreja, 2010 §3.1).

### LOBSTER parser (`data/lobster.rs`)

Parses the [LOBSTER](https://lobsterdata.com) historical equity order book format. Replays message streams against a live `OrderBook` via `LobsterReplayer`, handling the six LOBSTER event types (new limit, partial cancel, full cancel, execution visible, execution hidden, trading halt). Bypasses the matching engine on `NewLimitOrder` events to avoid double-matching — executions appear as separate Type 4/5 events in the data.

### Arrow & Parquet export (`data/arrow_export.rs`)

`OrderBookSnapshot` and `Trade` slices can be exported to Apache Arrow `RecordBatch`es and Parquet files for downstream analysis in Python (Polars, pandas, DuckDB).

---

## 8. PyO3 FFI Bindings (`crates/quantflow-ffi`)

The `quantflow-ffi` crate compiles to a Python extension module (`quantflow.so`) via [maturin](https://github.com/PyO3/maturin). It wraps the Rust core with zero-copy data transfer via the Arrow C Data Interface (PyCapsule protocol) — no serialisation between Rust and Python for order book snapshots or trade records.

### Exposed classes and functions

| Symbol | Type | Description |
|---|---|---|
| `PyOrderBook` | class | Live order book: `add_limit_order`, `cancel_order`, `best_bid/ask`, `snapshot(n)` → `pa.RecordBatch` |
| `PyHawkesSimulator` | class | Simulator: `reset(seed)`, `step()`, `place_limit_order`, `cancel_agent_order`, `get_book()`, `mid_price()` |
| `PyAvellanedaStoikov` | class | AS strategy: `compute_quotes(mid, inventory, t)`, `sigma`, `gamma`, `kappa` properties |
| `PyReplayEngine` | class | Historical playback: `next_event()`, `step_n(n)`, `snapshot(levels)`, `reset()`, `progress()`, `remaining()` |
| `calibrate_hawkes(events, d)` | function | Fit exponential Hawkes to a list of `{time, event_type}` dicts; returns μ, α, β, NLL |
| `load_lobster(msg_path, book_path)` | function | Parse LOBSTER CSV files; returns `{messages, snapshots}` as Arrow RecordBatches |

### Zero-copy snapshot format

`PyOrderBook.snapshot(n)` and `PyHawkesSimulator.get_book().snapshot(n)` return a `pyarrow.RecordBatch` with columns:

```
bid_price  float64  (NaN = empty level)
bid_qty    uint64
ask_price  float64  (NaN = empty level)
ask_qty    uint64
```

The RecordBatch is constructed in Rust and transferred via Arrow's PyCapsule Interface — the Python side holds a view into Rust-allocated memory with no copy.

```python
from quantflow import HawkesSimulator, AvellanedaStoikov

sim = HawkesSimulator.new({"t_max": 3600.0})
sim.reset(42)

strat = AvellanedaStoikov(gamma=0.05, kappa=0.5, t_end=3600.0)

while (event := sim.step()) is not None:
    mid = sim.mid_price()
    if mid:
        bid, ask = strat.compute_quotes(mid=mid, inventory=0, t=event["sim_time"])
```

---

## 9. Gymnasium MarketMakingEnv (`python/quantflow/envs/market_making.py`)

`MarketMakingEnv` wraps the Hawkes LOB simulator in a `gymnasium.Env`. The agent controls Avellaneda-Stoikov parameters each step; the Rust AS formula translates them into live limit orders on the book.

### Observation space (`gymnasium.spaces.Dict`)

**v1 — base features (8 keys, always present)**

| Key | Shape | Range | Description |
|---|---|---|---|
| `lob_state` | (20,) | [−1, 1] | 5 bid + 5 ask levels; each: (Δprice/mid, qty/max_qty_scale) |
| `volume_imbalance` | (1,) | [−1, 1] | (V_bid − V_ask) / (V_bid + V_ask) |
| `spread` | (1,) | [0, 1] | Bid-ask spread / mid |
| `mid_price_return` | (1,) | [−1, 1] | log(mid_t / mid_{t-1}), clipped to ±10% |
| `volatility` | (1,) | [0, 1] | Rolling realised vol (quadratic variation), clipped |
| `inventory` | (1,) | [−1, 1] | Signed inventory / inventory_limit |
| `pnl` | (1,) | [−1, 1] | Mark-to-market PnL / (initial_mid × inventory_limit) |
| `time_remaining` | (1,) | [0, 1] | 1 − t / T |

**v2 — additional regime-detection features (`obs_version="v2"`, 6 extra keys)**

| Key | Shape | Description |
|---|---|---|
| `ofi_short` | (1,) | Order-flow imbalance, 10-event rolling window |
| `ofi_long` | (1,) | Order-flow imbalance, 50-event rolling window |
| `trade_arrival_rate` | (1,) | Trades per second (last 30 events), normalised |
| `realized_vol` | (1,) | 20-event realised volatility (quadratic variation) |
| `spread_percentile` | (1,) | Rolling spread percentile vs last 200 observations |
| `agent_fill_imbalance` | (1,) | (bid_fills − ask_fills) / (total_fills + ε) |

### Action space (`Box(2,)`)

| Index | Parameter | Range | Effect |
|---|---|---|---|
| 0 | γ (risk aversion) | [0.01, 1.0] | Controls spread width and inventory skew |
| 1 | κ-offset | [−0.5, 0.5] | Multiplier: κ = κ_base × (1 + offset) |

### Reward

**v1 (default)**

```
R = ΔPnL  −  φ·|q|  −  ψ·q²  −  λ·max(0, |q|−K)
```

**v2 (`reward_version="v2"`)** — adds round-trip bonus, asymmetric inventory shaping, and terminal penalty:

```
R = ΔPnL  −  φ·|q|  −  ψ·q²  −  λ·max(0, |q|−K)
      +  rt_weight · round_trip_pnl          # bonus per completed round trip
      +  asymmetric_strength · q · ΔPnL      # extra penalty when inventory grows against PnL
      −  terminal_weight · q² / K²  (final step only)
```

Default v2 parameters: `phi=0.01, psi=0.001, lambda_breach=1.0, rt_weight=0.5, asymmetric_strength=0.3, terminal_weight=2.0`.

**Online reward normalisation** — enabled by default (`normalize_reward=True`). Uses Welford's one-pass algorithm to track running mean/variance across all steps (not reset between episodes). Rewards are clipped to ±10σ before normalisation. Set `normalize_reward=False` in eval environments to observe raw economic values.

### Fill detection

After each `events_per_step` Hawkes events, the env checks whether any trade `maker_id` matches the agent's resting bid or ask order ID. On a match, cash and inventory are updated. This is exact — no proxy fill model.

```python
import gymnasium as gym

env = gym.make("quantflow/MarketMaking-v0")
obs, _ = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

---

## 10. LOB Microstructure Features (`python/quantflow/features.py`)

Seven pure functions operating on `pyarrow.RecordBatch` snapshots and trade dicts. All functions are stateless and can be combined into a feature vector for the RL agent or any downstream model.

| Function | Input | Output | Formula |
|---|---|---|---|
| `volume_imbalance(snapshot, level)` | LOB snapshot | float ∈ (−1, 1) | (V_bid − V_ask) / (V_bid + V_ask) |
| `weighted_mid_price(snapshot)` | LOB snapshot | float | Σ(P_ask·V_bid + P_bid·V_ask) / Σ(V_bid + V_ask) — Stoikov microprice |
| `depth_ratio(snapshot, levels)` | LOB snapshot | float > 0 | Σ V^b / Σ V^a |
| `spread_bps(snapshot)` | LOB snapshot | float ≥ 0 | (ask − bid) / mid × 10,000 |
| `order_flow_imbalance(trades, window, mid)` | Trade list | float ∈ (−1, 1) | (V_buy − V_sell) / total; tick-rule fallback |
| `realized_volatility(trades, window, ann_factor)` | Trade list | float ≥ 0 | √(Σ r_k² · ann_factor / n) — quadratic variation |
| `trade_arrival_rate(trades, timestamps, window_s)` | Trade list + times | float ≥ 0 | Count(t ∈ [t_end − W, t_end]) / W |

`compute_all(snapshot, trades, timestamps, mid)` returns all eight features as a `float32` array of shape `(8,)` in the order of `FEATURE_NAMES`.

### RunningNormalizer

Online mean/variance tracking using Welford's (1962) one-pass algorithm. Numerically stable — avoids catastrophic cancellation from the two-pass `Σx² − n·mean²` formula.

```python
from quantflow.features import compute_all, RunningNormalizer

normalizer = RunningNormalizer(n_features=8)

# Each env step:
features = compute_all(snapshot, trades, timestamps=sim_times, mid=mid)
normalizer.update(features)
z = normalizer.normalize(features)   # ≈ N(0, 1) after warm-up

# Persist for inference deployment
normalizer.save("normalizer.json")
normalizer2 = RunningNormalizer.load("normalizer.json")
```

---

## 11. SAC Training Pipeline (`python/quantflow/training/`)

Trains a Soft Actor-Critic agent with `MultiInputPolicy` (required for Dict observation spaces). All hyperparameters are grouped in `SACConfig`.

### Configuration

```python
from quantflow.training.train import SACConfig

cfg = SACConfig(
    policy                  = "MultiInputPolicy",  # Dict obs → LobFeatureExtractor + MLP
    buffer_size             = 1_000_000,
    batch_size              = 256,
    tau                     = 0.005,
    gamma                   = 0.99,
    ent_coef                = "auto",              # automatic entropy tuning
    net_arch                = [128, 128],          # actor/critic heads after extractor
    target_update_interval  = 2,                   # delayed target updates for critic stability
    total_timesteps         = 1_000_000,           # 2M for final run (--final flag)
    eval_freq               = 10_000,
    n_eval_episodes         = 20,                  # fixed seeds 5000–5019
)
```

**Learning rate** decays linearly from 3×10⁻⁴ → 5×10⁻⁵ over training (schedule passed directly to SB3).

### LobFeatureExtractor

A custom `BaseFeaturesExtractor` that replaces SB3's default `CombinedExtractor`. Placed in `python/quantflow/training/feature_extractor.py`.

```
lob_state (20,)
  → reshape (batch, 10 levels, 2)   [Δprice, qty per level]
  → permute (batch, 2 channels, 10) for Conv1d
  → Conv1d(2→16, k=3) → ReLU       → (batch, 16, 8)
  → Conv1d(16→32, k=3) → ReLU      → (batch, 32, 6)
  → Flatten                         → (batch, 192)

scalar keys (all other obs, sorted, concatenated)
  → Linear(N→64) → ReLU            → (batch, 64)

combined
  → cat([192, 64])                  → (batch, 256)
  → Linear(256→128) → ReLU         → (batch, 128)  ← features_dim
```

The CNN learns spatial patterns across price levels (liquidity gaps, volume clusters). Compatible with both `obs_version="v1"` (7 scalar keys) and `obs_version="v2"` (13 scalar keys) — scalar dimension is inferred dynamically.

### QuantflowEvalCallback

Fires every `eval_freq` steps (default 10k). Runs `n_eval_episodes` (default 20) episodes with fixed seeds `5000–5019` using the current deterministic policy **and** a static AS baseline with the same env. Eval episodes use `normalize_reward=False` so metrics reflect raw economic values. Logs to SB3's stdout logger and optionally to W&B via direct `wandb.log()` — no TensorBoard dependency.

| Logged key | Description |
|---|---|
| `eval/mean_reward` | Mean cumulative episode reward (raw) |
| `eval/mean_pnl` | Mean mark-to-market PnL at episode end |
| `eval/mean_sharpe` | Per-episode Sharpe: mean(r) / std(r) · √n |
| `eval/mean_inventory_std` | Mean inventory standard deviation |
| `baseline/mean_reward` | Same metrics for static AS (γ=0.1) |
| `baseline/mean_pnl` | AS baseline PnL |
| `baseline/mean_sharpe` | AS baseline Sharpe |
| `delta/pnl` | SAC − AS baseline PnL |
| `delta/sharpe` | SAC − AS baseline Sharpe |

Best model (by mean reward) is saved to `<run_dir>/best_model.zip` automatically.

### W&B integration

No TensorBoard installation required. Pass `--wandb` to enable:

```bash
uv run python -m quantflow.training.train \
    --final \
    --wandb --wandb-project quantflow-mm \
    --run-dir runs/sac_final
```

---

## 12. Domain Randomization & Curriculum Learning

### Domain Randomization (`python/quantflow/envs/domain_randomizer.py`)

`DomainRandomizer` wraps any `gymnasium.Env` and randomly perturbs Hawkes simulator parameters at each episode reset. This prevents the agent from over-fitting to a single market regime.

| Parameter | Perturbation |
|---|---|
| `mu` (baseline intensities) | ×U(0.7, 1.3) per process |
| `alpha` (excitation matrix) | ×U(0.8, 1.2) per entry |
| `beta` (decay rates) | ×U(0.85, 1.15) |
| `tick_size` | sampled from {0.01, 0.02, 0.05} |

Parameters are clamped to maintain Hawkes stability (spectral radius < 1).

```python
from quantflow.envs.domain_randomizer import DomainRandomizer
from quantflow.envs.market_making import MarketMakingEnv

env = DomainRandomizer(MarketMakingEnv(), perturbation_scale=0.2)
```

### Curriculum Learning (`python/quantflow/envs/curriculum.py`)

`CurriculumWrapper` exposes three difficulty stages that progressively increase market complexity. `CurriculumCallback` auto-advances the stage when a rolling reward threshold is met.

| Stage | `episode_length` | `events_per_step` | `inventory_limit` | Advance threshold |
|---|---|---|---|---|
| `easy` | 250 | 25 | 5 | mean reward > −0.5 over 10 eval episodes |
| `medium` | 500 | 50 | 10 | mean reward > −0.2 |
| `hard` | 1000 | 100 | 20 | — (terminal stage) |

```python
from quantflow.envs.curriculum import CurriculumWrapper, CurriculumCallback
from quantflow.envs.market_making import MarketMakingEnv

env = CurriculumWrapper(MarketMakingEnv(), auto_advance=True)
callback = CurriculumCallback(env, eval_freq=5_000)
```

### A/B Comparison (`python/quantflow/training/ab_curriculum.py`)

Runs a controlled experiment: no-curriculum (hard env from start) vs `CurriculumWrapper`. Both conditions train for the same total timesteps. Three evaluation regimes (STANDARD, HIGH-VOL, LOW-LIQ) measure robustness via mean Sharpe across regimes.

```bash
uv run python -m quantflow.training.ab_curriculum --timesteps 200000
```

---

## 13. Evaluation Framework (`python/quantflow/evaluation/`)

Three-stage pipeline comparing SAC against three fixed-action baselines.

### Strategies compared

| Agent | Action | Description |
|---|---|---|
| Naive Symmetric | γ=0.5, κ-offset=0 | Wide symmetric quotes, no inventory adaptation |
| Static AS | γ=0.1, κ-offset=0 | Conservative risk aversion, fixed parameters |
| Optimized AS | γ=0.05, κ-offset=0 | Best γ from grid search |
| SAC | deterministic policy | Trained agent |

### Stage 1 — `compare.py`

Runs all four strategies for N episodes, collects per-step trajectories, computes PnL decomposition:

```
inventory_pnl = final_inventory × (final_mid − initial_mid)   [price drift exposure]
spread_pnl    = total_pnl − inventory_pnl                      [spread capture revenue]
```

Saves `results.parquet` (episode summaries) and `trajectories.parquet` (per-step data).

### Stage 2 — `report.py`

Loads `results.parquet` and prints:
- Performance table: PnL±std, Sharpe±std, MaxDD, FillRate, InvStd, Quote-to-Trade
- PnL decomposition table: Spread PnL vs Inventory PnL per strategy
- SAC vs Optimized AS delta summary

### Stage 3 — `plots.py`

Six matplotlib plots saved as PNG (seaborn-v0_8-whitegrid style, 150 dpi):

| File | Description |
|---|---|
| `cumulative_pnl.png` | Median PnL curve + 25/75 percentile band per strategy |
| `pnl_distribution.png` | Box plot of episode-end PnL |
| `inventory_trajectories.png` | Inventory paths for one seed, all strategies |
| `sharpe_comparison.png` | Bar chart with ±1 std error bars |
| `fillrate_vs_risk.png` | Scatter: mean fill rate vs mean inventory std |
| `pnl_decomposition.png` | Stacked bar: spread PnL + inventory PnL |

### Running the full pipeline

```bash
bash scripts/run_evaluation.sh runs/sac_test/best_model.zip 50
# → results/evaluation/results.parquet
# → results/evaluation/trajectories.parquet
# → results/evaluation/report.txt
# → results/evaluation/plots/*.png
```

---

## Test suite

185 Rust tests + 94 Python tests across all modules:

```bash
# Rust
cargo test -p quantflow-core

# Python (requires maturin develop --release first)
uv run pytest tests/ -v
```

**Rust (214 tests)**

| Module | Tests | Coverage |
|---|---|---|
| `orderbook::types` | 11 | Price arithmetic, display, ordering, tick round-trips |
| `orderbook::book` | 14 | BTreeMap ordering, cancel pruning, modify guards |
| `orderbook::matching` | 23 | All order types, FIFO priority, partial maker retention, IOC/FOK |
| `data::lobster` | 8 | Row parsing, error paths, replayer state machine |
| `data::arrow_export` | 10 | Schema, row counts, value extraction, Parquet round-trip |
| `data::binance` | 8 | Serde deserialisation, parsed_bids/asks, is_buy, ws_url spot/futures |
| `data::market_event` | 4 | hawkes_dim roundtrip, out-of-range None, batch schema, Parquet round-trip |
| `data::normalizer` | 6 | Trade mapping, depth diffing, best/deep classification, cancel detection |
| `data::replay` | 7 | from_parquet, cursor advance, apply_event, mid_price, snapshot |
| `data::recorder` | 3 | File creation, message counting, Parquet readability |
| `hawkes::kernel` | 29 | Kernel evaluate/integral closed forms, branching ratio, stability |
| `hawkes::process` | 22 | Constructor validation, intensity, log-likelihood, simulation rate |
| `hawkes::calibration` | 16 | Gradient correctness, convergence, KS test, goodness-of-fit |
| `simulator` | 9 | Initialisation, event dispatch, U-shape, snapshot structure |
| `metrics::stylized_facts` | 11 | Kurtosis, ACF, spread, signature plot, intraday pattern |
| `strategy::avellaneda` | 25 | Quote formulas, σ auto-cal, spread floor, PnL decomposition, adverse selection, backtest fill/inventory/drawdown |
| Integration | 7 | End-to-end LOBSTER replay, final book state |

**Python (120 tests)**

| Module | Tests | Coverage |
|---|---|---|
| `tests/test_ffi.py` | 27 | PyOrderBook (empty, insert, cross, cancel, snapshot), PyHawkesSimulator (create, reset, step, day), PyAvellanedaStoikov (construction, quotes, repr) |
| `tests/test_env.py` | 18 | Gymnasium compliance, observation/action spaces, reset contract, fills, inventory bounds, termination, `gym.make` |
| `tests/test_features.py` | 49 | All 7 feature functions (edge cases, window clipping, tick rule), `compute_all`, `RunningNormalizer` (Welford convergence, serialisation, constant-feature stability) |
| `tests/test_domain_randomizer.py` | 13 | Parameter perturbation, stability clamping, seed reproducibility, obs-space passthrough |
| `tests/test_curriculum.py` | 13 | Initial stage, auto-advance easy→medium→hard, no advance beyond hard, info keys, `set_stage`, reward clamping, `auto_advance=False` |

---

## References

- Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book* — optimal quotes under inventory risk (HJB approach)
- Bacry, Mastromatteo & Muzy (2015) — *Hawkes processes in finance* — calibration of multivariate Hawkes to equity/FX microstructure data
- Cont (2001) — *Empirical properties of asset returns* — the six stylized facts
- Cont, Stoikov & Talreja (2010) — *A stochastic model for order book dynamics* — Markov chain model; priority rules §3.1, §3.3
- Gopikrishnan et al. (2000) — *Statistical properties of share volume traded in financial markets* — log-normal order size distribution
- Hawkes (1971) — *Spectra of some self-exciting and mutually exciting point processes* — original Hawkes process paper
- Admati & Pfleiderer (1988) — *A theory of intraday patterns* — U-shape volume concentration
- Ogata (1981) — *On Lewis' simulation method for point processes* — thinning algorithm
- Spooner et al. (2018) — *Market making via reinforcement learning* — RL for market making with asymmetric information

---

## Roadmap

- [x] BTreeMap-based LOB with FIFO price-time priority
- [x] Matching engine: Limit, Market, IOC, FOK
- [x] LOBSTER historical data parser and replayer
- [x] Apache Arrow / Parquet export
- [x] Hawkes excitation kernel library (Exponential, Power-law)
- [x] Multivariate Hawkes process with Ogata thinning
- [x] MLE calibration via L-BFGS + goodness-of-fit
- [x] 12-dimensional Hawkes-LOB simulator with intraday U-shape
- [x] Stylized facts validation
- [x] Avellaneda-Stoikov market-making baseline + backtest
- [x] σ auto-calibration from realised mid-price volatility (quadratic variation)
- [x] Spread floor, inventory-aware pre-fill breach check
- [x] Full diagnostic report: PnL decomposition, adverse selection (1s/5s/10s), quote statistics, Calmar ratio
- [x] Parameter grid search (γ × κ × inv_limit, rayon-parallelised, 5 seeds, top-N by Sharpe)
- [x] PyO3 FFI bindings: `PyOrderBook`, `PyHawkesSimulator`, `PyAvellanedaStoikov`, `calibrate_hawkes`, `load_lobster`
- [x] Zero-copy Arrow RecordBatch transfer via PyCapsule Interface
- [x] Gymnasium `MarketMakingEnv` — Dict obs space, AS-parameterised actions, fill detection, warm-up
- [x] LOB microstructure feature engineering (7 functions, `RunningNormalizer`, `compute_all`)
- [x] SAC training pipeline (SB3 `MultiInputPolicy`, auto entropy, W&B integration, best-model checkpointing)
- [x] Evaluation framework: 4-strategy comparison, Parquet persistence, 6 matplotlib plots
- [x] `InventoryMode` enum + `compute_quotes_skewed()` in `AvellanedaStoikov` — γ doubling, spread shift, suppress, dump
- [x] `tick_size_f` config for `PyHawkesSimulator` — decimal tick size from Python
- [x] FastAPI backend with WebSocket streaming (`/ws/live`) and REST control endpoints
- [x] Bloomberg-terminal Next.js dashboard (F1 Live / F2 Arena / F3 Metrics / F4 Simulator)
- [x] F1: real-time LOB depth chart, trade-flow scatter, stats panel, agent fill feed — all data from Rust engine
- [x] Reward v2: round-trip bonus, asymmetric inventory shaping, terminal penalty
- [x] Obs v2: 6 regime-detection features (OFI short/long, trade arrival rate, realised vol, spread percentile, fill imbalance)
- [x] Online reward normalisation (Welford's algorithm, persists across episodes, eval bypass)
- [x] LobFeatureExtractor: 1D-CNN over LOB depth + scalar MLP, replaces SB3 CombinedExtractor
- [x] SAC training hardening: linear LR decay 3e-4 → 5e-5, delayed target updates, 20 eval episodes with fixed seeds
- [x] Domain Randomization: Hawkes param perturbation at episode reset, stability clamping
- [x] Curriculum Learning: 3-stage easy/medium/hard, auto-advance on rolling reward threshold
- [x] Binance Futures WebSocket client — combined stream (depth@100ms / aggTrade / bookTicker)
- [x] ParquetRecorder — BatchBuffer flushing to Arrow Parquet (depth, trades, tickers)
- [x] Unified MarketEvent format — 12-type taxonomy aligned with Hawkes dimensions
- [x] BinanceNormalizer — depth-snapshot diffing → Limit/Cancel events, aggTrade → Market events
- [x] ReplayEngine — Parquet playback through OrderBook with FIFO quantity reduction
- [x] PyReplayEngine FFI — historical replay accessible from Python
- [ ] Calibrate Hawkes parameters to real Binance data; retrain on calibrated simulator
- [ ] PPO comparison vs SAC
- [ ] Multi-asset extension (correlated LOBs)

---

## License

MIT — Josef Dukarevic
