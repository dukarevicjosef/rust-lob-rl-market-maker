# rust-lob-rl-market-maker

A high-performance limit order book (LOB) engine written in Rust, with a full quantitative market-microstructure stack: a Hawkes process library for realistic order-flow simulation, a Hawkes-driven LOB simulator, and an Avellaneda-Stoikov market-making strategy with backtest. The architecture is designed as the foundation for a reinforcement learning market-making agent: the Rust core handles matching and market data at low latency; a Python layer (via PyO3) will expose a Gymnasium-compatible environment for RL training.

## Repository layout

```
rust-lob-rl-market-maker/
├── crates/
│   ├── quantflow-core/
│   │   └── src/
│   │       ├── orderbook/
│   │       │   ├── types.rs         # Price, Quantity, OrderId, Side, OrderType, Timestamp
│   │       │   ├── order.rs         # Order, Trade, ExecutionReport, BookDelta
│   │       │   ├── book.rs          # OrderBook — BTreeMap-based, price-time priority
│   │       │   └── matching.rs      # Limit, Market, IOC, FOK matching engine
│   │       ├── data/
│   │       │   ├── lobster.rs       # LOBSTER message & snapshot CSV parser
│   │       │   └── arrow_export.rs  # Arrow RecordBatch + Parquet export
│   │       ├── hawkes/
│   │       │   ├── kernel.rs        # ExponentialKernel, PowerLawKernel, ExcitationKernel trait
│   │       │   ├── process.rs       # MultivariateHawkes — Ogata thinning simulation
│   │       │   └── calibration.rs   # MLE via L-BFGS, goodness-of-fit, KS test, Q-Q data
│   │       ├── simulator.rs         # HawkesLobSimulator — 12D Hawkes → LOB event dispatch
│   │       ├── metrics/
│   │       │   └── stylized_facts.rs # Return kurtosis, ACF, spread dist, signature plot, intraday
│   │       └── strategy/
│   │           └── avellaneda.rs    # Avellaneda-Stoikov (2008) + BacktestResult
│   └── quantflow-ffi/               # PyO3 extension module (wip)
├── scripts/
│   └── plot_qq.py                   # Q-Q plot visualisation (matplotlib)
└── pyproject.toml
```

## Building

```bash
# Full test suite (178 tests)
cargo test -p quantflow-core

# Release build
cargo build -p quantflow-core --release

# Q-Q plot: simulate, calibrate, visualise goodness-of-fit
cargo run -p quantflow-core --example hawkes_qq | python3 scripts/plot_qq.py

# Criterion benchmarks
cargo bench --bench orderbook_bench -p quantflow-core
```

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

### Practical parameter estimation

| Parameter | `estimate_sigma` | Realised volatility from recent trade prices |
|---|---|---|
| κ | `estimate_kappa` | (Depth at best bid + depth at best ask) / (2 × spread) |

The σ estimator uses log-returns of consecutive trades:
```
σ̂ = std_dev{log(p_k / p_{k-1}) : k = last_window trades}
```

The κ estimator is a heuristic: a deep, tight book implies rapid fill arrival near the touch. In production, κ would be calibrated from historical fill data by fitting the fill rate as a function of quote offset.

### Backtest mechanics

The backtest runs against the Hawkes LOB simulator, refreshing quotes every second of simulation time:

1. **Fill detection:** After each market event, check whether any of the agent's resting orders appear as `maker_id` in the returned trades. If so, update cash and inventory.

2. **Quote refresh:** Cancel the current bid/ask, compute new quotes via the AS formula, and place fresh limit orders — provided the inventory limit `|q| + quote_qty ≤ inventory_limit` is satisfied.

3. **Mark-to-market PnL:** At each book snapshot, compute `cash + inventory × mid_price`. This is the theoretical liquidation value assuming the agent could close the position at mid.

### Performance metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Sharpe ratio** | `mean(ΔPnL) / std(ΔPnL) · √n` | Risk-adjusted return; > 1.0 indicates a viable strategy |
| **Max drawdown** | `max(peak − current PnL)` | Worst peak-to-trough loss; measures tail risk |
| **Fill rate** | `fills / quotes_placed` | Fraction of placed orders that were executed; low rate = quotes too far from mid |

```rust
use quantflow_core::strategy::AvellanedaStoikov;
use quantflow_core::simulator::HawkesLobSimulator;

let mut sim = HawkesLobSimulator::default_12d().unwrap();
let strat = AvellanedaStoikov::default_params(sim.config.t_max);
let result = strat.run_backtest(&mut sim, /*seed=*/ 42);

println!("Final PnL:    {:.2}", result.final_pnl());
println!("Sharpe:       {:.3}", result.sharpe);
println!("Max drawdown: {:.2}", result.max_drawdown);
println!("Fill rate:    {:.1}%", result.fill_rate * 100.0);
println!("Max inventory: {}", result.max_inventory());
```

---

## 6. Limit Order Book Engine (`orderbook/`)

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

## Test suite

178 tests across all modules:

```bash
cargo test -p quantflow-core
```

| Module | Tests | Coverage |
|---|---|---|
| `orderbook::types` | 11 | Price arithmetic, display, ordering, tick round-trips |
| `orderbook::book` | 14 | BTreeMap ordering, cancel pruning, modify guards |
| `orderbook::matching` | 23 | All order types, FIFO priority, partial maker retention, IOC/FOK |
| `data::lobster` | 8 | Row parsing, error paths, replayer state machine |
| `data::arrow_export` | 10 | Schema, row counts, value extraction, Parquet round-trip |
| `hawkes::kernel` | 29 | Kernel evaluate/integral closed forms, branching ratio, stability |
| `hawkes::process` | 22 | Constructor validation, intensity, log-likelihood, simulation rate |
| `hawkes::calibration` | 16 | Gradient correctness, convergence, KS test, goodness-of-fit |
| `simulator` | 9 | Initialisation, event dispatch, U-shape, snapshot structure |
| `metrics::stylized_facts` | 11 | Kurtosis, ACF, spread, signature plot, intraday pattern |
| `strategy::avellaneda` | 18 | Quote formulas, σ/κ estimation, backtest fill/inventory/drawdown |
| Integration | 7 | End-to-end LOBSTER replay, final book state |

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
- [ ] PyO3 bindings: `PyOrderBook`, `PyHawkesLobSimulator`
- [ ] `python/env.py` — Gymnasium `MarketMakerEnv`
- [ ] `python/agent.py` — SB3 PPO/SAC wrappers
- [ ] RL training loop

---

## License

MIT — Josef Dukarevic
