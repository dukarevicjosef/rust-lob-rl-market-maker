# QuantFlow

**High-Performance Market Making with Deep Reinforcement Learning**

Rust LOB Engine · Hawkes Process Simulator · SAC Agent · Calibrated on Binance BTCUSDT · Live Paper Trading

A full-stack market microstructure engine: a sub-microsecond Rust order book at the core, a 12-dimensional Hawkes process calibrated on real Binance Futures data driving the simulator, and a SAC agent that learns to control Avellaneda-Stoikov quote parameters under inventory risk. The system runs end-to-end — from LOB matching to live paper trading on Binance Testnet.

---

## Key Results

| Metric | Value |
|--------|-------|
| LOB insert latency | 778 ns |
| LOB throughput | 20.3M ops/sec |
| SAC vs. AS baseline | +1,428 PnL (ΔSharpe +0.90) |
| Hawkes calibration | 10/12 dims active, all branching ratios < 1.0 |
| Paper trading (10 min Testnet) | +0.62 USDT, 95 fills, kill switch: not triggered |
| Training convergence | Positive PnL at 130K steps (with reward normalization) |

---

## Architecture

```
Binance WebSocket ──► Rust LOB Engine ──► SAC Agent (PyTorch)
        │                    │                    │
   aggTrade stream      Hawkes Simulator     Risk Manager
   depth20@100ms        (Ogata thinning)     (7 checks + kill switch)
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                    Paper Trading Runner
                    (asyncio, 5 loops)
                             │
                    Next.js Dashboard
                    (live LOB, trade feed)
```

| Layer | Technologies |
|-------|-------------|
| **Rust** | LOB engine, Hawkes simulator, data pipeline, Risk Manager |
| **Python** | SAC (SB3), Gymnasium environment, Hawkes calibration, evaluation |
| **TypeScript** | Next.js 14 dashboard, TradingView-style charts |
| **Bridge** | PyO3 + Apache Arrow (zero-copy IPC) |

---

## Features

**Systems Engineering**
- Sub-microsecond LOB with price-time priority (`BTreeMap<u64, VecDeque<Order>>`)
- 12-dimensional Hawkes process simulator with Ogata thinning (Ogata 1981)
- Zero-copy Rust↔Python bridge via PyO3 + Apache Arrow
- Binance WebSocket data pipeline (Depth, AggTrades → Parquet)

**Machine Learning**
- SAC agent controlling Avellaneda-Stoikov parameters (γ, κ) per step
- CNN feature extractor for LOB state (`Conv1d` over 5 bid + 5 ask levels)
- Reward v2: round-trip spread-capture bonus, asymmetric inventory penalties, terminal penalties
- Online reward normalization (Welford), LR decay schedule, delayed target updates
- Systematic A/B testing framework (4 experiments, W&B tracked)

**Quantitative Finance**
- Hawkes MLE calibration on real BTCUSDT Perpetuals data (L-BFGS)
- Stylized facts validation: spread distribution, return ACF, intraday intensity, Q-Q
- Avellaneda-Stoikov baseline with grid-searched optimal (γ, κ)
- Cross-excitation analysis of bid/ask order flow dynamics

**Production**
- Paper trading on Binance Futures Testnet with live REST order execution
- Risk Manager: 7 pre-trade checks (size, position, notional, daily loss, drawdown, rate, open orders)
- Safety rules: inventory hard/soft limits, quote pull on mid movement, volatility regime spread widening
- Bloomberg-terminal-style Next.js dashboard with live LOB depth visualization

---

## Dashboard

![Live Trading](docs/screenshots/live_trading.png)
*SAC Agent live on simulated BTCUSDT — LOB depth, price chart, agent state, trade feed*

![Replay Mode](docs/screenshots/replay_mode.png)
*Replaying real Binance BTCUSDT data through the Rust LOB engine*

---

## Training Progression

| Phase | Config | Best Eval PnL | Steps to Positive |
|-------|--------|---------------|-------------------|
| Baseline | Reward v1, no normalization | +920 @ 1.06M | ~1,000,000 |
| + Reward v2 | RT bonus, asymmetric penalty | +462 @ 130K | ~130,000 |
| + Normalization | Welford online, clip=10 | +462 @ 130K | ~130,000 |
| + LR Decay + CNN | 3e-4→5e-5, LOB Conv1d | +249 @ 910K | ~910,000 |
| + BTCUSDT Calibration | Real market Hawkes parameters | +249 @ 910K | ~910,000 |
| + Safety Rules | Inventory limits, quote pull, vol regime | +84.6 mean | immediate |

---

## Hawkes Calibration

Calibrated on 1h of Binance BTCUSDT Perpetuals (2026-04-02):

- 10/12 dimensions active (Limit Bid/Ask, Cancel Bid/Ask, Market Buy/Sell + cross-excitation)
- All branching ratios < 1.0 — subcritical process, stable simulation
- Spread median: 0.10 USD (empirical) vs. 0.10 USD (simulated) ✓
- Event rate: 282/s (empirical) vs. 257/s (simulated) ✓

Stylized facts plots: [`results/calibration/`](results/calibration/)

---

## Paper Trading Results

10-Minute Testnet Session (2026-04-05, BTCUSDT @ ~$67,400):

| Metric | Value |
|--------|-------|
| Total fills | 95 |
| Realized PnL | +0.62 USDT |
| Final position | 0.0000 BTC (flat) |
| Kill switch | Not triggered |
| Max inventory | 0.006 BTC |
| Avg spread captured | ~8 USD / round-trip |

---

## Quick Start

```bash
# Clone
git clone https://github.com/dukarevicjosef/rust-lob-rl-market-maker.git
cd rust-lob-rl-market-maker

# Build Rust extension + install Python deps
uv sync
maturin develop --release

# Run dashboard
cd dashboard && npm install && npm run dev

# Train agent
uv run python -m quantflow.training.train \
  --timesteps 1000000 \
  --hawkes-params data/btcusdt/calibration/hawkes_params.json \
  --wandb --wandb-project quantflow-mm

# Paper trade on Binance Testnet
cp .env.example .env   # fill in testnet API keys
uv run python -m quantflow.paper_trading \
  --model runs/sac_1M_btcusdt/best_model.zip \
  --duration 3600
```

---

## Project Structure

```
crates/quantflow-core/     # Rust: LOB engine, Hawkes simulator, data pipeline, Risk Manager
crates/quantflow-ffi/      # PyO3 bindings (maturin)
python/quantflow/
  ├── envs/                # Gymnasium environment (v1/v2 obs, reward, safety rules)
  ├── training/            # SAC training, A/B experiments, callbacks
  ├── calibration/         # Hawkes MLE, stylized facts, goodness-of-fit
  ├── evaluation/          # Multi-strategy comparison, Sharpe/drawdown reporting
  └── paper_trading/       # Live Binance paper trading runner
dashboard/                 # Next.js 14 + TradingView-style charts
data/btcusdt/              # Recorded market data + calibration output
results/                   # Evaluation plots, calibration figures
```

---

## Tech Stack

- **Rust**: LOB engine, Hawkes simulator, Binance WS connector, Risk Manager, Parquet pipeline
- **Python**: PyTorch, Stable-Baselines3, Gymnasium, Polars, SciPy, wandb
- **TypeScript**: Next.js 14, Tailwind CSS, shadcn/ui, Recharts
- **Bridge**: PyO3, Apache Arrow IPC (zero-copy array transfer)
- **Infra**: Weights & Biases (experiment tracking), Binance Futures WebSocket/REST

---

## Known Limitations & Roadmap

**Known Limitations**
- Exponential Hawkes kernels underfit volatility clustering (power-law kernels planned)
- Calibration on crash-period data distorts branching ratios
- Paper trading tested on Testnet only — Testnet liquidity is thinner than Mainnet
- No systematic hyperparameter optimization

**Roadmap**
- [ ] Optuna hyperparameter sweep
- [ ] Multi-seed ensemble policies
- [ ] Dashboard paper trading mode integration
- [ ] CI/CD pipeline (GitHub Actions: `cargo test` + `pytest` + `maturin`)
- [ ] Live trading with real capital (Binance Mainnet)

---

**Built by Josef Dukarevic** — IT Consultant & Private Trader, targeting quantitative finance roles in Frankfurt/Amsterdam.

[W&B Project](https://wandb.ai/dukarevicjosef-self-imployed/quantflow-mm)
