# rust-lob-rl-market-maker

A high-performance limit order book (LOB) engine written in Rust, with a LOBSTER data parser and Apache Arrow/Parquet export layer. The architecture is designed as the foundation for a reinforcement learning market-making agent: the Rust core handles matching and market data at low latency; a Python layer (via PyO3) will expose a Gymnasium-compatible environment for RL training.

## Repository layout

```
rust-lob-rl-market-maker/
├── crates/
│   ├── quantflow-core/          # Pure Rust library — LOB, data, serialization
│   │   ├── src/
│   │   │   ├── orderbook/
│   │   │   │   ├── types.rs     # Price, Quantity, OrderId, Side, OrderType, Timestamp
│   │   │   │   ├── order.rs     # Order, Trade, ExecutionReport, BookDelta
│   │   │   │   ├── book.rs      # OrderBook — BTreeMap-based, price-time priority
│   │   │   │   └── matching.rs  # Matching engine: Limit, Market, IOC, FOK
│   │   │   └── data/
│   │   │       ├── lobster.rs   # LOBSTER message & snapshot CSV parser
│   │   │       └── arrow_export.rs  # Arrow RecordBatch + Parquet export
│   │   ├── benches/
│   │   │   └── orderbook_bench.rs   # Criterion benchmarks
│   │   └── tests/
│   │       ├── lobster_replay.rs    # Integration tests against fixture CSVs
│   │       └── fixtures/
│   │           ├── sample_message.csv
│   │           └── sample_orderbook.csv
│   └── quantflow-ffi/           # PyO3 extension module (wip)
├── python/
│   └── quantflow/
│       └── __init__.py
└── pyproject.toml               # maturin build config
```

## Building

```bash
# Rust library + tests
cargo test -p quantflow-core

# Release build
cargo build -p quantflow-core --release

# Python extension (requires maturin)
maturin develop

# Criterion benchmarks
cargo bench --bench orderbook_bench -p quantflow-core
```

---

## Core: `quantflow-core`

### Price representation

All prices are stored as `i64` integer ticks with a fixed scale of `PRICE_SCALE = 100_000_000` (10⁸ ticks per dollar). This means $1.00 is represented as `100_000_000`, $99.90 as `9_990_000_000`, and so on.

**Why integer ticks?**
- Floating-point keys in a `BTreeMap` are unsound (`f64` does not implement `Ord`).
- Tick arithmetic is exact: `a + b - b == a` always holds, unlike float arithmetic.
- The fixed scale gives 8 decimal places of precision — more than sufficient for equities and most derivatives.

```rust
let p = Price::from_f64(100.25);  // → Price(10_025_000_000)
let q = Price::from_ticks(100_000_000);  // → $1.00
assert_eq!((p + q).to_f64(), 101.25);
```

`Price::from_f64` is intentionally labelled "lossy" in the docs and is for configuration and testing only. Hot-path code uses tick arithmetic directly.

---

### Order book (`orderbook/book.rs`)

`OrderBook` is a central limit order book backed by two `BTreeMap`s:

```
bids: BTreeMap<Reverse<Price>, VecDeque<Order>>   — highest price first
asks: BTreeMap<Price, VecDeque<Order>>            — lowest price first
order_index: HashMap<OrderId, (Side, Price)>      — O(1) cancel/modify lookup
```

Each price level holds a `VecDeque<Order>`. The front of the deque is the oldest (highest-priority) order at that level — FIFO price-time priority.

#### Public API

| Method | Description |
|---|---|
| `add_limit_order(order)` | Route through the matching engine; resting orders are inserted. |
| `cancel_order(id)` | O(1) index lookup → O(log L) BTreeMap → O(Q) deque scan. |
| `modify_order(id, new_qty)` | Downward modification only — in-place, no priority loss. |
| `best_bid()` / `best_ask()` | First entry of each BTreeMap; O(1). |
| `mid_price()` / `spread()` | Derived from best bid/ask; O(1). |
| `depth(side, levels)` | Aggregated `(price, total_qty)` per level. |
| `snapshot(levels)` | `OrderBookSnapshot` with bids, asks, and timestamp. |

**Cancel complexity note:** The `order_index` gives O(1) lookup of `(Side, Price)`, reducing the BTreeMap traversal to O(log L) where L is the number of price levels. Finding the order within a level's `VecDeque` is O(Q) where Q is the queue depth at that price. For a well-distributed book this is typically small; for a concentrated book with thousands of orders per level it becomes the bottleneck (see benchmarks).

**Modify semantics:** Only downward quantity adjustment is supported. Increasing quantity would require re-queuing the order at the back of its level, which is a de-facto cancel+replace and would lose time priority. This matches the convention described in Cont, Stoikov & Talreja (2010) §3.3.

---

### Matching engine (`orderbook/matching.rs`)

The matching engine implements four order types:

#### Limit
Match against the opposite side at the limit price or better. Residual rests on the book.

```
incoming bid @ 100.05
  book asks: [100.00 × 50], [100.10 × 80]

→ match 50 lots @ 100.00 (ask price, not bid price)
→ residual bid of (qty − 50) rests at 100.05
```

Execution price is always the **resting (maker) order's price**, not the aggressor's limit — consistent with continuous double auction mechanics.

#### Market
Match at any available price; any unfilled residual is cancelled silently (never rests).

#### IOC (Immediate-or-Cancel)
Identical to a limit order except the residual is cancelled rather than rested. Useful for aggressive quotes that must not leave unintended resting exposure.

#### FOK (Fill-or-Kill)
A pre-validation pass (`can_fill_fully`) walks the resting liquidity at the limit price to check whether the entire order can be filled **before** any state mutation. If not, the order is rejected atomically — zero trades, zero book impact.

```rust
fn can_fill_fully(book: &OrderBook, incoming: &Order) -> bool {
    let available: u64 = book.asks.iter()
        .take_while(|(p, _)| **p <= incoming.price)
        .flat_map(|(_, level)| level.iter())
        .map(|o| o.quantity.0)
        .sum();
    available >= incoming.quantity.0
}
```

#### Partial fills and maker priority

When a maker order is partially consumed, it is re-inserted at the **front** of its price level (`insert_resting_front`) so it retains its original time priority for subsequent sweeps. This is the standard continuous auction rule (Cont et al., 2010 §3.1).

```
level @ 100.00: [maker_A × 30, maker_B × 50]

incoming market buy × 20
→ fill 20 from maker_A (30 → 10)
→ re-insert maker_A × 10 at FRONT
→ level is now: [maker_A × 10, maker_B × 50]
```

#### ExecutionReport

Every `add_limit_order` call returns an `ExecutionReport`:

```rust
pub struct ExecutionReport {
    pub trades: Vec<Trade>,         // fills produced, in time order
    pub remaining: Option<Order>,   // resting residual (None = fully filled or cancelled)
}
```

---

### Types (`orderbook/types.rs`, `orderbook/order.rs`)

| Type | Representation | Notes |
|---|---|---|
| `Price` | `i64` ticks | Arithmetic via `Add`/`Sub`; display as `"100.05000000"` |
| `Quantity` | `u64` lots | Integer lots; no fractional shares |
| `OrderId` | `u64` | Caller-assigned; must be unique within a book |
| `Timestamp` | `u64` ns | Nanoseconds since epoch |
| `Side` | enum | `Bid` / `Ask`; `opposite()` helper |
| `OrderType` | enum | `Limit` / `Market` / `ImmediateOrCancel` / `FillOrKill` |
| `Trade` | struct | `(price, quantity, maker_id, taker_id, timestamp)` |
| `BookDelta` | enum | `Add` / `Remove` / `Modify` / `Trade` — incremental update stream |

`BookDelta` is designed for downstream consumers (feed handlers, replayers) that need to reconstruct book state from a delta stream without holding a full snapshot.

---

### LOBSTER parser (`data/lobster.rs`)

[LOBSTER](https://lobsterdata.com) is a widely-used source of historical limit order book data for US equities (NASDAQ). The format consists of two paired CSV files per symbol per day: a **message file** and an **order book file**.

#### Message file format

```
Time, Type, OrderID, Size, Price, Direction
34200.005, 1, 42, 100, 10010000, 1
```

| Column | Type | Notes |
|---|---|---|
| Time | `f64` | Seconds since midnight |
| Type | `i32` | See event types below |
| OrderID | `u64` | |
| Size | `u64` | For Type 2: reduction amount; for Type 4/5: executed shares |
| Price | `i64` | Raw integer (scale depends on provider; typically 10⁻⁴) |
| Direction | `i8` | `1` = buy (bid), `−1` = sell (ask) |

**Event types:**

| Code | Variant | Meaning |
|---|---|---|
| 1 | `NewLimitOrder` | New resting limit order added to the book |
| 2 | `PartialCancel` | Size reduced by `size` field (not set to `size`) |
| 3 | `FullCancel` | Order fully removed |
| 4 | `ExecutionVisible` | Visible resting order executed; `size` = executed shares |
| 5 | `ExecutionHidden` | Hidden/iceberg order executed |
| 7 | `TradingHalt` | Market-wide halt; no book state change |

#### Order book file format

```
AskPrice1, AskSize1, BidPrice1, BidSize1, AskPrice2, AskSize2, BidPrice2, BidSize2, ...
```

One row per event, L levels per side. Column count = 4 × L.

#### Streaming API

Both files are read lazily — no full-file allocation:

```rust
use quantflow_core::data::lobster::{replay_messages, load_snapshots};

// Streaming message iterator
for result in replay_messages(Path::new("AAPL_2012-06-21_34200000_57600000_message_10.csv"))? {
    let msg = result?;
    println!("{:?} order {} @ {}", msg.event_type, msg.order_id, msg.price);
}

// Streaming snapshot iterator
for result in load_snapshots(Path::new("AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"))? {
    let snap = result?;
    let l0 = &snap.levels[0];
    println!("best ask {} × {}", l0.ask_price, l0.ask_size);
}
```

#### LobsterReplayer

`LobsterReplayer` applies a message stream to a live `OrderBook`, maintaining an internal `sizes: HashMap<u64, u64>` to resolve whether a Type 4/5 execution fully consumed a resting order or left a partial:

```rust
use quantflow_core::data::lobster::LobsterReplayer;

// price_scale: how many file-ticks equal $1.00 in the source data.
// Standard LOBSTER = 10_000 (4 decimal places).
// Prices already in PRICE_SCALE = 100_000_000: pass 100_000_000.
let mut r = LobsterReplayer::new(10_000);

for result in replay_messages(&msg_path)? {
    r.apply(&result?);
}

let snap = r.book.snapshot(5);
println!("mid = {}", r.book.mid_price().unwrap());
```

**Design decision — bypass matching on Type 1:** When inserting a `NewLimitOrder` event during replay, the replayer calls `OrderBook::insert_resting` directly rather than `add_limit_order`. This bypasses the matching engine because, in LOBSTER data, any executions caused by the new order are already represented as separate Type 4/5 events. Running the matching engine on insert would cause double-matching and incorrect book state.

**Price scale conversion:**

```rust
// LOBSTER native: price=10_025_000 means $1002.50 (4dp scale)
// to_price(10_000) → 10_025_000 * (10^8 / 10_000) = 10_025_000 * 10_000 = 100_250_000_000 ticks
let internal_price = msg.to_price(10_000);
assert_eq!(internal_price.to_f64(), 1002.50);
```

---

### Arrow & Parquet export (`data/arrow_export.rs`)

`OrderBookSnapshot` and `Trade` slices can be serialized to Apache Arrow `RecordBatch`es and written to Parquet.

#### Snapshot schema — row-per-level layout

```
timestamp  UInt64   nanoseconds since epoch
level      Int32    1-based depth level (1 = best)
bid_price  Int64    ticks (PRICE_SCALE = 10^8)
bid_qty    UInt64   lots
ask_price  Int64    ticks
ask_qty    UInt64   lots
```

A snapshot of depth L produces L rows. Multiple snapshots in one call are concatenated into a single batch — efficient for time-series bulk export.

```rust
use quantflow_core::data::arrow_export::snapshots_to_parquet;

// Collect snapshots during a simulation or replay
let snapshots: Vec<OrderBookSnapshot> = /* ... */;
snapshots_to_parquet(&snapshots, Path::new("book_depth.parquet"))?;
```

#### Trade schema

```
timestamp  UInt64
price      Int64
quantity   UInt64
maker_id   UInt64
taker_id   UInt64
```

```rust
use quantflow_core::data::arrow_export::trades_to_parquet;

trades_to_parquet(&trades, Path::new("executions.parquet"))?;
```

#### Zero-copy notes

The Arrow arrays are built from `Vec<i64>` / `Vec<u64>` buffers via `Int64Array::from` / `UInt64Array::from`. This allocates one contiguous Arrow buffer per column but involves no intermediate copies beyond the initial extraction from the `(Price, Quantity)` tuples. True zero-copy would require the source data to already be in columnar Arrow layout — achievable if the book is redesigned as a struct-of-arrays rather than the current array-of-structs VecDeque model.

---

### Benchmarks (`benches/orderbook_bench.rs`)

Five Criterion benchmarks measure individual operations and aggregate throughput. Run with:

```bash
cargo bench --bench orderbook_bench -p quantflow-core
```

| Benchmark | Setup | Target |
|---|---|---|
| `insert_limit_order_no_match` | 10,000-order book | < 500 ns |
| `cancel_order_mid_queue` | 10,000-order book; cancel at queue position ~125 | < 200 ns |
| `market_order_5_level_sweep` | 100 levels × 10 lots; buy 50 sweeps exactly 5 asks | < 1 µs |
| `snapshot_10_levels` | 100-level book; shared across iterations (read-only) | < 10 µs |
| `throughput/1M_mixed_70ins_20can_10mkt` | Seeded 10k-order book; 1 M ops | ops/sec |

The throughput benchmark uses `Criterion::Throughput::Elements(1_000_000)` so the output is directly in operations/second. `sample_size(10)` keeps wall time under ~30 seconds.

**Cancel benchmark note:** `cancel_mid_queue` is designed to exercise the realistic case where the target sits at queue position ~125 in a 500-deep price level. The linear scan cost is O(Q/2) and is currently the dominant term for deep, concentrated levels — the `order_index` eliminates the BTreeMap traversal but cannot avoid the deque search. Replacing `VecDeque<Order>` with an `IndexMap` or a slab-allocated linked list would reduce this to O(1) at the cost of higher constant factors for insert.

---

## Test suite

72 tests across unit, integration, and property levels:

```
cargo test -p quantflow-core
```

| Module | Tests | What they cover |
|---|---|---|
| `orderbook::types` | 11 | Price arithmetic, display, ordering, tick round-trips |
| `orderbook::book` | 14 | BTreeMap ordering, cancel pruning, modify guards, depth aggregation |
| `orderbook::matching` | 23 | All 7 specified scenarios + FIFO priority, partial maker retention, IOC/FOK edge cases |
| `data::lobster` | 8 | Row parsing, error paths, replayer state machine |
| `data::arrow_export` | 10 | Schema correctness, row counts, value extraction, Parquet round-trip |
| `tests/lobster_replay` | 8 | End-to-end fixture replay, final book state, spread computation, snapshot file parsing |

**Key integration test — `replay_sample_messages_final_state`:**

The fixture `sample_message.csv` encodes a 10-event sequence exercising every event type (NewLimit, PartialCancel, FullCancel, ExecutionVisible). After replay, the test verifies the exact tick-level book state:

```
bids: [(99.80, 100), (99.75, 75)]    ← id=3 partial-cancelled; id=6 new
asks: [(100.05, 100), (100.20, 80)]  ← id=5 partially executed; id=4 untouched
spread = 100.05 − 99.80 = 0.25
```

---

## Design decisions and references

**Fixed-point prices** — Integer tick arithmetic is the standard approach in production exchange systems. Float keys in ordered maps are undefined behaviour in Rust and produce incorrect results in C++ (NaN ordering). The `i64` representation with `PRICE_SCALE = 10^8` allows 8 decimal places and supports negative prices (synthetic instruments, spreads).

**BTreeMap + VecDeque** — A `BTreeMap` gives O(log L) level access with minimal memory overhead. Within each level, a `VecDeque` gives O(1) front-pop (the hot path for matching) and O(1) push-back (new arrivals). Cancel requires a linear scan; acceptable for typical level depths, but the `order_index` means we only scan a single level rather than the whole side.

**No matching on LOBSTER replay** — Type 1 events in LOBSTER represent orders that were placed, not orders that were immediately executed. Executions appear as Type 4/5 events on the passive side. Routing Type 1 through the matching engine would double-count fills.

**Row-per-level Arrow schema** — Columnar analytics (e.g., "what is the average bid depth at level 3 during the first hour?") are most efficiently expressed over a row-per-level layout. An alternative row-per-snapshot layout would require array columns which are less friendly to SQL engines and Polars/pandas group-by operations.

**References:**
- Avellaneda & Stoikov (2008) — optimal market-making quotes under inventory risk
- Cont, Stoikov & Talreja (2010) — Markov chain model of LOB dynamics; priority rules §3.1, §3.3
- Ogata (1981) — thinning algorithm for Hawkes process simulation (planned: `sim/hawkes.rs`)
- Spooner et al. (2018) — RL for market making with asymmetric information

---

## Roadmap

- [ ] `sim/hawkes.rs` — multivariate Hawkes process for synthetic order arrival
- [ ] `sim/market.rs` — `MarketSimulator::step(dt)` returning `Vec<Event>`
- [ ] `agent/interface.rs` — observation/action types for Python FFI
- [ ] `python/env.py` — Gymnasium `MarketMakerEnv`
- [ ] `python/agent.py` — SB3 PPO/SAC wrappers
- [ ] PyO3 bindings for `OrderBook`, `MarketSimulator`

---

## License

MIT — Josef Dukarevic
